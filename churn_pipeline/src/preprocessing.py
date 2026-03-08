import os
import sys
import glob
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config as cfg
from src import db_connection as db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def preprocess(fetch_all: bool = False,
               scaler: StandardScaler = None,
               label_encoders: dict = None):
    """
    Fetch data from MySQL, clean, encode, scale.

    Args:
        fetch_all      : True  → fetch all processed records (full retrain)
                         False → fetch only unprocessed records (incremental)
        scaler         : Pass the active model's saved scaler for incremental
                         runs so new data lands on the same scale the model
                         was trained on. If None, a fresh scaler is fitted
                         (correct only for initial full train).
        label_encoders : Pass the active model's saved encoders for incremental
                         runs so categories map to the same integer codes.
                         If None, fresh encoders are fitted.

    Returns:
        X_scaled        : np.ndarray — scaled feature matrix
        feature_columns : list[str]  — column names in order
        scaler          : StandardScaler used (reused or freshly fitted)
        raw_ids         : list[int]  — DB row IDs to mark as processed
        label_encoders  : dict[str, LabelEncoder]

    FIX — CRITICAL:
        The old version always called scaler.fit_transform(df) regardless
        of whether a scaler was passed in. For incremental runs this fitted
        a fresh scaler on only the new batch — a completely different scale
        than the IsolationForest was trained on. The model received
        correctly-shaped but wrongly-scaled inputs and produced garbage
        risk scores with no error raised anywhere.

        Fix: if a scaler is passed in, call transform() only — never
        fit_transform(). The caller (scheduler/train.py) is responsible
        for passing the saved scaler from the active model package on
        incremental runs, and passing None only on a full initial train.
    """
    logger.info("=" * 50)
    logger.info("  PREPROCESSING STARTED")
    logger.info("=" * 50)

    # ── Step 1: Fetch ─────────────────────
    if fetch_all:
        df = db.fetch_all_processed_customers()
    else:
        df = db.fetch_unprocessed_customers()

    if df is None or df.empty:
        logger.warning("⚠️  No data to process.")
        return None, None, None, None, None

    logger.info(f"✅ Fetched {len(df)} records.")
    raw_ids = df["id"].tolist() if "id" in df.columns else []

    # ── Step 2: Drop system columns ───────
    drop = [c for c in ["id", "is_processed", "created_at"] if c in df.columns]
    df   = df.drop(columns=drop)

    # ── Step 3: Validate columns ──────────
    missing = [c for c in cfg.FEATURE_COLUMNS if c not in df.columns]
    if missing:
        logger.error(f"❌ Missing feature columns: {missing}")
        return None, None, None, None, None

    df = df[cfg.FEATURE_COLUMNS].copy()

    # ── Step 4: Fix TotalCharges ──────────
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_bad = df["TotalCharges"].isna().sum()
    if n_bad > 0:
        median = df["TotalCharges"].median()
        logger.warning(
            f"⚠️  {n_bad} rows had unparseable TotalCharges — "
            f"filling with median ({median:.2f})."
        )
        df["TotalCharges"] = df["TotalCharges"].fillna(median)

    # ── Step 5: Fill remaining nulls ──────
    for col in cfg.CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    for col in cfg.NUMERICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # ── Step 6: Encode categoricals ───────
    # FIX: if label_encoders are passed in, reuse them so each category
    # maps to the same integer the model was trained on.
    # Unknown values that weren't in training get -1 (safe sentinel).
    # Only fit fresh encoders when no saved encoders are provided
    # (i.e. first-time full train only).
    reusing_encoders = label_encoders is not None and len(label_encoders) > 0
    fitted_encoders  = {}

    for col in cfg.CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue

        if reusing_encoders and col in label_encoders:
            le = label_encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda v, _le=le: (
                    int(_le.transform([v])[0]) if v in _le.classes_ else -1
                )
            )
            fitted_encoders[col] = le
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            fitted_encoders[col] = le

    if reusing_encoders:
        logger.info(f"✅ Reused {len(fitted_encoders)} saved label encoders.")
    else:
        logger.info(f"✅ Fitted {len(fitted_encoders)} new label encoders.")

    # ── Step 7: Scale ─────────────────────
    # FIX: if a scaler is passed in, call transform() not fit_transform().
    # fit_transform() on a small incremental batch produces a completely
    # different scale than the full training set — model inputs become
    # meaningless and scores are silently wrong.
    if scaler is not None:
        X_scaled = scaler.transform(df)
        logger.info("✅ Applied existing scaler (transform only — scale preserved).")
    else:
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        logger.info("✅ Fitted new StandardScaler on full dataset.")

    # ── Step 8: Save processed CSV ────────
    # FIX: old version saved dated CSVs forever — unbounded disk growth.
    # Now keeps only the last 7 days and deletes older files.
    os.makedirs(cfg.DATA_PROC_DIR, exist_ok=True)
    today     = datetime.now().strftime("%Y_%m_%d")
    save_path = os.path.join(cfg.DATA_PROC_DIR, f"processed_{today}.csv")
    pd.DataFrame(X_scaled, columns=cfg.FEATURE_COLUMNS).to_csv(save_path, index=False)
    _cleanup_old_csvs(cfg.DATA_PROC_DIR, keep_days=7)
    logger.info(f"✅ Saved: {save_path}")

    logger.info("=" * 50)
    logger.info(f"  PREPROCESSING COMPLETE — {len(df)} records ready")
    logger.info("=" * 50)

    return X_scaled, cfg.FEATURE_COLUMNS, scaler, raw_ids, fitted_encoders


def _cleanup_old_csvs(directory: str, keep_days: int = 7):
    """Delete processed_YYYY_MM_DD.csv files older than keep_days."""
    cutoff = datetime.now() - timedelta(days=keep_days)
    for path in glob.glob(os.path.join(directory, "processed_*.csv")):
        fname = os.path.basename(path)
        try:
            date_str  = fname.replace("processed_", "").replace(".csv", "")
            file_date = datetime.strptime(date_str, "%Y_%m_%d")
            if file_date < cutoff:
                os.remove(path)
                logger.info(f"🗑️  Deleted old processed file: {fname}")
        except ValueError:
            pass  # filename doesn't match pattern — leave it alone


if __name__ == "__main__":
    result = preprocess(fetch_all=False)
    if result[0] is not None:
        X, cols, scaler, ids, encoders = result
        print(f"\nShape    : {X.shape}")
        print(f"Features : {cols}")
        print(f"Encoders : {list(encoders.keys())}")
        print(f"IDs      : {len(ids)} records to mark processed")