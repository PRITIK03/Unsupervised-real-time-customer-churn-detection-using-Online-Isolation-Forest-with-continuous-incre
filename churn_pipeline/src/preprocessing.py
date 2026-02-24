import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─────────────────────────────────────────
#  PATH FIX
# ─────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

cfg = load_module("config",        os.path.join(ROOT_DIR, "config.py"))
db  = load_module("db_connection", os.path.join(ROOT_DIR, "src", "db_connection.py"))

# ─────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
#  MAIN PREPROCESSING FUNCTION
# ─────────────────────────────────────────
def preprocess(fetch_all: bool = False):
    """
    Fetch data from MySQL, clean, encode, scale.
    fetch_all=False  → only unprocessed records (daily cron use)
    fetch_all=True   → all processed records (full retrain use)
    Returns: X_scaled (numpy array), feature_columns, scaler, raw_ids
    """
    logger.info("=" * 50)
    logger.info("  PREPROCESSING STARTED")
    logger.info("=" * 50)

    # ── Step 1: Fetch from MySQL ──────────
    if fetch_all:
        df = db.fetch_all_processed_customers()
    else:
        df = db.fetch_unprocessed_customers()

    if df.empty:
        logger.warning("⚠️ No data to process. Exiting.")
        return None, None, None, None

    logger.info(f"✅ Fetched {len(df)} records from database.")

    # Save original IDs to mark as processed later
    raw_ids = df["id"].tolist() if "id" in df.columns else []

    # ── Step 2: Drop non-feature columns ──
    drop_cols = ["id", "is_processed", "created_at"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    logger.info(f"✅ Dropped system columns.")

    # ── Step 3: Keep only feature columns ─
    feature_cols = [c for c in cfg.FEATURE_COLUMNS if c in df.columns]
    df = df[feature_cols]
    logger.info(f"✅ Using {len(feature_cols)} feature columns.")

    # ── Step 4: Fix TotalCharges ──────────
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # ── Step 5: Fill any remaining nulls ──
    for col in cfg.CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    for col in cfg.NUMERICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    logger.info(f"✅ Null values handled.")

    # ── Step 6: Encode categoricals ───────
    le = LabelEncoder()
    for col in cfg.CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    logger.info(f"✅ Categorical columns encoded.")

    # ── Step 7: Scale numericals ──────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    logger.info(f"✅ Features scaled with StandardScaler.")

    # ── Step 8: Save processed CSV ────────
    os.makedirs(cfg.DATA_PROC_DIR, exist_ok=True)
    today     = datetime.now().strftime("%Y_%m_%d")
    save_path = os.path.join(cfg.DATA_PROC_DIR, f"processed_{today}.csv")
    pd.DataFrame(X_scaled, columns=feature_cols).to_csv(save_path, index=False)
    logger.info(f"✅ Processed data saved to: {save_path}")

    logger.info("=" * 50)
    logger.info(f"  PREPROCESSING COMPLETE — {len(df)} records ready")
    logger.info("=" * 50)

    return X_scaled, feature_cols, scaler, raw_ids


# ─────────────────────────────────────────
#  TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    X, cols, scaler, ids = preprocess(fetch_all=False)
    if X is not None:
        print(f"\nShape of processed data : {X.shape}")
        print(f"Number of features      : {len(cols)}")
        print(f"Feature columns         : {cols}")
        print(f"Sample (first row)      : {X[0]}")