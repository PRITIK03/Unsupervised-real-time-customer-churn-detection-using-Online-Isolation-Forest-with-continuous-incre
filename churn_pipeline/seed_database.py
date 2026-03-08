import os
import sys
import pandas as pd
import logging
from sqlalchemy import text

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config as cfg
from src import db_connection as db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def seed(force: bool = False):
    """
    Load IBM Telco CSV into raw_customers table.

    Args:
        force : if True, truncate existing data and re-seed.
                if False (default), skip if records already exist.

    FIX 1 — CORRUPTED FILE:
        The previous seed_database.py had the entire contents of
        requirements.txt pasted into the middle of the Python source,
        causing an immediate SyntaxError on import. This is a clean
        rewrite of the file.

    FIX 2 — IDEMPOTENCY:
        Old version unconditionally inserted all 7,043 rows every run.
        Running it twice silently produced 14,086 rows — all with
        is_processed=0 — so the model trained on duplicated data.
        Now checks if rows already exist and skips unless --force is passed.

    FIX 3 — COLUMN VALIDATION:
        After dropping customerID and Churn, the remaining columns are
        validated against cfg.FEATURE_COLUMNS before inserting. Extra
        columns (e.g. added during Excel editing) are dropped with a
        warning instead of silently corrupting the DB schema.
    """
    logger.info("=" * 55)
    logger.info("  SEEDING DATABASE WITH IBM TELECOM DATA")
    logger.info("=" * 55)

    # ── Step 1: Idempotency check ─────────
    if not force:
        try:
            existing = pd.read_sql(
                "SELECT COUNT(*) AS cnt FROM raw_customers",
                con=db.get_engine()
            )
            count = int(existing.iloc[0]["cnt"])
            if count > 0:
                logger.info(
                    f"✅ raw_customers already has {count:,} rows. "
                    f"Skipping. Use --force to re-seed."
                )
                return
        except Exception:
            # Table may not exist yet — continue to insert
            pass

    # ── Step 2: Load CSV ──────────────────
    if not os.path.exists(cfg.RAW_CSV):
        logger.error(f"❌ CSV not found: {cfg.RAW_CSV}")
        logger.error("   Place telco_customers.csv inside data/raw/ and retry.")
        return

    logger.info(f"Loading: {cfg.RAW_CSV}")
    df = pd.read_csv(cfg.RAW_CSV)
    logger.info(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns.")

    # ── Step 3: Drop label / ID columns ───
    drop_cols = [c for c in ["customerID", "Churn"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info(f"✅ Dropped: {drop_cols}")

    # ── Step 4: Fix TotalCharges ──────────
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        bad = df[df["TotalCharges"].isna()]
        if not bad.empty:
            logger.warning(
                f"⚠️  {len(bad)} rows had invalid TotalCharges "
                f"(row indices: {bad.index.tolist()}) — dropping."
            )
        df = df.dropna(subset=["TotalCharges"])

    # ── Step 5: Validate columns ──────────
    expected = set(cfg.FEATURE_COLUMNS)
    present  = set(df.columns)

    extra   = present - expected
    missing = expected - present

    if extra:
        logger.warning(f"⚠️  Dropping unexpected columns: {sorted(extra)}")
        df = df.drop(columns=list(extra))

    if missing:
        logger.error(f"❌ CSV is missing required columns: {sorted(missing)}")
        logger.error("   Cannot seed. Check your CSV file.")
        return

    # Enforce exact column order from config
    df = df[cfg.FEATURE_COLUMNS].copy()

    # ── Step 6: Add is_processed flag ─────
    df["is_processed"] = 0

    logger.info(f"Final shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    # ── Step 7: Truncate if --force ───────
    if force:
        try:
            with db.get_engine().begin() as conn:
                conn.execute(text("TRUNCATE TABLE raw_customers"))
            logger.warning("⚠️  --force: raw_customers truncated.")
        except Exception as e:
            logger.error(f"❌ Failed to truncate raw_customers: {e}")
            return

    # ── Step 8: Insert ────────────────────
    logger.info("Inserting into raw_customers...")
    db.insert_raw_customers(df)

    logger.info("=" * 55)
    logger.info(f"  ✅ SEEDING COMPLETE — {len(df):,} records inserted.")
    logger.info("  Next: run the training pipeline:")
    logger.info("  > python src/train.py --full")
    logger.info("=" * 55)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Seed ChurnGuard database with IBM Telco data"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Truncate existing data and re-seed (default: skip if data exists)"
    )
    args = parser.parse_args()
    seed(force=args.force)