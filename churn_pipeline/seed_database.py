import os
import sys
import pandas as pd
import logging

# ─────────────────────────────────────────
#  PATH FIX
# ─────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

import importlib.util
spec = importlib.util.spec_from_file_location("config", os.path.join(ROOT_DIR, "config.py"))
cfg  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

spec2 = importlib.util.spec_from_file_location("db_connection", os.path.join(ROOT_DIR, "src", "db_connection.py"))
db    = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(db)

# ─────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
#  SEED FUNCTION
# ─────────────────────────────────────────
def seed():
    logger.info("=" * 50)
    logger.info("  SEEDING DATABASE WITH IBM TELECOM DATA")
    logger.info("=" * 50)

    # ── Load CSV ──────────────────────────
    csv_path = cfg.RAW_CSV
    if not os.path.exists(csv_path):
        logger.error(f"❌ CSV not found at: {csv_path}")
        logger.error("Please place telco_customers.csv inside data/raw/ folder.")
        return

    logger.info(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns.")

    # ── Drop columns not needed ───────────
    drop_cols = []
    if "customerID" in df.columns:
        drop_cols.append("customerID")
    if "Churn" in df.columns:
        drop_cols.append("Churn")

    df = df.drop(columns=drop_cols)
    logger.info(f"✅ Dropped columns: {drop_cols}")

    # ── Fix TotalCharges (sometimes has spaces) ──
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["TotalCharges"])
        after = len(df)
        if before != after:
            logger.info(f"✅ Dropped {before - after} rows with invalid TotalCharges.")

    # ── Add is_processed column ───────────
    df["is_processed"] = 0

    logger.info(f"Final shape before insert: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")

    # ── Insert into MySQL ─────────────────
    logger.info("Inserting into raw_customers table...")
    db.insert_raw_customers(df)

    logger.info("=" * 50)
    logger.info(f"  ✅ SEEDING COMPLETE — {len(df)} records inserted!")
    logger.info("=" * 50)


if __name__ == "__main__":
    seed()