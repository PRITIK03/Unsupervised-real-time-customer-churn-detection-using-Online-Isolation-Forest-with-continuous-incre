import os
import sys
import pymysql
import pandas as pd
import logging
from sqlalchemy import create_engine, text

# ─────────────────────────────────────────
#  PATH FIX — must be before any local imports
# ─────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import importlib.util
spec = importlib.util.spec_from_file_location("config", os.path.join(ROOT_DIR, "config.py"))
cfg  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfg)

DB_URL      = cfg.DB_URL
DB_CONFIG   = cfg.DB_CONFIG

# ─────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
#  ENGINE
# ─────────────────────────────────────────
def get_engine():
    try:
        engine = create_engine(DB_URL, echo=False)
        logger.info("Database engine created successfully.")
        return engine
    except Exception as e:
        logger.error(f"Failed to create engine: {e}")
        raise


def test_connection():
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection successful!")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


# ─────────────────────────────────────────
#  RAW CUSTOMERS
# ─────────────────────────────────────────
def insert_raw_customers(df: pd.DataFrame):
    try:
        engine = get_engine()
        df.to_sql(
            name      = "raw_customers",
            con       = engine,
            if_exists = "append",
            index     = False
        )
        logger.info(f"✅ Inserted {len(df)} records into raw_customers.")
    except Exception as e:
        logger.error(f"❌ Failed to insert raw customers: {e}")
        raise


def fetch_unprocessed_customers():
    try:
        engine = get_engine()
        df = pd.read_sql(
            "SELECT * FROM raw_customers WHERE is_processed = 0",
            con=engine
        )
        logger.info(f"✅ Fetched {len(df)} unprocessed records.")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to fetch unprocessed customers: {e}")
        raise


def mark_customers_processed(ids: list):
    try:
        engine = get_engine()
        ids_str = ", ".join(str(i) for i in ids)
        with engine.connect() as conn:
            conn.execute(text(
                f"UPDATE raw_customers SET is_processed = 1 WHERE id IN ({ids_str})"
            ))
            conn.commit()
        logger.info(f"✅ Marked {len(ids)} records as processed.")
    except Exception as e:
        logger.error(f"❌ Failed to mark customers as processed: {e}")
        raise


def fetch_all_processed_customers():
    try:
        engine = get_engine()
        df = pd.read_sql(
            "SELECT * FROM raw_customers WHERE is_processed = 1",
            con=engine
        )
        logger.info(f"✅ Fetched {len(df)} processed records.")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to fetch processed customers: {e}")
        raise


# ─────────────────────────────────────────
#  MODEL REGISTRY
# ─────────────────────────────────────────
def register_model(model_version: str, model_path: str, trained_on_records: int, notes: str = ""):
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("UPDATE model_registry SET is_active = 0"))
            conn.execute(text("""
                INSERT INTO model_registry
                    (model_version, model_path, trained_on_records, is_active, notes)
                VALUES
                    (:version, :path, :records, 1, :notes)
            """), {
                "version" : model_version,
                "path"    : model_path,
                "records" : trained_on_records,
                "notes"   : notes
            })
            conn.commit()
        logger.info(f"✅ Model {model_version} registered as active.")
    except Exception as e:
        logger.error(f"❌ Failed to register model: {e}")
        raise


def update_model_registry(model_version: str, model_path: str, trained_on_records: int, notes: str = ""):
    """
    Update existing active model row instead of inserting new one.
    - If the version row exists  → UPDATE trained_on_records, training_date, notes
    - If no row exists yet       → INSERT first row (first run only)
    Result: model_registry always stays at 1 row no matter how many cron runs.
    """
    try:
        engine = get_engine()
        with engine.begin() as conn:
            # FIX: use [0] index instead of ["cnt"] — MySQL Row objects
            # do not support string key access without mappings=True
            result = conn.execute(text(
                "SELECT COUNT(*) FROM model_registry WHERE model_version = :v"
            ), {"v": model_version})
            count = result.fetchone()[0]

            if count > 0:
                # Row exists — update it in place
                conn.execute(text("""
                    UPDATE model_registry
                    SET trained_on_records = :records,
                        training_date      = NOW(),
                        notes              = :notes
                    WHERE model_version = :v
                """), {"records": trained_on_records, "notes": notes, "v": model_version})
                logger.info(f"✅ Model registry updated: {model_version}")
            else:
                # First run only — insert new row
                conn.execute(text("""
                    INSERT INTO model_registry
                        (model_version, model_path, trained_on_records, is_active, notes)
                    VALUES
                        (:v, :path, :records, 1, :notes)
                """), {"v": model_version, "path": model_path,
                       "records": trained_on_records, "notes": notes})
                logger.info(f"✅ Model registry created: {model_version}")
    except Exception as e:
        logger.error(f"❌ update_model_registry error: {e}")


def get_active_model():
    try:
        engine = get_engine()
        df = pd.read_sql(
            "SELECT * FROM model_registry WHERE is_active = 1 ORDER BY training_date DESC LIMIT 1",
            con=engine
        )
        if df.empty:
            logger.warning("⚠️ No active model found in registry.")
            return None
        logger.info(f"✅ Active model: {df.iloc[0]['model_version']}")
        return df.iloc[0].to_dict()
    except Exception as e:
        logger.error(f"❌ Failed to fetch active model: {e}")
        raise


def get_all_models():
    try:
        engine = get_engine()
        df = pd.read_sql(
            "SELECT * FROM model_registry ORDER BY training_date DESC",
            con=engine
        )
        return df
    except Exception as e:
        logger.error(f"❌ Failed to fetch model registry: {e}")
        raise


# ─────────────────────────────────────────
#  MODEL METRICS
# ─────────────────────────────────────────
def log_model_metrics(metrics: dict):
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO model_metrics (
                    model_version, mean_anomaly_score, std_anomaly_score,
                    anomaly_rate_pct, ks_distance, trend_score,
                    total_records_scored, status
                ) VALUES (
                    :model_version, :mean_anomaly_score, :std_anomaly_score,
                    :anomaly_rate_pct, :ks_distance, :trend_score,
                    :total_records_scored, :status
                )
            """), metrics)
            conn.commit()
        logger.info(f"✅ Metrics logged — Status: {metrics['status']}")
    except Exception as e:
        logger.error(f"❌ Failed to log metrics: {e}")
        raise


def fetch_recent_metrics(days: int = 30):
    try:
        engine = get_engine()
        df = pd.read_sql(f"""
            SELECT * FROM model_metrics
            WHERE logged_at >= DATE_SUB(NOW(), INTERVAL {days} DAY)
            ORDER BY logged_at DESC
        """, con=engine)
        logger.info(f"✅ Fetched {len(df)} metric records.")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to fetch metrics: {e}")
        raise


def fetch_last_n_metrics(n: int = 7):
    try:
        engine = get_engine()
        df = pd.read_sql(
            f"SELECT * FROM model_metrics ORDER BY logged_at DESC LIMIT {n}",
            con=engine
        )
        return df
    except Exception as e:
        logger.error(f"❌ Failed to fetch last {n} metrics: {e}")
        raise


# ─────────────────────────────────────────
#  TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    test_connection()