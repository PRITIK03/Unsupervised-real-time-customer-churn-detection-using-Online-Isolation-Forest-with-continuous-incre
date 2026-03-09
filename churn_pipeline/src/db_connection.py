import os
import sys
import pandas as pd
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
#  ENGINE SINGLETON
#  FIX: old get_engine() created a brand-new engine
#  and connection pool on every single call. With 4
#  queries per /dashboard-stats request that meant
#  4 pools created and thrown away per request.
#  Now one engine is created at first call and reused
#  for the lifetime of the process.
# ─────────────────────────────────────────
_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            cfg.DB_URL,
            echo=False,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_recycle=1800,   # recycle connections after 30 min
            pool_pre_ping=True,  # test connection before use
        )
        logger.info("Database engine created (singleton, QueuePool).")
    return _engine


def test_connection():
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection OK.")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


# ─────────────────────────────────────────
#  RAW CUSTOMERS
# ─────────────────────────────────────────
def insert_raw_customers(df: pd.DataFrame):
    try:
        df.to_sql("raw_customers", con=get_engine(), if_exists="append", index=False)
        logger.info(f"✅ Inserted {len(df)} records into raw_customers.")
    except Exception as e:
        logger.error(f"❌ insert_raw_customers failed: {e}")
        raise


def fetch_unprocessed_customers() -> pd.DataFrame:
    try:
        df = pd.read_sql(
            "SELECT * FROM raw_customers WHERE is_processed = 0",
            con=get_engine(),
        )
        logger.info(f"✅ Fetched {len(df)} unprocessed records.")
        return df
    except Exception as e:
        logger.error(f"❌ fetch_unprocessed_customers failed: {e}")
        raise


def mark_customers_processed(ids: list):
    """
    FIX: old version built SQL with an f-string:
        f"WHERE id IN ({ids_str})"
    SQL injection vulnerability if IDs ever come from
    untrusted input. Fixed with a parameterised query.
    SQLAlchemy expands tuple bind params into a safe
    parameterised IN list automatically.
    """
    if not ids:
        return
    try:
        with get_engine().begin() as conn:
            conn.execute(
                text("UPDATE raw_customers SET is_processed = 1 WHERE id IN :ids"),
                {"ids": tuple(ids)},
            )
        logger.info(f"✅ Marked {len(ids)} records as processed.")
    except Exception as e:
        logger.error(f"❌ mark_customers_processed failed: {e}")
        raise


def fetch_all_processed_customers() -> pd.DataFrame:
    try:
        df = pd.read_sql(
            "SELECT * FROM raw_customers WHERE is_processed = 1",
            con=get_engine(),
        )
        logger.info(f"✅ Fetched {len(df)} processed records.")
        return df
    except Exception as e:
        logger.error(f"❌ fetch_all_processed_customers failed: {e}")
        raise


# ─────────────────────────────────────────
#  MODEL REGISTRY
# ─────────────────────────────────────────
def register_model(model_version: str, model_path: str,
                   trained_on_records: int, notes: str = ""):
    try:
        with get_engine().begin() as conn:
            conn.execute(text("UPDATE model_registry SET is_active = 0"))
            conn.execute(text("""
                INSERT INTO model_registry
                    (model_version, model_path, trained_on_records, is_active, notes)
                VALUES
                    (:version, :path, :records, 1, :notes)
            """), {"version": model_version, "path": model_path,
                   "records": trained_on_records, "notes": notes})
        logger.info(f"✅ Model {model_version} registered as active.")
    except Exception as e:
        logger.error(f"❌ register_model failed: {e}")
        raise


def update_model_registry(model_version: str, model_path: str,
                           trained_on_records: int, notes: str = ""):
    try:
        with get_engine().begin() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM model_registry WHERE model_version = :v"),
                {"v": model_version},
            )
            count = result.fetchone()[0]

            if count > 0:
                conn.execute(text("""
                    UPDATE model_registry
                    SET trained_on_records = :records,
                        training_date      = NOW(),
                        notes              = :notes
                    WHERE model_version = :v
                """), {"records": trained_on_records, "notes": notes, "v": model_version})
                logger.info(f"✅ Model registry updated: {model_version}")
            else:
                conn.execute(text("""
                    INSERT INTO model_registry
                        (model_version, model_path, trained_on_records, is_active, notes)
                    VALUES (:v, :path, :records, 1, :notes)
                """), {"v": model_version, "path": model_path,
                       "records": trained_on_records, "notes": notes})
                logger.info(f"✅ Model registry created: {model_version}")
    except Exception as e:
        logger.error(f"❌ update_model_registry failed: {e}")
        raise


def get_active_model():
    try:
        df = pd.read_sql(
            "SELECT * FROM model_registry WHERE is_active = 1 "
            "ORDER BY training_date DESC LIMIT 1",
            con=get_engine(),
        )
        if df.empty:
            logger.warning("⚠️ No active model found in registry.")
            return None
        return df.iloc[0].to_dict()
    except Exception as e:
        logger.error(f"❌ get_active_model failed: {e}")
        raise


def get_all_models() -> pd.DataFrame:
    try:
        return pd.read_sql(
            "SELECT * FROM model_registry ORDER BY training_date DESC",
            con=get_engine(),
        )
    except Exception as e:
        logger.error(f"❌ get_all_models failed: {e}")
        raise


# ─────────────────────────────────────────
#  MODEL METRICS
# ─────────────────────────────────────────
def log_model_metrics(metrics: dict):
    try:
        with get_engine().begin() as conn:
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
        logger.info(f"✅ Metrics logged — status: {metrics['status']}")
    except Exception as e:
        logger.error(f"❌ log_model_metrics failed: {e}")
        raise


def fetch_recent_metrics(days: int = 30) -> pd.DataFrame:
    """
    FIX: old version used f-string for {days} — SQL injection risk.
    Now uses a bound parameter. days is also cast to int() so any
    non-numeric value raises ValueError before touching the DB.
    """
    try:
        df = pd.read_sql(
            text(
                "SELECT * FROM model_metrics "
                "WHERE logged_at >= DATE_SUB(NOW(), INTERVAL :days DAY) "
                "ORDER BY logged_at DESC"
            ),
            con=get_engine(),
            params={"days": int(days)},
        )
        logger.info(f"✅ Fetched {len(df)} metric records.")
        return df
    except Exception as e:
        logger.error(f"❌ fetch_recent_metrics failed: {e}")
        raise


def fetch_last_n_metrics(n: int = 7) -> pd.DataFrame:
    """
    FIX: LIMIT clause can't take a bound param in MySQL via text().
    Safe alternative: cast to int() first — int() rejects any
    non-numeric input so injection is impossible before it reaches SQL.
    """
    try:
        safe_n = int(n)
        return pd.read_sql(
            f"SELECT * FROM model_metrics ORDER BY logged_at DESC LIMIT {safe_n}",
            con=get_engine(),
        )
    except Exception as e:
        logger.error(f"❌ fetch_last_n_metrics failed: {e}")
        raise


# ─────────────────────────────────────────
#  PIPELINE ALERTS
# ─────────────────────────────────────────
def log_pipeline_alert(alert_type: str, message: str,
                       model_version: str = "", severity: str = "WARNING"):
    """
    Insert a row into pipeline_alerts.
    alert_type: DEGRADED, WARNING, ROLLBACK, PIPELINE_FAIL
    severity  : INFO, WARNING, CRITICAL
    """
    try:
        with get_engine().begin() as conn:
            conn.execute(text("""
                INSERT INTO pipeline_alerts
                    (alert_type, message, model_version, severity)
                VALUES
                    (:alert_type, :message, :model_version, :severity)
            """), {
                "alert_type"   : alert_type,
                "message"      : message,
                "model_version": model_version,
                "severity"     : severity,
            })
        logger.info(f"🔔 Alert logged — [{severity}] {alert_type}: {message}")
    except Exception as e:
        logger.error(f"❌ log_pipeline_alert failed: {e}")


def fetch_recent_alerts(days: int = 30) -> pd.DataFrame:
    """Fetch alerts from the last N days."""
    try:
        df = pd.read_sql(
            text(
                "SELECT * FROM pipeline_alerts "
                "WHERE created_at >= DATE_SUB(NOW(), INTERVAL :days DAY) "
                "ORDER BY created_at DESC"
            ),
            con=get_engine(),
            params={"days": int(days)},
        )
        return df
    except Exception as e:
        logger.error(f"❌ fetch_recent_alerts failed: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    test_connection()