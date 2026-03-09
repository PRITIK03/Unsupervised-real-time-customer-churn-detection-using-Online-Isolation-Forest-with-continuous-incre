import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
from src.db_connection import get_engine
from sqlalchemy import text

def setup():
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS pipeline_alerts (
                id             INT AUTO_INCREMENT PRIMARY KEY,
                alert_type     VARCHAR(50)  NOT NULL  COMMENT 'DEGRADED, WARNING, ROLLBACK, PIPELINE_FAIL',
                message        TEXT         NOT NULL,
                model_version  VARCHAR(50),
                severity       VARCHAR(20)  DEFAULT 'WARNING'  COMMENT 'INFO, WARNING, CRITICAL',
                created_at     TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
            );
        """))
    print("Table pipeline_alerts created successfully!")

if __name__ == "__main__":
    setup()
