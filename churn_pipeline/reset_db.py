import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
from src.db_connection import get_engine
from sqlalchemy import text

def setup():
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("UPDATE raw_customers SET is_processed=0 LIMIT 5"))
    print("Reset 5 records to unprocessed")

if __name__ == "__main__":
    setup()
