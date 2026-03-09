import os
import sys
import logging
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

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

cfg   = load_module("config",        os.path.join(ROOT_DIR, "config.py"))
db    = load_module("db_connection",  os.path.join(ROOT_DIR, "src", "db_connection.py"))
prep  = load_module("preprocessing",  os.path.join(ROOT_DIR, "src", "preprocessing.py"))
train = load_module("train",          os.path.join(ROOT_DIR, "src", "train.py"))
eval_ = load_module("evaluate",       os.path.join(ROOT_DIR, "src", "evaluate.py"))

# ─────────────────────────────────────────
#  LOGGING — also writes to file
# ─────────────────────────────────────────
os.makedirs(cfg.LOGS_DIR, exist_ok=True)
log_file = os.path.join(cfg.LOGS_DIR, "scheduler.log")

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s - %(levelname)s - %(message)s",
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
#  PIPELINE JOB — runs every day at midnight
# ─────────────────────────────────────────
def run_pipeline():
    """
    Full daily pipeline job:
    1. Run incremental_update() — trains on new data while preserving old
       (falls back to initial_train() on first run automatically)
    2. Load the updated model package
    3. Evaluate and log metrics (KS drift, trend, anomaly rate)
    4. Handle degradation alerts if needed
    """
    logger.info("=" * 60)
    logger.info(f"  PIPELINE JOB STARTED — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    try:
        # ── Step 1: Train (incremental or initial) ────────────
        logger.info("STEP 1 — Running incremental training...")
        version = train.incremental_update()

        if version is None:
            logger.warning("⚠️ No new data found for today. Pipeline skipped.")
            return

        logger.info(f"✅ Step 1 complete — Model version: {version}")

        # ── Step 2: Load updated model for evaluation ─────────
        logger.info("STEP 2 — Loading updated model for evaluation...")
        active = db.get_active_model()
        if active is None:
            logger.error("❌ No active model found after training. Aborting.")
            return

        oif, scaler, feature_cols, package = train.load_model(active["model_path"])

        # Compute dynamic tier thresholds on full processed data
        all_processed = db.fetch_all_processed_customers()
        if all_processed is not None and not all_processed.empty:
            X_eval, _, _, _, _ = prep.preprocess(
                fetch_all=True,
                scaler=scaler,
                label_encoders=package.get("label_encoders", {}),
            )
            if X_eval is not None and len(X_eval) > 0:
                import numpy as np
                all_scores = oif._normalize(oif.model.score_samples(X_eval))
                oif.tier_thresholds = train.compute_dynamic_tiers(all_scores)
        else:
            X_eval = None

        if X_eval is None:
            logger.warning("⚠️ Could not load evaluation data. Skipping evaluation.")
            return

        logger.info(f"✅ Step 2 complete — Evaluating on {len(X_eval)} records.")

        # ── Step 3: Evaluate ──────────────
        logger.info("STEP 3 — Evaluating model...")
        metrics = eval_.evaluate(oif, X_eval, version)
        logger.info(f"✅ Step 3 complete — Status: {metrics['status']}")

        # ── Done ──────────────────────────
        logger.info("=" * 60)
        logger.info(f"  ✅ PIPELINE COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Model   : {version}")
        logger.info(f"  Records : {len(X_eval)}")
        logger.info(f"  Status  : {metrics['status']}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ PIPELINE FAILED: {e}", exc_info=True)
        try:
            db.log_pipeline_alert(
                alert_type    = "PIPELINE_FAIL",
                message       = f"Pipeline failed: {e}",
                model_version = "",
                severity      = "CRITICAL",
            )
        except Exception:
            pass  # don't let alert logging crash the error handler


# ─────────────────────────────────────────
#  MANUAL RUN — for testing pipeline now
# ─────────────────────────────────────────
def run_now():
    """Run the pipeline immediately — used for testing."""
    logger.info("Manual pipeline run triggered.")
    run_pipeline()


# ─────────────────────────────────────────
#  SCHEDULER SETUP
# ─────────────────────────────────────────
def start_scheduler():
    """
    Start the APScheduler cron job.
    Runs every day at midnight (00:00).
    Keeps running in background — don't close terminal.
    """
    scheduler = BlockingScheduler()

    scheduler.add_job(
        func    = run_pipeline,
        trigger = CronTrigger(
            hour   = cfg.CRON_HOUR,
            minute = cfg.CRON_MINUTE
        ),
        id      = "daily_churn_pipeline",
        name    = "Daily Churn Detection Pipeline",
        replace_existing = True
    )

    logger.info("=" * 60)
    logger.info("  SCHEDULER STARTED")
    logger.info(f"  Pipeline runs daily at {cfg.CRON_HOUR:02d}:{cfg.CRON_MINUTE:02d}")
    logger.info("  Logs saved to: " + log_file)
    logger.info("  Press Ctrl+C to stop.")
    logger.info("=" * 60)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user.")


# ─────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Churn Pipeline Scheduler")
    parser.add_argument(
        "--run-now",
        action  = "store_true",
        help    = "Run the pipeline immediately instead of waiting for cron"
    )
    args = parser.parse_args()

    if args.run_now:
        run_now()
    else:
        start_scheduler()