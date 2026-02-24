import os
import sys
import numpy as np
import logging
from scipy.stats import ks_2samp
from datetime import datetime

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
train = load_module("train",          os.path.join(ROOT_DIR, "src", "train.py"))

# ─────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
#  EVALUATION FUNCTION
# ─────────────────────────────────────────
def evaluate(oif, X: np.ndarray, model_version: str):
    """
    Compute all evaluation metrics after training and log to DB.
    
    Metrics computed:
    1. Mean anomaly score        — average risk across all customers
    2. Std anomaly score         — spread/consistency of scores
    3. Anomaly rate %            — % customers flagged as anomalous
    4. KS distance               — drift vs previous model
    5. Trend score               — 7-day rolling health score
    6. Status                    — GOOD / WARNING / DEGRADED
    """
    logger.info("=" * 50)
    logger.info("  EVALUATION STARTED")
    logger.info("=" * 50)

    # ── Step 1: Get predictions ───────────
    labels, scores = oif.predict(X)

    # ── Step 2: Core metrics ──────────────
    mean_score   = float(np.mean(scores))
    std_score    = float(np.std(scores))
    anomaly_rate = float(np.sum(labels == -1) / len(labels) * 100)
    total        = len(X)

    logger.info(f"  Mean anomaly score : {mean_score:.4f}")
    logger.info(f"  Std anomaly score  : {std_score:.4f}")
    logger.info(f"  Anomaly rate       : {anomaly_rate:.2f}%")
    logger.info(f"  Total records      : {total}")

    # ── Step 3: KS Distance (drift) ───────
    ks_distance = compute_ks_distance(oif, X, scores)

    # ── Step 4: Trend Score ───────────────
    trend_score = compute_trend_score()

    # ── Step 5: Determine Status ──────────
    status = determine_status(
        ks_distance  = ks_distance,
        anomaly_rate = anomaly_rate,
        trend_score  = trend_score
    )

    logger.info(f"  KS distance        : {ks_distance:.4f}")
    logger.info(f"  Trend score        : {trend_score:.4f}")
    logger.info(f"  Status             : {status}")

    # ── Step 6: Log to DB ─────────────────
    metrics = {
        "model_version"       : model_version,
        "mean_anomaly_score"  : mean_score,
        "std_anomaly_score"   : std_score,
        "anomaly_rate_pct"    : anomaly_rate,
        "ks_distance"         : ks_distance,
        "trend_score"         : trend_score,
        "total_records_scored": total,
        "status"              : status
    }

    db.log_model_metrics(metrics)

    # ── Step 7: Rollback if DEGRADED ──────
    if status == "DEGRADED":
        handle_degraded_model(model_version)

    logger.info("=" * 50)
    logger.info(f"  ✅ EVALUATION COMPLETE — Status: {status}")
    logger.info("=" * 50)

    return metrics


# ─────────────────────────────────────────
#  KS DISTANCE — MODEL DRIFT DETECTION
# ─────────────────────────────────────────
def compute_ks_distance(oif, X: np.ndarray, current_scores: np.ndarray):
    """
    Compare current score distribution vs previous model's scores.
    KS distance = 0    → identical distributions (perfect stability)
    KS distance = 1    → completely different (severe drift)
    """
    try:
        # Fetch last metric entry to get previous scores reference
        prev_metrics = db.fetch_last_n_metrics(n=1)

        if prev_metrics.empty:
            logger.info("  No previous metrics found. KS distance = 0.0 (first run)")
            return 0.0

        # Get previous mean and std to simulate previous distribution
        prev_mean = float(prev_metrics.iloc[0]["mean_anomaly_score"])
        prev_std  = float(prev_metrics.iloc[0]["std_anomaly_score"])
        prev_n    = int(prev_metrics.iloc[0]["total_records_scored"])

        # Simulate previous score distribution
        np.random.seed(42)
        prev_scores_simulated = np.random.normal(prev_mean, prev_std, prev_n)
        prev_scores_simulated = np.clip(prev_scores_simulated, 0, 1)

        # Compute KS statistic
        ks_stat, _ = ks_2samp(current_scores, prev_scores_simulated)
        logger.info(f"  KS distance computed: {ks_stat:.4f}")
        return float(ks_stat)

    except Exception as e:
        logger.warning(f"  Could not compute KS distance: {e}. Defaulting to 0.0")
        return 0.0


# ─────────────────────────────────────────
#  TREND SCORE — 7 DAY ROLLING HEALTH
# ─────────────────────────────────────────
def compute_trend_score():
    """
    Compute a 7-day rolling health score.
    Score = 1.0  → model consistently healthy
    Score = 0.0  → model consistently degraded
    """
    try:
        recent = db.fetch_last_n_metrics(n=cfg.EVAL_CONFIG["trend_window_days"])

        if recent.empty:
            return 1.0  # First run, assume healthy

        # Map status to numeric scores
        status_map = {"GOOD": 1.0, "WARNING": 0.5, "DEGRADED": 0.0}
        scores     = recent["status"].map(status_map).fillna(0.5).tolist()
        trend      = float(np.mean(scores))

        logger.info(f"  Trend score over last {len(scores)} runs: {trend:.4f}")
        return trend

    except Exception as e:
        logger.warning(f"  Could not compute trend score: {e}. Defaulting to 1.0")
        return 1.0


# ─────────────────────────────────────────
#  STATUS DETERMINATION
# ─────────────────────────────────────────
def determine_status(ks_distance: float, anomaly_rate: float, trend_score: float):
    """
    Determine model health status based on all metrics.
    
    Rules:
    - DEGRADED  : KS > critical threshold OR anomaly rate out of bounds
    - WARNING   : KS > warning threshold OR trend score < 0.5
    - GOOD      : everything within normal range
    """
    ks_warn     = cfg.EVAL_CONFIG["ks_warning_threshold"]
    ks_critical = cfg.EVAL_CONFIG["ks_critical_threshold"]
    min_rate    = cfg.EVAL_CONFIG["min_anomaly_rate"] * 100
    max_rate    = cfg.EVAL_CONFIG["max_anomaly_rate"] * 100

    # Critical conditions
    if ks_distance > ks_critical:
        logger.warning(f"  ⚠️ DEGRADED: KS distance {ks_distance:.4f} > {ks_critical}")
        return "DEGRADED"

    if anomaly_rate < min_rate or anomaly_rate > max_rate:
        logger.warning(f"  ⚠️ DEGRADED: Anomaly rate {anomaly_rate:.2f}% out of bounds [{min_rate}%, {max_rate}%]")
        return "DEGRADED"

    # Warning conditions
    if ks_distance > ks_warn:
        logger.warning(f"  ⚠️ WARNING: KS distance {ks_distance:.4f} > {ks_warn}")
        return "WARNING"

    if trend_score < 0.5:
        logger.warning(f"  ⚠️ WARNING: Trend score {trend_score:.4f} < 0.5")
        return "WARNING"

    return "GOOD"


# ─────────────────────────────────────────
#  ROLLBACK IF DEGRADED
# ─────────────────────────────────────────
def handle_degraded_model(current_version: str):
    """
    If current model is DEGRADED, rollback to previous good model.
    """
    logger.warning(f"  🔄 Model {current_version} is DEGRADED. Attempting rollback...")

    try:
        all_models = db.get_all_models()

        # Find last GOOD model that is not the current one
        good_models = all_models[
            (all_models["model_version"] != current_version) &
            (all_models["notes"].str.contains("GOOD", na=False) |
             all_models["is_active"] == 0)
        ]

        if good_models.empty:
            logger.warning("  No previous good model found for rollback. Keeping current.")
            return

        rollback_version = good_models.iloc[0]["model_version"]
        rollback_path    = good_models.iloc[0]["model_path"]

        # Re-register previous model as active
        db.register_model(
            model_version      = rollback_version,
            model_path         = rollback_path,
            trained_on_records = int(good_models.iloc[0]["trained_on_records"]),
            notes              = "ROLLED BACK due to degradation"
        )

        logger.warning(f"  ✅ Rolled back to: {rollback_version}")

    except Exception as e:
        logger.error(f"  ❌ Rollback failed: {e}")


# ─────────────────────────────────────────
#  TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    prep = load_module("preprocessing", os.path.join(ROOT_DIR, "src", "preprocessing.py"))

    # Use all processed data for evaluation
    X, feature_cols, scaler, _ = prep.preprocess(fetch_all=True)

    if X is not None:
        # Load active model
        active = db.get_active_model()
        if active:
            oif, _, _, _ = train.load_model(active["model_path"])
            metrics = evaluate(oif, X, active["model_version"])

            print("\n── Evaluation Results ──")
            for k, v in metrics.items():
                print(f"  {k:25s}: {v}")
        else:
            print("No active model found.")
    else:
        print("No data available.")