import os
import sys
import pickle
import logging
import numpy as np
from datetime import datetime
from collections import deque
from sklearn.ensemble import IsolationForest

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
#  ONLINE ISOLATION FOREST CLASS
# ─────────────────────────────────────────
class OnlineIsolationForest:
    """
    Production-grade Online Isolation Forest with:
    - Sliding window continuous learning
    - Model versioning
    - Score normalization
    - Auto save/load
    """

    def __init__(self,
                 n_estimators  = 100,
                 contamination  = 0.15,
                 window_size    = 1000,
                 random_state   = 42):

        self.n_estimators   = n_estimators
        self.contamination  = contamination
        self.window_size    = window_size
        self.random_state   = random_state
        self.window         = deque(maxlen=window_size)
        self.model          = None
        self.update_count   = 0
        self.global_min     = None
        self.global_max     = None

        logger.info(f"OnlineIsolationForest initialized — "
                    f"window={window_size}, contamination={contamination}")


    def _build_model(self):
        return IsolationForest(
            n_estimators  = self.n_estimators,
            contamination = self.contamination,
            random_state  = self.random_state
        )


    def initial_train(self, X: np.ndarray):
        """First time full training on entire dataset."""
        logger.info(f"Starting initial training on {len(X)} records...")

        # Fill window with last window_size records
        for row in X[-self.window_size:]:
            self.window.append(row)

        # Train model
        self.model = self._build_model()
        self.model.fit(X)
        self.update_count += 1

        # Compute score boundaries from full dataset
        raw_scores      = self.model.score_samples(X)
        self.global_min = float(np.percentile(raw_scores, 5))
        self.global_max = float(np.percentile(raw_scores, 95))

        logger.info(f"✅ Initial training complete.")
        logger.info(f"   Score range: [{self.global_min:.4f}, {self.global_max:.4f}]")
        logger.info(f"   Window filled: {len(self.window)} records")


    def update(self, X_new: np.ndarray):
        """
        Continuous learning — add new data to window and retrain.
        This is called by the cron job daily.
        """
        if self.model is None:
            logger.warning("No existing model found. Running initial training.")
            self.initial_train(X_new)
            return

        logger.info(f"Updating model with {len(X_new)} new records...")

        # Add new records to sliding window
        for row in X_new:
            self.window.append(row)

        # Retrain on current window
        window_data = np.array(self.window)
        self.model  = self._build_model()
        self.model.fit(window_data)
        self.update_count += 1

        # Recompute score boundaries
        raw_scores      = self.model.score_samples(window_data)
        self.global_min = float(np.percentile(raw_scores, 5))
        self.global_max = float(np.percentile(raw_scores, 95))

        logger.info(f"✅ Model updated — update count: {self.update_count}")
        logger.info(f"   Window size: {len(self.window)}")
        logger.info(f"   Score range: [{self.global_min:.4f}, {self.global_max:.4f}]")


    def predict(self, X: np.ndarray):
        """
        Returns:
        - labels : 1 = normal, -1 = anomaly (churn risk)
        - scores : normalized 0-1 (higher = more anomalous)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        raw_scores = self.model.score_samples(X)
        labels     = self.model.predict(X)

        # Normalize scores to 0-1
        scores = self._normalize(raw_scores)

        return labels, scores


    def _normalize(self, raw_scores: np.ndarray):
        """Normalize raw scores to 0-1 using p5-p95 boundaries."""
        scores = (self.global_max - np.clip(raw_scores, self.global_min, self.global_max)) \
                 / (self.global_max - self.global_min)
        return np.clip(scores, 0, 1)


    def get_risk_tier(self, score: float):
        """Convert normalized score to risk tier."""
        if score >= 0.75:
            return "Critical"
        elif score >= 0.50:
            return "High"
        elif score >= 0.25:
            return "Medium"
        else:
            return "Low"


# ─────────────────────────────────────────
#  SAVE / LOAD MODEL
# ─────────────────────────────────────────
def save_model(oif: OnlineIsolationForest, scaler, feature_columns: list):
    """Save model package to models/ folder with versioned filename."""
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)

    today         = datetime.now().strftime("%Y_%m_%d_%H%M")
    version       = f"model_v_{today}"
    filename      = f"{version}.pkl"
    save_path     = os.path.join(cfg.MODELS_DIR, filename)

    model_package = {
        "model"           : oif.model,
        "window"          : oif.window,
        "window_size"     : oif.window_size,
        "contamination"   : oif.contamination,
        "n_estimators"    : oif.n_estimators,
        "update_count"    : oif.update_count,
        "global_min"      : oif.global_min,
        "global_max"      : oif.global_max,
        "scaler"          : scaler,
        "feature_columns" : feature_columns,
        "version"         : version,
        "saved_at"        : datetime.now().isoformat()
    }

    with open(save_path, "wb") as f:
        pickle.dump(model_package, f)

    logger.info(f"✅ Model saved: {save_path}")
    return version, save_path


def load_model(model_path: str):
    """Load model package from .pkl file."""
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found: {model_path}")
        return None

    with open(model_path, "rb") as f:
        package = pickle.load(f)

    # Reconstruct OnlineIsolationForest object
    oif               = OnlineIsolationForest(
        n_estimators  = package["n_estimators"],
        contamination = package["contamination"],
        window_size   = package["window_size"]
    )
    oif.model         = package["model"]
    oif.window        = package["window"]
    oif.update_count  = package["update_count"]
    oif.global_min    = package["global_min"]
    oif.global_max    = package["global_max"]

    logger.info(f"✅ Model loaded: {package['version']}")
    return oif, package["scaler"], package["feature_columns"], package


# ─────────────────────────────────────────
#  MAIN TRAIN FUNCTION
# ─────────────────────────────────────────
def run_training(X: np.ndarray, feature_columns: list, scaler, record_ids: list):
    """
    Full training run:
    1. Check if active model exists
    2. If yes  → update (continuous learning)
    3. If no   → initial train
    4. Save model + register in DB
    5. Mark records as processed
    """
    logger.info("=" * 50)
    logger.info("  TRAINING STARTED")
    logger.info("=" * 50)

    # ── Check for existing model ──────────
    active_model_info = db.get_active_model()

    if active_model_info and os.path.exists(active_model_info["model_path"]):
        # Load existing model and update
        logger.info(f"Found existing model: {active_model_info['model_version']}")
        oif, _, _, _ = load_model(active_model_info["model_path"])
        oif.update(X)
    else:
        # Fresh training
        logger.info("No existing model found. Starting fresh training.")
        oif = OnlineIsolationForest(
            n_estimators  = cfg.MODEL_PARAMS["n_estimators"],
            contamination = cfg.MODEL_PARAMS["contamination"],
            window_size   = cfg.MODEL_PARAMS["window_size"],
            random_state  = cfg.MODEL_PARAMS["random_state"]
        )
        oif.initial_train(X)

    # ── Save model ────────────────────────
    version, save_path = save_model(oif, scaler, feature_columns)

    # ── Register in DB ────────────────────
    db.register_model(
        model_version      = version,
        model_path         = save_path,
        trained_on_records = len(X),
        notes              = f"Update #{oif.update_count}"
    )

    # ── Mark records as processed ─────────
    if record_ids:
        db.mark_customers_processed(record_ids)
        logger.info(f"✅ Marked {len(record_ids)} records as processed in DB.")

    logger.info("=" * 50)
    logger.info(f"  ✅ TRAINING COMPLETE — Version: {version}")
    logger.info("=" * 50)

    return oif, version


# ─────────────────────────────────────────
#  TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Import preprocessing
    prep = load_module("preprocessing", os.path.join(ROOT_DIR, "src", "preprocessing.py"))

    # Run preprocessing first
    X, feature_cols, scaler, raw_ids = prep.preprocess(fetch_all=False)

    if X is not None:
        # Run training
        oif, version = run_training(X, feature_cols, scaler, raw_ids)

        # Test prediction on first 5 records
        labels, scores = oif.predict(X[:5])
        print("\n── Sample Predictions ──")
        for i in range(5):
            tier = oif.get_risk_tier(scores[i])
            print(f"  Customer {i+1}: Score={scores[i]:.4f} | Risk={tier} | Label={labels[i]}")
    else:
        print("No data available for training.")