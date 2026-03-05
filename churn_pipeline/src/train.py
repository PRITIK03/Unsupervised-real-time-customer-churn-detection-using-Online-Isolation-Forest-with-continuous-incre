import os
import sys
import pickle
import logging
import numpy as np
from datetime import datetime
from collections import deque
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

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
#  DYNAMIC TIER COMPUTATION
#  Replaces hardcoded 0.75 / 0.50 / 0.25 cutoffs.
#  Thresholds are computed from the actual score
#  distribution on the training data, so they
#  automatically adapt every time the model retrains.
#
#  Example on 7032 customers:
#    critical_threshold = 90th percentile score
#    high_threshold     = 75th percentile score
#    medium_threshold   = 50th percentile score
#
#  This means:
#    Top 10% of customers  → Critical
#    Next 15%              → High
#    Next 25%              → Medium
#    Bottom 50%            → Low
# ─────────────────────────────────────────
def compute_dynamic_tiers(scores: np.ndarray) -> dict:
    """
    Compute risk tier boundaries from actual score distribution.

    Args:
        scores: normalised anomaly scores (0-1) for all training records

    Returns:
        dict with keys: critical_threshold, high_threshold, medium_threshold
        These are saved into the model package and used at inference time.
    """
    percentiles = cfg.TIER_PERCENTILES

    critical_threshold = float(np.percentile(scores, percentiles["critical_pct"]))
    high_threshold     = float(np.percentile(scores, percentiles["high_pct"]))
    medium_threshold   = float(np.percentile(scores, percentiles["medium_pct"]))

    logger.info("✅ Dynamic tier thresholds computed from training data:")
    logger.info(f"   Critical  : score >= {critical_threshold:.4f} "
                f"(top {100 - percentiles['critical_pct']}%)")
    logger.info(f"   High      : score >= {high_threshold:.4f} "
                f"(top {100 - percentiles['high_pct']}%)")
    logger.info(f"   Medium    : score >= {medium_threshold:.4f} "
                f"(top {100 - percentiles['medium_pct']}%)")
    logger.info(f"   Low       : score <  {medium_threshold:.4f}")

    return {
        "critical_threshold": critical_threshold,
        "high_threshold"    : high_threshold,
        "medium_threshold"  : medium_threshold,
    }


# ─────────────────────────────────────────
#  ONLINE ISOLATION FOREST CLASS
# ─────────────────────────────────────────
class OnlineIsolationForest:
    def __init__(self,
                 n_estimators  = 100,
                 contamination = 0.15,
                 window_size   = 1000,
                 random_state  = 42):

        self.n_estimators  = n_estimators
        self.contamination = contamination
        self.window_size   = window_size
        self.random_state  = random_state
        self.window        = deque(maxlen=window_size)
        self.model         = None
        self.update_count  = 0
        self.global_min    = None
        self.global_max    = None

        # ── Dynamic tier thresholds ──
        # Set to None on init — populated after first training
        # via compute_dynamic_tiers() and saved in model package.
        # At inference time, loaded from the saved package.
        self.tier_thresholds = None

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

        for row in X[-self.window_size:]:
            self.window.append(row)

        self.model = self._build_model()
        self.model.fit(X)
        self.update_count += 1

        raw_scores      = self.model.score_samples(X)
        self.global_min = float(np.percentile(raw_scores, 5))
        self.global_max = float(np.percentile(raw_scores, 95))

        # ── Compute dynamic tiers from training scores ──
        normalised_scores    = self._normalize(raw_scores)
        self.tier_thresholds = compute_dynamic_tiers(normalised_scores)

        logger.info(f"✅ Initial training complete.")
        logger.info(f"   Score range: [{self.global_min:.4f}, {self.global_max:.4f}]")
        logger.info(f"   Window filled: {len(self.window)} records")

    def update(self, X_new: np.ndarray):
        """Continuous learning — add new data to window and retrain."""
        if self.model is None:
            logger.warning("No existing model found. Running initial training.")
            self.initial_train(X_new)
            return

        logger.info(f"Updating model with {len(X_new)} new records...")

        for row in X_new:
            self.window.append(row)

        window_data = np.array(self.window)
        self.model  = self._build_model()
        self.model.fit(window_data)
        self.update_count += 1

        raw_scores      = self.model.score_samples(window_data)
        self.global_min = float(np.percentile(raw_scores, 5))
        self.global_max = float(np.percentile(raw_scores, 95))

        # ── Recompute tiers from updated window scores ──
        normalised_scores    = self._normalize(raw_scores)
        self.tier_thresholds = compute_dynamic_tiers(normalised_scores)

        logger.info(f"✅ Model updated — update count: {self.update_count}")
        logger.info(f"   Window size: {len(self.window)}")
        logger.info(f"   Score range: [{self.global_min:.4f}, {self.global_max:.4f}]")

    def predict(self, X: np.ndarray):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        raw_scores = self.model.score_samples(X)
        labels     = self.model.predict(X)
        scores     = self._normalize(raw_scores)
        return labels, scores

    def _normalize(self, raw_scores: np.ndarray):
        scores = (self.global_max - np.clip(raw_scores, self.global_min, self.global_max)) \
                 / (self.global_max - self.global_min)
        return np.clip(scores, 0, 1)

    def get_risk_tier(self, score: float) -> str:
        """
        Assign risk tier using data-driven thresholds.

        If tier_thresholds were computed from training data, use them.
        Falls back to hardcoded values only if model was loaded from
        an old package that predates this change.
        """
        if self.tier_thresholds is not None:
            t = self.tier_thresholds
            if score >= t["critical_threshold"]: return "Critical"
            if score >= t["high_threshold"]:     return "High"
            if score >= t["medium_threshold"]:   return "Medium"
            return "Low"

        # ── Fallback for old model packages ──
        logger.warning("tier_thresholds not found — using fallback. Re-run training to fix.")
        if score >= 0.75:   return "Critical"
        elif score >= 0.50: return "High"
        elif score >= 0.25: return "Medium"
        else:               return "Low"


# ─────────────────────────────────────────
#  BUILD & SAVE LABEL ENCODERS
# ─────────────────────────────────────────
def build_label_encoders(df_raw):
    """
    Fit one LabelEncoder per categorical column on the full training data.
    Saved into the model package so the API reuses exact same encodings
    at inference time.
    """
    encoders = {}
    for col in cfg.CATEGORICAL_COLUMNS:
        if col in df_raw.columns:
            le = LabelEncoder()
            le.fit(df_raw[col].astype(str).fillna("Unknown"))
            encoders[col] = le
            logger.info(f"  Encoder for '{col}': {list(le.classes_)}")
    logger.info(f"✅ Built {len(encoders)} label encoders.")
    return encoders


# ─────────────────────────────────────────
#  SAVE MODEL
# ─────────────────────────────────────────
def save_model(oif: OnlineIsolationForest, scaler, feature_columns: list,
               label_encoders: dict,
               existing_version: str = None, existing_path: str = None):
    """
    Save model — always overwrites the same file.
    First time: creates model_v_1.pkl
    Every time after: overwrites the same existing file.
    Now also saves tier_thresholds into the package.
    """
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)

    if existing_path and os.path.exists(existing_path):
        save_path = existing_path
        version   = existing_version
        logger.info(f"Overwriting existing model file: {save_path}")
    else:
        version   = "model_v_1"
        filename  = f"{version}.pkl"
        save_path = os.path.join(cfg.MODELS_DIR, filename)
        logger.info(f"Creating new model file: {save_path}")

    model_package = {
        "model"           : oif.model,
        "window"          : oif.window,
        "window_size"     : oif.window_size,
        "contamination"   : oif.contamination,
        "n_estimators"    : oif.n_estimators,
        "update_count"    : oif.update_count,
        "global_min"      : oif.global_min,
        "global_max"      : oif.global_max,
        "tier_thresholds" : oif.tier_thresholds,   # ← NEW: data-driven tiers
        "scaler"          : scaler,
        "label_encoders"  : label_encoders,
        "feature_columns" : feature_columns,
        "version"         : version,
        "last_updated"    : datetime.now().isoformat()
    }

    with open(save_path, "wb") as f:
        pickle.dump(model_package, f)

    logger.info(f"✅ Model saved: {save_path}")
    return version, save_path


# ─────────────────────────────────────────
#  LOAD MODEL
# ─────────────────────────────────────────
def load_model(model_path: str):
    """
    Load model package from .pkl file.
    Returns: (oif, scaler, feature_columns, full_package)
    """
    if not os.path.exists(model_path):
        logger.error(f"❌ Model file not found: {model_path}")
        return None

    with open(model_path, "rb") as f:
        package = pickle.load(f)

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

    # ── Load dynamic tier thresholds ──
    # Falls back to None for old packages — get_risk_tier() handles this gracefully
    oif.tier_thresholds = package.get("tier_thresholds", None)
    if oif.tier_thresholds:
        logger.info(f"   Tier thresholds loaded: "
                    f"Critical>={oif.tier_thresholds['critical_threshold']:.3f}, "
                    f"High>={oif.tier_thresholds['high_threshold']:.3f}, "
                    f"Medium>={oif.tier_thresholds['medium_threshold']:.3f}")
    else:
        logger.warning("⚠️ No tier_thresholds in package — re-run training to generate them.")

    # Backward compat: older packages may not have label_encoders
    if "label_encoders" not in package:
        logger.warning("⚠️ Model package has no label_encoders. Re-run training.")
        package["label_encoders"] = {}

    logger.info(f"✅ Model loaded: {package['version']}")
    return oif, package["scaler"], package["feature_columns"], package


# ─────────────────────────────────────────
#  MAIN TRAIN FUNCTION
# ─────────────────────────────────────────
def run_training(X: np.ndarray, feature_columns: list, scaler,
                 record_ids: list, label_encoders: dict = None):
    """
    Full training run:
    1. Check if active model exists
    2. If yes  → update same model, overwrite same file, update same DB row
    3. If no   → initial train, create model_v_1, insert first DB row
    4. Mark records as processed
    """
    logger.info("=" * 50)
    logger.info("  TRAINING STARTED")
    logger.info("=" * 50)

    active_model_info = db.get_active_model()
    existing_version  = None
    existing_path     = None

    if active_model_info and os.path.exists(active_model_info["model_path"]):
        logger.info(f"Found existing model: {active_model_info['model_version']}")
        existing_version = active_model_info["model_version"]
        existing_path    = active_model_info["model_path"]
        oif, _, _, _     = load_model(existing_path)
        oif.update(X)
    else:
        logger.info("No existing model found. Starting fresh training.")
        oif = OnlineIsolationForest(
            n_estimators  = cfg.MODEL_PARAMS["n_estimators"],
            contamination = cfg.MODEL_PARAMS["contamination"],
            window_size   = cfg.MODEL_PARAMS["window_size"],
            random_state  = cfg.MODEL_PARAMS["random_state"]
        )
        oif.initial_train(X)

    version, save_path = save_model(
        oif, scaler, feature_columns,
        label_encoders   = label_encoders or {},
        existing_version = existing_version,
        existing_path    = existing_path
    )

    db.update_model_registry(
        model_version      = version,
        model_path         = save_path,
        trained_on_records = len(X),
        notes              = f"Update #{oif.update_count}"
    )

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
    prep = load_module("preprocessing", os.path.join(ROOT_DIR, "src", "preprocessing.py"))
    X, feature_cols, scaler, raw_ids, label_encoders = prep.preprocess(fetch_all=False)

    if X is not None:
        oif, version = run_training(X, feature_cols, scaler, raw_ids, label_encoders)
        labels, scores = oif.predict(X[:5])
        print("\n── Sample Predictions ──")
        for i in range(5):
            tier = oif.get_risk_tier(scores[i])
            print(f"  Customer {i+1}: Score={scores[i]:.4f} | Risk={tier} | Label={labels[i]}")
        print(f"\n── Tier Thresholds ──")
        print(f"  {oif.tier_thresholds}")
    else:
        print("No data available for training.")