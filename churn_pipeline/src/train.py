import os
import sys
import pickle
import logging
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config as cfg
from src import db_connection as db
from src.preprocessing import preprocess
from sklearn.ensemble import IsolationForest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
#  LOAD ACTIVE MODEL PACKAGE
#  Returns model, scaler, label_encoders,
#  global_min, global_max from the saved .pkl
#  Returns None for all if no model exists yet.
# ─────────────────────────────────────────
def load_active_model_package():
    """
    Load the full model package (model + scaler + encoders + score bounds)
    from the path registered in model_registry.
    Returns (model, scaler, label_encoders, global_min, global_max)
    or (None, None, None, None, None) if no model is registered yet.
    """
    try:
        active = db.get_active_model()
        if active is None:
            return None, None, None, None, None

        path = active.get("model_path")
        if not path or not os.path.exists(path):
            logger.warning(f"⚠️  Model path not found: {path}")
            return None, None, None, None, None

        with open(path, "rb") as f:
            package = pickle.load(f)

        model          = package.get("model")
        scaler         = package.get("scaler")
        label_encoders = package.get("label_encoders", {})
        global_min     = package.get("global_min", 0.0)
        global_max     = package.get("global_max", 1.0)

        logger.info(f"✅ Loaded model package: {path}")
        return model, scaler, label_encoders, global_min, global_max

    except Exception as e:
        logger.error(f"❌ Failed to load model package: {e}")
        return None, None, None, None, None


# ─────────────────────────────────────────
#  SAVE MODEL PACKAGE
#  Saves model + scaler + encoders + score bounds
#  together in one .pkl so they are always in sync.
# ─────────────────────────────────────────
def save_model_package(model, scaler, label_encoders,
                       global_min, global_max, version: str) -> str:
    """
    Save all model artefacts together in one file.
    Keeping them together guarantees the scaler and
    encoders are always the ones the model was trained with.
    FIX: old code saved only the model — scaler and encoders
    were never persisted, so incremental runs always had to
    refit them on the new batch (wrong scale every time).
    """
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    path = os.path.join(cfg.MODELS_DIR, f"{version}.pkl")

    package = {
        "model"         : model,
        "scaler"        : scaler,
        "label_encoders": label_encoders,
        "global_min"    : global_min,
        "global_max"    : global_max,
        "version"       : version,
    }
    with open(path, "wb") as f:
        pickle.dump(package, f)

    logger.info(f"✅ Model package saved: {path}")
    return path


# ─────────────────────────────────────────
#  NEXT VERSION NAME
# ─────────────────────────────────────────
def _next_version() -> str:
    """
    FIX: old logic searched for the highest existing
    model_vN.pkl file, but if model_v1.pkl was manually
    deleted it would create another model_v1 and silently
    overwrite the registry entry.
    Now reads the highest version number from model_registry
    in the DB instead — source of truth is the DB not the
    filesystem.
    """
    try:
        all_models = db.get_all_models()
        if all_models.empty:
            return "model_v1"

        versions = all_models["model_version"].tolist()
        numbers  = []
        for v in versions:
            try:
                numbers.append(int(v.replace("model_v", "")))
            except ValueError:
                pass

        next_n = max(numbers) + 1 if numbers else 1
        return f"model_v{next_n}"
    except Exception:
        return "model_v1"


# ─────────────────────────────────────────
#  INITIAL TRAIN
#  Fits a fresh model on all processed records.
#  Called once on first run or after a full reset.
# ─────────────────────────────────────────
def initial_train():
    """
    Full train from scratch on all records in raw_customers.
    Fits fresh scaler and label encoders — correct because
    we have the full dataset available.
    Saves model + scaler + encoders together in one package.
    """
    logger.info("=" * 55)
    logger.info("  INITIAL TRAIN STARTED")
    logger.info("=" * 55)

    # preprocess with no scaler/encoders → fits fresh ones on full data
    X, feature_cols, scaler, raw_ids, label_encoders = preprocess(
        fetch_all=True,
        scaler=None,
        label_encoders=None,
    )

    if X is None:
        logger.error("❌ Preprocessing returned no data. Aborting.")
        return None

    logger.info(f"Training IsolationForest on {X.shape[0]} records...")
    model = IsolationForest(
        n_estimators =cfg.MODEL_PARAMS["n_estimators"],
        contamination=cfg.MODEL_PARAMS["contamination"],
        random_state =cfg.MODEL_PARAMS["random_state"],
    )
    model.fit(X)

    # Score bounds — computed once on full training set
    # FIX: old code recomputed global_min/max from each
    # incremental window, making score ranges shift every
    # retrain. Storing them here keeps the scale stable.
    raw_scores = model.decision_function(X)
    scores_inv = -raw_scores                        # higher = more anomalous
    global_min = float(scores_inv.min())
    global_max = float(scores_inv.max())
    logger.info(f"Score bounds — min: {global_min:.4f}, max: {global_max:.4f}")

    version = _next_version()
    path    = save_model_package(
        model, scaler, label_encoders, global_min, global_max, version
    )

    db.register_model(
        model_version      =version,
        model_path         =path,
        trained_on_records =X.shape[0],
        notes              ="Initial train",
    )

    logger.info("=" * 55)
    logger.info(f"  ✅ INITIAL TRAIN COMPLETE — version: {version}")
    logger.info("=" * 55)
    return version


# ─────────────────────────────────────────
#  INCREMENTAL UPDATE
#  Processes only new (unprocessed) records.
#  Reuses the saved scaler and encoders so the
#  new data is on exactly the same scale as the
#  model's training data.
# ─────────────────────────────────────────
def incremental_update():
    """
    Update model with new unprocessed records only.

    FIX — CRITICAL: the old version called preprocess()
    with no scaler argument, so it always fitted a fresh
    scaler on the new batch alone. 50 new customers → scaler
    fitted on 50 rows → completely different scale than the
    7043-row training distribution → model scores garbage.

    Fix: load the saved scaler and label_encoders from the
    active model package and pass them into preprocess() so
    new data is transformed on the original training scale.
    """
    logger.info("=" * 55)
    logger.info("  INCREMENTAL UPDATE STARTED")
    logger.info("=" * 55)

    # Load current model + its scaler + its encoders
    model, scaler, label_encoders, global_min, global_max = load_active_model_package()

    if model is None:
        logger.warning("⚠️  No active model found. Running initial train instead.")
        return initial_train()

    # preprocess with saved scaler + encoders → transform() only, no refit
    X, feature_cols, scaler, raw_ids, label_encoders = preprocess(
        fetch_all=False,
        scaler=scaler,                  # ← reuse saved scaler
        label_encoders=label_encoders,  # ← reuse saved encoders
    )

    if X is None or len(X) == 0:
        logger.info("ℹ️  No new records to process.")
        return None

    logger.info(f"Updating model with {X.shape[0]} new records...")

    # Partial fit: add new trees to existing forest
    # IsolationForest does not have partial_fit natively —
    # we retrain on the window of existing + new data.
    window_size = cfg.MODEL_PARAMS.get("window_size", 8000)

    # Fetch recent processed data to form the training window
    all_processed = db.fetch_all_processed_customers()
    if all_processed is not None and not all_processed.empty:
        # Preprocess the full window using the same saved scaler/encoders
        X_all, _, _, _, _ = preprocess(
            fetch_all=True,
            scaler=scaler,
            label_encoders=label_encoders,
        )
        if X_all is not None and len(X_all) > window_size:
            X_window = X_all[-window_size:]
        elif X_all is not None:
            X_window = X_all
        else:
            X_window = X
    else:
        X_window = X

    model = IsolationForest(
        n_estimators =cfg.MODEL_PARAMS["n_estimators"],
        contamination=cfg.MODEL_PARAMS["contamination"],
        random_state =cfg.MODEL_PARAMS["random_state"],
    )
    model.fit(X_window)

    # FIX: do NOT recompute global_min/max from the window —
    # that would shift the score scale every update and make
    # historical comparisons meaningless. Keep the bounds from
    # the initial train and only expand them if the new data
    # genuinely exceeds the original range.
    raw_scores = model.decision_function(X_window)
    scores_inv = -raw_scores
    new_min    = float(scores_inv.min())
    new_max    = float(scores_inv.max())
    global_min = min(global_min, new_min)
    global_max = max(global_max, new_max)
    logger.info(f"Score bounds — min: {global_min:.4f}, max: {global_max:.4f}")

    # Mark new records as processed
    if raw_ids:
        db.mark_customers_processed(raw_ids)

    # Save updated package with same scaler/encoders
    active     = db.get_active_model()
    version    = active["model_version"] if active else _next_version()
    path       = save_model_package(
        model, scaler, label_encoders, global_min, global_max, version
    )

    db.update_model_registry(
        model_version      =version,
        model_path         =path,
        trained_on_records =X_window.shape[0],
        notes              =f"Incremental update +{X.shape[0]} records",
    )

    logger.info("=" * 55)
    logger.info(f"  ✅ INCREMENTAL UPDATE COMPLETE — {X.shape[0]} new records processed")
    logger.info("=" * 55)
    return version


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train ChurnGuard model")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force a full retrain from scratch (default: incremental update)"
    )
    args = parser.parse_args()

    if args.full:
        initial_train()
    else:
        incremental_update()