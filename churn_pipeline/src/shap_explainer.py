import os
import sys
import numpy as np
import logging

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

cfg = load_module("config", os.path.join(ROOT_DIR, "config.py"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_explainer(oif, X_background: np.ndarray):
    """
    Build a SHAP TreeExplainer from the trained IsolationForest.
    Returns shap.TreeExplainer or None.
    """
    try:
        import shap

        if X_background is None or len(X_background) == 0:
            logger.warning("⚠️ SHAP: X_background is empty — cannot build explainer")
            return None

        n_bg = min(200, len(X_background))
        rng  = np.random.default_rng(seed=42)
        idx  = rng.choice(len(X_background), size=n_bg, replace=False)
        bg   = X_background[idx]

        # oif.model is the underlying sklearn IsolationForest
        if not hasattr(oif, "model"):
            logger.warning("⚠️ SHAP: oif has no .model attribute — cannot build TreeExplainer")
            return None

        explainer = shap.TreeExplainer(oif.model, data=bg)
        logger.info(f"✅ SHAP TreeExplainer built — background size: {n_bg}")
        return explainer

    except ImportError:
        logger.warning("⚠️ shap not installed. Run: pip install shap")
        return None
    except Exception as e:
        import traceback
        logger.warning(f"⚠️ Could not build SHAP explainer: {e}")
        logger.warning(traceback.format_exc())
        return None


def explain_single(explainer, X_single: np.ndarray,
                   feature_names: list, top_n: int = 5) -> list:
    """
    Compute SHAP explanation for one customer row.
    Returns list of dicts sorted by absolute SHAP value descending.
    Returns empty list if explainer is None or fails.
    """
    if explainer is None:
        logger.warning("⚠️ explain_single called with None explainer")
        return []

    if X_single is None or len(X_single) == 0:
        logger.warning("⚠️ explain_single called with empty X_single")
        return []

    try:
        import shap

        shap_values = explainer.shap_values(X_single)

        # Handle both old shap (ndarray) and new shap (Explanation object)
        if hasattr(shap_values, "values"):
            sv_raw = np.array(shap_values.values)
        else:
            sv_raw = np.array(shap_values)

        # sv_raw could be (1, n_features) or (n_features,) or
        # (n_estimators, 1, n_features) for some shap versions
        sv_raw = sv_raw.squeeze()

        # If still multi-dimensional after squeeze, take mean across first axis
        if sv_raw.ndim > 1:
            sv_raw = sv_raw.mean(axis=0)

        sv = sv_raw.flatten()

        if len(sv) == 0:
            logger.warning("⚠️ SHAP values array is empty after processing")
            return []

        pairs = []
        for i, feat in enumerate(feature_names):
            val = float(sv[i]) if i < len(sv) else 0.0
            pairs.append({
                "feature"   : feat,
                "shap_value": round(val, 6),
                "abs_val"   : abs(val),
                "direction" : "increases_risk" if val > 0 else "decreases_risk"
            })

        pairs.sort(key=lambda x: x["abs_val"], reverse=True)

        top = pairs[:top_n]
        for rank, item in enumerate(top, start=1):
            item["rank"] = rank
            del item["abs_val"]

        if top:
            logger.info(f"✅ SHAP explanation computed — top: "
                        f"{top[0]['feature']} ({top[0]['shap_value']:+.4f})")
        return top

    except Exception as e:
        import traceback
        logger.warning(f"⚠️ SHAP explanation failed: {e}")
        logger.warning(traceback.format_exc())
        return []


def explain_batch(explainer, X_batch: np.ndarray,
                  feature_names: list, top_n: int = 3) -> list:
    if explainer is None:
        return [[] for _ in range(len(X_batch))]
    results = []
    for i in range(len(X_batch)):
        exp = explain_single(explainer, X_batch[i:i+1], feature_names, top_n=top_n)
        results.append(exp)
    return results


def summarise_explanation(explanation: list) -> str:
    if not explanation:
        return "Explanation not available."
    parts = []
    for item in explanation:
        arrow = "↑" if item["direction"] == "increases_risk" else "↓"
        parts.append(f"{item['feature']} ({arrow})")
    return "Risk driven by: " + ", ".join(parts)


if __name__ == "__main__":
    train = load_module("train", os.path.join(ROOT_DIR, "src", "train.py"))
    prep  = load_module("preprocessing", os.path.join(ROOT_DIR, "src", "preprocessing.py"))
    db    = load_module("db_connection", os.path.join(ROOT_DIR, "src", "db_connection.py"))
    active = db.get_active_model()

    if active:
        oif, scaler, feature_cols, package = train.load_model(active["model_path"])
        X, _, _, _, _ = prep.preprocess(fetch_all=True)
        if X is not None:
            explainer = build_explainer(oif, X)
            exp       = explain_single(explainer, X[0:1], feature_cols, top_n=5)
            summary   = summarise_explanation(exp)
            print("\n── SHAP Explanation for Customer 1 ──")
            for item in exp:
                print(f"  #{item['rank']} {item['feature']:20s} "
                      f"SHAP={item['shap_value']:+.4f}  [{item['direction']}]")
            print(f"\n  Summary: {summary}")
    else:
        print("No active model found. Run training first.")