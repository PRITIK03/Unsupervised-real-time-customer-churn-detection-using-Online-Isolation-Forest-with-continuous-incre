import os
import sys
import numpy as np
import logging

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

cfg = load_module("config", os.path.join(ROOT_DIR, "config.py"))

# ─────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
#  BUILD SHAP EXPLAINER
#  Called once after training and saved into
#  the model package alongside the IsoForest.
#
#  Why TreeExplainer?
#  IsolationForest is a tree-based ensemble.
#  TreeExplainer is the correct and efficient
#  SHAP method for tree models — it computes
#  exact Shapley values, not approximations.
# ─────────────────────────────────────────
def build_explainer(oif, X_background: np.ndarray):
    """
    Build a SHAP TreeExplainer from the trained IsolationForest.

    Args:
        oif          : OnlineIsolationForest instance (must be trained)
        X_background : numpy array of training data used as background
                       for SHAP — a sample of 200 rows is enough

    Returns:
        shap.TreeExplainer instance, or None if shap not installed
    """
    try:
        import shap

        # Use a subsample as background to keep it fast
        n_bg = min(200, len(X_background))
        rng  = np.random.default_rng(seed=42)
        idx  = rng.choice(len(X_background), size=n_bg, replace=False)
        bg   = X_background[idx]

        explainer = shap.TreeExplainer(oif.model, data=bg)
        logger.info(f"✅ SHAP TreeExplainer built — background size: {n_bg}")
        return explainer

    except ImportError:
        logger.warning("⚠️ shap not installed. Run: pip install shap")
        return None
    except Exception as e:
        logger.warning(f"⚠️ Could not build SHAP explainer: {e}")
        return None


# ─────────────────────────────────────────
#  EXPLAIN SINGLE CUSTOMER
#  Given one preprocessed row (1 x 19),
#  returns the top N features driving the
#  anomaly score — with direction and magnitude.
#
#  SHAP value interpretation for IsoForest:
#  Positive SHAP → feature INCREASES anomaly (pushes toward risk)
#  Negative SHAP → feature DECREASES anomaly (pushes toward normal)
# ─────────────────────────────────────────
def explain_single(explainer, X_single: np.ndarray,
                   feature_names: list, top_n: int = 5) -> list:
    """
    Compute SHAP explanation for one customer row.

    Args:
        explainer    : shap.TreeExplainer (from build_explainer)
        X_single     : (1, 19) numpy array — preprocessed customer features
        feature_names: list of 19 feature column names
        top_n        : number of top features to return

    Returns:
        List of dicts sorted by absolute SHAP value descending:
        [
          {
            "feature"   : "Contract",
            "value"     : "Month-to-month",   ← raw encoded value
            "shap_value": 0.142,              ← contribution to anomaly
            "direction" : "increases_risk",   ← or "decreases_risk"
            "rank"      : 1
          },
          ...
        ]
        Returns empty list if explainer is None.
    """
    if explainer is None:
        return []

    try:
        import shap

        shap_values = explainer.shap_values(X_single)

        # shap_values shape: (1, 19) or (19,) depending on shap version
        if hasattr(shap_values, "values"):
            sv = np.array(shap_values.values).flatten()
        else:
            sv = np.array(shap_values).flatten()

        # Pair each feature with its SHAP value
        pairs = []
        for i, feat in enumerate(feature_names):
            val = float(sv[i]) if i < len(sv) else 0.0
            pairs.append({
                "feature"   : feat,
                "shap_value": round(val, 6),
                "abs_val"   : abs(val),
                "direction" : "increases_risk" if val > 0 else "decreases_risk"
            })

        # Sort by absolute contribution descending
        pairs.sort(key=lambda x: x["abs_val"], reverse=True)

        # Take top N and add rank
        top = pairs[:top_n]
        for rank, item in enumerate(top, start=1):
            item["rank"] = rank
            del item["abs_val"]   # clean up internal field

        logger.info(f"✅ SHAP explanation computed — top feature: "
                    f"{top[0]['feature']} ({top[0]['shap_value']:+.4f})" if top else "no features")
        return top

    except Exception as e:
        logger.warning(f"⚠️ SHAP explanation failed: {e}")
        return []


# ─────────────────────────────────────────
#  EXPLAIN BATCH
#  Same as explain_single but for N rows.
#  Returns a list of explanation lists.
#  Used by /predict-batch endpoint.
# ─────────────────────────────────────────
def explain_batch(explainer, X_batch: np.ndarray,
                  feature_names: list, top_n: int = 3) -> list:
    """
    Compute SHAP explanations for a batch of customers.
    Returns top_n=3 by default (lighter for batch responses).
    """
    if explainer is None:
        return [[] for _ in range(len(X_batch))]

    results = []
    for i in range(len(X_batch)):
        exp = explain_single(explainer, X_batch[i:i+1], feature_names, top_n=top_n)
        results.append(exp)
    return results


# ─────────────────────────────────────────
#  HUMAN-READABLE SUMMARY
#  Converts raw SHAP output into plain English
#  sentences for display in the frontend.
# ─────────────────────────────────────────
def summarise_explanation(explanation: list) -> str:
    """
    Convert top SHAP features into a plain English summary.

    Example output:
      "Risk driven by: Contract (↑), tenure (↓), MonthlyCharges (↑)"
    """
    if not explanation:
        return "Explanation not available."

    parts = []
    for item in explanation:
        arrow = "↑" if item["direction"] == "increases_risk" else "↓"
        parts.append(f"{item['feature']} ({arrow})")

    return "Risk driven by: " + ", ".join(parts)


# ─────────────────────────────────────────
#  TEST
# ─────────────────────────────────────────
if __name__ == "__main__":
    import pickle

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