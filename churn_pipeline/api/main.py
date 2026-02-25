import os
import sys
import numpy as np
import pandas as pd
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import importlib.util

# ─────────────────────────────────────────
#  PATH FIX
# ─────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

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
    level  = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
#  MODEL CACHE
# ─────────────────────────────────────────
MODEL_CACHE = {
    "oif"            : None,
    "scaler"         : None,
    "feature_columns": None,
    "label_encoders" : None,   # ← NEW: stores fitted encoders from training
    "version"        : None
}

def load_active_model():
    """
    Load the currently active model from DB registry.
    FIX: also loads label_encoders from the model package into MODEL_CACHE
    so preprocess_input() can use them for correct inference encoding.
    """
    try:
        active = db.get_active_model()
        if active is None:
            logger.warning("No active model found in registry.")
            return False

        oif, scaler, feature_columns, package = train.load_model(active["model_path"])
        MODEL_CACHE["oif"]             = oif
        MODEL_CACHE["scaler"]          = scaler
        MODEL_CACHE["feature_columns"] = feature_columns
        MODEL_CACHE["label_encoders"]  = package.get("label_encoders", {})
        MODEL_CACHE["version"]         = active["model_version"]

        encoder_count = len(MODEL_CACHE["label_encoders"])
        logger.info(f"✅ Model loaded: {active['model_version']} "
                    f"({encoder_count} label encoders available)")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False


# ─────────────────────────────────────────
#  LIFESPAN — replaces deprecated @app.on_event("startup")
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_active_model()
    yield
    # Shutdown (nothing to clean up)


# ─────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────
app = FastAPI(
    title       = "Customer Churn Detection API",
    description = "Unsupervised churn detection using Online Isolation Forest",
    version     = "1.0.0",
    lifespan    = lifespan
)

# Serve frontend static files
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ─────────────────────────────────────────
#  INPUT SCHEMAS
# ─────────────────────────────────────────
class CustomerInput(BaseModel):
    gender           : str
    SeniorCitizen    : int
    Partner          : str
    Dependents       : str
    tenure           : int
    PhoneService     : str
    MultipleLines    : str
    InternetService  : str
    OnlineSecurity   : str
    OnlineBackup     : str
    DeviceProtection : str
    TechSupport      : str
    StreamingTV      : str
    StreamingMovies  : str
    Contract         : str
    PaperlessBilling : str
    PaymentMethod    : str
    MonthlyCharges   : float
    TotalCharges     : float

class BatchInput(BaseModel):
    # FIX: use List[CustomerInput] from typing — compatible with Python 3.8+
    # Original list[CustomerInput] requires Python 3.9+
    customers: List[CustomerInput]

class ActivateModelInput(BaseModel):
    model_version      : str
    model_path         : str
    trained_on_records : int


# ─────────────────────────────────────────
#  PREPROCESSING FOR INFERENCE
# ─────────────────────────────────────────
def preprocess_input(data: dict) -> np.ndarray:
    """
    Preprocess a single customer dict for inference.

    FIX: uses the saved label_encoders from MODEL_CACHE instead of
    creating a fresh LabelEncoder and fitting it on a single row.

    The old approach was critically broken:
      le = LabelEncoder()
      le.fit_transform(["Female"])  →  [0]   ← "Female" = 0
      le.fit_transform(["Male"])    →  [0]   ← "Male" = 0 too!
    A fresh encoder fitted on 1 value always returns 0.

    The fix: encoders were fitted on the full training set in
    preprocessing.py, saved in train.py's model package, and loaded
    into MODEL_CACHE["label_encoders"] at startup. Unknown values at
    inference time fall back to -1 (handled safely below).
    """
    df = pd.DataFrame([data])

    label_encoders = MODEL_CACHE.get("label_encoders") or {}

    for col in cfg.CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        if col in label_encoders:
            le       = label_encoders[col]
            val      = str(df[col].iloc[0])
            # Handle unseen categories gracefully — map to -1
            if val in le.classes_:
                df[col] = le.transform([val])
            else:
                logger.warning(f"Unseen category '{val}' in column '{col}'. "
                               f"Encoding as -1.")
                df[col] = -1
        else:
            # Fallback for old model packages without saved encoders
            logger.warning(f"No saved encoder for '{col}'. "
                           f"Re-run training to fix inference encoding.")
            from sklearn.preprocessing import LabelEncoder as _LE
            _le    = _LE()
            df[col] = _le.fit_transform(df[col].astype(str))

    # Ensure correct column order then scale
    df = df[cfg.FEATURE_COLUMNS]
    X  = MODEL_CACHE["scaler"].transform(df)
    return X


# ─────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────

# ── Serve Frontend ────────────────────────
@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# ── Health Check ──────────────────────────
@app.get("/health")
async def health():
    return {
        "status"          : "running",
        "model_loaded"    : MODEL_CACHE["oif"] is not None,
        "model_version"   : MODEL_CACHE["version"],
        "encoders_loaded" : len(MODEL_CACHE["label_encoders"] or {}),
        "timestamp"       : datetime.now().isoformat()
    }


# ── Model Info ────────────────────────────
@app.get("/model-info")
async def model_info():
    active = db.get_active_model()
    if active is None:
        raise HTTPException(status_code=404, detail="No active model found")
    return {
        "model_version"     : active["model_version"],
        "training_date"     : str(active["training_date"]),
        "trained_on_records": int(active["trained_on_records"]),
        "is_active"         : bool(active["is_active"]),
        "notes"             : active["notes"]
    }


# ── Single Prediction ─────────────────────
@app.post("/predict")
async def predict(customer: CustomerInput):
    if MODEL_CACHE["oif"] is None:
        if not load_active_model():
            raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X              = preprocess_input(customer.dict())
        labels, scores = MODEL_CACHE["oif"].predict(X)
        score          = float(scores[0])
        label          = int(labels[0])
        risk_tier      = MODEL_CACHE["oif"].get_risk_tier(score)

        return {
            "churn_risk_score": round(score * 100, 2),
            "risk_tier"       : risk_tier,
            "anomaly_label"   : label,
            "is_anomaly"      : label == -1,
            "model_version"   : MODEL_CACHE["version"],
            "timestamp"       : datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Batch Prediction ──────────────────────
@app.post("/predict-batch")
async def predict_batch(batch: BatchInput):
    if MODEL_CACHE["oif"] is None:
        if not load_active_model():
            raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = []
        for customer in batch.customers:
            X              = preprocess_input(customer.dict())
            labels, scores = MODEL_CACHE["oif"].predict(X)
            score          = float(scores[0])
            label          = int(labels[0])
            risk_tier      = MODEL_CACHE["oif"].get_risk_tier(score)
            results.append({
                "churn_risk_score": round(score * 100, 2),
                "risk_tier"       : risk_tier,
                "anomaly_label"   : label,
                "is_anomaly"      : label == -1
            })

        return {
            "total"        : len(results),
            "predictions"  : results,
            "model_version": MODEL_CACHE["version"],
            "timestamp"    : datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Monitoring Metrics ────────────────────
@app.get("/monitoring")
async def monitoring(days: int = 30):
    try:
        df = db.fetch_recent_metrics(days=days)
        if df.empty:
            return {"metrics": [], "total": 0}

        if "logged_at" in df.columns:
            df["logged_at"] = df["logged_at"].astype(str)

        return {
            "metrics": df.to_dict(orient="records"),
            "total"  : len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Model Registry ────────────────────────
@app.get("/registry")
async def registry():
    try:
        df = db.get_all_models()
        if df.empty:
            return {"models": [], "total": 0}

        if "training_date" in df.columns:
            df["training_date"] = df["training_date"].astype(str)

        return {
            "models": df.to_dict(orient="records"),
            "total" : len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Dashboard Stats ───────────────────────
@app.get("/dashboard-stats")
async def dashboard_stats():
    try:
        engine       = db.get_engine()
        total_df     = pd.read_sql(
            "SELECT COUNT(*) as total FROM raw_customers", engine)
        processed_df = pd.read_sql(
            "SELECT COUNT(*) as total FROM raw_customers WHERE is_processed=1", engine)

        metrics_df = db.fetch_last_n_metrics(n=7)
        active     = db.get_active_model()
        risk_dist  = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}

        if active and MODEL_CACHE["oif"]:
            raw_df = pd.read_sql(
                "SELECT * FROM raw_customers WHERE is_processed=1 LIMIT 500",
                engine
            )
            if not raw_df.empty:
                drop_cols = ["id", "is_processed", "created_at"]
                raw_df    = raw_df.drop(
                    columns=[c for c in drop_cols if c in raw_df.columns])
                raw_df    = raw_df[cfg.FEATURE_COLUMNS]

                # FIX: coerce TotalCharges to numeric before scaling
                # IBM Telco dataset has blank TotalCharges for new customers
                # which are stored as strings — will crash scaler.transform()
                raw_df["TotalCharges"] = pd.to_numeric(
                    raw_df["TotalCharges"], errors="coerce"
                )
                raw_df["TotalCharges"] = raw_df["TotalCharges"].fillna(
                    raw_df["TotalCharges"].median()
                )

                # FIX: use saved label_encoders from MODEL_CACHE instead of
                # fresh LabelEncoder — same fix as preprocess_input()
                label_encoders = MODEL_CACHE.get("label_encoders") or {}
                for col in cfg.CATEGORICAL_COLUMNS:
                    if col not in raw_df.columns:
                        continue
                    if col in label_encoders:
                        le = label_encoders[col]
                        raw_df[col] = raw_df[col].astype(str).apply(
                            lambda v: le.transform([v])[0]
                            if v in le.classes_ else -1
                        )
                    else:
                        from sklearn.preprocessing import LabelEncoder as _LE
                        _le        = _LE()
                        raw_df[col] = _le.fit_transform(raw_df[col].astype(str))

                X_sample  = MODEL_CACHE["scaler"].transform(raw_df)
                _, scores = MODEL_CACHE["oif"].predict(X_sample)

                for s in scores:
                    tier = MODEL_CACHE["oif"].get_risk_tier(s)
                    risk_dist[tier] += 1

        trend_data = []
        if not metrics_df.empty:
            for _, row in metrics_df.iterrows():
                trend_data.append({
                    "date"              : str(row["logged_at"]),
                    "mean_anomaly_score": float(row["mean_anomaly_score"] or 0),
                    "anomaly_rate"      : float(row["anomaly_rate_pct"]   or 0),
                    "ks_distance"       : float(row["ks_distance"]        or 0),
                    "status"            : str(row["status"]               or "GOOD")
                })

        return {
            "total_customers"    : int(total_df.iloc[0]["total"]),
            "processed_customers": int(processed_df.iloc[0]["total"]),
            "model_version"      : MODEL_CACHE["version"] or "None",
            "risk_distribution"  : risk_dist,
            "trend_data"         : trend_data,
            "latest_status"      : trend_data[0]["status"] if trend_data else "GOOD"
        }
    except Exception as e:
        logger.error(f"Dashboard stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Reload Model ──────────────────────────
@app.post("/reload-model")
async def reload_model():
    success = load_active_model()
    if success:
        return {"message": f"Model reloaded: {MODEL_CACHE['version']}"}
    raise HTTPException(status_code=500, detail="Failed to reload model")


# ── Activate Model Version ────────────────
@app.post("/activate-model")
async def activate_model(data: ActivateModelInput):
    """
    Manually activate a specific model version.
    Uses register_model() (INSERT) intentionally here — this endpoint
    is for switching to a different model version, which is a legitimate
    case for creating a new active row rather than updating the current one.
    """
    try:
        db.register_model(
            model_version      = data.model_version,
            model_path         = data.model_path,
            trained_on_records = data.trained_on_records,
            notes              = "Manually activated"
        )
        load_active_model()
        return {"message": f"Activated: {data.model_version}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)