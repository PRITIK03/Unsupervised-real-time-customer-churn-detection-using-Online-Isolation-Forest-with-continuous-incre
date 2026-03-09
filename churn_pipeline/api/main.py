import os
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from datetime import timedelta, datetime
from passlib.context import CryptContext

# ─────────────────────────────────────────
#  AUTH CONFIG
#  FIX 1: SECRET_KEY was "supersecretkey" hardcoded
#  in source code committed to a public GitHub repo.
#  Now loaded from .env via config.py.
#
#  FIX 2: ACCESS_TOKEN_EXPIRE_MINUTES was hardcoded.
#  Now loaded from config.py.
# ─────────────────────────────────────────
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import importlib.util
def _load_cfg():
    spec = importlib.util.spec_from_file_location("config", os.path.join(ROOT_DIR, "config.py"))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
cfg_auth = _load_cfg()

SECRET_KEY                  = cfg_auth.SECRET_KEY
ALGORITHM                   = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = cfg_auth.ACCESS_TOKEN_EXPIRE_MINUTES

# ─────────────────────────────────────────
#  FIX 3: verify_password previously did plain
#  string comparison:
#      return plain_password == hashed_password
#  "adminpass" was stored as plaintext — no hashing.
#  Now uses bcrypt via passlib. Timing-safe.
#
#  FIX 4: fake_users_db now loads the password hash
#  from .env via config.py — never plaintext in source.
#  Generate a hash with:
#    python -c "from passlib.context import CryptContext; \
#               ctx=CryptContext(schemes=['bcrypt']); \
#               print(ctx.hash('your_password'))"
#  Then set ADMIN_PASSWORD_HASH in your .env file.
# ─────────────────────────────────────────
pwd_context   = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

fake_users_db = {
    cfg_auth.ADMIN_USERNAME: {
        "username"       : cfg_auth.ADMIN_USERNAME,
        "full_name"      : "Admin User",
        "hashed_password": cfg_auth.ADMIN_PASSWORD_HASH,
        "disabled"       : False,
    }
}

def verify_password(plain_password, hashed_password):
    # FIX: was `return plain_password == hashed_password`
    # Plain string comparison with no hashing whatsoever.
    # Now uses bcrypt verify — hash never leaves .env.
    if not hashed_password:
        return False  # no hash configured — reject all logins safely
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = fake_users_db.get(username)
    if user is None:
        raise credentials_exception
    return user

def require_auth(user: dict = Depends(get_current_user)):
    return user


import numpy as np
import pandas as pd
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import importlib.util

# ─────────────────────────────────────────
#  PATH FIX
# ─────────────────────────────────────────
# ROOT_DIR and sys.path already set above in auth section

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

cfg     = load_module("config",         os.path.join(ROOT_DIR, "config.py"))
db      = load_module("db_connection",  os.path.join(ROOT_DIR, "src", "db_connection.py"))
train   = load_module("train",          os.path.join(ROOT_DIR, "src", "train.py"))
shap_ex = load_module("shap_explainer", os.path.join(ROOT_DIR, "src", "shap_explainer.py"))

# ─────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
#  MODEL CACHE
# ─────────────────────────────────────────
MODEL_CACHE = {
    "oif"            : None,
    "scaler"         : None,
    "feature_columns": None,
    "label_encoders" : None,
    "version"        : None,
    "shap_explainer" : None,
    "X_background"   : None,
}


# ─────────────────────────────────────────
#  BACKGROUND DATA BUILDER
#  Shared helper — builds encoded X_bg from DB.
#  Called at startup and lazily on demand.
# ─────────────────────────────────────────
def _build_background_data():
    """
    Fetch up to 300 processed customers from DB,
    encode them, scale them, return numpy array.
    Returns None if no data or encoding fails.
    """
    try:
        engine = db.get_engine()
        bg_df = pd.read_sql(
            "SELECT * FROM raw_customers WHERE is_processed=1 LIMIT 300",
            engine
        )
        if bg_df.empty:
            logger.warning("⚠️ SHAP: raw_customers table has no processed rows (is_processed=1). "
                           "Run the pipeline first to populate data.")
            return None

        drop_cols = ["id", "is_processed", "created_at"]
        bg_df = bg_df.drop(
            columns=[c for c in drop_cols if c in bg_df.columns], errors="ignore"
        )

        missing = [c for c in cfg.FEATURE_COLUMNS if c not in bg_df.columns]
        if missing:
            logger.warning(f"⚠️ SHAP background missing columns: {missing}")
            return None

        bg_df = bg_df[cfg.FEATURE_COLUMNS].copy()
        bg_df["TotalCharges"] = pd.to_numeric(bg_df["TotalCharges"], errors="coerce")
        bg_df["TotalCharges"] = bg_df["TotalCharges"].fillna(bg_df["TotalCharges"].median())

        label_encoders = MODEL_CACHE.get("label_encoders") or {}
        for col in cfg.CATEGORICAL_COLUMNS:
            if col not in bg_df.columns:
                continue
            if col in label_encoders:
                le = label_encoders[col]
                bg_df[col] = bg_df[col].astype(str).apply(
                    lambda v, _le=le: int(_le.transform([v])[0]) if v in _le.classes_ else -1
                )
            else:
                from sklearn.preprocessing import LabelEncoder as _LE
                _le = _LE()
                bg_df[col] = _le.fit_transform(bg_df[col].astype(str))

        X_bg = MODEL_CACHE["scaler"].transform(bg_df)
        logger.info(f"✅ SHAP background data built — shape: {X_bg.shape}")
        return X_bg

    except Exception as e:
        import traceback
        logger.warning(f"⚠️ SHAP background build failed: {e}")
        logger.warning(traceback.format_exc())
        return None


# ─────────────────────────────────────────
#  BUILD SHAP EXPLAINER
# ─────────────────────────────────────────
def _build_shap_explainer(X_bg):
    """Try to build SHAP explainer from background data. Returns explainer or None."""
    if X_bg is None or MODEL_CACHE["oif"] is None:
        return None
    try:
        explainer = shap_ex.build_explainer(MODEL_CACHE["oif"], X_bg)
        if explainer is not None:
            logger.info("✅ SHAP explainer built via shap_ex.build_explainer")
            return explainer
    except Exception as e:
        logger.warning(f"⚠️ shap_ex.build_explainer failed: {e}, trying inline...")

    try:
        import shap as _shap
        n_bg = min(200, len(X_bg))
        idx = np.random.default_rng(42).choice(len(X_bg), size=n_bg, replace=False)
        explainer = _shap.TreeExplainer(MODEL_CACHE["oif"].model, data=X_bg[idx])
        logger.info(f"✅ SHAP explainer built inline — bg={n_bg}")
        return explainer
    except Exception as e:
        import traceback
        logger.warning(f"⚠️ Inline SHAP build also failed: {e}")
        logger.warning(traceback.format_exc())
        return None


def load_active_model():
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
        MODEL_CACHE["shap_explainer"]  = None
        MODEL_CACHE["X_background"]    = None

        X_bg = _build_background_data()
        if X_bg is not None:
            MODEL_CACHE["X_background"]   = X_bg
            MODEL_CACHE["shap_explainer"] = _build_shap_explainer(X_bg)

            if oif.tier_thresholds is None:
                try:
                    shap_exp = MODEL_CACHE.get("shap_explainer")
                    raw_scores = oif.model.score_samples(X_bg)
                    norm_scores = oif._normalize(raw_scores)

                    if shap_exp is not None:
                        n_sample = min(50, len(X_bg))
                        idx = np.random.default_rng(42).choice(len(X_bg), size=n_sample, replace=False)
                        adj_scores = []
                        for i in idx:
                            sv = shap_exp.shap_values(X_bg[i:i+1])
                            sv = np.array(sv.values if hasattr(sv, "values") else sv).flatten()
                            rsk = float(np.sum(sv[sv > 0]))
                            sfe = float(np.abs(np.sum(sv[sv < 0])))
                            if (rsk + sfe) > 0:
                                rr = rsk / (rsk + sfe)
                                f = (0.20 + 1.20 * rr) if rr < 0.5 else (1.0 + 0.30 * (rr - 0.5))
                            else:
                                f = 1.0
                            adj_scores.append(float(np.clip(norm_scores[i] * f, 0, 1)))
                        adj_scores = np.array(adj_scores)
                        oif.tier_thresholds = train.compute_dynamic_tiers(adj_scores)
                        logger.info(f"✅ Dynamic tiers from SHAP-adjusted scores (n={n_sample})")
                    else:
                        oif.tier_thresholds = train.compute_dynamic_tiers(norm_scores)
                        logger.info(f"✅ Dynamic tiers from raw scores (n={len(X_bg)})")
                except Exception as e:
                    import traceback
                    logger.warning(f"⚠️ Could not compute tiers: {e}")
                    logger.warning(traceback.format_exc())

        shap_status   = "ready" if MODEL_CACHE["shap_explainer"] is not None else "unavailable"
        encoder_count = len(MODEL_CACHE["label_encoders"])
        logger.info(f"✅ Model loaded: {active['model_version']} "
                    f"({encoder_count} label encoders, SHAP={shap_status})")
        return True

    except Exception as e:
        import traceback
        logger.error(f"❌ Failed to load model: {e}")
        logger.error(traceback.format_exc())
        return False


def _ensure_shap_explainer():
    """Lazily build SHAP explainer if not yet available."""
    if MODEL_CACHE["shap_explainer"] is not None:
        return
    if MODEL_CACHE["oif"] is None or MODEL_CACHE["scaler"] is None:
        return
    logger.info("🔄 Lazy SHAP build triggered...")
    X_bg = MODEL_CACHE["X_background"] or _build_background_data()
    if X_bg is not None:
        MODEL_CACHE["X_background"]   = X_bg
        MODEL_CACHE["shap_explainer"] = _build_shap_explainer(X_bg)


# ─────────────────────────────────────────
#  LIFESPAN
# ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_active_model()
    yield


# ─────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Detection API",
    description="Unsupervised churn detection using Online Isolation Forest + SHAP",
    version="2.0.0",
    lifespan=lifespan
)

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ─────────────────────────────────────────
#  VALID CATEGORY VALUES
# ─────────────────────────────────────────
VALID_GENDER         = ["Male", "Female"]
VALID_YES_NO         = ["Yes", "No"]
VALID_MULTIPLE_LINES = ["Yes", "No", "No phone service"]
VALID_INTERNET       = ["DSL", "Fiber optic", "No"]
VALID_INTERNET_ADDON = ["Yes", "No", "No internet service"]
VALID_CONTRACT       = ["Month-to-month", "One year", "Two year"]
VALID_PAYMENT_METHOD = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
]


# ─────────────────────────────────────────
#  INPUT SCHEMAS
# ─────────────────────────────────────────
class CustomerInput(BaseModel):
    gender           : str   = Field(..., description="Male or Female")
    SeniorCitizen    : int   = Field(..., ge=0, le=1)
    Partner          : str
    Dependents       : str
    tenure           : int   = Field(..., ge=0, le=120)
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
    MonthlyCharges   : float = Field(..., ge=0, le=500)
    TotalCharges     : float = Field(..., ge=0, le=100000)

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        if v not in VALID_GENDER:
            raise ValueError(f"gender must be one of {VALID_GENDER}")
        return v

    @field_validator("Partner", "Dependents", "PhoneService", "PaperlessBilling")
    @classmethod
    def validate_yes_no(cls, v):
        if v not in VALID_YES_NO:
            raise ValueError(f"Value must be one of {VALID_YES_NO}")
        return v

    @field_validator("MultipleLines")
    @classmethod
    def validate_multiple_lines(cls, v):
        if v not in VALID_MULTIPLE_LINES:
            raise ValueError(f"MultipleLines must be one of {VALID_MULTIPLE_LINES}")
        return v

    @field_validator("InternetService")
    @classmethod
    def validate_internet(cls, v):
        if v not in VALID_INTERNET:
            raise ValueError(f"InternetService must be one of {VALID_INTERNET}")
        return v

    @field_validator("OnlineSecurity", "OnlineBackup", "DeviceProtection",
                     "TechSupport", "StreamingTV", "StreamingMovies")
    @classmethod
    def validate_internet_addon(cls, v):
        if v not in VALID_INTERNET_ADDON:
            raise ValueError(f"Value must be one of {VALID_INTERNET_ADDON}")
        return v

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v):
        if v not in VALID_CONTRACT:
            raise ValueError(f"Contract must be one of {VALID_CONTRACT}")
        return v

    @field_validator("PaymentMethod")
    @classmethod
    def validate_payment_method(cls, v):
        if v not in VALID_PAYMENT_METHOD:
            raise ValueError(f"PaymentMethod must be one of {VALID_PAYMENT_METHOD}")
        return v


class BatchInput(BaseModel):
    customers: List[CustomerInput]

class ActivateModelInput(BaseModel):
    model_version      : str
    model_path         : str
    trained_on_records : int

class TimelineInput(BaseModel):
    churn_risk_score : float = Field(..., ge=0, le=100)
    contract         : str
    tenure           : int   = Field(..., ge=0, le=120)
    monthly_charges  : float = Field(..., ge=0, le=500)

    @field_validator("contract")
    @classmethod
    def validate_contract(cls, v):
        if v not in VALID_CONTRACT:
            raise ValueError(f"contract must be one of {VALID_CONTRACT}")
        return v


# ─────────────────────────────────────────
#  SHARED HELPER — encode + score raw_df
# ─────────────────────────────────────────
def _encode_and_score(raw_df: pd.DataFrame):
    if MODEL_CACHE["oif"] is None:
        return None, None

    drop_cols    = ["id", "is_processed", "created_at"]
    df           = raw_df.drop(columns=[c for c in drop_cols if c in raw_df.columns], errors="ignore")
    contract_col = df["Contract"].copy() if "Contract" in df.columns else None
    df           = df[cfg.FEATURE_COLUMNS].copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    label_encoders = MODEL_CACHE.get("label_encoders") or {}
    for col in cfg.CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        if col in label_encoders:
            le = label_encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda v: le.transform([v])[0] if v in le.classes_ else -1
            )
        else:
            from sklearn.preprocessing import LabelEncoder as _LE
            _le = _LE()
            df[col] = _le.fit_transform(df[col].astype(str))

    X = MODEL_CACHE["scaler"].transform(df)
    _, scores = MODEL_CACHE["oif"].predict(X)
    return scores, contract_col


# ─────────────────────────────────────────
#  PREPROCESSING FOR INFERENCE
# ─────────────────────────────────────────
def preprocess_input(data: dict) -> np.ndarray:
    df             = pd.DataFrame([data])
    label_encoders = MODEL_CACHE.get("label_encoders") or {}

    for col in cfg.CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        if col in label_encoders:
            le  = label_encoders[col]
            val = str(df[col].iloc[0])
            df[col] = le.transform([val]) if val in le.classes_ else -1
        else:
            from sklearn.preprocessing import LabelEncoder as _LE
            _le = _LE()
            df[col] = _le.fit_transform(df[col].astype(str))

    df = df[cfg.FEATURE_COLUMNS]
    X  = MODEL_CACHE["scaler"].transform(df)
    return X


# ─────────────────────────────────────────
#  TRAJECTORY SIMULATION
# ─────────────────────────────────────────
def _simulate_trajectory(base_score, contract, tenure, monthly_charges):
    tc = cfg.TRAJECTORY_CONFIG

    monthly_drift   = tc["drift_rates"].get(contract, tc["drift_rates"]["default"])
    charge_pressure = max(
        0.0,
        (monthly_charges - tc["charge_pressure_threshold"]) / tc["charge_pressure_scale"]
    )
    loyalty_reduction = min(
        tenure * tc["loyalty_reduction_per_month"],
        tc["loyalty_reduction_cap"]
    )

    trajectory = []
    score      = float(base_score)
    rng        = np.random.default_rng(seed=int(base_score * 100))

    for _ in range(tc["n_months"]):
        noise       = rng.uniform(-tc["noise_range"], tc["noise_range"])
        month_delta = monthly_drift + charge_pressure - loyalty_reduction + noise
        score       = float(np.clip(score + month_delta, tc["score_min"], tc["score_max"]))
        trajectory.append(round(score, 2))

    return trajectory


# ─────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    response = FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"]        = "no-cache"
    response.headers["Expires"]       = "0"
    return response


@app.get("/health")
async def health():
    shap_ready = MODEL_CACHE["shap_explainer"] is not None
    return {
        "status"         : "running",
        "model_loaded"   : MODEL_CACHE["oif"] is not None,
        "model_version"  : MODEL_CACHE["version"],
        "encoders_loaded": len(MODEL_CACHE["label_encoders"] or {}),
        "shap_ready"     : shap_ready,
        "timestamp"      : datetime.now().isoformat()
    }

@app.get("/health-check")
def health_check():
    return {"status": "Server is healthy"}


@app.get("/model-info")
async def model_info():
    active = db.get_active_model()
    if active is None:
        raise HTTPException(status_code=404, detail="No active model found")
    oif = MODEL_CACHE["oif"]
    return {
        "model_version"      : active["model_version"],
        "training_date"      : str(active["training_date"]),
        "trained_on_records" : int(active["trained_on_records"]),
        "is_active"          : bool(active["is_active"]),
        "notes"              : active["notes"],
        "tier_thresholds"    : oif.tier_thresholds if oif else None,
    }


# ── Single Prediction ─────────────────────
# NOTE: require_auth removed — frontend sends
# no token. Re-add Depends(require_auth) later
# once you add login flow to the frontend.
@app.post("/predict")
async def predict(customer: CustomerInput):
    if MODEL_CACHE["oif"] is None:
        if not load_active_model():
            raise HTTPException(status_code=503, detail="Model not loaded")

    _ensure_shap_explainer()

    try:
        customer_dict  = customer.dict()
        X              = preprocess_input(customer_dict)
        labels, scores = MODEL_CACHE["oif"].predict(X)
        raw_score      = float(scores[0])
        label          = int(labels[0])

        explanation = []
        summary     = ""
        exp_obj     = MODEL_CACHE.get("shap_explainer")
        shap_values_all = None

        if exp_obj is not None:
            try:
                sv_raw     = exp_obj.shap_values(X)
                sv         = np.array(
                    sv_raw.values if hasattr(sv_raw, "values") else sv_raw
                ).flatten()
                shap_values_all = sv
                feat_names = MODEL_CACHE["feature_columns"] or cfg.FEATURE_COLUMNS

                pairs = []
                for i, feat in enumerate(feat_names):
                    val = float(sv[i]) if i < len(sv) else 0.0
                    pairs.append({
                        "feature"   : feat,
                        "shap_value": round(val, 6),
                        "abs_val"   : abs(val),
                        "direction" : "increases_risk" if val > 0 else "decreases_risk"
                    })

                pairs.sort(key=lambda x: x["abs_val"], reverse=True)
                explanation = pairs[:5]
                for rank, item in enumerate(explanation, start=1):
                    item["rank"] = rank
                    del item["abs_val"]

                parts   = [f"{it['feature']} ({'↑' if it['direction'] == 'increases_risk' else '↓'})" for it in explanation]
                summary = "Risk driven by: " + ", ".join(parts)
                logger.info(f"✅ SHAP OK — top feature: {explanation[0]['feature']}")

            except Exception as e:
                import traceback
                logger.warning(f"⚠️ SHAP values failed: {e}")
                logger.warning(traceback.format_exc())
        else:
            logger.warning("⚠️ SHAP explainer is None at predict time — explanation will be empty")

        score = raw_score
        if shap_values_all is not None and len(shap_values_all) > 0:
            risk_sum = float(np.sum(shap_values_all[shap_values_all > 0]))
            safe_sum = float(np.abs(np.sum(shap_values_all[shap_values_all < 0])))

            if (risk_sum + safe_sum) > 0:
                risk_ratio = risk_sum / (risk_sum + safe_sum)

                if risk_ratio < 0.5:
                    factor = 0.20 + 1.20 * risk_ratio
                else:
                    factor = 1.0 + 0.30 * (risk_ratio - 0.5)

                dampened = np.clip(score * factor, 0, 1)
                logger.info(f"📊 Score adjustment: raw={score:.4f}, risk_ratio={risk_ratio:.3f}, "
                            f"factor={factor:.3f}, adjusted={dampened:.4f}")
                score = dampened

        risk_tier = MODEL_CACHE["oif"].get_risk_tier(score)

        return {
            "churn_risk_score"   : round(score * 100, 2),
            "risk_tier"          : risk_tier,
            "anomaly_label"      : label,
            "is_anomaly"         : label == -1,
            "model_version"      : MODEL_CACHE["version"],
            "tier_thresholds"    : MODEL_CACHE["oif"].tier_thresholds,
            "explanation"        : explanation,
            "explanation_summary": summary,
            "timestamp"          : datetime.now().isoformat()
        }

    except Exception as e:
        import traceback
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
async def explain(customer: CustomerInput):
    if MODEL_CACHE["oif"] is None:
        if not load_active_model():
            raise HTTPException(status_code=503, detail="Model not loaded")

    _ensure_shap_explainer()

    try:
        customer_dict  = customer.dict()
        X              = preprocess_input(customer_dict)
        labels, scores = MODEL_CACHE["oif"].predict(X)
        score          = float(scores[0])
        risk_tier      = MODEL_CACHE["oif"].get_risk_tier(score)

        exp_obj = MODEL_CACHE.get("shap_explainer")
        if exp_obj is None:
            raise HTTPException(
                status_code=503,
                detail="SHAP explainer not available. Run pipeline first to populate data."
            )

        explanation = shap_ex.explain_single(
            exp_obj,
            X,
            MODEL_CACHE["feature_columns"] or cfg.FEATURE_COLUMNS,
            top_n=10
        )
        summary = shap_ex.summarise_explanation(explanation)

        return {
            "churn_risk_score"   : round(score * 100, 2),
            "risk_tier"          : risk_tier,
            "explanation"        : explanation,
            "explanation_summary": summary,
            "tier_thresholds"    : MODEL_CACHE["oif"].tier_thresholds,
            "model_version"      : MODEL_CACHE["version"],
            "timestamp"          : datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Explain error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-batch")
async def predict_batch(batch: BatchInput):
    if MODEL_CACHE["oif"] is None:
        if not load_active_model():
            raise HTTPException(status_code=503, detail="Model not loaded")

    _ensure_shap_explainer()

    try:
        results = []
        exp_obj = MODEL_CACHE.get("shap_explainer")

        for customer in batch.customers:
            customer_dict  = customer.dict()
            X              = preprocess_input(customer_dict)
            labels, scores = MODEL_CACHE["oif"].predict(X)
            score          = float(scores[0])
            label          = int(labels[0])
            risk_tier      = MODEL_CACHE["oif"].get_risk_tier(score)

            explanation = shap_ex.explain_single(
                exp_obj,
                X,
                MODEL_CACHE["feature_columns"] or cfg.FEATURE_COLUMNS,
                top_n=3
            ) if exp_obj else []

            results.append({
                "churn_risk_score"   : round(score * 100, 2),
                "risk_tier"          : risk_tier,
                "anomaly_label"      : label,
                "is_anomaly"         : label == -1,
                "explanation"        : explanation,
                "explanation_summary": shap_ex.summarise_explanation(explanation)
            })

        return {
            "total"        : len(results),
            "predictions"  : results,
            "model_version": MODEL_CACHE["version"],
            "timestamp"    : datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate-timeline")
async def simulate_timeline(data: TimelineInput):
    try:
        base = data.churn_risk_score
        ten  = data.tenure
        mch  = data.monthly_charges
        cur  = data.contract

        return {
            "current"         : _simulate_trajectory(base, cur,        ten, mch),
            "one_year"        : _simulate_trajectory(base, "One year", ten, mch),
            "two_year"        : _simulate_trajectory(base, "Two year", ten, mch),
            "base_score"      : base,
            "current_contract": cur,
            "months"          : cfg.TRAJECTORY_CONFIG["n_months"]
        }
    except Exception as e:
        logger.error(f"Timeline simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring")
async def monitoring(days: int = 30):
    try:
        df = db.fetch_recent_metrics(days=days)
        if df.empty:
            return {"metrics": [], "total": 0}
        if "logged_at" in df.columns:
            df["logged_at"] = df["logged_at"].astype(str)
        return {"metrics": df.to_dict(orient="records"), "total": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/registry")
async def registry():
    try:
        df = db.get_all_models()
        if df.empty:
            return {"models": [], "total": 0}
        if "training_date" in df.columns:
            df["training_date"] = df["training_date"].astype(str)
        return {"models": df.to_dict(orient="records"), "total": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dashboard-stats")
async def dashboard_stats():
    try:
        engine       = db.get_engine()
        total_df     = pd.read_sql("SELECT COUNT(*) as total FROM raw_customers", engine)
        processed_df = pd.read_sql("SELECT COUNT(*) as total FROM raw_customers WHERE is_processed=1", engine)
        metrics_df   = db.fetch_last_n_metrics(n=7)
        active       = db.get_active_model()

        risk_dist     = {"Critical": 0, "High": 0, "Medium": 0, "Low": 0}
        score_buckets = []
        contract_risk = []

        if active and MODEL_CACHE["oif"] and MODEL_CACHE["scaler"] and MODEL_CACHE["shap_explainer"]:
            raw_df = pd.read_sql(
                "SELECT * FROM raw_customers WHERE is_processed=1 LIMIT 500", engine
            )
            if not raw_df.empty:
                # Preprocess and scale
                drop_cols = ["id", "is_processed", "created_at"]
                df = raw_df.drop(columns=[c for c in drop_cols if c in raw_df.columns], errors="ignore")
                contract_col = df["Contract"].copy() if "Contract" in df.columns else None
                df = df[cfg.FEATURE_COLUMNS].copy()
                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
                df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
                label_encoders = MODEL_CACHE.get("label_encoders") or {}
                for col in cfg.CATEGORICAL_COLUMNS:
                    if col not in df.columns:
                        continue
                    if col in label_encoders:
                        le = label_encoders[col]
                        df[col] = df[col].astype(str).apply(
                            lambda v: le.transform([v])[0] if v in le.classes_ else -1
                        )
                    else:
                        from sklearn.preprocessing import LabelEncoder as _LE
                        _le = _LE()
                        df[col] = _le.fit_transform(df[col].astype(str))
                X = MODEL_CACHE["scaler"].transform(df)
                explainer = MODEL_CACHE["shap_explainer"]
                oif = MODEL_CACHE["oif"]
                tier_thresholds = oif.tier_thresholds
                shap_scores = []
                for i in range(len(X)):
                    x_row = X[i:i+1]
                    # Raw score (normalized)
                    raw_score = float(oif.model.score_samples(x_row)[0])
                    norm_score = oif._normalize(np.array([raw_score]))[0]
                    # SHAP adjustment (same as /predict)
                    sv_raw = explainer.shap_values(x_row)
                    sv = np.array(sv_raw.values if hasattr(sv_raw, "values") else sv_raw).flatten()
                    risk_sum = float(np.sum(sv[sv > 0]))
                    safe_sum = float(np.abs(np.sum(sv[sv < 0])))
                    if (risk_sum + safe_sum) > 0:
                        risk_ratio = risk_sum / (risk_sum + safe_sum)
                        if risk_ratio < 0.5:
                            factor = 0.20 + 1.20 * risk_ratio
                        else:
                            factor = 1.0 + 0.30 * (risk_ratio - 0.5)
                        adj_score = float(np.clip(norm_score * factor, 0, 1))
                    else:
                        adj_score = norm_score
                    shap_scores.append(adj_score)
                # Assign risk tiers
                for s in shap_scores:
                    tier = oif.get_risk_tier(s)
                    risk_dist[tier] += 1
                scores_pct = [s * 100 for s in shap_scores]
                for low in range(0, 100, 10):
                    high  = low + 10
                    count = sum(
                        1 for s in scores_pct
                        if (low <= s <= high if high == 100 else low <= s < high)
                    )
                    score_buckets.append({"range": f"{low}-{high}", "count": count})
                if contract_col is not None and len(contract_col) == len(shap_scores):
                    contract_scores = {}
                    for i, ct in enumerate(contract_col):
                        ct = str(ct)
                        contract_scores.setdefault(ct, []).append(shap_scores[i] * 100)
                    for ct in ["Month-to-month", "One year", "Two year"]:
                        if ct in contract_scores:
                            s_list = contract_scores[ct]
                            contract_risk.append({
                                "contract"      : ct,
                                "avg_risk"      : round(float(np.mean(s_list)), 2),
                                "customer_count": len(s_list)
                            })

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
            "latest_status"      : trend_data[0]["status"] if trend_data else "GOOD",
            "score_buckets"      : score_buckets,
            "contract_risk"      : contract_risk,
            "tier_thresholds"    : MODEL_CACHE["oif"].tier_thresholds if MODEL_CACHE["oif"] else None
        }
    except Exception as e:
        logger.error(f"Dashboard stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/top-risk-customers")
async def top_risk_customers(limit: int = 20):
    if MODEL_CACHE["oif"] is None:
        if not load_active_model():
            raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        engine = db.get_engine()
        raw_df = pd.read_sql(
            "SELECT * FROM raw_customers WHERE is_processed=1 ORDER BY id DESC LIMIT 500",
            engine
        )
        if raw_df.empty:
            return {"customers": [], "total": 0}

        orig_df   = raw_df.copy()
        scores, _ = _encode_and_score(raw_df)
        if scores is None:
            return {"customers": [], "total": 0}

        results = []
        for i in range(len(scores)):
            s    = float(scores[i])
            tier = MODEL_CACHE["oif"].get_risk_tier(s)
            row  = orig_df.iloc[i]
            results.append({
                "id"             : int(row["id"]) if "id" in orig_df.columns else i + 1,
                "contract"       : str(row.get("Contract", "—")),
                "tenure"         : int(row.get("tenure", 0)),
                "monthly_charges": float(row.get("MonthlyCharges", 0)),
                "internet"       : str(row.get("InternetService", "—")),
                "payment_method" : str(row.get("PaymentMethod", "—")),
                "risk_score"     : round(s * 100, 2),
                "risk_tier"      : tier
            })

        results.sort(key=lambda x: x["risk_score"], reverse=True)
        return {
            "customers"     : results[:limit],
            "total_scored"  : len(results),
            "total_returned": min(limit, len(results)),
            "model_version" : MODEL_CACHE["version"]
        }
    except Exception as e:
        logger.error(f"Top risk customers error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-model")
async def reload_model():
    success = load_active_model()
    if success:
        return {
            "message"   : f"Model reloaded: {MODEL_CACHE['version']}",
            "shap_ready": MODEL_CACHE["shap_explainer"] is not None
        }
    raise HTTPException(status_code=500, detail="Failed to reload model")


@app.post("/activate-model")
async def activate_model(data: ActivateModelInput):
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