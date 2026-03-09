import os
from dotenv import load_dotenv

# ─────────────────────────────────────────
#  LOAD .env
#  FIX: DB password and SECRET_KEY were hardcoded
#  in plain source code committed to a public repo.
#  All secrets now live in .env which is gitignored.
# ─────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Check your .env file."
        )
    return val

# ── Database ──────────────────────────────
DB_URL = (
    f"mysql+pymysql://{_require('DB_USER')}:{_require('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '3306')}"
    f"/{_require('DB_NAME')}"
)

# ── JWT Auth ──────────────────────────────
SECRET_KEY                  = _require("SECRET_KEY")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# ── Admin credentials ─────────────────────
ADMIN_USERNAME      = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "")

# ── Paths ─────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
LOGS_DIR      = os.path.join(BASE_DIR, "logs")
RAW_CSV       = os.path.join(DATA_RAW_DIR, "telco_customers.csv")

# ── Model parameters ──────────────────────
MODEL_PARAMS = {
    "n_estimators" : 100,
    "contamination": 0.15,
    "window_size"  : 8000,
    "random_state" : 42,
}

# ── Risk tier percentiles ─────────────────
TIER_PERCENTILES = {
    "critical_pct": 98,   # Top 2% are Critical
    "high_pct"    : 90,   # Next 8% are High
    "medium_pct"  : 70,   # Next 20% are Medium
}

# ── Evaluation thresholds ─────────────────
EVAL_CONFIG = {
    "ks_warning_threshold" : 0.15,
    "ks_critical_threshold": 0.30,
    "min_anomaly_rate"     : 0.05,
    "max_anomaly_rate"     : 0.40,
    "trend_window_days"    : 7,
}

# ── Scheduler ─────────────────────────────
CRON_HOUR   = 0
CRON_MINUTE = 0

# ── Features ──────────────────────────────
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges",
]
CATEGORICAL_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERICAL_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# ── Trajectory config ─────────────────────
TRAJECTORY_CONFIG = {
    "drift_rates": {
        "Month-to-month": +0.9,
        "One year"      : -1.2,
        "Two year"      : -1.8,
        "default"       : +0.5,
    },
    "charge_pressure_threshold"  : 70,
    "charge_pressure_scale"      : 100,
    "loyalty_reduction_per_month": 0.04,
    "loyalty_reduction_cap"      : 1.5,
    "noise_range"                : 0.8,
    "score_min"                  : 1.0,
    "score_max"                  : 99.0,
    "n_months"                   : 12,
}