import os

# ─────────────────────────────────────────
#  DATABASE
# ─────────────────────────────────────────
DB_CONFIG = {
    "host"    : "localhost",
    "port"    : 3306,
    "user"    : "root",
    "password": "1234",
    "database": "churn_db"
}
DB_URL = "mysql+pymysql://root:1234@localhost:3306/churn_db"

# ─────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR   = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")
RAW_CSV         = os.path.join(DATA_RAW_DIR, "telco_customers.csv")

# ─────────────────────────────────────────
#  MODEL PARAMETERS
# ─────────────────────────────────────────
MODEL_PARAMS = {
    "n_estimators"  : 100,
    "contamination" : 0.15,
    "window_size"   : 8000,
    "random_state"  : 42
}

SCORE_BOUNDARIES = {
    "global_min": -0.599399,
    "global_max": -0.484276
}

# ─────────────────────────────────────────
#  DYNAMIC RISK TIER PERCENTILES
#  Thresholds are computed from actual score
#  distribution at training time — not hardcoded.
#  Top 10% → Critical, next 15% → High,
#  next 25% → Medium, rest → Low
# ─────────────────────────────────────────
TIER_PERCENTILES = {
    "critical_pct" : 90,   # top 10% of scores
    "high_pct"     : 75,   # top 10–25%
    "medium_pct"   : 50,   # top 25–50%
                           # below 50th → Low
}

# ─────────────────────────────────────────
#  EVALUATION THRESHOLDS
# ─────────────────────────────────────────
EVAL_CONFIG = {
    "ks_warning_threshold"  : 0.15,
    "ks_critical_threshold" : 0.30,
    "min_anomaly_rate"      : 0.05,
    "max_anomaly_rate"      : 0.40,
    "trend_window_days"     : 7
}

# ─────────────────────────────────────────
#  SCHEDULER
# ─────────────────────────────────────────
CRON_HOUR   = 0
CRON_MINUTE = 0

# ─────────────────────────────────────────
#  FEATURES
# ─────────────────────────────────────────
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges"
]
CATEGORICAL_COLUMNS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]
NUMERICAL_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# ─────────────────────────────────────────
#  TRAJECTORY CONFIG
#  Monthly drift rates per contract type.
#  Moved out of hardcoded function into config
#  so business team can adjust without touching code.
# ─────────────────────────────────────────
TRAJECTORY_CONFIG = {
    "drift_rates": {
        "Month-to-month": +0.9,
        "One year"       : -1.2,
        "Two year"       : -1.8,
        "default"        : +0.5,
    },
    "charge_pressure_threshold": 70,    # monthly charges above this add pressure
    "charge_pressure_scale"    : 100,   # divisor for normalising charge pressure
    "loyalty_reduction_per_month": 0.04,# risk reduction per month of tenure
    "loyalty_reduction_cap"    : 1.5,   # max loyalty reduction (pp)
    "noise_range"              : 0.8,   # ± noise per month
    "score_min"                : 1.0,   # floor for trajectory scores
    "score_max"                : 99.0,  # ceiling for trajectory scores
    "n_months"                 : 12,    # projection horizon
}