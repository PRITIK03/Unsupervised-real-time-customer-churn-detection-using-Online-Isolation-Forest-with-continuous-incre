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