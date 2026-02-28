import os
import sys
import numpy as np
import pandas as pd
import logging

# ─────────────────────────────────────────
#  PATH FIX — same pattern as seed_database.py
# ─────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
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
    level  = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
#  VALID CATEGORY VALUES
#  Pulled directly from IBM Telco dataset —
#  must match exactly what the model was trained on.
# ─────────────────────────────────────────
VALID = {
    "gender"         : ["Male", "Female"],
    "Partner"        : ["Yes", "No"],
    "Dependents"     : ["Yes", "No"],
    "PhoneService"   : ["Yes", "No"],
    "MultipleLines"  : ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity" : ["Yes", "No", "No internet service"],
    "OnlineBackup"   : ["Yes", "No", "No internet service"],
    "DeviceProtection":["Yes", "No", "No internet service"],
    "TechSupport"    : ["Yes", "No", "No internet service"],
    "StreamingTV"    : ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract"       : ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling":["Yes", "No"],
    "PaymentMethod"  : [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}

# ─────────────────────────────────────────
#  HELPER
# ─────────────────────────────────────────
def _pick(col, n, weights=None):
    """Random sample from valid categories for a column."""
    return np.random.choice(VALID[col], size=n, p=weights)

def _make_df(rows: list) -> pd.DataFrame:
    """
    Convert list-of-dicts to DataFrame with correct column order
    matching cfg.FEATURE_COLUMNS, plus is_processed=0.
    """
    df = pd.DataFrame(rows)
    df = df[cfg.FEATURE_COLUMNS]   # enforce correct column order
    df["is_processed"] = 0
    return df


# ─────────────────────────────────────────
#  TYPE 1 — NORMAL CUSTOMERS
#  Should score: LOW risk
#  Profile: medium-long tenure, average charges,
#  stable contract, multiple services, auto payment.
#  These look exactly like the bulk of IBM Telco data
#  the model was trained on → score near the center of
#  the normal distribution → LOW tier.
# ─────────────────────────────────────────
def generate_normal_batch(n: int = 50, seed: int = None) -> pd.DataFrame:
    """
    Generate n typical / normal customers.
    Expected model output: LOW risk score (< 0.25).
    """
    if seed is not None:
        np.random.seed(seed)

    logger.info(f"Generating {n} NORMAL customers (expected: LOW risk)...")
    rows = []

    for _ in range(n):
        tenure          = int(np.random.randint(24, 65))       # established customers
        monthly_charges = round(float(np.random.uniform(45, 80)), 2)
        total_charges   = round(monthly_charges * tenure + np.random.uniform(-50, 50), 2)
        total_charges   = max(total_charges, 0)

        # Internet customers with lots of add-ons — typical mid-tier profile
        internet = np.random.choice(["DSL", "Fiber optic"], p=[0.55, 0.45])
        addon    = "Yes"  # most add-ons purchased — low churn signal

        row = {
            "gender"          : _pick("gender", 1)[0],
            "SeniorCitizen"   : int(np.random.choice([0, 1], p=[0.84, 0.16])),
            "Partner"         : np.random.choice(["Yes", "No"], p=[0.55, 0.45]),
            "Dependents"      : np.random.choice(["Yes", "No"], p=[0.40, 0.60]),
            "tenure"          : tenure,
            "PhoneService"    : "Yes",
            "MultipleLines"   : np.random.choice(["Yes", "No"], p=[0.55, 0.45]),
            "InternetService" : internet,
            "OnlineSecurity"  : addon,
            "OnlineBackup"    : addon,
            "DeviceProtection": addon,
            "TechSupport"     : addon,
            "StreamingTV"     : np.random.choice(["Yes", "No"], p=[0.50, 0.50]),
            "StreamingMovies" : np.random.choice(["Yes", "No"], p=[0.50, 0.50]),
            "Contract"        : np.random.choice(
                                    ["One year", "Two year"],
                                    p=[0.45, 0.55]
                                ),   # stable contract — strong low-churn signal
            "PaperlessBilling": np.random.choice(["Yes", "No"], p=[0.45, 0.55]),
            "PaymentMethod"   : np.random.choice(
                                    ["Bank transfer (automatic)", "Credit card (automatic)"],
                                    p=[0.50, 0.50]
                                ),   # auto-pay — low churn signal
            "MonthlyCharges"  : monthly_charges,
            "TotalCharges"    : total_charges
        }
        rows.append(row)

    df = _make_df(rows)
    logger.info(f"✅ Generated {len(df)} NORMAL customers.")
    return df


# ─────────────────────────────────────────
#  TYPE 2 — HIGH CHURN RISK CUSTOMERS
#  Should score: MEDIUM-HIGH risk (0.25 – 0.75)
#  Profile: new customers, high charges, month-to-month
#  contract, fiber optic, no add-ons, electronic check.
#  This is the textbook churn profile from IBM Telco research.
# ─────────────────────────────────────────
def generate_churn_risk_batch(n: int = 30, seed: int = None) -> pd.DataFrame:
    """
    Generate n high-churn-risk customers.
    Expected model output: MEDIUM to HIGH risk score (0.25 – 0.75).
    """
    if seed is not None:
        np.random.seed(seed)

    logger.info(f"Generating {n} HIGH CHURN RISK customers (expected: MEDIUM-HIGH)...")
    rows = []

    for _ in range(n):
        tenure          = int(np.random.randint(1, 12))        # very new customers
        monthly_charges = round(float(np.random.uniform(80, 110)), 2)  # expensive plan
        total_charges   = round(monthly_charges * tenure + np.random.uniform(0, 20), 2)

        no_addon = "No"  # no add-ons purchased — strong churn signal

        row = {
            "gender"          : _pick("gender", 1)[0],
            "SeniorCitizen"   : int(np.random.choice([0, 1], p=[0.70, 0.30])),
            "Partner"         : "No",   # single — higher churn tendency
            "Dependents"      : "No",   # no dependents — less sticky
            "tenure"          : tenure,
            "PhoneService"    : "Yes",
            "MultipleLines"   : "No",
            "InternetService" : "Fiber optic",   # expensive, dissatisfied subset
            "OnlineSecurity"  : no_addon,
            "OnlineBackup"    : no_addon,
            "DeviceProtection": no_addon,
            "TechSupport"     : no_addon,
            "StreamingTV"     : np.random.choice(["Yes", "No"], p=[0.60, 0.40]),
            "StreamingMovies" : np.random.choice(["Yes", "No"], p=[0.60, 0.40]),
            "Contract"        : "Month-to-month",  # easiest to cancel
            "PaperlessBilling": "Yes",
            "PaymentMethod"   : "Electronic check",  # highest churn payment method
            "MonthlyCharges"  : monthly_charges,
            "TotalCharges"    : total_charges
        }
        rows.append(row)

    df = _make_df(rows)
    logger.info(f"✅ Generated {len(df)} HIGH CHURN RISK customers.")
    return df


# ─────────────────────────────────────────
#  TYPE 3 — EXTREME ANOMALIES
#  Should score: CRITICAL (> 0.75), flagged as anomaly
#  Profile: statistically impossible or extremely rare
#  combinations the Isolation Forest has never encountered.
#  These are designed to be far outside the training distribution.
# ─────────────────────────────────────────
def generate_anomaly_batch(n: int = 20, seed: int = None) -> pd.DataFrame:
    """
    Generate n extreme anomaly customers.
    Expected model output: CRITICAL risk score (> 0.75), is_anomaly=True.

    4 anomaly subtypes — each ~n/4 records:
      A) Ghost customer    — 0 tenure, impossibly high TotalCharges
      B) Dormant veteran   — 72 months tenure, zero services, $0 charges
      C) Charge spike      — 1 month tenure, $500+ total charges
      D) Service paradox   — No internet but all internet add-ons = "Yes"
    """
    if seed is not None:
        np.random.seed(seed)

    logger.info(f"Generating {n} EXTREME ANOMALY customers (expected: CRITICAL)...")
    rows = []

    per_type = max(1, n // 4)
    remainder = n - (per_type * 4)

    # ── Subtype A: Ghost customer ─────────
    # tenure=0 but TotalCharges is huge — impossible in real data
    for _ in range(per_type):
        rows.append({
            "gender"          : _pick("gender", 1)[0],
            "SeniorCitizen"   : 0,
            "Partner"         : "No",
            "Dependents"      : "No",
            "tenure"          : 0,                               # ← impossible: new customer
            "PhoneService"    : "Yes",
            "MultipleLines"   : "Yes",
            "InternetService" : "Fiber optic",
            "OnlineSecurity"  : "Yes",
            "OnlineBackup"    : "Yes",
            "DeviceProtection": "Yes",
            "TechSupport"     : "Yes",
            "StreamingTV"     : "Yes",
            "StreamingMovies" : "Yes",
            "Contract"        : "Two year",
            "PaperlessBilling": "Yes",
            "PaymentMethod"   : "Electronic check",
            "MonthlyCharges"  : round(float(np.random.uniform(90, 120)), 2),
            "TotalCharges"    : round(float(np.random.uniform(400, 600)), 2)  # ← impossible
        })

    # ── Subtype B: Dormant veteran ────────
    # max tenure, no services at all, $0 charges
    for _ in range(per_type):
        rows.append({
            "gender"          : _pick("gender", 1)[0],
            "SeniorCitizen"   : 1,
            "Partner"         : "Yes",
            "Dependents"      : "Yes",
            "tenure"          : 72,                              # ← maximum tenure
            "PhoneService"    : "No",
            "MultipleLines"   : "No phone service",
            "InternetService" : "No",                            # ← no internet
            "OnlineSecurity"  : "No internet service",
            "OnlineBackup"    : "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport"     : "No internet service",
            "StreamingTV"     : "No internet service",
            "StreamingMovies" : "No internet service",
            "Contract"        : "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod"   : "Mailed check",
            "MonthlyCharges"  : 0.0,                             # ← $0 charges
            "TotalCharges"    : 0.0                              # ← impossible for 72 months
        })

    # ── Subtype C: Charge spike ───────────
    # 1 month tenure but $500+ total — billing system glitch pattern
    for _ in range(per_type):
        monthly = round(float(np.random.uniform(90, 115)), 2)
        rows.append({
            "gender"          : _pick("gender", 1)[0],
            "SeniorCitizen"   : int(np.random.choice([0, 1])),
            "Partner"         : "No",
            "Dependents"      : "No",
            "tenure"          : 1,                               # ← brand new
            "PhoneService"    : "Yes",
            "MultipleLines"   : "No",
            "InternetService" : "Fiber optic",
            "OnlineSecurity"  : "No",
            "OnlineBackup"    : "No",
            "DeviceProtection": "No",
            "TechSupport"     : "No",
            "StreamingTV"     : "Yes",
            "StreamingMovies" : "Yes",
            "Contract"        : "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod"   : "Electronic check",
            "MonthlyCharges"  : monthly,
            "TotalCharges"    : round(float(np.random.uniform(500, 800)), 2)  # ← impossible
        })

    # ── Subtype D: Service paradox ────────
    # InternetService=No but all internet add-ons are "Yes"
    # This is a data impossibility — the model has never seen this combination
    for i in range(per_type + remainder):
        monthly = round(float(np.random.uniform(20, 30)), 2)  # low charge but "Yes" everywhere
        rows.append({
            "gender"          : _pick("gender", 1)[0],
            "SeniorCitizen"   : 0,
            "Partner"         : np.random.choice(["Yes", "No"]),
            "Dependents"      : np.random.choice(["Yes", "No"]),
            "tenure"          : int(np.random.randint(10, 40)),
            "PhoneService"    : "Yes",
            "MultipleLines"   : "Yes",
            "InternetService" : "No",                 # ← no internet...
            "OnlineSecurity"  : "Yes",                # ← ...but security = Yes (impossible)
            "OnlineBackup"    : "Yes",                # ← impossible
            "DeviceProtection": "Yes",                # ← impossible
            "TechSupport"     : "Yes",                # ← impossible
            "StreamingTV"     : "Yes",                # ← impossible
            "StreamingMovies" : "Yes",                # ← impossible
            "Contract"        : np.random.choice(["Month-to-month", "One year"]),
            "PaperlessBilling": "Yes",
            "PaymentMethod"   : "Electronic check",
            "MonthlyCharges"  : monthly,
            "TotalCharges"    : round(monthly * np.random.randint(10, 40), 2)
        })

    df = _make_df(rows)
    logger.info(f"✅ Generated {len(df)} EXTREME ANOMALY customers.")
    return df


# ─────────────────────────────────────────
#  MAIN — INSERT ALL 3 BATCHES
# ─────────────────────────────────────────
def generate_and_insert(
    n_normal   : int = 50,
    n_churn    : int = 30,
    n_anomaly  : int = 20,
    seed       : int = 42
):
    """
    Generate all 3 synthetic batches and insert into raw_customers table.
    Records go in with is_processed=0 so the pipeline picks them up
    on the next scheduler run (or manual train run).

    Args:
        n_normal  : number of normal customers to generate
        n_churn   : number of high-churn-risk customers to generate
        n_anomaly : number of extreme anomaly customers to generate
        seed      : random seed for reproducibility

    Returns:
        total records inserted (int)
    """
    logger.info("=" * 55)
    logger.info("  SYNTHETIC DATA GENERATION STARTED")
    logger.info(f"  Normal: {n_normal} | Churn Risk: {n_churn} | Anomalies: {n_anomaly}")
    logger.info("=" * 55)

    # ── Generate ──────────────────────────
    df_normal  = generate_normal_batch    (n=n_normal,  seed=seed)
    df_churn   = generate_churn_risk_batch(n=n_churn,   seed=seed + 1)
    df_anomaly = generate_anomaly_batch   (n=n_anomaly, seed=seed + 2)

    # ── Label each batch for logging ──────
    df_normal ["_type"] = "NORMAL"
    df_churn  ["_type"] = "CHURN_RISK"
    df_anomaly["_type"] = "ANOMALY"

    # Drop the label column before inserting (not a real DB column)
    all_df = pd.concat([df_normal, df_churn, df_anomaly], ignore_index=True)
    all_df = all_df.drop(columns=["_type"])

    total = len(all_df)
    logger.info(f"Total synthetic records to insert: {total}")
    logger.info(f"  Columns: {all_df.columns.tolist()}")

    # ── Insert into raw_customers ─────────
    logger.info("Inserting into raw_customers table (is_processed=0)...")
    db.insert_raw_customers(all_df)

    logger.info("=" * 55)
    logger.info(f"  ✅ DONE — {total} synthetic records inserted.")
    logger.info("  Run the training pipeline to process them:")
    logger.info("  > python src/scheduler.py --run-now")
    logger.info("  OR manually trigger via API:")
    logger.info("  > POST /reload-model")
    logger.info("=" * 55)

    return total


# ─────────────────────────────────────────
#  PREVIEW — show sample rows without inserting
# ─────────────────────────────────────────
def preview(n_each: int = 3):
    """
    Print a sample of each batch type to console.
    Useful to verify data looks correct before inserting.
    Does NOT touch the database.
    """
    print("\n" + "=" * 55)
    print("  SYNTHETIC DATA PREVIEW (not inserted)")
    print("=" * 55)

    batches = [
        ("NORMAL (expected: LOW risk)",        generate_normal_batch    (n=n_each, seed=99)),
        ("CHURN RISK (expected: MEDIUM-HIGH)",  generate_churn_risk_batch(n=n_each, seed=99)),
        ("ANOMALY (expected: CRITICAL)",        generate_anomaly_batch   (n=n_each, seed=99)),
    ]

    for label, df in batches:
        print(f"\n── {label} ──")
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(df.drop(columns=["is_processed"]).to_string(index=False))

    print("\n" + "=" * 55)
    print("  Run with --insert to actually insert into DB")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic churn data for model testing"
    )
    parser.add_argument(
        "--insert",
        action  = "store_true",
        help    = "Insert generated data into raw_customers table (default: preview only)"
    )
    parser.add_argument("--normal",  type=int, default=50, help="Number of normal customers (default: 50)")
    parser.add_argument("--churn",   type=int, default=30, help="Number of churn-risk customers (default: 30)")
    parser.add_argument("--anomaly", type=int, default=20, help="Number of anomaly customers (default: 20)")
    parser.add_argument("--seed",    type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    if args.insert:
        generate_and_insert(
            n_normal  = args.normal,
            n_churn   = args.churn,
            n_anomaly = args.anomaly,
            seed      = args.seed
        )
    else:
        preview()
        print("👆 Add --insert flag to insert into database.")
        print("   Example: python generate_synthetic.py --insert")
        print("   Custom:  python generate_synthetic.py --insert --normal 100 --churn 50 --anomaly 30")