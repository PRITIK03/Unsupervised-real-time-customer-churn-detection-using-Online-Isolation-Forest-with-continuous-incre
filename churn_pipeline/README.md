# ChurnGuard — AI Churn Detection System

Unsupervised real-time customer churn detection using **Online Isolation Forest** with continuous incremental learning, SHAP explainability, FastAPI backend, and a modern glassmorphism frontend.

> **Live URL**: http://localhost:8000

---

## Architecture Overview

```
churn_pipeline/
├── api/
│   └── main.py              # FastAPI app — endpoints, SHAP scoring, model serving
├── src/
│   ├── db_connection.py      # MySQL connection + model registry (CRUD)
│   ├── preprocessing.py      # Feature engineering, encoding, scaling
│   ├── train.py              # OnlineIsolationForest + incremental training
│   ├── evaluate.py           # Model evaluation — KS drift, anomaly rate
│   ├── shap_explainer.py     # SHAP TreeExplainer — build, explain, summarise
│   └── scheduler.py          # APScheduler — periodic retraining
├── frontend/
│   ├── index.html            # Single-page app (all views inline)
│   ├── css/style.css         # Dark glassmorphism design system
│   └── js/
│       ├── dashboard.js      # Dashboard charts + stats
│       ├── predict.js        # Prediction form + SHAP panel
│       ├── monitoring.js     # Model drift monitoring
│       └── registry.js       # Model version management
├── config.py                 # All settings — DB, paths, model params, tiers
├── seed_database.py          # One-time script to load CSV into MySQL
├── generate_synthetic.py     # Synthetic data generator for testing
├── requirements.txt          # Python dependencies
├── data/raw/                 # Place telco_customers.csv here
├── data/processed/           # Auto-generated processed CSVs
├── models/                   # Versioned .pkl model files
└── logs/                     # Auto-generated log files
```

---

## Features

### Backend
- **Online Isolation Forest** — unsupervised anomaly detection with incremental (windowed) retraining
- **SHAP Explainability** — TreeExplainer computes per-feature contributions for every prediction
- **SHAP-Adjusted Scoring** — raw anomaly scores adjusted by SHAP risk direction so scores reflect actual churn risk, not just statistical unusualness
- **Dynamic Tier Thresholds** — Critical/High/Medium/Low tiers computed from SHAP-adjusted score distribution at startup (percentile-based, not hardcoded)
- **Model Registry** — versioned model storage with activate/deactivate via API
- **Drift Monitoring** — KS-test based distribution shift detection
- **Scheduled Retraining** — APScheduler cron for automatic model updates
- **Data-Driven Business Logic** — risk tiers, trajectory drift rates, and scoring thresholds all configurable via `config.py`

### Frontend
- **Dashboard** — live model stats, anomaly rate, score distribution charts
- **Predict** — customer feature form with risk meter, SHAP feature bars (top 5), full SHAP modal (top 10), and 12-month risk trajectory
- **Monitoring** — drift metrics, model health, retraining history
- **Model Registry** — view, activate, and manage model versions
- **Design** — dark theme, glassmorphism, particle canvas, smooth animations

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- MySQL server running on `localhost:3306`
- Database `churn_db` created

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install shap python-jose[cryptography] passlib[bcrypt]
```

### 3. Configure
Edit `config.py` with your MySQL credentials:
```python
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "your_password",
    "database": "churn_db"
}
```

### 4. Seed Database
Place `telco_customers.csv` in `data/raw/`, then:
```bash
python seed_database.py
```

### 5. Train Model
```bash
python src/train.py
```

### 6. Run Server
```bash
cd api
python main.py
```
Open http://localhost:8000 in your browser.

> **Note**: After editing backend files, always do a clean restart (`Ctrl+C` → `python main.py`). Uvicorn's hot-reloader can sometimes fail to reinitialise the SHAP explainer properly.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve frontend |
| `GET` | `/health` | Server status, model info, SHAP readiness |
| `POST` | `/predict` | Single customer churn prediction + SHAP explanation |
| `POST` | `/explain` | Full SHAP breakdown (top 10 features) |
| `GET` | `/model-info` | Active model details + tier thresholds |
| `GET` | `/top-risk-customers` | Highest risk customers from DB |
| `POST` | `/reload-model` | Hot-reload active model |
| `POST` | `/token` | JWT authentication |

---

## How Scoring Works

1. **IsolationForest** computes a raw anomaly score (how statistically unusual a customer is)
2. **Normalisation** maps raw scores to 0–1 using training data min/max
3. **SHAP Adjustment** checks whether the features driving the anomaly actually *increase* or *decrease* churn risk:
   - If features net-reduce risk → score is dampened (stable customer who is just statistically rare)
   - If features net-increase risk → score is preserved/boosted
4. **Tier Assignment** uses percentile-based thresholds computed from SHAP-adjusted training scores

This ensures that a loyal, long-tenure customer doesn't get flagged as "Critical" just because they're statistically uncommon in the dataset.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, Uvicorn |
| ML Model | scikit-learn IsolationForest |
| Explainability | SHAP (TreeExplainer) |
| Database | MySQL + SQLAlchemy |
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Auth | JWT (python-jose) |
| Scheduling | APScheduler |