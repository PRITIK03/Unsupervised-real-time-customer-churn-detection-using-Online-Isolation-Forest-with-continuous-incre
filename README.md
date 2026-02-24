Unsupervised Real-Time Customer Churn Detection using Online Isolation Forest 🚀
A robust pipeline for real-time, unsupervised customer churn detection leveraging Online Isolation Forest with continuous incremental learning. The system features a Flask API for predictions and monitoring, and integrates with MySQL for persistent storage.

✨ Features
Unsupervised Learning: Detects churn without labeled data using Online Isolation Forest
Real-Time & Incremental: Continuously learns from new data, adapting to changing customer behavior
Flask API: RESTful endpoints for predictions, monitoring, and model management
MySQL Integration: Stores customer data, predictions, and model metadata
Modular Frontend: Dashboard, prediction form, monitoring, and model registry
Easy Deployment: Modular structure for quick setup and extensibility

🗂️ Project Structure
churn_pipeline/
├── config.py                # Configuration (DB credentials, paths, params)
├── requirements.txt         # Python dependencies
├── seed_database.py         # Script to load CSV into MySQL
├── README.md
├── data/
│   ├── raw/                 # Raw datasets (e.g., telco_customers.csv)
│   └── processed/           # Processed datasets
├── models/                  # Saved model files
├── logs/                    # Log files
├── src/
│   ├── db_connection.py     # MySQL connection logic
│   ├── preprocessing.py     # Data cleaning & feature engineering
│   ├── train.py             # Online Isolation Forest training
│   ├── evaluate.py          # Model evaluation utilities
│   ├── scheduler.py         # Incremental learning scheduler
│   └── __init__.py
├── api/
│   ├── main.py              # Flask API entrypoint
│   └── __init__.py
└── frontend/
    ├── index.html           # Main UI shell
    ├── css/
    │   └── style.css
    ├── js/
    │   ├── dashboard.js
    │   ├── predict.js
    │   ├── monitoring.js
    │   └── registry.js
    └── assets/
        └── logo.svg

        🚦 Getting Started
Clone the Repository
git clone https://github.com/PRITIK03/churn_pipeline.git
cd churn_pipeline/churn_pipeline

Install Dependencies
pip install -r requirements.txt

Configure Database
Update config.py with your MySQL credentials.
Seed the database:
python seed_database.py

Run the Scheduler (Incremental Learning)
python src/scheduler.py --run-now

Start the Flask API
python api/main.py

Access the Frontend
Open frontend/index.html in your browser.

🔗 API Endpoints
POST /predict — Predict churn for a customer
GET /monitor — Get live model stats
GET /registry — List model versions and metadata
🛠️ Customization
Add new features or preprocessing steps in src/preprocessing.py
Extend the API in api/main.py
Modify the UI in frontend/
🏷️ Topics
unsupervised-learning real-time churn-detection online-isolation-forest incremental-learning flask mysql api machine-learning customer-analytics

📄 License
Copyright © 2026 Sciqus. All rights reserved.
This project is proprietary and confidential. Unauthorized copying, distribution, or use is prohibited without explicit permission from Sciqus.
