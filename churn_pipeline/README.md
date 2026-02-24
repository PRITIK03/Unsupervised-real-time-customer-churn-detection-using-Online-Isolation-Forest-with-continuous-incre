# Churn Pipeline

This project contains a churn prediction pipeline with the following structure:

- **data/raw/**: Place your IBM dataset (`telco_customers.csv`) here.
- **data/processed/**: Auto-generated processed CSVs will be saved here.
- **models/**: Versioned `.pkl` model files are saved here automatically.
- **logs/**: Log files are auto-generated here.
- **src/**: Source code for database connection, preprocessing, training, evaluation, and scheduling.
- **api/**: FastAPI app for serving endpoints.
- **frontend/**: Inference UI.
- **config.py**: All settings (DB credentials, paths, params).
- **seed_database.py**: One-time script to load CSV into MySQL.
- **requirements.txt**: Python dependencies.
