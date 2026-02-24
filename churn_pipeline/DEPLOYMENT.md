# Deployment Instructions

1. **Clone the Repository**
   ```
   git clone https://github.com/PRITIK03/Unsupervised-real-time-customer-churn-detection-using-Online-Isolation-Forest-with-continuous-incremental-learning.git
   cd Unsupervised-real-time-customer-churn-detection-using-Online-Isolation-Forest-with-continuous-incremental-learning/churn_pipeline
   ```

2. **Set Up Python Environment (Recommended)**
   - Using `venv`:
     ```
     python -m venv venv
     source venv/bin/activate  # On Windows: venv\Scripts\activate
     ```
   - Or, create an `environment.yml` for conda if needed.

3. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```
   - For a fully reproducible environment, you can freeze dependencies:
     ```
     pip freeze > requirements.txt
     ```

4. **Configure Database**
   - Edit `config.py` with your MySQL credentials and settings.
   - Seed the database:
     ```
     python seed_database.py
     ```

5. **Run Incremental Learning Scheduler**
   ```
   python src/scheduler.py --run-now
   ```

6. **Start the Flask API**
   ```
   python api/main.py
   ```

7. **Access the Frontend**
   - Open `frontend/index.html` in your browser.
