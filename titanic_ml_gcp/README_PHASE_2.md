# Phase 2: Local MLOps with MLflow & Optuna

This phase focuses on interactive model retraining and hyperparameter optimization using MLflow and Optuna, running locally.

## Setup

1. **Install Dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

2. **Directory Structure**:
   - `mlflow/`: Contains local SQLite databases (`mlflow.db` for tracking, `optuna.db` for study state).
   - `mlruns/`: Default location for MLflow artifacts (unless configured otherwise).

## Running the App

Start the Streamlit application:
```bash
streamlit run src/app/streamlit_app.py
```

## Features

### 1. Feature Selection
- Use the sidebar to select/deselect features.
- Defaults to all available features.

### 2. Model Retraining
- Navigate to "Model Training" tab.
- Configure Optuna trials (e.g., 10-20) and timeout.
- Click "Train New Model".
- Watch progress in the terminal (logs) and UI.

### 3. Metrics & Visualization
- After training, view:
  - Best AUC score.
  - Accuracy on validation set.
  - Confusion Matrix.
  - Feature Importance plot.
  - Best hyperparameters.

### 4. MLflow Tracking
- All runs are logged to MLflow.
- To view the MLflow UI:
  ```bash
  mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
  ```

### 5. Inference
- Navigate to "Inference" tab.
- View registered models (local registry).
- (Future) Load model and predict on new data.


