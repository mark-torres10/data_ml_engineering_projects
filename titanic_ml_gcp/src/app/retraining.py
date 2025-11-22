import pandas as pd
import xgboost as xgb
import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow
from src.models.optuna_objective import titanic_objective
from src.utils.mlflow_utils import MLFlowManager
import logging
from typing import List, Tuple, Optional, Dict
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_local_data(data_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load processed data from local directory.
    
    Args:
        data_dir: Directory containing processed CSVs
        
    Returns:
        X, y: Features and target
    """
    # Resolve path relative to project root if needed
    base_path = Path(data_dir)
    
    # Try multiple potential paths
    potential_paths = [
        base_path / "train_processed.csv",
        Path("titanic_ml_gcp") / base_path / "train_processed.csv", # If running from outside
        Path("..") / base_path / "train_processed.csv", # If running from src/app
        Path("data/processed/train_processed.csv") # Default absolute-ish
    ]
    
    train_path = None
    for p in potential_paths:
        if p.exists():
            train_path = p
            break
            
    if not train_path:
        # List current directory for debugging
        cwd = os.getcwd()
        logger.error(f"Current working directory: {cwd}")
        logger.error(f"Could not find train_processed.csv in {potential_paths}")
        raise FileNotFoundError(f"Training data not found. Checked: {potential_paths}")
        
    logger.info(f"Loading data from {train_path}")
    df = pd.read_csv(train_path)
    
    # Assuming 'Survived' is the target
    target = "Survived"
    if target not in df.columns:
         raise ValueError(f"Target column {target} not found in data")
         
    X = df.drop(columns=[target])
    y = df[target]
    
    return X, y

def run_retraining(
    selected_features: List[str],
    n_trials: int = 20,
    timeout: int = 600,
    experiment_name: str = "titanic_retraining_local",
    test_size: float = 0.2
):
    """
    Run the retraining workflow with Optuna optimization.
    
    Args:
        selected_features: List of feature names to use
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds
        experiment_name: MLflow experiment name
        test_size: Validation split size
        
    Returns:
        final_model, best_trial
    """
    logger.info("Starting retraining workflow...")
    
    # 1. Load Data
    X, y = load_local_data()
    
    # 2. Filter Features
    # Check if features exist
    missing_features = [f for f in selected_features if f not in X.columns]
    if missing_features:
        # If some are missing, maybe they are derived? 
        # For now, assume features passed match columns in processed data.
        # Or we just log warning and use intersection.
        logger.warning(f"Selected features not in dataset: {missing_features}")
        valid_features = [f for f in selected_features if f in X.columns]
        if not valid_features:
            raise ValueError("No valid features selected")
        X = X[valid_features]
    else:
        X = X[selected_features]
        
    logger.info(f"Using features: {selected_features}")
    
    # 3. Split Data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # 4. Setup MLflow
    mlflow_manager = MLFlowManager(experiment_name=experiment_name)
    
    # 5. Run Optimization
    # Create unique study name to avoid conflicts if reusing DB
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"{experiment_name}_{timestamp}"
    
    # Ensure mlflow directory exists for DB
    Path("mlflow").mkdir(exist_ok=True)
    # Use absolute path for SQLite
    db_path = Path("mlflow/optuna.db").absolute()
    storage_url = f"sqlite:///{db_path}"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    
    # Callback for MLflow logging
    # Nest trials under the parent run
    mlflow_callback = MLflowCallback(
        tracking_uri=mlflow_manager.tracking_uri,
        metric_name="auc",
        nest_trials=True
    )
    
    logger.info(f"Running Optuna optimization for {n_trials} trials...")
    
    # Start parent run
    with mlflow_manager.start_run(run_name=f"retrain_{timestamp}", tags={"phase": "2", "type": "retraining"}) as run:
        
        # Log basic info
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("timeout", timeout)
        mlflow.log_param("features_count", len(X.columns))
        mlflow.log_param("features", str(list(X.columns)))
        
        # Run optimization
        study.optimize(
            lambda trial: titanic_objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            timeout=timeout,
            callbacks=[mlflow_callback]
        )
        
        best_trial = study.best_trial
        logger.info(f"Optimization complete. Best AUC: {best_trial.value}")
        logger.info(f"Best params: {best_trial.params}")
        
        # Log best params to parent run explicitly for easy access
        mlflow.log_params(best_trial.params)
        mlflow.log_metric("best_auc", best_trial.value)
        
        # 6. Train Final Model
        logger.info("Training final model with best parameters...")
        best_params = best_trial.params
        # Add fixed params
        best_params.update({
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "verbosity": 0,
            "tree_method": "auto",
            "random_state": 42
        })
        
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X_train, y_train)
        
        # 7. Register Model
        # Log model to the parent run
        mlflow.xgboost.log_model(final_model, "model")
        
        # Register in local registry
        model_uri = f"runs:/{run.info.run_id}/model"
        reg_name = "titanic_xgboost_local"
        
        logger.info(f"Registering model to {reg_name}")
        mlflow_manager.register_model(model_uri, reg_name, tags={"features": str(list(X.columns)), "auc": str(best_trial.value)})
        
        return final_model, best_trial

