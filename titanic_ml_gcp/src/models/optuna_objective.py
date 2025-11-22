import optuna
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import mlflow
import logging

logger = logging.getLogger(__name__)

def titanic_objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function for Titanic XGBoost model.
    
    Args:
        trial: Optuna trial object
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        float: AUC score (maximize)
    """
    
    # Define search space
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "verbosity": 0,
        "tree_method": "auto",
        "random_state": 42,
        "n_jobs": -1, # Use all cores
        
        # Hyperparameters to tune
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0),
    }
    
    # Pruning callback
    # Note: validation_0 is the default name for the first eval_set
    # pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_0-auc")
    
    model = xgb.XGBClassifier(**params)
    
    # Fit model
    # Removing callbacks for now to resolve compatibility issues with XGBClassifier wrapper
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
        # callbacks=[pruning_callback] 
    )
    
    # Predict
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    
    return auc

