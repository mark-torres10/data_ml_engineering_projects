import sys
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from src.app.retraining import run_retraining, load_local_data
from src.utils.mlflow_utils import MLFlowManager

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def train_baseline(X_train, y_train, X_val, y_val):
    """Train a baseline XGBoost model with default parameters."""
    print("Training Baseline Model (Default Parameters)...")
    
    # Default parameters (similar to what we might have used in Phase 1 without tuning)
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    acc = accuracy_score(y_val, model.predict(X_val))
    
    print(f"Baseline - AUC: {auc:.4f}, Accuracy: {acc:.4f}")
    return model, auc, acc

def main():
    output_dir = "outputs/phase2_results"
    ensure_dir(output_dir)
    
    # 1. Load Data
    print("Loading data...")
    try:
        X, y = load_local_data()
    except Exception as e:
        # Fallback for script execution location
        if os.path.exists("titanic_ml_gcp/data/processed/train_processed.csv"):
             X, y = load_local_data("titanic_ml_gcp/data/processed")
        else:
            raise e

    # Select all features for this run
    selected_features = list(X.columns)
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Train Baseline
    # We'll log this to MLflow manually to compare
    mlflow_manager = MLFlowManager(experiment_name="titanic_retraining_local")
    
    with mlflow_manager.start_run(run_name="baseline_run", tags={"type": "baseline"}) as run:
        baseline_model, base_auc, base_acc = train_baseline(X_train, y_train, X_val, y_val)
        
        mlflow.log_metrics({"auc": base_auc, "accuracy": base_acc})
        mlflow.xgboost.log_model(baseline_model, "model")
        mlflow.log_params({"max_depth": "default", "learning_rate": "default"})
        
    # 3. Run Optuna Optimization (The "New" Way)
    print("\nRunning Optuna Optimization (20 trials)...")
    # This function handles its own MLflow logging (parent/child runs)
    final_model, best_trial = run_retraining(
        selected_features=selected_features,
        n_trials=20,
        timeout=600,
        experiment_name="titanic_retraining_local"
    )
    
    opt_auc = best_trial.value
    # Re-evaluate final model to get accuracy (Optuna only maximized AUC)
    opt_preds_val = final_model.predict(X_val)
    opt_acc = accuracy_score(y_val, opt_preds_val)
    
    print(f"\nOptimized - AUC: {opt_auc:.4f}, Accuracy: {opt_acc:.4f}")
    print(f"Improvement in AUC: {opt_auc - base_auc:.4f}")
    
    # 4. Generate Visualizations & Report
    
    # Comparison Plot
    metrics_df = pd.DataFrame({
        "Model": ["Baseline", "Optimized"],
        "AUC": [base_auc, opt_auc],
        "Accuracy": [base_acc, opt_acc]
    })
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=metrics_df.melt(id_vars="Model"), x="variable", y="value", hue="Model")
    plt.title("Model Performance Comparison: Baseline vs Optuna Optimized")
    plt.ylim(0.7, 1.0)
    plt.savefig(f"{output_dir}/comparison_plot.png")
    plt.close()
    
    # Confusion Matrix (Optimized)
    cm = confusion_matrix(y_val, opt_preds_val)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix (Optimized Model)")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"{output_dir}/confusion_matrix_optimized.png")
    plt.close()
    
    # Optimization History (Simulated view)
    # In a real scenario we'd pull from study, but here we just state the best params
    
    # Save Results JSON
    results = {
        "baseline": {
            "auc": base_auc,
            "accuracy": base_acc,
            "params": "default"
        },
        "optimized": {
            "auc": opt_auc,
            "accuracy": opt_acc,
            "params": best_trial.params,
            "improvement_auc": opt_auc - base_auc
        }
    }
    
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()

