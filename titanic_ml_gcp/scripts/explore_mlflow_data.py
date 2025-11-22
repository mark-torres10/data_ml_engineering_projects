import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

def explore_mlflow_data():
    """Explore what MLflow captured during our experiments."""
    
    # Set tracking URI
    tracking_uri = "sqlite:///mlflow/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    
    print("=" * 60)
    print("MLflow Experiment Analysis")
    print("=" * 60)
    
    # 1. List all experiments
    experiments = client.search_experiments()
    print(f"\n📊 EXPERIMENTS FOUND: {len(experiments)}")
    for exp in experiments:
        print(f"  - {exp.name} (ID: {exp.experiment_id})")
    
    # 2. Get runs from our main experiment
    exp_name = "titanic_retraining_local"
    experiment = client.get_experiment_by_name(exp_name)
    
    if not experiment:
        print(f"❌ Experiment '{exp_name}' not found!")
        return
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    print(f"\n🏃 RUNS IN '{exp_name}': {len(runs)}")
    
    # 3. Analyze runs
    run_data = []
    for run in runs:
        run_info = {
            'run_id': run.info.run_id[:8],  # Short ID
            'run_name': run.data.tags.get('mlflow.runName', 'unnamed'),
            'status': run.info.status,
            'start_time': pd.to_datetime(run.info.start_time, unit='ms'),
            'duration_sec': (run.info.end_time - run.info.start_time) / 1000 if run.info.end_time else None,
            'type': run.data.tags.get('type', 'unknown'),
            'phase': run.data.tags.get('phase', 'unknown')
        }
        
        # Add key metrics
        for metric_name in ['auc', 'accuracy', 'best_auc']:
            if metric_name in run.data.metrics:
                run_info[metric_name] = round(run.data.metrics[metric_name], 4)
        
        # Add key parameters
        for param_name in ['max_depth', 'learning_rate', 'n_estimators']:
            if param_name in run.data.params:
                run_info[param_name] = run.data.params[param_name]
                
        run_data.append(run_info)
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(run_data)
    df = df.sort_values('start_time')
    
    print("\n📈 RUN SUMMARY:")
    print(df.to_string(index=False))
    
    # 4. Find the optimization run (parent run with nested trials)
    optimization_runs = [r for r in runs if r.data.tags.get('type') == 'retraining']
    
    if optimization_runs:
        opt_run = optimization_runs[0]  # Most recent
        print(f"\n🎯 OPTIMIZATION RUN DETAILS:")
        print(f"   Run ID: {opt_run.info.run_id}")
        print(f"   Run Name: {opt_run.data.tags.get('mlflow.runName')}")
        print(f"   Duration: {(opt_run.info.end_time - opt_run.info.start_time) / 1000:.1f} seconds")
        
        # Parameters logged
        print(f"\n📋 PARAMETERS LOGGED ({len(opt_run.data.params)}):")
        for key, value in sorted(opt_run.data.params.items()):
            if key in ['features', 'n_trials', 'timeout', 'features_count']:
                print(f"   {key}: {value}")
            elif key.startswith(('max_depth', 'learning_rate', 'n_estimators')):
                print(f"   {key}: {value}")
        
        # Metrics logged
        print(f"\n📊 METRICS LOGGED ({len(opt_run.data.metrics)}):")
        for key, value in sorted(opt_run.data.metrics.items()):
            print(f"   {key}: {value:.4f}")
        
        # Artifacts
        artifacts = client.list_artifacts(opt_run.info.run_id)
        print(f"\n📦 ARTIFACTS STORED ({len(artifacts)}):")
        for artifact in artifacts:
            print(f"   {artifact.path} ({artifact.file_size} bytes)")
    
    # 5. Model Registry
    print(f"\n🏛️ MODEL REGISTRY:")
    try:
        registered_models = client.search_registered_models()
        if registered_models:
            for model in registered_models:
                print(f"   Model: {model.name}")
                versions = client.search_model_versions(f"name='{model.name}'")
                for version in versions:
                    print(f"     Version {version.version}: {version.current_stage}")
                    print(f"       Source: {version.source}")
                    if version.tags:
                        print(f"       Tags: {version.tags}")
        else:
            print("   No registered models found")
    except Exception as e:
        print(f"   Error accessing model registry: {e}")
    
    # 6. What MLflow gives us vs. what we had before
    print(f"\n" + "=" * 60)
    print("🔍 WHAT MLFLOW PROVIDES VS. BEFORE")
    print("=" * 60)
    
    print("\n❌ BEFORE MLflow (Traditional Approach):")
    print("   • Hyperparameters: Lost after script ends")
    print("   • Results: Only in terminal output or manual logs")
    print("   • Models: Saved to random file paths")
    print("   • Comparison: Manual spreadsheet tracking")
    print("   • Reproducibility: Hope you remember the exact command")
    print("   • Collaboration: Email model files around")
    
    print("\n✅ WITH MLflow:")
    print("   • Hyperparameters: Automatically logged and searchable")
    print("   • Results: Persistent database with web UI")
    print("   • Models: Centralized registry with versioning")
    print("   • Comparison: Built-in run comparison tools")
    print("   • Reproducibility: Exact environment + code captured")
    print("   • Collaboration: Shared tracking server")
    
    print(f"\n📁 FILE ORGANIZATION:")
    print(f"   • Database: mlflow/mlflow.db ({os.path.getsize('mlflow/mlflow.db')} bytes)")
    print(f"   • Artifacts: mlruns/ directory structure")
    print(f"   • Models: Automatically versioned and tagged")
    
    return df

if __name__ == "__main__":
    explore_mlflow_data()

