import mlflow
import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class MLFlowManager:
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None, artifact_location: Optional[str] = None):
        """
        Initialize MLflow manager.
        
        Args:
            experiment_name: Name of the experiment
            tracking_uri: URI for tracking server. Defaults to local sqlite DB in ./mlflow/mlflow.db.
            artifact_location: Location for artifacts. Defaults to ./mlruns.
        """
        self.experiment_name = experiment_name
        
        # Default to local SQLite DB if not specified
        if not tracking_uri:
            # Ensure mlflow directory exists
            mlflow_dir = Path("mlflow")
            mlflow_dir.mkdir(exist_ok=True)
            
            # Use absolute path for SQLite to avoid issues
            db_path = mlflow_dir.absolute() / "mlflow.db"
            self.tracking_uri = f"sqlite:///{db_path}"
        else:
            self.tracking_uri = tracking_uri
            
        logger.info(f"Setting MLflow tracking URI to: {self.tracking_uri}")
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        try:
            # Check if experiment exists
            experiment = mlflow.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                logger.info(f"Creating experiment: {experiment_name}")
                self.experiment_id = mlflow.create_experiment(
                    name=experiment_name,
                    artifact_location=artifact_location
                )
            else:
                self.experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name} (ID: {self.experiment_id})")
                
            mlflow.set_experiment(experiment_name=experiment_name)
            
        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {e}")
            raise

    @contextmanager
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, Any]] = None, nested: bool = False):
        """
        Context manager for MLflow run.
        
        Args:
            run_name: Optional name for the run
            tags: Optional dictionary of tags
            nested: Whether this run is nested inside another run
        """
        try:
            run = mlflow.start_run(run_name=run_name, tags=tags, nested=nested)
            logger.info(f"Started MLflow run: {run.info.run_id} (Name: {run_name})")
            yield run
        except Exception as e:
            logger.error(f"Error in MLflow run: {e}")
            raise
        finally:
            mlflow.end_run()
            logger.info("Ended MLflow run")

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to current run."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to current run."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a local file or directory as an artifact."""
        if not os.path.exists(local_path):
            logger.warning(f"Artifact not found at {local_path}, skipping log_artifact")
            return
        mlflow.log_artifact(local_path, artifact_path)

    def log_sklearn_model(self, model, artifact_path: str, registered_model_name: Optional[str] = None):
        """Log a scikit-learn model."""
        mlflow.sklearn.log_model(model, artifact_path, registered_model_name=registered_model_name)

    def log_xgboost_model(self, model, artifact_path: str, registered_model_name: Optional[str] = None):
        """Log an XGBoost model."""
        mlflow.xgboost.log_model(model, artifact_path, registered_model_name=registered_model_name)

    def register_model(self, model_uri: str, name: str, tags: Optional[Dict[str, str]] = None):
        """
        Register a model in the Model Registry.
        
        Args:
            model_uri: URI of the model (e.g., runs:/<run_id>/model)
            name: Name of the registered model
            tags: Optional tags for the model version
        """
        logger.info(f"Registering model {name} from {model_uri}")
        result = mlflow.register_model(model_uri, name)
        
        if tags:
            client = mlflow.tracking.MlflowClient()
            for key, value in tags.items():
                client.set_model_version_tag(
                    name=name,
                    version=result.version,
                    key=key,
                    value=value
                )
        return result

