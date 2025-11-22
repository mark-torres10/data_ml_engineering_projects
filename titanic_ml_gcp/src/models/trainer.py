"""
XGBoost model trainer for Titanic survival prediction.

This module provides the XGBoostTrainer class for training models with:
- Data loading from GCS or Feature Store
- Train/validation splitting with stratification
- Cross-validation support
- Model training with early stopping
- Artifact management and logging
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

from src.config import config

# W&B imports - conditionally used based on config
try:
    import wandb
    from wandb.integration.xgboost import WandbCallback
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    WandbCallback = None
from src.models.model_config import XGBoostParams, TrainingConfig
from src.models.model_utils import (
    ModelArtifactManager,
    create_model_version_dir,
    save_training_artifacts,
    validate_feature_consistency,
    setup_cloud_logging,
)
from src.utils.gcs_utils import load_dataframe_from_gcs, upload_dataframe_to_gcs

logger = logging.getLogger(__name__)


class XGBoostTrainer:
    """
    Trainer class for XGBoost models.
    
    Handles the complete training workflow including data loading,
    preprocessing, model training, and artifact saving.
    """
    
    def __init__(
        self,
        model_params: Optional[XGBoostParams] = None,
        training_config: Optional[TrainingConfig] = None,
        job_name: Optional[str] = None
    ):
        """
        Initialize XGBoost trainer.
        
        Args:
            model_params: XGBoost hyperparameters
            training_config: Training configuration
            job_name: Name for the training job (for logging)
        """
        self.model_params = model_params or XGBoostParams()
        self.training_config = training_config or TrainingConfig()
        self.job_name = job_name or f"xgboost_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize model and artifact manager
        self.model: Optional[XGBClassifier] = None
        self.artifact_manager = ModelArtifactManager()
        
        # Training data
        self.X_train: Optional[pd.DataFrame] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_val: Optional[pd.Series] = None
        self.feature_names: Optional[list] = None
        
        # Training metrics
        self.training_metrics: Dict[str, Any] = {}
        
        # W&B run tracking
        self.wandb_run = None
        self.use_wandb = config.USE_WANDB and WANDB_AVAILABLE
        
        if self.use_wandb and not WANDB_AVAILABLE:
            logger.warning("W&B is enabled in config but wandb package is not installed. Disabling W&B.")
            self.use_wandb = False
        
        logger.info(f"Initialized XGBoostTrainer for job: {self.job_name}")
        if self.use_wandb:
            logger.info("W&B tracking enabled")
    
    def _init_wandb_run(self) -> None:
        """Initialize Weights & Biases run for experiment tracking."""
        if not self.use_wandb:
            return
        
        try:
            # Explicit login if API key is provided (useful inside containers)
            if config.WANDB_API_KEY:
                try:
                    wandb.login(key=config.WANDB_API_KEY)
                    logger.info("Authenticated with W&B using provided API key")
                except Exception as login_err:
                    logger.error(f"Failed to authenticate with W&B: {login_err}")
                    # Fail fast if we cannot authenticate with W&B
                    raise
            
            # Initialize W&B run
            self.wandb_run = wandb.init(
                project=config.WANDB_PROJECT,
                entity=config.WANDB_ENTITY if config.WANDB_ENTITY else None,
                name=self.job_name,
                config={
                    "model_type": "XGBoost",
                    "model_params": self.model_params.to_dict(),
                    "training_config": self.training_config.to_dict(),
                    "gcp_project": config.GCP_PROJECT_ID,
                    "gcp_region": config.GCP_REGION,
                },
                tags=["xgboost", "titanic", "training", "vertex-ai"],
                reinit=True  # Allow multiple runs in same process
            )
            logger.info(f"W&B run initialized: {self.wandb_run.name} (ID: {self.wandb_run.id})")
        except Exception as e:
            # Do not silently disable W&B; surface the error so the job fails loudly
            logger.error(f"Failed to initialize W&B run, aborting training: {e}")
            raise
    
    def _finish_wandb_run(self) -> None:
        """Finish Weights & Biases run."""
        if self.wandb_run is not None:
            try:
                wandb.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.error(f"Error finishing W&B run: {e}")
            finally:
                self.wandb_run = None
    
    def load_data_from_gcs(
        self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and test data from GCS.
        
        Args:
            train_path: GCS path to training data
            test_path: GCS path to test data
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_path = train_path or self.training_config.gcs_train_path
        test_path = test_path or self.training_config.gcs_test_path
        
        if not train_path:
            raise ValueError("Training data path must be provided")
        
        logger.info(f"Loading training data from: {train_path}")
        train_df = load_dataframe_from_gcs(train_path)
        logger.info(f"Loaded training data: {train_df.shape}")
        
        test_df = None
        if test_path:
            logger.info(f"Loading test data from: {test_path}")
            test_df = load_dataframe_from_gcs(test_path)
            logger.info(f"Loaded test data: {test_df.shape}")
        
        return train_df, test_df
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str = "Survived",
        drop_columns: Optional[list] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target from dataframe.
        
        Args:
            df: Input dataframe
            target_column: Name of target column
            drop_columns: Additional columns to drop
            
        Returns:
            Tuple of (X, y)
        """
        # Default columns to drop
        default_drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
        drop_columns = drop_columns or []
        all_drop_cols = list(set(default_drop_cols + drop_columns))
        
        # Remove target from drop list if present
        if target_column in all_drop_cols:
            all_drop_cols.remove(target_column)
        
        # Separate features and target
        y = df[target_column] if target_column in df.columns else None
        
        # Drop unnecessary columns
        X = df.drop(columns=[col for col in all_drop_cols if col in df.columns])
        
        # Drop target from features if present
        if target_column in X.columns:
            X = X.drop(columns=[target_column])
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Prepared features: {X.shape}, target: {y.shape if y is not None else 'None'}")
        logger.info(f"Feature names: {self.feature_names}")
        
        return X, y
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: Optional[float] = None,
        stratify: Optional[bool] = None,
        random_state: Optional[int] = None
    ) -> None:
        """
        Split data into training and validation sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            validation_split: Fraction for validation (0-1)
            stratify: Whether to use stratified split
            random_state: Random seed
        """
        validation_split = validation_split or self.training_config.validation_split
        stratify = stratify if stratify is not None else self.training_config.stratify
        random_state = random_state or self.training_config.random_state
        
        if validation_split > 0:
            stratify_arg = y if stratify else None
            
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X, y,
                test_size=validation_split,
                random_state=random_state,
                stratify=stratify_arg
            )
            
            logger.info(f"Split data - Train: {self.X_train.shape}, Val: {self.X_val.shape}")
            logger.info(f"Train target distribution: {dict(self.y_train.value_counts())}")
            logger.info(f"Val target distribution: {dict(self.y_val.value_counts())}")
        else:
            self.X_train = X
            self.y_train = y
            self.X_val = None
            self.y_val = None
            
            logger.info(f"No validation split - Train: {self.X_train.shape}")
    
    def perform_cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Perform cross-validation to estimate model performance.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary with CV metrics
        """
        if not self.training_config.use_cross_validation:
            logger.info("Cross-validation disabled")
            return {}
        
        logger.info(f"Performing {self.training_config.cv_folds}-fold cross-validation")
        
        # Create XGBoost model for cross-validation.
        # Early stopping requires an eval_set, which isn't used in cross_val_score,
        # so we disable it here to avoid configuration errors.
        cv_params = self.model_params.to_dict().copy()
        cv_params.pop("early_stopping_rounds", None)
        cv_model = XGBClassifier(**cv_params)
        
        # Stratified K-Fold
        cv = StratifiedKFold(
            n_splits=self.training_config.cv_folds,
            shuffle=True,
            random_state=self.training_config.random_state
        )
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            cv_model, X, y,
            cv=cv,
            scoring=self.training_config.cv_scoring,
            n_jobs=-1
        )
        
        cv_results = {
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "cv_scores": cv_scores.tolist()
        }
        
        logger.info(f"CV {self.training_config.cv_scoring}: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']:.4f})")
        
        # Log to W&B
        if self.use_wandb and self.wandb_run is not None:
            try:
                wandb.log({
                    "cv/mean_score": cv_results["cv_mean"],
                    "cv/std_score": cv_results["cv_std"],
                })
                
                # Log fold scores as histogram
                wandb.log({"cv/fold_scores": wandb.Histogram(cv_results["cv_scores"])})
                
                # Log individual fold scores
                for i, score in enumerate(cv_results["cv_scores"]):
                    wandb.log({f"cv/fold_{i+1}_score": score})
                
                logger.info("Cross-validation results logged to W&B")
            except Exception as e:
                logger.error(f"Error logging CV results to W&B: {e}")
        
        return cv_results
    
    def train(
        self,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> XGBClassifier:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features (uses self.X_train if None)
            y_train: Training target (uses self.y_train if None)
            X_val: Validation features (uses self.X_val if None)
            y_val: Validation target (uses self.y_val if None)
            
        Returns:
            Trained model
        """
        X_train = X_train if X_train is not None else self.X_train
        y_train = y_train if y_train is not None else self.y_train
        X_val = X_val if X_val is not None else self.X_val
        y_val = y_val if y_val is not None else self.y_val
        
        if X_train is None or y_train is None:
            raise ValueError("Training data not provided")
        
        logger.info("Starting model training...")
        start_time = datetime.now()
        
        # Create model
        self.model = XGBClassifier(**self.model_params.to_dict())
        
        # Prepare eval set for early stopping
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Configure callbacks on the model (XGBoost 2.x uses the `callbacks` attribute)
        if self.use_wandb and self.wandb_run is not None and WandbCallback is not None:
            try:
                self.model.set_params(
                    callbacks=[
                        WandbCallback(
                            log_model=True,  # Log model as artifact
                            log_feature_importance=True,  # Log feature importance
                            define_metric=True,  # Define custom metrics
                        )
                    ]
                )
                logger.info("W&B callback configured on XGBoost model")
            except Exception as e:
                logger.error(f"Error configuring W&B callback on model: {e}")
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=self.training_config.verbose
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        
        self.training_metrics = {
            "train_accuracy": float(accuracy_score(y_train, train_pred)),
            "train_auc": float(roc_auc_score(y_train, train_pred_proba)),
            "training_time_seconds": training_time,
            "n_estimators": self.model.n_estimators,
            "best_iteration": getattr(self.model, "best_iteration", None)
        }
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_pred_proba = self.model.predict_proba(X_val)[:, 1]
            
            self.training_metrics.update({
                "val_accuracy": float(accuracy_score(y_val, val_pred)),
                "val_auc": float(roc_auc_score(y_val, val_pred_proba))
            })
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Train accuracy: {self.training_metrics['train_accuracy']:.4f}")
        logger.info(f"Train AUC: {self.training_metrics['train_auc']:.4f}")
        
        if "val_accuracy" in self.training_metrics:
            logger.info(f"Val accuracy: {self.training_metrics['val_accuracy']:.4f}")
            logger.info(f"Val AUC: {self.training_metrics['val_auc']:.4f}")
        
        # Log final training metrics to W&B
        if self.use_wandb and self.wandb_run is not None:
            try:
                wandb.log({
                    "train/final_accuracy": self.training_metrics["train_accuracy"],
                    "train/final_auc": self.training_metrics["train_auc"],
                    "train/duration_seconds": self.training_metrics["training_time_seconds"],
                    "train/n_estimators": self.training_metrics["n_estimators"],
                })
                
                if self.training_metrics.get("best_iteration"):
                    wandb.log({"train/best_iteration": self.training_metrics["best_iteration"]})
                
                if "val_accuracy" in self.training_metrics:
                    wandb.log({
                        "val/final_accuracy": self.training_metrics["val_accuracy"],
                        "val/final_auc": self.training_metrics["val_auc"],
                    })
                
                logger.info("Training metrics logged to W&B")
            except Exception as e:
                logger.error(f"Error logging training metrics to W&B: {e}")
        
        return self.model
    
    def save_model(
        self,
        output_dir: Optional[Path] = None,
        version: Optional[str] = None,
        upload_to_gcs: bool = True
    ) -> Dict[str, Any]:
        """
        Save trained model and artifacts.
        
        Args:
            output_dir: Directory to save artifacts
            version: Version string for the model
            upload_to_gcs: Whether to upload to GCS
            
        Returns:
            Dictionary with paths to saved artifacts
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        
        # Create version directory
        base_dir = Path(output_dir or self.training_config.output_dir)
        version_dir = create_model_version_dir(base_dir, version)
        
        # Prepare metadata
        metadata = {
            "job_name": self.job_name,
            "timestamp": datetime.now().isoformat(),
            "model_params": self.model_params.to_dict(),
            "training_config": self.training_config.to_dict(),
            "training_metrics": self.training_metrics,
            "feature_names": self.feature_names,
            "n_features": len(self.feature_names) if self.feature_names else 0,
        }
        
        # Save artifacts locally
        artifact_paths = save_training_artifacts(
            model=self.model,
            output_dir=version_dir,
            metadata=metadata,
            feature_names=self.feature_names,
            save_format=self.training_config.save_format
        )
        
        # Upload to GCS if requested
        if upload_to_gcs:
            gcs_paths = {}
            for artifact_type, local_path in artifact_paths.items():
                gcs_path = f"{config.GCS_MODEL_PATH}/{version or 'latest'}/{local_path.name}"
                gcs_uri = self.artifact_manager.upload_to_gcs(local_path, gcs_path)
                gcs_paths[artifact_type] = gcs_uri
            
            logger.info(f"Artifacts uploaded to GCS: {config.GCS_MODEL_URI}")
            return {"local": artifact_paths, "gcs": gcs_paths}
        
        return {"local": artifact_paths}
    
    def run_training_pipeline(
        self,
        train_path: Optional[str] = None,
        test_path: Optional[str] = None,
        target_column: str = "Survived",
        output_dir: Optional[Path] = None,
        version: Optional[str] = None,
        upload_to_gcs: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Args:
            train_path: GCS path to training data
            test_path: GCS path to test data
            target_column: Name of target column
            output_dir: Directory to save artifacts
            version: Model version
            upload_to_gcs: Whether to upload to GCS
            
        Returns:
            Dictionary with training results and artifact paths
        """
        logger.info(f"Starting training pipeline: {self.job_name}")
        
        # Initialize W&B run
        self._init_wandb_run()
        
        try:
            # Load data
            train_df, test_df = self.load_data_from_gcs(train_path, test_path)
            
            # Log dataset info to W&B
            if self.use_wandb and self.wandb_run is not None:
                try:
                    wandb.log({
                        "data/train_samples": len(train_df),
                        "data/train_features": len(train_df.columns),
                    })
                    if test_df is not None:
                        wandb.log({
                            "data/test_samples": len(test_df),
                        })
                except Exception as e:
                    logger.error(f"Error logging data info to W&B: {e}")
            
            # Prepare features and target
            X, y = self.prepare_data(train_df, target_column)
            
            # Log feature info to W&B
            if self.use_wandb and self.wandb_run is not None:
                try:
                    wandb.log({
                        "data/n_features": len(self.feature_names),
                        "data/class_balance": float(y.mean()),
                    })
                except Exception as e:
                    logger.error(f"Error logging feature info to W&B: {e}")
            
            # Perform cross-validation
            cv_results = self.perform_cross_validation(X, y)
            
            # Split data
            self.split_data(X, y)
            
            # Train model
            self.train()
            
            # Save model
            artifact_paths = self.save_model(output_dir, version, upload_to_gcs)
            
            # Compile results
            results = {
                "job_name": self.job_name,
                "training_metrics": self.training_metrics,
                "cv_results": cv_results,
                "artifact_paths": artifact_paths,
                "feature_names": self.feature_names,
                "completed_at": datetime.now().isoformat()
            }
            
            # Log summary to W&B
            if self.use_wandb and self.wandb_run is not None:
                try:
                    wandb.summary.update({
                        "final_train_accuracy": self.training_metrics.get("train_accuracy"),
                        "final_train_auc": self.training_metrics.get("train_auc"),
                        "final_val_accuracy": self.training_metrics.get("val_accuracy"),
                        "final_val_auc": self.training_metrics.get("val_auc"),
                        "cv_mean": cv_results.get("cv_mean") if cv_results else None,
                        "training_time": self.training_metrics.get("training_time_seconds"),
                    })
                except Exception as e:
                    logger.error(f"Error updating W&B summary: {e}")
            
            logger.info("Training pipeline completed successfully")
            
            return results
        
        finally:
            # Always finish W&B run, even if pipeline fails
            self._finish_wandb_run()


def main():
    """Main function for running training from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train XGBoost model for Titanic prediction")
    parser.add_argument("--train-path", type=str, required=True, help="GCS path to training data")
    parser.add_argument("--test-path", type=str, help="GCS path to test data")
    parser.add_argument("--output-dir", type=str, default="outputs/models", help="Output directory")
    parser.add_argument("--version", type=str, help="Model version")
    parser.add_argument("--no-upload", action="store_true", help="Skip GCS upload")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_cloud_logging("xgboost_training", args.log_level)
    
    # Create trainer
    trainer = XGBoostTrainer()
    
    # Run training pipeline
    results = trainer.run_training_pipeline(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=Path(args.output_dir),
        version=args.version,
        upload_to_gcs=not args.no_upload
    )
    
    print("\n" + "="*60)
    print("Training Results")
    print("="*60)
    print(f"Job Name: {results['job_name']}")
    print(f"Training Accuracy: {results['training_metrics']['train_accuracy']:.4f}")
    print(f"Training AUC: {results['training_metrics']['train_auc']:.4f}")
    if "val_accuracy" in results['training_metrics']:
        print(f"Validation Accuracy: {results['training_metrics']['val_accuracy']:.4f}")
        print(f"Validation AUC: {results['training_metrics']['val_auc']:.4f}")
    if results['cv_results']:
        print(f"CV Score: {results['cv_results']['cv_mean']:.4f} (+/- {results['cv_results']['cv_std']:.4f})")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

