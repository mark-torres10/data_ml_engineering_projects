"""
Model configuration and hyperparameters for XGBoost training.

This module provides configuration dataclasses for model hyperparameters,
training parameters, and evaluation settings.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class XGBoostParams:
    """XGBoost model hyperparameters."""
    
    # Core XGBoost parameters
    objective: str = "binary:logistic"
    max_depth: int = 5
    learning_rate: float = 0.1
    n_estimators: int = 100
    
    # Regularization
    min_child_weight: int = 1
    gamma: float = 0.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    
    # Training behavior
    random_state: int = 42
    n_jobs: int = -1
    tree_method: str = "hist"
    
    # Early stopping
    early_stopping_rounds: int = 10
    eval_metric: str = "auc"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary for XGBoost."""
        return {
            "objective": self.objective,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "n_estimators": self.n_estimators,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "tree_method": self.tree_method,
            "early_stopping_rounds": self.early_stopping_rounds,
            "eval_metric": self.eval_metric,
        }
    
    def save(self, path: Path) -> None:
        """Save parameters to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "XGBoostParams":
        """Load parameters from JSON file."""
        with open(path, "r") as f:
            params = json.load(f)
        return cls(**params)


@dataclass
class TrainingConfig:
    """Configuration for training process."""
    
    # Data split
    validation_split: float = 0.2
    test_split: float = 0.0  # If 0, assumes test data is separate
    stratify: bool = True
    random_state: int = 42
    
    # Training settings
    use_cross_validation: bool = True
    cv_folds: int = 5
    cv_scoring: str = "roc_auc"
    
    # Feature selection
    feature_selection: bool = False
    max_features: Optional[int] = None
    
    # Data source
    data_source: str = "gcs"  # "gcs" or "feature_store"
    gcs_train_path: Optional[str] = None
    gcs_test_path: Optional[str] = None
    
    # Output settings
    output_dir: str = "outputs/models"
    model_name: str = "xgboost_model"
    save_format: str = "json"  # "json" or "pickle"
    
    # Logging
    log_interval: int = 10
    verbose: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "validation_split": self.validation_split,
            "test_split": self.test_split,
            "stratify": self.stratify,
            "random_state": self.random_state,
            "use_cross_validation": self.use_cross_validation,
            "cv_folds": self.cv_folds,
            "cv_scoring": self.cv_scoring,
            "feature_selection": self.feature_selection,
            "max_features": self.max_features,
            "data_source": self.data_source,
            "gcs_train_path": self.gcs_train_path,
            "gcs_test_path": self.gcs_test_path,
            "output_dir": self.output_dir,
            "model_name": self.model_name,
            "save_format": self.save_format,
            "log_interval": self.log_interval,
            "verbose": self.verbose,
        }
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            config = json.load(f)
        return cls(**config)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Metrics to calculate
    calculate_accuracy: bool = True
    calculate_precision: bool = True
    calculate_recall: bool = True
    calculate_f1: bool = True
    calculate_auc_roc: bool = True
    calculate_confusion_matrix: bool = True
    
    # Classification threshold
    classification_threshold: float = 0.5
    
    # Visualization
    plot_roc_curve: bool = True
    plot_confusion_matrix: bool = True
    plot_feature_importance: bool = True
    plot_precision_recall: bool = True
    
    # Output settings
    save_plots: bool = True
    save_metrics: bool = True
    output_dir: str = "outputs/evaluation"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "calculate_accuracy": self.calculate_accuracy,
            "calculate_precision": self.calculate_precision,
            "calculate_recall": self.calculate_recall,
            "calculate_f1": self.calculate_f1,
            "calculate_auc_roc": self.calculate_auc_roc,
            "calculate_confusion_matrix": self.calculate_confusion_matrix,
            "classification_threshold": self.classification_threshold,
            "plot_roc_curve": self.plot_roc_curve,
            "plot_confusion_matrix": self.plot_confusion_matrix,
            "plot_feature_importance": self.plot_feature_importance,
            "plot_precision_recall": self.plot_precision_recall,
            "save_plots": self.save_plots,
            "save_metrics": self.save_metrics,
            "output_dir": self.output_dir,
        }

