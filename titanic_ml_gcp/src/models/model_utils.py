"""
Utility functions for model training, saving, loading, and management.

This module provides helper functions for:
- Model serialization and deserialization
- GCS model artifact management
- Feature engineering for training
- Logging and monitoring
"""

import json
import joblib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from google.cloud import storage

from src.config import config

# Configure logging
logger = logging.getLogger(__name__)


class ModelArtifactManager:
    """Manages model artifacts including saving, loading, and GCS operations."""
    
    def __init__(self, bucket_name: Optional[str] = None):
        """
        Initialize model artifact manager.
        
        Args:
            bucket_name: GCS bucket name. If None, uses config.GCS_BUCKET_NAME
        """
        self.bucket_name = bucket_name or config.GCS_BUCKET_NAME
        self.storage_client = storage.Client(project=config.GCP_PROJECT_ID)
        self.bucket = self.storage_client.bucket(self.bucket_name)
        
    def save_model_local(
        self,
        model: XGBClassifier,
        model_path: Path,
        save_format: str = "json"
    ) -> None:
        """
        Save XGBoost model to local file.
        
        Args:
            model: Trained XGBoost model
            model_path: Path to save model
            save_format: "json" or "pickle"
        """
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_format == "json":
            model.save_model(str(model_path.with_suffix(".json")))
            logger.info(f"Model saved to {model_path.with_suffix('.json')}")
        elif save_format == "pickle":
            joblib.dump(model, model_path.with_suffix(".pkl"))
            logger.info(f"Model saved to {model_path.with_suffix('.pkl')}")
        else:
            raise ValueError(f"Unsupported save format: {save_format}")
    
    def load_model_local(
        self,
        model_path: Path,
        load_format: str = "json"
    ) -> XGBClassifier:
        """
        Load XGBoost model from local file.
        
        Args:
            model_path: Path to model file
            load_format: "json" or "pickle"
            
        Returns:
            Loaded XGBoost model
        """
        if load_format == "json":
            model = XGBClassifier()
            model.load_model(str(model_path.with_suffix(".json")))
            logger.info(f"Model loaded from {model_path.with_suffix('.json')}")
        elif load_format == "pickle":
            model = joblib.load(model_path.with_suffix(".pkl"))
            logger.info(f"Model loaded from {model_path.with_suffix('.pkl')}")
        else:
            raise ValueError(f"Unsupported load format: {load_format}")
        
        return model
    
    def upload_to_gcs(
        self,
        local_path: Path,
        gcs_path: str,
        make_public: bool = False
    ) -> str:
        """
        Upload file to GCS.
        
        Args:
            local_path: Local file path
            gcs_path: GCS destination path (without gs://bucket/)
            make_public: Whether to make the blob publicly readable
            
        Returns:
            GCS URI of uploaded file
        """
        blob = self.bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path))
        
        if make_public:
            blob.make_public()
        
        gcs_uri = f"gs://{self.bucket_name}/{gcs_path}"
        logger.info(f"Uploaded {local_path} to {gcs_uri}")
        return gcs_uri
    
    def download_from_gcs(
        self,
        gcs_path: str,
        local_path: Path
    ) -> Path:
        """
        Download file from GCS.
        
        Args:
            gcs_path: GCS source path (without gs://bucket/)
            local_path: Local destination path
            
        Returns:
            Local file path
        """
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        blob = self.bucket.blob(gcs_path)
        blob.download_to_filename(str(local_path))
        
        logger.info(f"Downloaded gs://{self.bucket_name}/{gcs_path} to {local_path}")
        return local_path
    
    def save_metadata(
        self,
        metadata: Dict[str, Any],
        output_path: Path
    ) -> None:
        """
        Save model metadata to JSON file.
        
        Args:
            metadata: Dictionary containing metadata
            output_path: Path to save metadata JSON
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.now().isoformat()
        
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {output_path}")
    
    def load_metadata(self, metadata_path: Path) -> Dict[str, Any]:
        """
        Load model metadata from JSON file.
        
        Args:
            metadata_path: Path to metadata JSON
            
        Returns:
            Metadata dictionary
        """
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        logger.info(f"Metadata loaded from {metadata_path}")
        return metadata
    
    def list_gcs_models(self, prefix: str = "models/") -> List[str]:
        """
        List all model files in GCS bucket.
        
        Args:
            prefix: GCS prefix to search
            
        Returns:
            List of GCS paths
        """
        blobs = self.bucket.list_blobs(prefix=prefix)
        model_paths = [blob.name for blob in blobs if blob.name.endswith(('.json', '.pkl'))]
        
        logger.info(f"Found {len(model_paths)} model files in gs://{self.bucket_name}/{prefix}")
        return model_paths


def create_model_version_dir(
    base_dir: Path,
    version: Optional[str] = None
) -> Path:
    """
    Create a versioned directory for model artifacts.
    
    Args:
        base_dir: Base directory for models
        version: Version string. If None, uses timestamp
        
    Returns:
        Path to versioned directory
    """
    if version is None:
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    version_dir = base_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created model version directory: {version_dir}")
    return version_dir


def save_training_artifacts(
    model: XGBClassifier,
    output_dir: Path,
    metadata: Dict[str, Any],
    feature_names: List[str],
    save_format: str = "json"
) -> Dict[str, Path]:
    """
    Save all training artifacts (model, metadata, feature names).
    
    Args:
        model: Trained XGBoost model
        output_dir: Directory to save artifacts
        metadata: Training metadata
        feature_names: List of feature names
        save_format: Model save format
        
    Returns:
        Dictionary mapping artifact type to file path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_manager = ModelArtifactManager()
    
    # Save model
    model_path = output_dir / "model"
    artifact_manager.save_model_local(model, model_path, save_format)
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    artifact_manager.save_metadata(metadata, metadata_path)
    
    # Save feature names
    features_path = output_dir / "features.json"
    with open(features_path, "w") as f:
        json.dump({"feature_names": feature_names}, f, indent=2)
    
    logger.info(f"All training artifacts saved to {output_dir}")
    
    return {
        "model": model_path.with_suffix(f".{save_format}"),
        "metadata": metadata_path,
        "features": features_path
    }


def load_training_artifacts(
    artifact_dir: Path,
    load_format: str = "json"
) -> Tuple[XGBClassifier, Dict[str, Any], List[str]]:
    """
    Load all training artifacts.
    
    Args:
        artifact_dir: Directory containing artifacts
        load_format: Model load format
        
    Returns:
        Tuple of (model, metadata, feature_names)
    """
    artifact_manager = ModelArtifactManager()
    
    # Load model
    model_path = artifact_dir / "model"
    model = artifact_manager.load_model_local(model_path, load_format)
    
    # Load metadata
    metadata_path = artifact_dir / "metadata.json"
    metadata = artifact_manager.load_metadata(metadata_path)
    
    # Load feature names
    features_path = artifact_dir / "features.json"
    with open(features_path, "r") as f:
        feature_data = json.load(f)
        feature_names = feature_data["feature_names"]
    
    logger.info(f"All training artifacts loaded from {artifact_dir}")
    
    return model, metadata, feature_names


def get_feature_importance(
    model: XGBClassifier,
    feature_names: List[str],
    importance_type: str = "weight"
) -> pd.DataFrame:
    """
    Get feature importance from trained model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        importance_type: Type of importance ("weight", "gain", "cover")
        
    Returns:
        DataFrame with feature names and importance scores
    """
    importance_dict = model.get_booster().get_score(importance_type=importance_type)
    
    # Create DataFrame
    importance_df = pd.DataFrame([
        {"feature": feature, "importance": importance_dict.get(f"f{i}", 0.0)}
        for i, feature in enumerate(feature_names)
    ])
    
    # Sort by importance
    importance_df = importance_df.sort_values("importance", ascending=False)
    
    return importance_df


def validate_feature_consistency(
    train_features: List[str],
    test_features: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate that training and test features match.
    
    Args:
        train_features: Features used in training
        test_features: Features in test data
        
    Returns:
        Tuple of (is_valid, list_of_mismatches)
    """
    train_set = set(train_features)
    test_set = set(test_features)
    
    missing_in_test = train_set - test_set
    extra_in_test = test_set - train_set
    
    mismatches = []
    if missing_in_test:
        mismatches.append(f"Missing in test: {missing_in_test}")
    if extra_in_test:
        mismatches.append(f"Extra in test: {extra_in_test}")
    
    is_valid = len(mismatches) == 0
    
    if is_valid:
        logger.info("✓ Feature consistency check passed")
    else:
        logger.warning(f"✗ Feature consistency check failed: {mismatches}")
    
    return is_valid, mismatches


def setup_cloud_logging(
    job_name: str,
    log_level: str = "INFO"
) -> logging.Logger:
    """
    Set up Cloud Logging for training jobs.
    
    Args:
        job_name: Name of the training job
        log_level: Logging level
        
    Returns:
        Configured logger
    """
    try:
        from google.cloud import logging as cloud_logging
        
        # Create Cloud Logging client
        logging_client = cloud_logging.Client(project=config.GCP_PROJECT_ID)
        
        # Set up Cloud Logging handler
        handler = cloud_logging.handlers.CloudLoggingHandler(
            logging_client,
            name=job_name
        )
        
        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, log_level.upper()))
        logger.addHandler(handler)
        
        logger.info(f"Cloud Logging configured for job: {job_name}")
        
    except Exception as e:
        logger.warning(f"Could not set up Cloud Logging: {e}")
        logger.info("Using console logging instead")
        
        # Fall back to console logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger()
    
    return logger

