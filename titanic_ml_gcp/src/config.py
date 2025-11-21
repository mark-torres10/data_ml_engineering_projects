"""
Configuration management for Titanic ML GCP project.

This module loads environment variables from .env file and provides
a centralized configuration object for the entire application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
# The .env file should be in the project root
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"

if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ Loaded environment variables from {env_path}")
else:
    print(f"⚠ Warning: .env file not found at {env_path}")
    print("  Using system environment variables only")


class Config:
    """
    Configuration class for GCP and application settings.
    
    All values are loaded from environment variables defined in .env file.
    """
    
    # GCP Project Configuration
    GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "titanic-ml-gcp")
    GCP_REGION = os.getenv("GCP_REGION", "us-central1")
    GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
    
    # Service Account
    SERVICE_ACCOUNT_EMAIL = os.getenv(
        "SERVICE_ACCOUNT_EMAIL",
        f"titanic-ml-sa@{GCP_PROJECT_ID}.iam.gserviceaccount.com"
    )
    
    # GCS Configuration
    GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", f"titanic-ml-data-{GCP_PROJECT_ID}")
    GCS_DATA_PATH = os.getenv("GCS_DATA_PATH", "data")
    GCS_MODEL_PATH = os.getenv("GCS_MODEL_PATH", "models")
    GCS_PREDICTIONS_PATH = os.getenv("GCS_PREDICTIONS_PATH", "predictions")
    
    # Construct full GCS URIs
    @property
    def GCS_BUCKET_URI(self):
        return f"gs://{self.GCS_BUCKET_NAME}"
    
    @property
    def GCS_DATA_URI(self):
        return f"{self.GCS_BUCKET_URI}/{self.GCS_DATA_PATH}"
    
    @property
    def GCS_MODEL_URI(self):
        return f"{self.GCS_BUCKET_URI}/{self.GCS_MODEL_PATH}"
    
    @property
    def GCS_PREDICTIONS_URI(self):
        return f"{self.GCS_BUCKET_URI}/{self.GCS_PREDICTIONS_PATH}"
    
    # Feature Store Configuration
    FEATURE_STORE_ID = os.getenv("FEATURE_STORE_ID", "titanic_featurestore")
    ENTITY_TYPE_ID = os.getenv("ENTITY_TYPE_ID", "passenger")
    
    # Model Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "titanic-xgboost")
    MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
    ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "titanic-prediction-endpoint")
    
    # Application Settings
    DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def display_config(cls):
        """Display current configuration (useful for debugging)."""
        print("\n" + "="*60)
        print("Current Configuration")
        print("="*60)
        print(f"GCP Project ID:      {cls.GCP_PROJECT_ID}")
        print(f"GCP Region:          {cls.GCP_REGION}")
        print(f"Service Account:     {cls.SERVICE_ACCOUNT_EMAIL}")
        print(f"GCS Bucket:          {cls.GCS_BUCKET_NAME}")
        print(f"Feature Store:       {cls.FEATURE_STORE_ID}")
        print(f"Model Name:          {cls.MODEL_NAME}")
        print(f"Model Version:       {cls.MODEL_VERSION}")
        print(f"Endpoint Name:       {cls.ENDPOINT_NAME}")
        print(f"Debug Mode:          {cls.DEBUG}")
        print("="*60 + "\n")


# Create a singleton config instance
config = Config()


if __name__ == "__main__":
    # When run directly, display the configuration
    config.display_config()

