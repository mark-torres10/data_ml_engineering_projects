#!/usr/bin/env python3
"""
Script to load Titanic dataset and save to local and GCS storage.

This script:
1. Loads the Titanic dataset from HuggingFace
2. Saves raw data to local data/raw directory
3. Uploads data to Google Cloud Storage
4. Displays dataset information

Usage:
    python scripts/01_load_and_save_data.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import TitanicDataLoader
from src.utils.gcp_client import get_storage_client
from src.config import config
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    
    print("\n" + "="*70)
    print("Titanic ML GCP - Data Loading Pipeline")
    print("="*70 + "\n")
    
    # Initialize data loader
    logger.info("Step 1: Loading Titanic dataset from HuggingFace...")
    loader = TitanicDataLoader()
    train_df, test_df = loader.load_data()
    
    # Display dataset information
    print("\n")
    loader.display_info()
    
    # Save to local files
    logger.info("\nStep 2: Saving data to local files...")
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = data_dir / "titanic_train.csv"
    test_path = data_dir / "titanic_test.csv"
    
    loader.save_to_csv(train_path, test_path)
    
    print(f"✓ Saved training data: {train_path}")
    print(f"✓ Saved test data: {test_path}")
    
    # Upload to GCS
    logger.info("\nStep 3: Uploading data to Google Cloud Storage...")
    try:
        upload_to_gcs(train_path, test_path)
        print("✓ Data uploaded to GCS successfully")
    except Exception as e:
        logger.warning(f"Could not upload to GCS: {e}")
        print(f"⚠ GCS upload skipped: {e}")
        print("  You can upload manually later when GCS bucket is created.")
    
    # Summary
    print("\n" + "="*70)
    print("Data Loading Complete!")
    print("="*70)
    print(f"\nLocal files:")
    print(f"  - {train_path}")
    print(f"  - {test_path}")
    print(f"\nGCS bucket (when created):")
    print(f"  - {config.GCS_BUCKET_URI}")
    print("="*70 + "\n")


def upload_to_gcs(train_path: Path, test_path: Path):
    """
    Upload data files to Google Cloud Storage.
    
    Args:
        train_path: Path to training data CSV
        test_path: Path to test data CSV
    """
    # Get GCS client
    storage_client = get_storage_client()
    
    # Get or create bucket
    bucket_name = config.GCS_BUCKET_NAME
    try:
        bucket = storage_client.get_bucket(bucket_name)
        logger.info(f"Using existing bucket: {bucket_name}")
    except Exception:
        logger.info(f"Creating bucket: {bucket_name}")
        bucket = storage_client.create_bucket(
            bucket_name,
            location=config.GCP_REGION
        )
    
    # Upload training data
    train_blob = bucket.blob(f"{config.GCS_DATA_PATH}/raw/titanic_train.csv")
    train_blob.upload_from_filename(str(train_path))
    logger.info(f"✓ Uploaded: gs://{bucket_name}/{train_blob.name}")
    
    # Upload test data
    test_blob = bucket.blob(f"{config.GCS_DATA_PATH}/raw/titanic_test.csv")
    test_blob.upload_from_filename(str(test_path))
    logger.info(f"✓ Uploaded: gs://{bucket_name}/{test_blob.name}")


if __name__ == "__main__":
    main()

