import logging
import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add project root to python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.data.loader import TitanicDataLoader
from src.features.preprocess import TitanicPreprocessor
from src.utils.gcs_utils import GCSManager
from src.config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Feature Store data preparation...")
    
    # 1. Load Raw Data (to get PassengerId)
    loader = TitanicDataLoader()
    train_df, _ = loader.load_data()
    
    # 2. Preprocess Features
    # Load the saved preprocessor to ensure consistency
    preprocessor_path = os.path.join("data", "artifacts", "preprocessor.joblib")
    if os.path.exists(preprocessor_path):
        preprocessor = TitanicPreprocessor.load(preprocessor_path)
        logger.info("Loaded existing preprocessor.")
    else:
        logger.info("Fitting new preprocessor...")
        preprocessor = TitanicPreprocessor()
        preprocessor.fit(train_df)
        
    X_train = preprocessor.transform_with_scaling(train_df)
    
    # 3. Construct Feature Store DataFrame
    # We need: entity_id (PassengerId), feature_timestamp, and features
    
    fs_df = X_train.copy()
    
    # Add Entity ID
    fs_df['passenger_id'] = train_df['PassengerId'].astype(str)
    
    # Add Timestamp (using current time for this batch)
    # In a real scenario, this might be the time the data was recorded
    current_time = datetime.now().isoformat()
    fs_df['feature_timestamp'] = current_time
    
    # Add Target (Optional, but useful for training data generation from FS)
    fs_df['Survived'] = train_df['Survived']
    
    # Lowercase all columns to match Feature Store requirements
    fs_df.columns = fs_df.columns.str.lower()
    
    logger.info(f"Feature Store DataFrame shape: {fs_df.shape}")
    logger.info(f"Columns: {fs_df.columns.tolist()}")
    
    # 4. Save locally
    local_fs_dir = os.path.join("data", "feature_store")
    os.makedirs(local_fs_dir, exist_ok=True)
    local_path = os.path.join(local_fs_dir, "titanic_features.csv")
    
    fs_df.to_csv(local_path, index=False)
    logger.info(f"Saved feature store data to {local_path}")
    
    # 5. Upload to GCS
    try:
        gcs = GCSManager()
        gcs_path = "data/feature_store/titanic_features.csv"
        gcs.upload_file(local_path, gcs_path)
        logger.info(f"Uploaded to gs://{config.GCS_BUCKET_NAME}/{gcs_path}")
    except Exception as e:
        logger.error(f"Failed to upload to GCS: {e}")

if __name__ == "__main__":
    main()

