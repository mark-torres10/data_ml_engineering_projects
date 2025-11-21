import logging
import os
import sys
from pathlib import Path
import pandas as pd

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
    logger.info("Starting data preprocessing pipeline...")
    
    # 1. Load Data
    loader = TitanicDataLoader()
    train_df, test_df = loader.load_data()
    
    # 2. Initialize Preprocessor
    preprocessor = TitanicPreprocessor()
    
    # 3. Process Training Data
    logger.info("Processing training data...")
    X_train = preprocessor.fit_transform(train_df)
    
    # Add target variable back for saving (Vertex AI AutoML/training often expects target in same file)
    # But for custom training we often split X and y. 
    # For Step 3.6 "Save processed training data", we usually want the complete dataset or X/y separate.
    # Let's save X_train with 'Survived' included for flexibility.
    
    # Re-attach target
    train_processed = X_train.copy()
    train_processed['Survived'] = train_df['Survived'].values
    
    logger.info(f"Processed training shape: {train_processed.shape}")
    
    # 4. Process Test Data
    logger.info("Processing test data...")
    X_test = preprocessor.transform_with_scaling(test_df)
    
    logger.info(f"Processed test shape: {X_test.shape}")
    
    # 5. Save Locally
    local_processed_dir = os.path.join("data", "processed")
    os.makedirs(local_processed_dir, exist_ok=True)
    
    train_path = os.path.join(local_processed_dir, "train_processed.csv")
    test_path = os.path.join(local_processed_dir, "test_processed.csv")
    
    train_processed.to_csv(train_path, index=False)
    X_test.to_csv(test_path, index=False) # Test set usually doesn't have target
    
    logger.info(f"Saved processed data locally to {local_processed_dir}")
    
    # 6. Save Preprocessor Artifact
    preprocessor_path = os.path.join("data", "artifacts", "preprocessor.joblib")
    preprocessor.save(preprocessor_path)
    
    # 7. Upload to GCS
    try:
        gcs = GCSManager()
        
        # Upload Data
        gcs_processed_path = os.path.join(config.GCS_DATA_PATH, "processed")
        gcs.upload_file(train_path, f"{gcs_processed_path}/train_processed.csv")
        gcs.upload_file(test_path, f"{gcs_processed_path}/test_processed.csv")
        
        # Upload Artifacts
        # We'll save artifacts in a 'model_artifacts' or similar folder, or just 'artifacts'
        # implementation_1.md says: gs://YOUR_BUCKET/data/processed/ for data
        # It also mentions saving scaler objects.
        gcs_artifact_path = "artifacts/preprocessor.joblib" 
        gcs.upload_file(preprocessor_path, gcs_artifact_path)
        
        logger.info("✓ All processed data and artifacts uploaded to GCS.")
        
    except Exception as e:
        logger.error(f"Failed to upload to GCS: {e}")
        logger.warning("Continuing without GCS upload. Check your credentials.")

if __name__ == "__main__":
    main()

