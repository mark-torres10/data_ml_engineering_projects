import logging
import sys
import pandas as pd
from pathlib import Path
from google.cloud import aiplatform

# Add project root to python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_serving():
    aiplatform.init(project=config.GCP_PROJECT_ID, location=config.GCP_REGION)
    
    fs_name = config.FEATURE_STORE_ID
    entity_name = config.ENTITY_TYPE_ID
    
    logger.info("Initializing Feature Store Client...")
    fs = aiplatform.Featurestore(featurestore_name=fs_name)
    entity_type = fs.get_entity_type(entity_type_id=entity_name)
    
    # Sample Entity IDs
    entity_ids = ["1", "2", "3"]
    
    logger.info(f"Fetching features for entities: {entity_ids}")
    
    try:
        # read_features returns a Pandas DataFrame
        # entity_type.read uses feature_ids list
        features_df = entity_type.read(
            entity_ids=entity_ids,
            feature_ids=["age", "sex_male", "survived", "pclass"]
        )
        
        logger.info("Features retrieved successfully!")
        print("\nFeature Values:")
        print(features_df)
        
        # Validate
        # Passenger 1: Male (1), Pclass 3, Survived 0
        p1 = features_df[features_df.index == "1"]
        if not p1.empty:
            age_val = p1['age'].values[0]
            if age_val is None:
                 logger.warning("Features returned None. Ingestion might still be in progress or Entity ID mismatch.")
                 logger.info("Please wait a few minutes and try again.")
            else:
                logger.info("Validating Passenger 1...")
                logger.info(f"Age: {age_val}")
                logger.info(f"Sex_male: {p1['sex_male'].values[0]}")
                logger.info(f"Survived: {p1['survived'].values[0]}")
            
    except Exception as e:
        logger.error(f"Feature serving failed: {e}")

if __name__ == "__main__":
    test_serving()

