import logging
import sys
from pathlib import Path
from google.cloud import aiplatform

# Add project root to python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_status():
    aiplatform.init(project=config.GCP_PROJECT_ID, location=config.GCP_REGION)
    fs_name = config.FEATURE_STORE_ID
    
    logger.info(f"Checking status of Feature Store: {fs_name}...")
    
    try:
        fs = aiplatform.Featurestore(featurestore_name=fs_name)
        logger.info(f"Feature Store exists: {fs.resource_name}")
        
        # Check Entity Type
        entity_name = config.ENTITY_TYPE_ID
        try:
            entity_type = fs.get_entity_type(entity_type_id=entity_name)
            logger.info(f"Entity Type '{entity_name}' exists.")
            
            # Check Features
            features = entity_type.list_features()
            logger.info(f"Found {len(features)} features:")
            for f in features:
                logger.info(f" - {f.name}")
                
        except Exception as e:
            logger.warning(f"Entity Type '{entity_name}' issue: {e}")
            
    except Exception as e:
        logger.error(f"Could not retrieve Feature Store: {e}")

if __name__ == "__main__":
    check_status()
