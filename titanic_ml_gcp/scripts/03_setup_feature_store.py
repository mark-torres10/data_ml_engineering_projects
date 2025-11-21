import logging
import time
import sys
import os
from pathlib import Path
from google.cloud import aiplatform

# Add project root to python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_feature_store():
    aiplatform.init(project=config.GCP_PROJECT_ID, location=config.GCP_REGION)
    
    fs_name = config.FEATURE_STORE_ID
    
    logger.info(f"Creating/Retrieving Feature Store: {fs_name}...")
    
    try:
        # Check if exists by trying to get it
        # Resource name format: projects/{project}/locations/{location}/featurestores/{featurestore_id}
        # But SDK usually abstracts this via get() if initialized with project/location?
        # Actually aiplatform.Featurestore(featurestore_name=name) gets it.
        
        try:
            fs = aiplatform.Featurestore(featurestore_name=fs_name)
            logger.info(f"Feature Store {fs_name} already exists.")
            return fs
        except Exception:
            # If it doesn't exist or other error, try creating
            logger.info(f"Feature Store {fs_name} not found (or accessible). Creating...")
            
        fs = aiplatform.Featurestore.create(
            featurestore_id=fs_name,
            online_store_fixed_node_count=1,
        )
        logger.info(f"Feature Store {fs_name} creation initiated. Waiting...")
        fs.wait() # Wait for creation
        logger.info(f"Feature Store {fs_name} created.")
            
    except Exception as e:
        logger.error(f"Error creating Feature Store: {e}")
        raise

    return fs

def create_entity_type(fs):
    entity_name = config.ENTITY_TYPE_ID
    logger.info(f"Creating/Retrieving Entity Type: {entity_name}...")
    
    try:
        # Check if exists (SDK doesn't have direct check on entity type object easily without listing)
        # We can try to get it
        try:
            entity_type = fs.get_entity_type(entity_type_id=entity_name)
            logger.info(f"Entity Type {entity_name} already exists.")
        except:
            entity_type = fs.create_entity_type(
                entity_type_id=entity_name,
                description="Titanic passenger features"
            )
            logger.info(f"Entity Type {entity_name} creation initiated. Waiting...")
            entity_type.wait()
            logger.info(f"Entity Type {entity_name} created.")
            
    except Exception as e:
        logger.error(f"Error creating Entity Type: {e}")
        raise
        
    return entity_type

def create_features(entity_type):
    logger.info("Defining features...")
    
    # Define features and their types
    # Feature IDs must be lower case
    
    feature_specs = {
        'pclass': 'INT64',
        'age': 'DOUBLE',
        'fare': 'DOUBLE',
        'family_size': 'INT64',
        'is_alone': 'INT64',
        'has_cabin': 'INT64',
        'sex_male': 'INT64',
        'embarked_q': 'INT64',
        'embarked_s': 'INT64',
        'title_miss': 'INT64',
        'title_mr': 'INT64',
        'title_mrs': 'INT64',
        'title_officer': 'INT64',
        'title_royalty': 'INT64',
        'survived': 'INT64',
    }
    
    existing_features = entity_type.list_features()
    existing_feature_ids = [f.name for f in existing_features]
    
    features_to_create = {}
    for feat_name, feat_type in feature_specs.items():
        if feat_name not in existing_feature_ids:
            features_to_create[feat_name] = {
                "value_type": feat_type,
                "description": f"Titanic feature {feat_name}"
            }
    
    if features_to_create:
        logger.info(f"Creating {len(features_to_create)} features...")
        entity_type.batch_create_features(feature_configs=features_to_create)
        logger.info("Feature creation initiated.")
        time.sleep(10) 
    else:
        logger.info("All features already exist.")

def ingest_data(entity_type):
    logger.info("Starting batch ingestion...")
    
    gcs_source_uri = f"gs://{config.GCS_BUCKET_NAME}/data/feature_store/titanic_features.csv"
    
    # Define mapping: Feature ID -> CSV Column Name
    # Note: ingest_from_gcs takes feature_ids list if CSV columns match feature IDs
    # OR feature_ids dict mapping if they differ.
    # Our CSV has mixed case, Feature Store has lower case.
    # We should use a list if we change CSV header, or a dict if we map.
    # The SDK documentation for feature_ids says:
    # "A list of feature ids to be ingested... If the source column name does not match the feature id, 
    # you can assume the source column name is same as feature id." - Wait, usually there's a mapping support?
    # Checking SDK source/docs: usually ingest methods allow specifying source columns if different?
    # Actually, for CSV ingestion, Vertex AI often expects header to match feature IDs or requires a separate mapping config?
    # ingest_from_gcs params: feature_ids (List[str]).
    # It seems it expects CSV headers to MATCH feature IDs exactly.
    
    # If CSV headers are Pclass, Age... and Feature IDs are pclass, age...
    # We might have a problem.
    # Solution: Modify the CSV generation script to output lowercase columns
    # OR check if SDK supports mapping. SDK docs for `ingest_from_gcs` usually just take `feature_ids` list.
    
    # Safest bet: Update CSV generation to use lowercase columns.
    # But let's check if we can pass a dictionary? Type hint is Iterable[str].
    
    # Let's assume we need to update the CSV.
    # I will update `scripts/03_prepare_feature_store_data.py` to lowercase columns first.
    # But I can't re-upload easily without re-running that script.
    
    # Let's try to proceed assuming I will update the CSV generation script next.
    # For this function, I'll assume the CSV HAS lowercase headers.
    
    feature_specs = [
        'pclass', 'age', 'fare', 'family_size', 'is_alone', 'has_cabin', 'sex_male',
        'embarked_q', 'embarked_s', 'title_miss', 'title_mr', 'title_mrs', 'title_officer',
        'title_royalty', 'survived'
    ]
    
    try:
        entity_type.ingest_from_gcs(
            feature_ids=feature_specs,
            feature_time="feature_timestamp",
            entity_id_field="passenger_id",
            gcs_source_uris=[gcs_source_uri],
            gcs_source_type="csv",
            worker_count=1,
            sync=True
        )
        logger.info("Ingestion completed successfully.")
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise

def main():
    try:
        fs = create_feature_store()
        entity_type = create_entity_type(fs)
        create_features(entity_type)
        ingest_data(entity_type)
        logger.info("Feature Store setup and ingestion complete!")
    except Exception as e:
        logger.error(f"Feature Store setup failed: {e}")
        # Don't exit with error code to allow pipeline to continue if it's just a "already exists" or minor issue,
        # but for initial setup we want to know.
        sys.exit(1)

if __name__ == "__main__":
    import sys
    main()

