"""
Submit training job to Vertex AI Custom Training.

This script creates and submits a custom training job to Vertex AI,
using the containerized training code.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from google.cloud import aiplatform
from google.cloud.aiplatform import gapic as aip_gapic

from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_custom_training_job(
    display_name: str,
    container_image_uri: str,
    train_data_path: str,
    output_dir: str,
    machine_type: str = "n1-standard-4",
    replica_count: int = 1,
    accelerator_type: Optional[str] = None,
    accelerator_count: int = 0,
    args: Optional[list] = None,
    environment_variables: Optional[Dict[str, str]] = None,
    service_account: Optional[str] = None
) -> aiplatform.CustomContainerTrainingJob:
    """
    Create a Vertex AI Custom Container Training Job.
    
    Args:
        display_name: Display name for the training job
        container_image_uri: URI of the training container image
        train_data_path: GCS path to training data
        output_dir: GCS path for output artifacts
        machine_type: Machine type for training
        replica_count: Number of training replicas
        accelerator_type: Type of accelerator (e.g., "NVIDIA_TESLA_T4")
        accelerator_count: Number of accelerators per replica
        args: Additional arguments to pass to container
        environment_variables: Environment variables for container
        service_account: Service account email
        
    Returns:
        CustomContainerTrainingJob instance
    """
    # Initialize Vertex AI
    aiplatform.init(
        project=config.GCP_PROJECT_ID,
        location=config.GCP_REGION,
        staging_bucket=config.GCS_BUCKET_URI
    )
    
    # Create training job
    job = aiplatform.CustomContainerTrainingJob(
        display_name=display_name,
        container_uri=container_image_uri,
        model_serving_container_image_uri=None,  # Not deploying directly
    )
    
    logger.info(f"Created training job: {display_name}")
    
    return job


def submit_training_job(
    job_name: Optional[str] = None,
    container_image_uri: Optional[str] = None,
    train_data_path: Optional[str] = None,
    test_data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_version: Optional[str] = None,
    machine_type: str = "n1-standard-4",
    replica_count: int = 1,
    use_gpu: bool = False,
    service_account: Optional[str] = None,
    wait_for_completion: bool = True
) -> aiplatform.CustomContainerTrainingJob:
    """
    Submit training job to Vertex AI.
    
    Args:
        job_name: Name for the training job
        container_image_uri: URI of training container
        train_data_path: GCS path to training data
        test_data_path: GCS path to test data
        output_dir: GCS output directory
        model_version: Version string for model
        machine_type: Machine type for training
        replica_count: Number of replicas
        use_gpu: Whether to use GPU acceleration
        service_account: Service account email
        wait_for_completion: Whether to wait for job completion
        
    Returns:
        Training job instance
    """
    # Set defaults
    if job_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"xgboost_training_{timestamp}"
    
    if container_image_uri is None:
        # Use default training image URI from central config
        container_image_uri = config.TRAINING_IMAGE_URI
    
    # Default to the preprocessed training data created in Step 3.6
    # See scripts/02_preprocess_data.py which writes `train_processed.csv`
    # and uploads it to `data/processed/train_processed.csv` in GCS.
    if train_data_path is None:
        train_data_path = f"{config.GCS_DATA_URI}/processed/train_processed.csv"
    
    if output_dir is None:
        output_dir = f"{config.GCS_MODEL_URI}/{model_version or 'latest'}"
    
    if service_account is None:
        service_account = config.SERVICE_ACCOUNT_EMAIL
    
    # Configure GPU if requested
    accelerator_type = None
    accelerator_count = 0
    if use_gpu:
        accelerator_type = "NVIDIA_TESLA_T4"
        accelerator_count = 1
        logger.info("GPU acceleration enabled")
    
    # Prepare container arguments
    container_args = [
        "--train-path", train_data_path,
        "--output-dir", output_dir,
    ]
    
    if test_data_path:
        container_args.extend(["--test-path", test_data_path])
    
    if model_version:
        container_args.extend(["--version", model_version])
    
    logger.info("Submitting training job to Vertex AI")
    logger.info(f"Job Name: {job_name}")
    logger.info(f"Container URI: {container_image_uri}")
    logger.info(f"Train Data: {train_data_path}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Machine Type: {machine_type}")
    logger.info(f"Service Account: {service_account}")
    
    # Initialize Vertex AI
    aiplatform.init(
        project=config.GCP_PROJECT_ID,
        location=config.GCP_REGION,
        staging_bucket=config.GCS_BUCKET_URI
    )
    
    # Create custom container training job
    job = aiplatform.CustomContainerTrainingJob(
        display_name=job_name,
        container_uri=container_image_uri,
    )
    
    # Define environment variables for the container
    # Explicitly disable W&B for Vertex AI execution to prevent auth errors
    env_vars = {
        "USE_WANDB": "false",
        "WANDB_API_KEY": config.WANDB_API_KEY or "",
        "WANDB_PROJECT": config.WANDB_PROJECT,
        "GCP_PROJECT_ID": config.GCP_PROJECT_ID,
        "GCP_REGION": config.GCP_REGION,
    }
    
    logger.info(f"Environment variables: {env_vars}")

    # Run the training job
    logger.info("Starting training job execution...")
    
    run_kwargs = {
        "args": container_args,
        "environment_variables": env_vars,
        "replica_count": replica_count,
        "machine_type": machine_type,
        "base_output_dir": output_dir,
        "service_account": service_account,
        "sync": wait_for_completion,
    }
    
    if accelerator_type:
        run_kwargs["accelerator_type"] = accelerator_type
        run_kwargs["accelerator_count"] = accelerator_count
    
    job.run(**run_kwargs)
    
    if wait_for_completion:
        logger.info("✓ Training job completed successfully")
        logger.info(f"Job resource name: {job.resource_name}")
        logger.info(f"Output directory: {output_dir}")
    else:
        logger.info("Training job submitted (running asynchronously)")
        logger.info(f"Monitor job at: https://console.cloud.google.com/vertex-ai/training/custom-jobs")
    
    return job


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(
        description="Submit XGBoost training job to Vertex AI"
    )
    
    parser.add_argument(
        "--job-name",
        type=str,
        help="Name for the training job"
    )
    
    parser.add_argument(
        "--container-uri",
        type=str,
        help="Container image URI"
    )
    
    parser.add_argument(
        "--train-path",
        type=str,
        help="GCS path to training data"
    )
    
    parser.add_argument(
        "--test-path",
        type=str,
        help="GCS path to test data"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="GCS output directory"
    )
    
    parser.add_argument(
        "--version",
        type=str,
        help="Model version"
    )
    
    parser.add_argument(
        "--machine-type",
        type=str,
        default="n1-standard-4",
        help="Machine type for training"
    )
    
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration"
    )
    
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for job completion"
    )
    
    args = parser.parse_args()
    
    # Submit training job
    job = submit_training_job(
        job_name=args.job_name,
        container_image_uri=args.container_uri,
        train_data_path=args.train_path,
        test_data_path=args.test_path,
        output_dir=args.output_dir,
        model_version=args.version,
        machine_type=args.machine_type,
        use_gpu=args.use_gpu,
        wait_for_completion=not args.no_wait
    )
    
    print("\n" + "="*60)
    print("Training Job Submitted")
    print("="*60)
    print(f"Job Name: {job.display_name}")
    print(f"Resource Name: {job.resource_name}")
    print(f"Console URL: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={config.GCP_PROJECT_ID}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

