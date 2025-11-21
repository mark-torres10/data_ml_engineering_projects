"""
Train XGBoost model locally for testing.

This script allows you to test the training pipeline locally
before submitting to Vertex AI.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.trainer import XGBoostTrainer
from src.models.model_config import XGBoostParams, TrainingConfig
from src.models.evaluate import ModelEvaluator
from src.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_local(
    train_path: str,
    test_path: str = None,
    output_dir: str = "outputs/models/local",
    version: str = None,
    upload_to_gcs: bool = False,
    run_evaluation: bool = True,
    **kwargs
) -> dict:
    """
    Train model locally.
    
    Args:
        train_path: Path or GCS URI to training data
        test_path: Path or GCS URI to test data
        output_dir: Local output directory
        version: Model version
        upload_to_gcs: Whether to upload artifacts to GCS
        run_evaluation: Whether to run evaluation after training
        **kwargs: Additional training parameters
        
    Returns:
        Dictionary with training and evaluation results
    """
    logger.info("="*60)
    logger.info("Starting Local Training")
    logger.info("="*60)
    logger.info(f"Training Data: {train_path}")
    logger.info(f"Test Data: {test_path}")
    logger.info(f"Output Directory: {output_dir}")
    logger.info(f"Upload to GCS: {upload_to_gcs}")
    logger.info("="*60)
    
    # Create custom training config if parameters provided
    training_config = TrainingConfig()
    if kwargs:
        for key, value in kwargs.items():
            if hasattr(training_config, key):
                setattr(training_config, key, value)
    
    # Initialize trainer
    trainer = XGBoostTrainer(
        model_params=XGBoostParams(),
        training_config=training_config
    )
    
    # Run training pipeline
    logger.info("\n📊 Starting training pipeline...")
    training_results = trainer.run_training_pipeline(
        train_path=train_path,
        test_path=test_path,
        output_dir=Path(output_dir),
        version=version,
        upload_to_gcs=upload_to_gcs
    )
    
    logger.info("\n✓ Training completed successfully")
    
    # Print training results
    print("\n" + "="*60)
    print("Training Results")
    print("="*60)
    print(f"Job Name: {training_results['job_name']}")
    print(f"\nTraining Metrics:")
    for metric, value in training_results['training_metrics'].items():
        print(f"  {metric}: {value}")
    
    if training_results.get('cv_results'):
        cv_mean = training_results['cv_results']['cv_mean']
        cv_std = training_results['cv_results']['cv_std']
        print(f"\nCross-Validation Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    print("="*60)
    
    # Run evaluation if test data provided
    evaluation_results = None
    if run_evaluation and test_path:
        logger.info("\n📈 Starting model evaluation...")
        
        # Load test data
        from src.utils.gcs_utils import load_dataframe_from_gcs
        test_df = load_dataframe_from_gcs(test_path)
        
        # Prepare test features
        y_test = test_df['Survived']
        X_test = test_df.drop(columns=['Survived'])
        
        # Ensure feature consistency
        X_test = X_test[training_results['feature_names']]
        
        # Create evaluator
        evaluator = ModelEvaluator(model=trainer.model)
        
        # Run evaluation
        eval_output_dir = Path(output_dir).parent / "evaluation" / (version or "latest")
        evaluation_results = evaluator.evaluate(
            X_test, y_test,
            feature_names=training_results['feature_names'],
            output_dir=eval_output_dir
        )
        
        logger.info("\n✓ Evaluation completed successfully")
        
        # Print evaluation results
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)
        print(f"Test Samples: {evaluation_results['n_samples']}")
        print(f"Correct Predictions: {evaluation_results['n_correct']}")
        print(f"\nTest Metrics:")
        for metric, value in evaluation_results['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        print("="*60)
    
    return {
        "training": training_results,
        "evaluation": evaluation_results
    }


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost model locally"
    )
    
    # Data paths
    parser.add_argument(
        "--train-path",
        type=str,
        required=True,
        help="Path or GCS URI to training data"
    )
    parser.add_argument(
        "--test-path",
        type=str,
        help="Path or GCS URI to test data"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/models/local",
        help="Output directory for model artifacts"
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Model version identifier"
    )
    parser.add_argument(
        "--upload-to-gcs",
        action="store_true",
        help="Upload artifacts to GCS after training"
    )
    
    # Training settings
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Validation split ratio"
    )
    parser.add_argument(
        "--use-cv",
        action="store_true",
        default=True,
        help="Use cross-validation"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds"
    )
    
    # Evaluation
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation on test set"
    )
    
    args = parser.parse_args()
    
    # Display configuration
    config.display_config()
    
    # Run training
    results = train_local(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        version=args.version,
        upload_to_gcs=args.upload_to_gcs,
        run_evaluation=not args.skip_evaluation and args.test_path is not None,
        validation_split=args.validation_split,
        use_cross_validation=args.use_cv,
        cv_folds=args.cv_folds
    )
    
    # Save results summary
    import json
    summary_path = Path(args.output_dir) / (args.version or "latest") / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n✓ Training summary saved to: {summary_path}")
    
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Model artifacts saved to: {args.output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

