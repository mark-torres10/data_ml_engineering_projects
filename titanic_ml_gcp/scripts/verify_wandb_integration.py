"""
Verification script for Weights & Biases integration.

This script verifies that:
1. W&B configuration is properly set up
2. All imports work correctly
3. Trainer and Evaluator can be instantiated with W&B support
4. W&B can be enabled/disabled via config
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def verify_config():
    """Verify W&B configuration is loaded."""
    print("=" * 60)
    print("Verifying W&B Configuration")
    print("=" * 60)
    
    try:
        from src.config import config
        
        print(f"✓ Config module imported successfully")
        print(f"  USE_WANDB: {config.USE_WANDB}")
        print(f"  WANDB_PROJECT: {config.WANDB_PROJECT}")
        print(f"  WANDB_ENTITY: {config.WANDB_ENTITY if config.WANDB_ENTITY else '(not set)'}")
        
        # Check if W&B is available
        try:
            import wandb
            print(f"✓ wandb package is installed (version: {wandb.__version__})")
        except ImportError:
            print(f"⚠ wandb package not installed (will be skipped at runtime)")
        
        return True
    except Exception as e:
        print(f"✗ Error verifying config: {e}")
        return False


def verify_trainer():
    """Verify Trainer with W&B integration."""
    print("\n" + "=" * 60)
    print("Verifying Trainer with W&B Integration")
    print("=" * 60)
    
    try:
        from src.models.trainer import XGBoostTrainer
        from src.models.model_config import XGBoostParams, TrainingConfig
        
        print(f"✓ Trainer module imported successfully")
        
        # Create trainer instance
        trainer = XGBoostTrainer(
            model_params=XGBoostParams(),
            training_config=TrainingConfig(),
            job_name="test_wandb_verification"
        )
        
        print(f"✓ Trainer instantiated successfully")
        print(f"  W&B enabled: {trainer.use_wandb}")
        print(f"  Job name: {trainer.job_name}")
        
        # Check if W&B methods are present
        assert hasattr(trainer, '_init_wandb_run'), "Missing _init_wandb_run method"
        assert hasattr(trainer, '_finish_wandb_run'), "Missing _finish_wandb_run method"
        print(f"✓ W&B methods present in Trainer")
        
        return True
    except Exception as e:
        print(f"✗ Error verifying trainer: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_evaluator():
    """Verify Evaluator with W&B logging."""
    print("\n" + "=" * 60)
    print("Verifying Evaluator with W&B Logging")
    print("=" * 60)
    
    try:
        from src.models.evaluate import ModelEvaluator
        from src.models.model_config import EvaluationConfig
        
        print(f"✓ Evaluator module imported successfully")
        
        # Create evaluator instance
        evaluator = ModelEvaluator(
            model=None,
            eval_config=EvaluationConfig()
        )
        
        print(f"✓ Evaluator instantiated successfully")
        
        # Check if W&B methods are present
        assert hasattr(evaluator, 'log_to_wandb'), "Missing log_to_wandb method"
        print(f"✓ W&B logging method present in Evaluator")
        
        return True
    except Exception as e:
        print(f"✗ Error verifying evaluator: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_docker_config():
    """Verify Docker configuration includes W&B settings."""
    print("\n" + "=" * 60)
    print("Verifying Docker Configuration")
    print("=" * 60)
    
    try:
        dockerfile_path = project_root / "deployment" / "docker" / "Dockerfile.training"
        
        if not dockerfile_path.exists():
            print(f"⚠ Dockerfile not found at {dockerfile_path}")
            return False
        
        with open(dockerfile_path, 'r') as f:
            content = f.read()
        
        checks = {
            "USE_WANDB": "USE_WANDB" in content,
            "WANDB_PROJECT": "WANDB_PROJECT" in content,
            "WANDB_API_KEY": "WANDB_API_KEY" in content,
            "WANDB_DIR": "WANDB_DIR" in content,
        }
        
        all_passed = all(checks.values())
        
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"{status} {check} configuration present")
        
        return all_passed
    except Exception as e:
        print(f"✗ Error verifying Docker config: {e}")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("Weights & Biases Integration Verification")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run all checks
    results.append(("Config", verify_config()))
    results.append(("Trainer", verify_trainer()))
    results.append(("Evaluator", verify_evaluator()))
    results.append(("Docker", verify_docker_config()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All verifications passed!")
        print("\nNext steps:")
        print("1. Set WANDB_API_KEY environment variable")
        print("2. Test training with W&B enabled")
        print("3. Check W&B dashboard for logged metrics")
        return 0
    else:
        print("\n✗ Some verifications failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

