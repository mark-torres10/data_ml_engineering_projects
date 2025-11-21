"""
Tests for model training infrastructure structure.

These tests validate the basic structure and imports without requiring actual data.
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all model modules can be imported."""
    try:
        from src.models import XGBoostTrainer, ModelEvaluator
        from src.models.model_config import XGBoostParams, TrainingConfig, EvaluationConfig
        from src.models.model_utils import ModelArtifactManager
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import model modules: {e}")


def test_xgboost_params_creation():
    """Test XGBoostParams can be instantiated."""
    from src.models.model_config import XGBoostParams
    
    params = XGBoostParams()
    assert params.objective == "binary:logistic"
    assert params.max_depth == 5
    assert params.learning_rate == 0.1
    assert params.n_estimators == 100
    
    # Test to_dict
    params_dict = params.to_dict()
    assert isinstance(params_dict, dict)
    assert "max_depth" in params_dict
    assert "learning_rate" in params_dict


def test_training_config_creation():
    """Test TrainingConfig can be instantiated."""
    from src.models.model_config import TrainingConfig
    
    config = TrainingConfig()
    assert config.validation_split == 0.2
    assert config.use_cross_validation is True
    assert config.cv_folds == 5
    assert config.stratify is True
    
    # Test to_dict
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert "validation_split" in config_dict


def test_evaluation_config_creation():
    """Test EvaluationConfig can be instantiated."""
    from src.models.model_config import EvaluationConfig
    
    config = EvaluationConfig()
    assert config.calculate_accuracy is True
    assert config.calculate_auc_roc is True
    assert config.plot_roc_curve is True
    
    # Test to_dict
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)


def test_trainer_initialization():
    """Test XGBoostTrainer can be instantiated."""
    from src.models.trainer import XGBoostTrainer
    from src.models.model_config import XGBoostParams, TrainingConfig
    
    # Create with defaults
    trainer = XGBoostTrainer()
    assert trainer.model is None
    assert trainer.X_train is None
    assert trainer.job_name is not None
    
    # Create with custom params
    params = XGBoostParams(max_depth=7)
    config = TrainingConfig(validation_split=0.3)
    
    trainer = XGBoostTrainer(
        model_params=params,
        training_config=config,
        job_name="test_job"
    )
    assert trainer.model_params.max_depth == 7
    assert trainer.training_config.validation_split == 0.3
    assert trainer.job_name == "test_job"


def test_evaluator_initialization():
    """Test ModelEvaluator can be instantiated."""
    from src.models.evaluate import ModelEvaluator
    from src.models.model_config import EvaluationConfig
    
    # Create with defaults
    evaluator = ModelEvaluator()
    assert evaluator.model is None
    assert evaluator.metrics == {}
    
    # Create with custom config
    config = EvaluationConfig(plot_roc_curve=False)
    evaluator = ModelEvaluator(eval_config=config)
    assert evaluator.eval_config.plot_roc_curve is False


def test_artifact_manager_initialization():
    """Test ModelArtifactManager can be instantiated."""
    from src.models.model_utils import ModelArtifactManager
    
    manager = ModelArtifactManager()
    assert manager.bucket_name is not None
    assert manager.storage_client is not None


def test_version_dir_creation():
    """Test create_model_version_dir function."""
    from src.models.model_utils import create_model_version_dir
    import tempfile
    import shutil
    
    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Test with version
        version_dir = create_model_version_dir(temp_dir, version="v1")
        assert version_dir.exists()
        assert version_dir.name == "v1"
        
        # Test without version (uses timestamp)
        version_dir2 = create_model_version_dir(temp_dir)
        assert version_dir2.exists()
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)


def test_config_serialization():
    """Test configuration save/load."""
    from src.models.model_config import XGBoostParams, TrainingConfig
    import tempfile
    import json
    
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_file = Path(f.name)
    
    try:
        # Test XGBoostParams
        params = XGBoostParams(max_depth=7, learning_rate=0.05)
        params.save(temp_file)
        
        loaded_params = XGBoostParams.load(temp_file)
        assert loaded_params.max_depth == 7
        assert loaded_params.learning_rate == 0.05
        
        # Test TrainingConfig
        config = TrainingConfig(validation_split=0.25, cv_folds=10)
        config.save(temp_file)
        
        loaded_config = TrainingConfig.load(temp_file)
        assert loaded_config.validation_split == 0.25
        assert loaded_config.cv_folds == 10
        
    finally:
        # Cleanup
        temp_file.unlink()


def test_file_structure():
    """Test that expected files exist."""
    project_root = Path(__file__).parent.parent
    
    # Check model files
    assert (project_root / "src/models/__init__.py").exists()
    assert (project_root / "src/models/trainer.py").exists()
    assert (project_root / "src/models/evaluate.py").exists()
    assert (project_root / "src/models/model_config.py").exists()
    assert (project_root / "src/models/model_utils.py").exists()
    assert (project_root / "src/models/README.md").exists()
    
    # Check scripts
    assert (project_root / "scripts/04_train_model_local.py").exists()
    assert (project_root / "scripts/04_submit_training_job.py").exists()
    
    # Check Docker files
    assert (project_root / "deployment/docker/Dockerfile.training").exists()
    assert (project_root / "deployment/docker/build_training_image.sh").exists()
    
    # Check documentation
    assert (project_root / "STEP_5_IMPLEMENTATION_SUMMARY.md").exists()
    assert (project_root / "TRAINING_QUICKSTART.md").exists()


def test_config_module():
    """Test updated config module."""
    from src.config import config
    
    # Test new attributes exist
    assert hasattr(config, 'ARTIFACT_REGISTRY_REPO')
    assert hasattr(config, 'TRAINING_IMAGE_NAME')
    assert hasattr(config, 'TRAINING_IMAGE_URI')
    assert hasattr(config, 'TRAINING_MACHINE_TYPE')
    
    # Test property methods
    assert isinstance(config.TRAINING_IMAGE_URI, str)
    assert isinstance(config.PREDICTION_IMAGE_URI, str)
    assert "docker.pkg.dev" in config.TRAINING_IMAGE_URI


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

