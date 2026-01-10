# Models Module

This module provides production-ready infrastructure for training, evaluating, and deploying XGBoost models for Titanic survival prediction.

## Overview

The models module consists of:

- **`trainer.py`**: Main training orchestration with `XGBoostTrainer` class
- **`evaluate.py`**: Comprehensive model evaluation with `ModelEvaluator` class  
- **`model_config.py`**: Configuration dataclasses for hyperparameters and settings
- **`model_utils.py`**: Utility functions for model management and artifact handling

## Quick Start

### Local Training

Train a model locally for testing:

```bash
# Basic usage
python scripts/04_train_model_local.py \
    --train-path gs://your-bucket/data/processed/train.csv \
    --test-path gs://your-bucket/data/processed/test.csv \
    --output-dir outputs/models/local \
    --version v1

# With custom settings
python scripts/04_train_model_local.py \
    --train-path data/processed/train.csv \
    --test-path data/processed/test.csv \
    --validation-split 0.2 \
    --cv-folds 5 \
    --upload-to-gcs
```

### Vertex AI Training

Submit a training job to Vertex AI:

```bash
# First, build and push the training container
cd deployment/docker
./build_training_image.sh

# Then submit the training job
python scripts/04_submit_training_job.py \
    --train-path gs://your-bucket/data/processed/train.csv \
    --test-path gs://your-bucket/data/processed/test.csv \
    --version v1
```

## Module Components

### XGBoostTrainer

The main training class that handles the complete training workflow.

**Features:**
- Data loading from GCS or local files
- Train/validation splitting with stratification
- Cross-validation for performance estimation
- Model training with early stopping
- Artifact management and versioning
- Cloud Logging integration

**Example Usage:**

```python
from src.models.trainer import XGBoostTrainer
from src.models.model_config import XGBoostParams, TrainingConfig

# Configure training
model_params = XGBoostParams(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100
)

training_config = TrainingConfig(
    validation_split=0.2,
    use_cross_validation=True,
    cv_folds=5
)

# Initialize trainer
trainer = XGBoostTrainer(
    model_params=model_params,
    training_config=training_config
)

# Run training pipeline
results = trainer.run_training_pipeline(
    train_path="gs://bucket/data/train.csv",
    test_path="gs://bucket/data/test.csv",
    output_dir="outputs/models",
    version="v1",
    upload_to_gcs=True
)
```

### ModelEvaluator

Comprehensive evaluation class for classification models.

**Features:**
- Multiple classification metrics (accuracy, precision, recall, F1, AUC-ROC)
- Confusion matrix visualization
- ROC curve and precision-recall curve plotting
- Feature importance analysis
- Classification reports
- Automated visualization generation

**Example Usage:**

```python
from src.models.evaluate import ModelEvaluator
from pathlib import Path

# Initialize evaluator
evaluator = ModelEvaluator()

# Load trained model
evaluator.load_model(Path("outputs/models/v1"))

# Run evaluation
results = evaluator.evaluate(
    X=X_test,
    y=y_test,
    feature_names=feature_names,
    output_dir=Path("outputs/evaluation/v1")
)

# Access metrics
print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
print(f"AUC-ROC: {results['metrics']['auc_roc']:.4f}")
```

### Configuration Classes

#### XGBoostParams

Defines XGBoost model hyperparameters:

```python
from src.models.model_config import XGBoostParams

params = XGBoostParams(
    objective="binary:logistic",
    max_depth=5,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=10
)
```

#### TrainingConfig

Defines training process configuration:

```python
from src.models.model_config import TrainingConfig

config = TrainingConfig(
    validation_split=0.2,
    stratify=True,
    use_cross_validation=True,
    cv_folds=5,
    data_source="gcs",
    output_dir="outputs/models"
)
```

#### EvaluationConfig

Defines evaluation settings:

```python
from src.models.model_config import EvaluationConfig

eval_config = EvaluationConfig(
    calculate_accuracy=True,
    calculate_auc_roc=True,
    plot_roc_curve=True,
    plot_confusion_matrix=True,
    plot_feature_importance=True
)
```

### Model Utilities

Helper functions for model management:

```python
from src.models.model_utils import (
    ModelArtifactManager,
    save_training_artifacts,
    load_training_artifacts,
    get_feature_importance,
    validate_feature_consistency
)

# Artifact management
artifact_manager = ModelArtifactManager()

# Upload model to GCS
gcs_uri = artifact_manager.upload_to_gcs(
    local_path=Path("model.json"),
    gcs_path="models/v1/model.json"
)

# Get feature importance
importance_df = get_feature_importance(
    model=trained_model,
    feature_names=feature_names,
    importance_type="gain"
)
```

## Docker Training Container

### Building the Container

```bash
# Navigate to docker directory
cd deployment/docker

# Build and optionally push to Artifact Registry
./build_training_image.sh
```

### Container Structure

The training container (`Dockerfile.training`) includes:
- Python 3.12 slim base image
- UV package manager for fast dependency installation
- All project source code and dependencies
- Entrypoint configured to run trainer.py

### Running Container Locally

```bash
docker run \
    -v ~/.config/gcloud:/root/.config/gcloud \
    -e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json \
    xgboost-training:latest \
    --train-path gs://bucket/data/train.csv \
    --output-dir outputs/models
```

## Vertex AI Integration

### Submitting Training Jobs

Use the `04_submit_training_job.py` script:

```bash
python scripts/04_submit_training_job.py \
    --job-name "xgboost-training-v1" \
    --train-path gs://bucket/data/processed/train.csv \
    --test-path gs://bucket/data/processed/test.csv \
    --output-dir gs://bucket/models/v1 \
    --version v1 \
    --machine-type n1-standard-4
```

**Options:**
- `--machine-type`: Machine type (n1-standard-4, n1-standard-8, etc.)
- `--use-gpu`: Enable GPU acceleration (adds NVIDIA_TESLA_T4)
- `--no-wait`: Submit job without waiting for completion

### Monitoring Training

Monitor training jobs in the Vertex AI console:

```
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=YOUR_PROJECT_ID
```

Or use the SDK:

```python
from google.cloud import aiplatform

aiplatform.init(project="your-project-id", location="us-central1")

# List recent training jobs
jobs = aiplatform.CustomJob.list(
    filter='display_name:"xgboost-training"',
    order_by='create_time desc'
)

for job in jobs[:5]:
    print(f"{job.display_name}: {job.state}")
```

## Model Artifacts

### Directory Structure

```
outputs/models/
├── v1/
│   ├── model.json                  # XGBoost model file
│   ├── metadata.json               # Training metadata
│   ├── features.json               # Feature names and info
│   └── training_summary.json       # Complete training results
├── v2/
│   └── ...
└── latest/                         # Symlink or latest version
    └── ...
```

### Metadata Schema

Each model includes metadata:

```json
{
  "job_name": "xgboost_training_20250121_143022",
  "timestamp": "2025-01-21T14:30:22",
  "model_params": {
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_estimators": 100
  },
  "training_metrics": {
    "train_accuracy": 0.8567,
    "train_auc": 0.9234,
    "val_accuracy": 0.8234,
    "val_auc": 0.8976
  },
  "feature_names": ["Age", "Fare", "Pclass", ...],
  "n_features": 15
}
```

## Best Practices

### 1. Version Control

Always version your models:

```python
# Use semantic versioning
version = "v1.0.0"  # MAJOR.MINOR.PATCH

# Or timestamp-based
from datetime import datetime
version = datetime.now().strftime("%Y%m%d_%H%M%S")
```

### 2. Feature Consistency

Ensure training and serving features match:

```python
from src.models.model_utils import validate_feature_consistency

is_valid, mismatches = validate_feature_consistency(
    train_features=train_feature_names,
    test_features=test_feature_names
)

if not is_valid:
    raise ValueError(f"Feature mismatch: {mismatches}")
```

### 3. Cross-Validation

Always use cross-validation for reliable performance estimates:

```python
training_config = TrainingConfig(
    use_cross_validation=True,
    cv_folds=5,  # 5-fold is standard
    cv_scoring="roc_auc"  # Use AUC for imbalanced data
)
```

### 4. Early Stopping

Prevent overfitting with early stopping:

```python
model_params = XGBoostParams(
    early_stopping_rounds=10,
    eval_metric="auc"
)
```

### 5. Artifact Management

Save all training artifacts:

```python
# Local training
results = trainer.run_training_pipeline(
    ...,
    upload_to_gcs=True  # Upload to GCS for backup
)

# Include metadata
metadata = {
    "git_commit": "abc123",
    "data_version": "v2",
    "experiment_id": "exp-001"
}
```

## Troubleshooting

### Issue: Out of Memory during Training

**Solution:** Use a larger machine type:

```bash
python scripts/04_submit_training_job.py \
    --machine-type n1-highmem-8 \
    ...
```

### Issue: Training Too Slow

**Solutions:**
1. Use GPU acceleration: `--use-gpu`
2. Reduce dataset size for testing
3. Use tree_method="hist" (default, fastest)
4. Reduce n_estimators or max_depth

### Issue: Container Build Fails

**Solution:** Check Docker daemon and dependencies:

```bash
# Check Docker is running
docker info

# Clean build (no cache)
docker build --no-cache -f deployment/docker/Dockerfile.training .
```

### Issue: GCS Permission Denied

**Solution:** Check service account permissions:

```bash
# Verify authentication
gcloud auth list

# Check bucket access
gsutil ls gs://your-bucket/

# Grant required roles
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member=serviceAccount:SERVICE_ACCOUNT_EMAIL \
    --role=roles/storage.admin
```

## Testing

Run tests for the models module:

```bash
# Run all model tests
pytest tests/test_trainer.py tests/test_evaluate.py -v

# Run with coverage
pytest tests/test_*.py --cov=src/models --cov-report=html
```

## Performance Benchmarks

Typical performance on Titanic dataset:

| Metric | Expected Value |
|--------|---------------|
| Training Time (local) | 2-5 seconds |
| Training Time (Vertex AI) | 10-15 minutes (includes setup) |
| Accuracy | 78-82% |
| AUC-ROC | 0.85-0.90 |
| CV Score | 0.82-0.85 |

## Next Steps

After training a model:

1. **Evaluate Performance**: Use ModelEvaluator for comprehensive metrics
2. **Register Model**: Upload to Vertex AI Model Registry
3. **Deploy to Endpoint**: Create prediction endpoint (see deployment docs)
4. **Monitor Production**: Set up monitoring and alerting

## Additional Resources

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Vertex AI Custom Training](https://cloud.google.com/vertex-ai/docs/training/custom-training)
- [Model Evaluation Best Practices](https://developers.google.com/machine-learning/guides/model-evaluation)
- [Implementation Plan](../../implementation_1.md) - Complete Phase 1 guide

