# Training Quick Start Guide

Quick reference for training XGBoost models in the Titanic ML GCP project.

## Prerequisites

```bash
# Activate virtual environment
source .venv/bin/activate

# Ensure environment variables are set
source .env  # or ensure .env file exists

# Verify GCP authentication
gcloud auth application-default login
```

## Local Training (Development)

### Basic Training

```bash
python scripts/04_train_model_local.py \
    --train-path gs://YOUR_BUCKET/data/processed/train_processed.csv \
    --test-path gs://YOUR_BUCKET/data/processed/test_processed.csv \
    --version v1
```

### With Custom Parameters

```bash
python scripts/04_train_model_local.py \
    --train-path data/processed/train.csv \
    --test-path data/processed/test.csv \
    --output-dir outputs/models/experiment_1 \
    --version v1.0.0 \
    --validation-split 0.2 \
    --cv-folds 5 \
    --upload-to-gcs
```

## Vertex AI Training (Production)

### 1. Build and Push Container

```bash
cd deployment/docker
./build_training_image.sh
# Answer 'y' when prompted to push to Artifact Registry
```

### 2. Create Artifact Registry (First Time Only)

```bash
gcloud artifacts repositories create titanic-ml-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="Titanic ML containers"
```

### 3. Submit Training Job

```bash
python scripts/04_submit_training_job.py \
    --job-name "xgboost-training-v1" \
    --train-path gs://YOUR_BUCKET/data/processed/train_processed.csv \
    --test-path gs://YOUR_BUCKET/data/processed/test_processed.csv \
    --version v1 \
    --machine-type n1-standard-4
```

### 4. Monitor Training

```bash
# In console
https://console.cloud.google.com/vertex-ai/training/custom-jobs

# Or via CLI
gcloud ai custom-jobs list --region=us-central1

# View logs
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

## Common Commands

### Check Training Results

```bash
# List local models
ls -la outputs/models/

# Check GCS models
gsutil ls gs://YOUR_BUCKET/models/

# View training metrics
cat outputs/models/v1/metadata.json | jq '.training_metrics'
```

### View Evaluation Results

```bash
# View metrics
cat outputs/evaluation/v1/metrics.json

# Open plots
open outputs/evaluation/v1/confusion_matrix.png
open outputs/evaluation/v1/roc_curve.png
open outputs/evaluation/v1/feature_importance.png
```

### Download Model from GCS

```bash
# Download specific version
gsutil -m cp -r gs://YOUR_BUCKET/models/v1/ outputs/models/

# Download latest
gsutil -m cp -r gs://YOUR_BUCKET/models/latest/ outputs/models/
```

## Python API Usage

### Quick Training

```python
from src.models.trainer import XGBoostTrainer

trainer = XGBoostTrainer()
results = trainer.run_training_pipeline(
    train_path="gs://bucket/data/processed/train_processed.csv",
    version="v1"
)

print(f"Train AUC: {results['training_metrics']['train_auc']:.4f}")
```

### Custom Configuration

```python
from src.models.trainer import XGBoostTrainer
from src.models.model_config import XGBoostParams, TrainingConfig

# Configure model
model_params = XGBoostParams(
    max_depth=7,
    learning_rate=0.05,
    n_estimators=200,
    subsample=0.8
)

# Configure training
training_config = TrainingConfig(
    validation_split=0.25,
    use_cross_validation=True,
    cv_folds=10
)

# Train
trainer = XGBoostTrainer(model_params, training_config)
results = trainer.run_training_pipeline(
    train_path="gs://bucket/data/train.csv",
    upload_to_gcs=True
)
```

### Evaluation

```python
from src.models.evaluate import ModelEvaluator
from pathlib import Path

evaluator = ModelEvaluator()
evaluator.load_model(Path("outputs/models/v1"))

results = evaluator.evaluate(
    X_test, y_test,
    feature_names=feature_names,
    output_dir=Path("outputs/evaluation/v1")
)
```

## Troubleshooting

### "Module not found" Error
```bash
# Check you're in project root and venv is activated
pwd
which python  # Should show .venv/bin/python
```

### "Permission denied" Error
```bash
# Re-authenticate
gcloud auth application-default login

# Check service account permissions
gcloud projects get-iam-policy YOUR_PROJECT_ID
```

### "Container not found" Error
```bash
# Rebuild and push container
cd deployment/docker
./build_training_image.sh

# Verify image exists
gcloud artifacts docker images list \
    us-central1-docker.pkg.dev/YOUR_PROJECT_ID/titanic-ml-repo
```

### Training Takes Too Long
```bash
# Use larger machine
python scripts/04_submit_training_job.py \
    --machine-type n1-highmem-8 \
    ...

# Or enable GPU
python scripts/04_submit_training_job.py \
    --use-gpu \
    ...
```

## Configuration Files

### `.env` Template
```bash
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
GCS_BUCKET_NAME=titanic-ml-data-your-project-id
ARTIFACT_REGISTRY_REPO=titanic-ml-repo
SERVICE_ACCOUNT_EMAIL=titanic-ml-sa@your-project-id.iam.gserviceaccount.com
```

### Model Parameters (src/models/model_config.py)
```python
XGBoostParams(
    objective="binary:logistic",  # Classification
    max_depth=5,                   # Tree depth
    learning_rate=0.1,             # Step size
    n_estimators=100,              # Number of trees
    subsample=0.8,                 # Row sampling
    colsample_bytree=0.8,          # Column sampling
    early_stopping_rounds=10       # Stop if no improvement
)
```

## Expected Performance

| Metric | Good | Excellent |
|--------|------|-----------|
| Accuracy | 78-80% | 80-82% |
| AUC-ROC | 0.85-0.87 | 0.87-0.90 |
| Precision | 0.75-0.80 | 0.80-0.85 |
| Recall | 0.70-0.75 | 0.75-0.80 |

## Next Steps After Training

1. ✅ Review metrics and plots
2. ✅ Validate model performance
3. 📦 Register model in Vertex AI Model Registry (Step 6.1)
4. 🚀 Deploy to Vertex AI Endpoint (Step 6.2-6.3)
5. 🧪 Test endpoint predictions (Step 6.4)
6. 📊 Set up monitoring (Step 6.5)

## Useful Links

- [Full Implementation Guide](implementation_1.md)
- [Models README](src/models/README.md)
- [Step 5 Summary](STEP_5_IMPLEMENTATION_SUMMARY.md)
- [Vertex AI Console](https://console.cloud.google.com/vertex-ai)
- [GCS Browser](https://console.cloud.google.com/storage)

---

**Quick Help**: For detailed usage, see `src/models/README.md`

