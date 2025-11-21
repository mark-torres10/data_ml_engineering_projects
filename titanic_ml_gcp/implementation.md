# Titanic ML GCP Project - Implementation Plan

## Project Overview

This project implements an end-to-end machine learning solution on Google Cloud Platform (GCP) using the Titanic dataset. The goal is to predict passenger survival using a classification model, with a focus on learning GCP services, MLOps practices, and Kubernetes orchestration.

**Target Audience:** Developers familiar with AWS but new to GCP, AutoML, Optuna, and Kubernetes.

**Dataset:** Titanic passenger data from HuggingFace ([paulopontesm/titanic](https://huggingface.co/datasets/paulopontesm/titanic))

**Primary ML Model:** XGBoost classifier

---

## Project Phases

This project is divided into three distinct phases, each building upon the previous one:

### Phase 1: Core ML Pipeline with Vertex AI and Feature Store
**File:** [implementation_1.md](implementation_1.md)

Build the foundational ML infrastructure:
- Set up GCP project and enable necessary services
- Ingest and preprocess Titanic dataset from HuggingFace
- Register features with Vertex AI Feature Store
- Train XGBoost classification model
- Deploy model as online inference endpoint
- Create Streamlit UI for predictions with model explanations (SHAP, feature importance)

**Key Learning Objectives:**
- GCP project setup and IAM (vs AWS IAM)
- Vertex AI ecosystem (vs AWS SageMaker)
- Feature Store concepts (vs AWS Feature Store)
- Model deployment and serving (vs SageMaker endpoints)
- Explainable AI integration

**Estimated Time:** 2-3 days

---

### Phase 2: Interactive Retraining with MLOps Tools
**File:** [implementation_2.md](implementation_2.md)

Add model experimentation and retraining capabilities:
- Enhance Streamlit UI for feature selection
- Implement model retraining workflow
- Integrate Optuna for hyperparameter optimization
- Register experiments and artifacts with MLflow
- Store models in Google Cloud Storage (GCS)
- Display comprehensive metrics (AUC/ROC, F1, precision, recall, latency)

**Key Learning Objectives:**
- Optuna hyperparameter tuning framework
- MLflow experiment tracking and model registry
- GCS integration (vs AWS S3)
- Interactive ML workflows
- Model performance monitoring

**Estimated Time:** 2-3 days

---

### Phase 3: Kubernetes Deployment with GKE Autopilot
**File:** [implementation_3.md](implementation_3.md)

Scale and orchestrate the application:
- Containerize Streamlit UI and inference service
- Deploy to Google Kubernetes Engine (GKE) Autopilot
- Configure service mesh and networking
- Implement auto-scaling policies
- Set up monitoring and logging

**Key Learning Objectives:**
- Kubernetes fundamentals
- GKE Autopilot (managed Kubernetes)
- Container orchestration
- Service discovery and load balancing
- Kubernetes scaling strategies

**Estimated Time:** 2-3 days

---

## Technology Choices

### Why Python 3.12?
- Latest stable Python version with performance improvements
- Enhanced error messages for better debugging
- Modern type hinting features
- Better async/await support
- Industry standard for new projects in 2024-2025

### Why uv instead of pip?
- **10-100x faster** than pip for package installation and resolution
- Written in Rust for maximum performance
- Drop-in replacement for pip (same commands, same workflow)
- Built-in virtual environment management
- Better dependency resolution and conflict detection
- Gaining rapid adoption in the Python community
- Fully compatible with existing requirements.txt and pip workflows

**Installation**: `curl -LsSf https://astral.sh/uv/install.sh | sh`

**Usage**: Replace `pip install` with `uv pip install` - that's it!

## Prerequisites

### Required Accounts and Tools
1. **Google Cloud Platform Account**
   - Free tier available: $300 credit for 90 days
   - Credit card required for verification
   - Project billing must be enabled

2. **Local Development Environment**
   - Python 3.12
   - `uv` package manager ([installation guide](https://github.com/astral-sh/uv))
   - Docker Desktop installed
   - Git installed
   - `gcloud` CLI installed ([installation guide](https://cloud.google.com/sdk/docs/install))
   - `kubectl` CLI installed ([installation guide](https://kubernetes.io/docs/tasks/tools/))

3. **Python Packages** (will be detailed in each phase)
   - `google-cloud-aiplatform`
   - `google-cloud-storage`
   - `pandas`, `numpy`, `scikit-learn`
   - `xgboost`
   - `streamlit`
   - `datasets` (HuggingFace)
   - `mlflow`
   - `optuna`
   - `shap`

### GCP Services Used (and AWS Equivalents)

| GCP Service | Purpose | AWS Equivalent |
|------------|---------|----------------|
| Vertex AI | ML platform | SageMaker |
| Vertex AI Feature Store | Feature management | SageMaker Feature Store |
| Cloud Storage (GCS) | Object storage | S3 |
| Cloud Build | CI/CD | CodeBuild |
| Artifact Registry | Container registry | ECR |
| GKE Autopilot | Managed Kubernetes | EKS |
| Cloud Run | Serverless containers | Fargate |
| Cloud IAM | Identity & access | AWS IAM |
| Cloud Logging | Centralized logging | CloudWatch Logs |
| Cloud Monitoring | Metrics & alerting | CloudWatch |

---

## Cost Estimates

**Phase 1:** ~$5-10 (mostly Vertex AI training and endpoint costs)
**Phase 2:** ~$3-5 (model training iterations)
**Phase 3:** ~$10-15 (GKE cluster running for testing)

**Total Estimated Cost:** $20-30 for complete implementation

**Cost Saving Tips:**
- Delete endpoints when not in use ($0.50-1/hour for running endpoints)
- Use preemptible/spot instances for training
- Delete GKE clusters when not actively testing
- Use Autopilot to avoid paying for unused node capacity
- Set up billing alerts in GCP Console

---

## Project Structure

```
titanic_ml_gcp/
├── implementation.md           # This file - overview and roadmap
├── implementation_1.md         # Phase 1: Core ML pipeline
├── implementation_2.md         # Phase 2: Retraining and MLOps
├── implementation_3.md         # Phase 3: Kubernetes deployment
├── data/
│   ├── raw/                    # Raw Titanic data from HuggingFace
│   ├── processed/              # Preprocessed features
│   └── feature_store/          # Feature Store exports
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # HuggingFace data loading
│   │   ├── preprocessor.py    # Data cleaning and feature engineering
│   │   └── feature_store.py   # Vertex AI Feature Store integration
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py         # XGBoost training logic
│   │   ├── predictor.py       # Prediction interface
│   │   └── explainer.py       # SHAP and feature importance
│   ├── app/
│   │   ├── __init__.py
│   │   ├── streamlit_app.py   # Main Streamlit UI
│   │   ├── inference.py       # Endpoint integration
│   │   └── retraining.py      # Retraining workflow
│   └── utils/
│       ├── __init__.py
│       ├── gcp_config.py      # GCP configuration
│       └── metrics.py         # Evaluation metrics
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile.streamlit
│   │   └── Dockerfile.inference
│   ├── kubernetes/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── ingress.yaml
│   └── terraform/             # (Optional) IaC for GCP resources
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
├── mlflow/
│   └── mlruns/                # MLflow tracking data
├── configs/
│   ├── model_config.yaml      # Model hyperparameters
│   ├── feature_config.yaml    # Feature definitions
│   └── deployment_config.yaml # Deployment settings
├── tests/
│   ├── test_preprocessor.py
│   ├── test_trainer.py
│   └── test_predictor.py
├── pyproject.toml             # Python project metadata and dependencies (modern)
├── requirements.txt           # Python dependencies (traditional, for compatibility)
├── Makefile                   # Common commands
├── .env.example               # Environment variables template
├── .gitignore
└── README.md
```

---

## Getting Started

1. **Clone or create the project directory**
   ```bash
   mkdir titanic_ml_gcp
   cd titanic_ml_gcp
   ```

2. **Set up Python environment**
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Create virtual environment with Python 3.12
   uv venv --python 3.12
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Option A: Using pyproject.toml (recommended for modern Python projects)
   uv pip install -e ".[dev]"
   
   # Option B: Using requirements.txt (traditional approach)
   uv pip install -r requirements.txt
   ```

4. **Install gcloud CLI and authenticate**
   ```bash
   gcloud init
   gcloud auth login
   gcloud auth application-default login
   ```

5. **Proceed to Phase 1**
   - Read [implementation_1.md](implementation_1.md) in detail
   - Follow each step sequentially
   - Test each component before moving to the next

---

## Key Differences: AWS vs GCP (Quick Reference)

### Authentication
- **AWS:** Access Keys (Access Key ID + Secret Access Key)
- **GCP:** Service Account Keys (JSON key files) or Application Default Credentials

### IAM
- **AWS:** Policies attached to users/roles
- **GCP:** IAM roles granted to principals (users/service accounts)

### Storage
- **AWS S3:** `s3://bucket-name/path`
- **GCP GCS:** `gs://bucket-name/path`

### CLI
- **AWS:** `aws s3 cp`, `aws sagemaker create-training-job`
- **GCP:** `gsutil cp`, `gcloud ai models create`

### Regions
- **AWS:** `us-east-1`, `eu-west-1`
- **GCP:** `us-central1`, `europe-west1`

### Container Registries
- **AWS ECR:** `<account-id>.dkr.ecr.<region>.amazonaws.com/<repo>`
- **GCP Artifact Registry:** `<region>-docker.pkg.dev/<project-id>/<repo>`

---

## Success Criteria

By the end of this project, you will have:

✅ **Phase 1:**
- Working Vertex AI Feature Store with Titanic features
- Trained XGBoost model deployed as an endpoint
- Streamlit UI that makes predictions with explanations
- Understanding of GCP ML infrastructure

✅ **Phase 2:**
- Interactive retraining interface
- Optuna-optimized hyperparameters
- MLflow tracking with models stored in GCS
- Comprehensive metrics dashboard

✅ **Phase 3:**
- Containerized application running on GKE
- Auto-scaling Kubernetes deployment
- Production-ready ML service
- End-to-end MLOps pipeline

---

## Next Steps

Start with **[implementation_1.md](implementation_1.md)** to begin Phase 1 of the project.

Each implementation file contains:
- Detailed step-by-step instructions
- Code examples and templates
- GCP-specific explanations for AWS users
- Troubleshooting tips
- Verification checkpoints
- Links to official documentation

Good luck! 🚀

