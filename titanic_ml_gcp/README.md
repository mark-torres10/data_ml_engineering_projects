# Titanic ML on GCP - Complete ML Engineering Project

A comprehensive end-to-end machine learning project demonstrating MLOps practices on Google Cloud Platform. This project predicts Titanic passenger survival using XGBoost, with a focus on production-ready deployment, experiment tracking, and Kubernetes orchestration.

## 🎯 Project Goals

- Learn GCP ML services (Vertex AI, Feature Store, GKE) for developers familiar with AWS
- Implement production-grade MLOps practices with MLflow and Optuna
- Deploy scalable ML applications using Kubernetes
- Build interactive UIs for model inference and retraining

## 📋 Table of Contents

- [Overview](#overview)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Implementation Phases](#implementation-phases)
- [Getting Started](#getting-started)
- [Cost Estimates](#cost-estimates)
- [Key Learning Outcomes](#key-learning-outcomes)

## 🔍 Overview

This project implements a complete ML pipeline for the classic Titanic dataset, predicting passenger survival based on features like age, sex, passenger class, and family information. The implementation is divided into three progressive phases:

1. **Core ML Pipeline**: Data ingestion, Feature Store, model training, and deployment
2. **MLOps Tools**: Interactive retraining, hyperparameter optimization, and experiment tracking
3. **Kubernetes Deployment**: Production-ready containerized deployment with auto-scaling

## 🛠 Technology Stack

### Machine Learning
- **Dataset**: Titanic from HuggingFace (`paulopontesm/titanic`)
- **Model**: XGBoost classifier
- **Explainability**: SHAP values and feature importance
- **Experiment Tracking**: MLflow
- **Hyperparameter Optimization**: Optuna

### Google Cloud Platform
- **Vertex AI**: ML platform (training and serving)
- **Vertex AI Feature Store**: Centralized feature management
- **Cloud Storage (GCS)**: Object storage for models and artifacts
- **Artifact Registry**: Container image registry
- **GKE Autopilot**: Managed Kubernetes
- **Cloud Monitoring & Logging**: Observability

### Application Stack
- **UI Framework**: Streamlit
- **Containerization**: Docker
- **Orchestration**: Kubernetes
- **Python**: 3.12
- **Package Manager**: uv (fast Rust-based pip alternative)

## 📁 Project Structure

```
titanic_ml_gcp/
├── implementation.md           # Main overview and roadmap
├── implementation_1.md         # Phase 1: Core ML pipeline
├── implementation_2.md         # Phase 2: MLOps and retraining
├── implementation_3.md         # Phase 3: Kubernetes deployment
├── README.md                   # This file
├── data/                       # Datasets (raw and processed)
├── notebooks/                  # Jupyter notebooks for EDA
├── src/                        # Source code
│   ├── data/                   # Data loading and preprocessing
│   ├── models/                 # Model training and inference
│   ├── app/                    # Streamlit application
│   └── utils/                  # Helper utilities
├── deployment/                 # Deployment configurations
│   ├── docker/                 # Dockerfiles
│   └── kubernetes/             # Kubernetes manifests
├── mlflow/                     # MLflow tracking data
├── configs/                    # Configuration files
└── tests/                      # Unit and integration tests
```

## 🚀 Implementation Phases

### Phase 1: Core ML Pipeline (2-3 days)

**Build the foundational ML infrastructure**

- Set up GCP project and authenticate
- Load and preprocess Titanic dataset from HuggingFace
- Register features with Vertex AI Feature Store
- Train XGBoost classifier with 78%+ accuracy
- Deploy model to Vertex AI endpoint
- Create Streamlit UI for predictions
- Integrate model explanations (SHAP)

**Key Deliverables:**
- Working Feature Store with Titanic features
- Deployed model endpoint with <500ms latency
- Interactive prediction UI with explanations

**📄 See [implementation_1.md](implementation_1.md) for detailed instructions**

### Phase 2: MLOps and Interactive Retraining (2-3 days)

**Add experiment tracking and optimization**

- Set up MLflow for experiment tracking
- Configure Optuna for hyperparameter optimization
- Enhance Streamlit UI for feature selection
- Implement automated retraining workflow
- Display comprehensive metrics (AUC, F1, precision, recall)
- Register models in MLflow Model Registry
- Store artifacts in GCS

**Key Deliverables:**
- Interactive retraining interface
- Optimized hyperparameters via Optuna (50-100 trials)
- Complete experiment history in MLflow
- Model versioning and registry

**📄 See [implementation_2.md](implementation_2.md) for detailed instructions**

### Phase 3: Kubernetes Deployment (2-3 days)

**Deploy to production-ready infrastructure**

- Containerize Streamlit application with Docker
- Push images to Artifact Registry
- Create GKE Autopilot cluster
- Write Kubernetes manifests (Deployments, Services, Ingress)
- Implement Horizontal Pod Autoscaling
- Configure monitoring and logging
- Set up alerting policies
- Implement security best practices

**Key Deliverables:**
- Containerized application in Artifact Registry
- Running GKE cluster with auto-scaling
- External access via Load Balancer
- Production monitoring and alerting

**📄 See [implementation_3.md](implementation_3.md) for detailed instructions**

## 🏁 Getting Started

### Prerequisites

**Accounts:**
- Google Cloud Platform account with billing enabled
- $300 free credit available for new users

**Local Tools:**
- Python 3.12
- uv package manager ([installation guide](https://github.com/astral-sh/uv))
- Docker Desktop
- gcloud CLI ([installation guide](https://cloud.google.com/sdk/docs/install))
- kubectl ([installation guide](https://kubernetes.io/docs/tasks/tools/))
- Git

**Knowledge:**
- Basic Python programming
- Familiarity with machine learning concepts
- Basic command-line usage
- (Optional) AWS experience for comparisons

### Installation Options

This project supports both modern (`pyproject.toml`) and traditional (`requirements.txt`) dependency management:

**Option A - Modern (recommended):**
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

**Option B - Traditional:**
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

Both options install the same dependencies. Choose based on your preference.

### Quick Start

1. **Read the overview**
   - Start with [implementation.md](implementation.md)
   - Understand the three-phase structure
   - Review prerequisites and cost estimates

2. **Begin Phase 1**
   - Follow [implementation_1.md](implementation_1.md) step-by-step
   - Set up GCP project
   - Build core ML pipeline
   - Complete all verification steps

3. **Progress through phases**
   - Only move to next phase after completing previous one
   - Test thoroughly at each step
   - Document any issues or deviations

4. **Iterate and learn**
   - Experiment with different features
   - Try different hyperparameters
   - Scale up/down as needed

## 💰 Cost Estimates

**Phase 1: Core ML Pipeline**
- Vertex AI training: $2-5
- Vertex AI endpoint: $2-5 (if kept running)
- GCS storage: <$1
- **Total: ~$5-10**

**Phase 2: MLOps and Retraining**
- Training iterations: $2-3
- MLflow storage in GCS: <$1
- **Total: ~$3-5**

**Phase 3: Kubernetes Deployment**
- GKE Autopilot cluster: $5-10
- Load Balancer: $2-3
- Monitoring: <$1
- **Total: ~$10-15**

**Complete Project Cost: $20-30**

### Cost-Saving Tips

- Delete Vertex AI endpoints when not in use ($0.50-1/hour)
- Delete GKE clusters during inactive periods
- Use GCP free tier credits for first 90 days
- Set up billing alerts to track spending
- Clean up resources after completing each phase

## 🎓 Key Learning Outcomes

### GCP Services (AWS Equivalents)

| Service | Purpose | AWS Equivalent |
|---------|---------|----------------|
| Vertex AI | ML platform | SageMaker |
| Vertex AI Feature Store | Feature management | SageMaker Feature Store |
| Cloud Storage (GCS) | Object storage | S3 |
| Artifact Registry | Container registry | ECR |
| GKE Autopilot | Managed Kubernetes | EKS |
| Cloud Monitoring | Metrics & alerting | CloudWatch |

### Technical Skills

**Machine Learning:**
- Feature engineering for tabular data
- XGBoost classification
- Model evaluation metrics (AUC, F1, precision, recall)
- Model explainability with SHAP
- Feature importance analysis

**MLOps:**
- Experiment tracking with MLflow
- Hyperparameter optimization with Optuna
- Model versioning and registry
- Automated retraining pipelines
- Model monitoring and drift detection

**Cloud & Infrastructure:**
- GCP authentication and IAM
- Vertex AI training and deployment
- Feature Store concepts
- Container building and registry
- Kubernetes fundamentals
- Auto-scaling strategies
- Production monitoring and alerting

**Software Engineering:**
- Docker containerization
- Kubernetes manifests (YAML)
- Streamlit web applications
- API integration
- Configuration management
- Logging and debugging

## 📚 Additional Resources

### GCP Documentation
- [Vertex AI Overview](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform)
- [Feature Store Guide](https://cloud.google.com/vertex-ai/docs/featurestore)
- [GKE Autopilot](https://cloud.google.com/kubernetes-engine/docs/concepts/autopilot-overview)

### Tool Documentation
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

### Tutorials and Guides
- [Kubernetes Basics](https://kubernetes.io/docs/tutorials/kubernetes-basics/)
- [Docker Get Started](https://docs.docker.com/get-started/)
- [SHAP Explainability](https://shap.readthedocs.io/)

## 🤝 Contributing

This is a learning project, but improvements are welcome! Consider:
- Adding more ML models for comparison
- Implementing additional features
- Enhancing the UI/UX
- Adding more comprehensive tests
- Improving documentation

## 📝 License

This project is for educational purposes. Feel free to use and modify for your own learning.

## 🙋 Support

If you encounter issues:
1. Check the troubleshooting sections in each implementation guide
2. Review GCP Cloud Logging for detailed error messages
3. Verify all prerequisites are installed correctly
4. Ensure GCP billing is enabled and quotas are sufficient

## ✅ Project Checklist

Track your progress through the project:

**Phase 1:**
- [ ] GCP project set up with APIs enabled
- [ ] Titanic dataset loaded and preprocessed
- [ ] Feature Store created and populated
- [ ] XGBoost model trained (78%+ accuracy)
- [ ] Model deployed to Vertex AI endpoint
- [ ] Streamlit UI functional with predictions
- [ ] SHAP explanations working

**Phase 2:**
- [ ] MLflow tracking configured
- [ ] Optuna optimization working
- [ ] Feature selection UI implemented
- [ ] Retraining workflow functional
- [ ] Metrics dashboard complete
- [ ] Models registered in MLflow
- [ ] Artifacts stored in GCS

**Phase 3:**
- [ ] Application containerized
- [ ] Images in Artifact Registry
- [ ] GKE Autopilot cluster created
- [ ] Application deployed to Kubernetes
- [ ] External access working
- [ ] Auto-scaling configured
- [ ] Monitoring and alerting set up
- [ ] Security best practices implemented

## 🎉 Conclusion

This project provides a comprehensive introduction to ML engineering on GCP, covering data processing, model training, deployment, and production operations. By completing all three phases, you'll have hands-on experience with modern MLOps practices and cloud-native application development.

**Happy learning and building!** 🚀

