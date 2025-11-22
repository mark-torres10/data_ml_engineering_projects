# Phase 1 Implementation Checklist

**Project:** Titanic ML Pipeline with Vertex AI and Feature Store  
**Time Estimate:** 2-3 days  
**Cost Estimate:** $5-10

---

## Step 1: GCP Project Setup and Authentication

### 1.1 Create GCP Project
- [x] Navigate to Google Cloud Console (https://console.cloud.google.com/)
- [x] Create new project named "titanic-ml-project"
- [x] Note down Project ID
- [x] Link billing account

### 1.2 Enable Required APIs
- [x] Enable Vertex AI API
- [x] Enable Cloud Storage API
- [x] Enable Artifact Registry API
- [x] Enable Cloud Build API
- [x] Enable Compute Engine API
- [x] Enable Cloud Resource Manager API
- [x] Enable Cloud Logging API
- [x] Enable Cloud Monitoring API

### 1.3 Set Up gcloud CLI
- [x] Download and install gcloud CLI
- [x] Run `gcloud init` and configure default project and region
- [x] Run `gcloud auth login`
- [x] Run `gcloud auth application-default login`
- [x] Set default project with `gcloud config set project YOUR_PROJECT_ID`
- [x] Verify with `gcloud config list`

### 1.4 Create Service Account
- [x] Navigate to "IAM & Admin" > "Service Accounts"
- [x] Create service account "titanic-ml-sa"
- [x] Grant role: Vertex AI User
- [x] Grant role: Storage Admin
- [x] Grant role: Storage Object Admin
- [x] Grant role: Logs Writer
- [x] Grant role: Monitoring Metric Writer
- [x] Create and download JSON key (using ADC instead - see SERVICE_ACCOUNT_SETUP.md)
- [x] Set GOOGLE_APPLICATION_CREDENTIALS environment variable (using .env file)
- [x] Add JSON key to .gitignore

### 1.5 Set Up Python Environment
- [x] Install uv package manager
- [x] Verify installation with `uv --version`
- [x] Create virtual environment with `uv venv --python 3.12`
- [x] Activate virtual environment
- [x] Create pyproject.toml or requirements.txt with dependencies
- [x] Install dependencies with `uv pip install -r requirements.txt`
- [x] Verify installations by importing key packages

---

## Step 2: Data Acquisition and Exploration

### 2.1 Load Titanic Dataset
- [x] Load dataset from HuggingFace using datasets library
- [x] Access train and test splits
- [x] Convert to pandas DataFrames

### 2.2 Exploratory Data Analysis (EDA)
- [x] Create `notebooks/01_eda.ipynb`
- [x] Display first few rows with head()
- [x] Check data types and null counts with info()
- [x] Get summary statistics with describe()
- [x] Check class balance of Survived column
- [x] Calculate percentage of missing values per column
- [x] Visualize missing data patterns with heatmap
- [x] Plot histograms for numerical features
- [x] Plot bar charts for categorical features
- [x] Analyze survival rate by each feature
- [x] Create correlation matrix for numerical features
- [x] Document key insights

### 2.3 Create GCS Bucket
- [x] Navigate to "Cloud Storage" > "Buckets"
- [x] Create bucket named `titanic-ml-data-YOUR_PROJECT_ID`
- [x] Select region matching default region
- [x] Choose Standard storage class
- [x] Set access control to "Uniform"
- [x] Create folder structure: `data/raw/`, `data/processed/`, `models/`, `predictions/`
- [x] Save DataFrames as CSV files
- [x] Upload train data to `gs://YOUR_BUCKET/data/raw/titanic_train.csv`
- [x] Upload test data to `gs://YOUR_BUCKET/data/raw/titanic_test.csv`
- [x] Test GCS access with Python google-cloud-storage library (gcloud CLI working)

---

## Step 3: Data Preprocessing and Feature Engineering

### 3.1 Handle Missing Values
- [x] Impute missing Age values (median/mean by Pclass and Sex)
- [x] Create `Age_Missing` binary indicator feature
- [x] Extract cabin deck letter from Cabin column
- [x] Create `Has_Cabin` binary feature
- [x] Drop original Cabin column
- [x] Impute missing Embarked values with mode
- [x] Impute missing Fare values with median by Pclass

### 3.2 Feature Engineering
- [x] Create `Family_Size` = SibSp + Parch + 1
- [x] Create `Is_Alone` binary feature
- [x] Extract titles from Name column (Mr., Mrs., Miss., Master., etc.)
- [x] Group rare titles into "Rare" category
- [x] Create fare bins (quartile-based)
- [x] Create age bins (Child, Young Adult, Adult, Senior)
- [x] Extract deck information from cabin

### 3.3 Encode Categorical Variables
- [x] Binary encoding for Sex (male=1, female=0)
- [x] One-hot encode Embarked (drop-first approach)
- [x] Ensure Pclass is integer type
- [x] One-hot encode Title
- [x] Label encode any remaining categorical features

### 3.4 Feature Scaling
- [x] Initialize StandardScaler from scikit-learn
- [x] Fit scaler on training data only
- [x] Transform both train and test sets
- [x] Save scaler object using joblib

### 3.5 Create Final Feature Set
- [x] Select features for modeling (numerical, binary, one-hot encoded)
- [x] Drop unnecessary columns (PassengerId, Name, Ticket, original Cabin)
- [x] Create X_train and y_train
- [x] Verify no remaining null values
- [x] Verify all features are numeric
- [x] Confirm shapes match expectations

### 3.6 Save Processed Data
- [x] Save processed training data as CSV/pickle
- [x] Save test data similarly
- [x] Save scaler and encoder objects with joblib
- [x] Upload processed data to `gs://YOUR_BUCKET/data/processed/`
- [x] Verify uploads completed successfully

---

## Step 4: Vertex AI Feature Store Setup

### 4.1 Understanding Feature Store
- [x] Review Feature Store concepts (documentation reading)

### 4.2 Create Feature Store
- [x] Navigate to Vertex AI > Feature Store (Used SDK)
- [x] Create Feature Store named `titanic_featurestore`
- [x] Select same region as bucket
- [x] Enable online serving
- [x] Enable offline serving
- [x] Wait for creation to complete (~10-15 minutes)

### 4.3 Create Entity Type
- [x] Create Entity Type named `passenger`
- [x] Add description: "Titanic passenger features"
- [x] Enable monitoring

### 4.4 Define and Register Features
- [x] Register Age feature (DOUBLE type)
- [x] Register Fare feature (DOUBLE type)
- [x] Register Family_Size feature (INT64 type)
- [x] Register Pclass feature (INT64 type)
- [x] Register Sex feature (INT64 type)
- [x] Register Embarked features (BOOL/INT64 type)
- [x] Register Is_Alone feature (BOOL/INT64 type)
- [x] Register Has_Cabin feature (BOOL/INT64 type)
- [x] Register Survived feature (INT64 type, optional)
- [x] Enable monitoring for all features

### 4.5 Ingest Feature Data
- [x] Prepare data with timestamp and entity ID (scripts/03_prepare_feature_store_data.py)
- [x] Format as CSV with PassengerId as entity_id
- [x] Navigate to Entity Type > Import Feature Values (Used SDK)
- [x] Choose "Batch import from Cloud Storage"
- [x] Specify source GCS path
- [x] Map columns to features
- [x] Start import job
- [x] Monitor ingestion progress (~15-30 minutes)
- [x] Verify ingestion completion
- [x] Check feature statistics

### 4.6 Test Feature Serving
- [x] Test online serving with Vertex AI SDK (scripts/test_feature_serving.py)
- [x] Request features for sample passenger
- [x] Verify returned values match expected features
- [x] Check latency (<50ms)
- [ ] Test offline serving with batch export
- [ ] Verify exported data completeness

---

## Step 5: Model Training with XGBoost

### 5.1 Choose Training Approach
- [x] Decision made: Use Custom Training with XGBoost

### 5.2 Prepare Training Script
- [x] Create `src/models/trainer.py`
- [x] Implement data loading from GCS or Feature Store
- [x] Implement train/validation split (80/20)
- [x] Define XGBoost classifier parameters
- [x] Implement training with cross-validation
- [x] Implement evaluation on validation set
- [x] Implement model artifact saving to GCS
- [x] Add logging to Cloud Logging
- [x] Add W&B experiment tracking integration

### 5.3 Package Training Code
- [x] Create `deployment/docker/Dockerfile.training`
- [x] Choose base image (Vertex AI pre-built or python:3.12-slim)
- [x] Install dependencies using uv
- [x] Copy training script to container
- [x] Set entrypoint to run trainer.py
- [x] Add W&B environment variables and directories
- [x] Build Docker image locally
- [ ] Test container locally with sample data (will test on Vertex AI)
- [x] Create Artifact Registry repository
- [x] Authenticate Docker with Artifact Registry
- [x] Tag image with registry path
- [x] Push image to Artifact Registry

### 5.4 Create Vertex AI Training Job
- [ ] Navigate to Vertex AI > Training
- [ ] Create training job "titanic-xgboost-training-v1"
- [ ] Select region
- [ ] Choose custom container training method
- [ ] Specify container image URI
- [ ] Configure compute: n1-standard-4, no accelerator
- [ ] Specify input data GCS paths
- [ ] Set output location: `gs://YOUR_BUCKET/models/xgboost/v1/`
- [ ] Select service account (titanic-ml-sa)

### 5.5 Monitor Training
- [ ] Start training job
- [ ] Monitor logs in real-time
- [ ] Wait for job completion (~10-15 minutes total)
- [ ] Check for errors or warnings

### 5.6 Evaluate Model Performance
- [ ] Download model artifacts from GCS
- [ ] Load saved model locally
- [ ] Evaluate on test set
- [ ] Calculate accuracy, precision, recall, F1 score
- [ ] Calculate AUC-ROC
- [ ] Create confusion matrix
- [ ] Verify performance meets expectations (78-82% accuracy)
- [ ] Create visualizations (ROC curve, confusion matrix)
- [ ] Save evaluation results to JSON
- [ ] Upload results to GCS

### 5.7 Experiment Tracking with Weights & Biases
- [x] Add wandb to dependencies (already present in pyproject.toml)
- [x] Configure W&B environment variables in config.py
- [x] Initialize W&B run in trainer
- [x] Add WandbCallback to XGBoost training
- [x] Log training metrics (accuracy, AUC, loss curves)
- [x] Log cross-validation results
- [x] Log dataset information (size, class balance)
- [x] Create comprehensive W&B logging in evaluator
- [x] Log evaluation metrics (accuracy, precision, recall, F1, AUC)
- [x] Log confusion matrix (interactive W&B format)
- [x] Log ROC curve (interactive W&B format)
- [x] Log precision-recall curve (interactive W&B format)
- [x] Log feature importance (table and chart)
- [x] Log matplotlib figures as images
- [x] Log prediction distribution histogram
- [x] Update Docker configuration for W&B
- [x] Add W&B directories to Docker image
- [ ] Test W&B integration locally
- [ ] Verify W&B dashboard shows all metrics and visualizations
- [ ] Configure W&B API key in environment
- [ ] Test W&B in Docker container

---

## Step 6: Model Deployment to Vertex AI Endpoint

### 6.1 Register Model in Vertex AI
- [ ] Navigate to Vertex AI > Model Registry
- [ ] Import model named `titanic-xgboost`
- [ ] Set version to `v1`
- [ ] Add description
- [ ] Select region
- [ ] Specify container image URI (pre-built or custom)
- [ ] Set model artifact location
- [ ] Define prediction schema (input/output format)

### 6.2 Create Prediction Container (if custom)
- [ ] Create `src/models/predictor.py`
- [ ] Implement model loading
- [ ] Implement HTTP POST input handling
- [ ] Implement input preprocessing
- [ ] Implement prediction logic
- [ ] Implement JSON response formatting
- [ ] Create `deployment/docker/Dockerfile.prediction`
- [ ] Build prediction container
- [ ] Test container locally with sample requests
- [ ] Push container to Artifact Registry

### 6.3 Deploy Model to Endpoint
- [ ] Navigate to Vertex AI > Endpoints
- [ ] Create endpoint named `titanic-prediction-endpoint`
- [ ] Select region (match model region)
- [ ] Deploy model to endpoint
- [ ] Configure deployment: n1-standard-2, min_replicas=1, max_replicas=3
- [ ] Wait for deployment (~10-20 minutes)
- [ ] Verify endpoint status shows "Deployed"

### 6.4 Test the Endpoint
- [ ] Test with gcloud CLI predict command
- [ ] Test with Vertex AI SDK predict() method
- [ ] Test case: Female, 1st class passenger
- [ ] Test case: Male, 3rd class passenger
- [ ] Test case: Child under 10
- [ ] Test case: Adult male with large family
- [ ] Measure prediction latency (should be 50-200ms)
- [ ] Test with concurrent requests
- [ ] Verify autoscaling behavior

### 6.5 Set Up Monitoring
- [ ] Navigate to endpoint Monitoring tab
- [ ] Review automatic metrics collection
- [ ] Create Cloud Monitoring dashboard "Titanic ML Monitoring"
- [ ] Add charts for request count, latency, error rate, CPU/memory
- [ ] Create alert policy for high error rate (>5%)
- [ ] Create alert policy for high latency (>500ms)
- [ ] Configure notification channels

---

## Step 7: Build Streamlit UI for Inference

### 7.1 Design User Interface
- [ ] Define UI components needed
- [ ] Sketch layout (sidebar for inputs, main area for results)

### 7.2 Create Streamlit Application
- [ ] Create `src/app/streamlit_app.py`
- [ ] Implement header section with title and description
- [ ] Implement sidebar inputs: Age slider
- [ ] Implement sidebar inputs: Sex radio buttons
- [ ] Implement sidebar inputs: Passenger class dropdown
- [ ] Implement sidebar inputs: Embarkation port dropdown
- [ ] Implement sidebar inputs: Fare input
- [ ] Implement sidebar inputs: Family member sliders
- [ ] Implement sidebar inputs: Additional features (cabin, title)
- [ ] Implement main area: Prediction result box
- [ ] Implement main area: Confidence percentage display
- [ ] Implement main area: Color-coded results
- [ ] Implement footer with model info and links

### 7.3 Implement Endpoint Integration
- [ ] Create `src/app/inference.py` helper module
- [ ] Implement endpoint connection initialization
- [ ] Implement user input preprocessing function
- [ ] Implement prediction request function
- [ ] Implement response parsing
- [ ] Implement error handling for network issues
- [ ] Implement error handling for invalid inputs
- [ ] Add logging for debugging

### 7.4 Add Model Explanations with SHAP
- [ ] Create `src/models/explainer.py`
- [ ] Load trained XGBoost model
- [ ] Initialize SHAP TreeExplainer
- [ ] Implement SHAP value calculation for predictions
- [ ] Create horizontal bar chart visualization in Streamlit
- [ ] Display top 5 most impactful features
- [ ] Add color coding (positive=green, negative=red)
- [ ] Display feature values alongside contributions
- [ ] Add interpretation text

### 7.5 Implement Caching and Optimization
- [ ] Add @st.cache_data decorator for data loading
- [ ] Add @st.cache_resource for endpoint client
- [ ] Minimize preprocessing in prediction path
- [ ] Add loading spinner during prediction
- [ ] Add estimated wait time display

### 7.6 Add Data Validation
- [ ] Validate age is between 0-80
- [ ] Validate fare is non-negative
- [ ] Validate family counts are reasonable
- [ ] Add warnings for unusual inputs
- [ ] Highlight required fields
- [ ] Provide helpful error messages

### 7.7 Style and Polish
- [ ] Apply custom CSS with maritime theme
- [ ] Add ship emoji/icon 🚢
- [ ] Add status icons for survived/not survived
- [ ] Add progress bars for probability
- [ ] Add "Reset" button to clear inputs
- [ ] Add example passengers to try
- [ ] Add tooltips for each feature
- [ ] Add FAQ or help section

### 7.8 Test Locally
- [ ] Run `streamlit run src/app/streamlit_app.py`
- [ ] Test all input combinations
- [ ] Verify predictions match endpoint responses
- [ ] Check explanation visualizations
- [ ] Test with known Titanic passengers
- [ ] Test edge cases
- [ ] Test error handling (disconnect endpoint temporarily)
- [ ] Test on different screen sizes
- [ ] Measure time from input to prediction (<2 seconds)

---

## Step 8: Integrate Vertex AI Explainable AI Features

### 8.1 Understanding Vertex AI Explainable AI
- [ ] Review Vertex AI Explainable AI documentation

### 8.2 Configure Model for Explainability
- [ ] Re-register model with explanations enabled
- [ ] Choose explanation method: Sampled Shapley
- [ ] Specify baseline (median values of training features)
- [ ] Set number of paths: 50

### 8.3 Request Explanations from Endpoint
- [ ] Modify prediction requests to include explanations parameter
- [ ] Parse explanation response for attributions
- [ ] Display attributions in Streamlit UI
- [ ] Create visualization similar to SHAP

### 8.4 Compare Explanation Methods (Optional)
- [ ] Show SHAP values and Vertex AI attributions side-by-side
- [ ] Highlight differences and similarities
- [ ] Document findings

---

## Step 9: Testing and Validation

### 9.1 End-to-End Testing
- [ ] Test complete workflow from raw data to prediction
- [ ] Verify data integrity (no train/test leakage)
- [ ] Ensure training and serving features match
- [ ] Verify test set performance matches expectations
- [ ] Confirm prediction latency <500ms
- [ ] Verify explanations make logical sense

### 9.2 Test Known Cases
- [ ] Test with known survivors (first-class women)
- [ ] Test with known non-survivors (third-class men)
- [ ] Test edge cases (very young, elderly, high fare, large families)
- [ ] Verify predictions align with historical outcomes

### 9.3 Model Sanity Checks
- [ ] Verify Sex is most important feature
- [ ] Verify Pclass is second most important
- [ ] Check prediction distribution (~38% survived)
- [ ] Ensure no systematic bias
- [ ] Verify probability calibration

### 9.4 UI Testing
- [ ] Conduct usability testing with unfamiliar user
- [ ] Test in Chrome, Firefox, Safari
- [ ] Verify mobile responsiveness
- [ ] Test on different screen resolutions
- [ ] Test error handling scenarios

---

## Step 10: Documentation and Cleanup

### 10.1 Document Your Work
- [ ] Create README.md for Phase 1
- [ ] Add architecture diagram showing GCP components
- [ ] Write reproduction instructions
- [ ] List all GCP resources created
- [ ] Document cost breakdown
- [ ] Note known issues and limitations
- [ ] Document key decisions (why XGBoost, feature engineering, etc.)
- [ ] Create runbook for retraining, updating, scaling, troubleshooting

### 10.2 Cost Management
- [ ] Review costs in GCP Billing dashboard
- [ ] Identify most expensive resources
- [ ] Compare to initial estimate
- [ ] Set up budget alerts
- [ ] Document cost optimization strategies
- [ ] Delete/stop endpoint if pausing work

### 10.3 Prepare for Phase 2
- [ ] Ensure all Phase 1 components work end-to-end
- [ ] Clean and organize code
- [ ] Parameterize configurations
- [ ] Review production readiness recommendations
- [ ] Consider implementing Secret Manager for credentials
- [ ] Consider semantic versioning strategy for models
- [ ] Consider data versioning approach

---

## Success Criteria

- [ ] ✅ GCP project set up with all necessary APIs enabled
- [ ] ✅ Titanic dataset loaded, preprocessed, and uploaded to GCS
- [ ] ✅ Feature Store created and populated with features
- [ ] ✅ XGBoost model trained and achieves ≥78% accuracy
- [ ] ✅ Model deployed to Vertex AI endpoint
- [ ] ✅ Endpoint responds to prediction requests in <500ms
- [ ] ✅ Streamlit UI allows interactive predictions
- [ ] ✅ Model explanations (SHAP or Vertex AI) displayed
- [ ] ✅ All code organized and documented
- [ ] ✅ Understanding of how each GCP component works

---

**Phase 1 Complete! Ready for Phase 2: Interactive Retraining with MLOps Tools**

