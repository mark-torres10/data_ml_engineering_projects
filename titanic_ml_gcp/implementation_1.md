# Phase 1: Core ML Pipeline with Vertex AI and Feature Store

## Overview

This phase establishes the foundational ML infrastructure on GCP. You'll learn to work with Vertex AI services, understand Feature Store concepts, deploy models to endpoints, and create an interactive UI for predictions with model explanations.

**Time Estimate:** 2-3 days

**Cost Estimate:** $5-10 (primarily Vertex AI training and endpoint hosting)

---

## Step 1: GCP Project Setup and Authentication

### 1.1 Create a New GCP Project

**Navigate to Google Cloud Console:**
- Go to https://console.cloud.google.com/
- Click the project dropdown in the top navigation bar
- Click "New Project"
- Enter a project name (e.g., "titanic-ml-project")
- Note your Project ID (auto-generated, will be needed later)
- Select a billing account (required for Vertex AI usage)

**AWS Comparison:** Similar to creating an AWS account and setting up a new AWS Organization or account, but GCP uses a simpler project-based hierarchy.

### 1.2 Enable Required APIs

**Navigate to APIs & Services:**
- In the Cloud Console, go to "APIs & Services" > "Library"
- Search for and enable each of the following APIs:
  - **Vertex AI API** - Core ML platform
  - **Cloud Storage API** - Object storage (like S3)
  - **Artifact Registry API** - Container registry (like ECR)
  - **Cloud Build API** - CI/CD builds
  - **Compute Engine API** - VM instances
  - **Cloud Resource Manager API** - Project management
  - **Cloud Logging API** - Centralized logging
  - **Cloud Monitoring API** - Metrics and alerting

**Why this matters:** Unlike AWS where services are enabled by default, GCP requires explicit API activation for cost control and security.

### 1.3 Set Up gcloud CLI

**Install gcloud CLI:**
- Download from https://cloud.google.com/sdk/docs/install
- Follow the installation wizard for your OS
- Restart your terminal after installation

**Initialize and authenticate:**
- Run `gcloud init` to configure your default project and region
- Select your newly created project
- Choose a default region (recommended: `us-central1` for lowest latency in US)
- Run `gcloud auth login` to authenticate with your Google account
- Run `gcloud auth application-default login` for application authentication

**Set default project:**
- Use `gcloud config set project YOUR_PROJECT_ID` to set your working project
- Verify with `gcloud config list`

**AWS Comparison:** This is equivalent to configuring AWS CLI with `aws configure`, but GCP uses OAuth for authentication instead of access keys.

### 1.4 Create a Service Account

**Purpose:** Service accounts are GCP's equivalent to AWS IAM roles - they provide credentials for applications to access GCP services.

**Create the service account:**
- Navigate to "IAM & Admin" > "Service Accounts"
- Click "Create Service Account"
- Name it "titanic-ml-sa"
- Description: "Service account for Titanic ML pipeline"
- Click "Create and Continue"

**Grant necessary roles:**
- Add the following roles (click "Add Another Role" for each):
  - **Vertex AI User** - For training and deploying models
  - **Storage Admin** - For creating and managing GCS buckets
  - **Storage Object Admin** - For reading/writing objects
  - **Logs Writer** - For writing logs
  - **Monitoring Metric Writer** - For writing metrics

**Create and download JSON key:**
- After creation, click on the service account
- Go to "Keys" tab
- Click "Add Key" > "Create new key" > "JSON"
- Save the JSON file securely (treat it like an AWS secret access key)
- Set environment variable: `export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"`

**Security Note:** Never commit this JSON key to version control. Add it to .gitignore immediately.

### 1.5 Set Up Python Environment

**Install uv package manager:**
- uv is a fast Python package installer and resolver (written in Rust)
- Install with: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- On Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
- Verify installation: `uv --version`

**Why uv over pip:**
- 10-100x faster than pip for package installation
- Better dependency resolution
- Built-in virtual environment management
- Compatible with pip and existing requirements.txt files
- Rapidly becoming the standard in modern Python projects

**Create a virtual environment:**
- Navigate to your project directory
- Create virtual environment with Python 3.12: `uv venv --python 3.12`
- This creates a `.venv` directory (standard name for uv)
- Activate it: `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\activate` (Windows)

**Create pyproject.toml (recommended) or requirements.txt:**
- uv works best with pyproject.toml but supports requirements.txt
- List all necessary packages with version constraints:
  - google-cloud-aiplatform>=1.35.0 (for Vertex AI)
  - google-cloud-storage>=2.10.0 (for GCS operations)
  - pandas>=2.0.0, numpy>=1.24.0 (data manipulation)
  - scikit-learn>=1.3.0 (preprocessing utilities)
  - xgboost>=2.0.0 (ML model)
  - datasets>=2.14.0 (HuggingFace datasets library)
  - streamlit>=1.27.0 (web UI)
  - shap>=0.42.0 (model explanations)
  - matplotlib>=3.7.0, seaborn>=0.12.0 (visualizations)
  - python-dotenv>=1.0.0 (environment management)

**Install dependencies:**
- Install all packages: `uv pip install -r requirements.txt`
- Or with pyproject.toml: `uv pip install -e .`
- Installation is significantly faster than pip (especially on first run)
- Verify installation by importing key packages in a Python shell

---

## Step 2: Data Acquisition and Exploration

### 2.1 Load Titanic Dataset from HuggingFace

**About the dataset:**
- HuggingFace hosts the classic Titanic dataset with train/test splits
- Primary dataset: `paulopontesm/titanic` or `Tomate/Kaggle-Titanic`
- Features include: PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked
- Target variable: Survived (0 = No, 1 = Yes)

**Load using the datasets library:**
- Import the load_dataset function from the datasets library
- Load the titanic dataset specifying the correct dataset identifier
- Access the train and test splits
- Convert to pandas DataFrames for easier manipulation

**Alternative if HuggingFace is unavailable:**
- Download from Kaggle Titanic competition
- Load CSV files directly with pandas

### 2.2 Exploratory Data Analysis (EDA)

**Create a Jupyter notebook for EDA:**
- Create `notebooks/01_eda.ipynb`
- Load the training data
- Perform the following analyses:

**Basic statistics:**
- Display first few rows using head()
- Check data types and null counts with info()
- Get summary statistics with describe()
- Check the class balance of the Survived column

**Missing data analysis:**
- Calculate percentage of missing values per column
- Notable issues in Titanic dataset:
  - Age: ~20% missing
  - Cabin: ~77% missing
  - Embarked: ~0.2% missing
- Visualize missing data patterns with a heatmap

**Feature distributions:**
- Plot histograms for numerical features (Age, Fare, SibSp, Parch)
- Plot bar charts for categorical features (Sex, Pclass, Embarked)
- Analyze survival rate by each feature

**Correlation analysis:**
- Create a correlation matrix for numerical features
- Identify which features correlate most with survival
- Note that Sex (when encoded) and Pclass have strong correlations

**Key insights to document:**
- Women had higher survival rates (~74%) than men (~19%)
- 1st class passengers had higher survival rates than 3rd class
- Age distribution shows many young children survived
- Fare correlates with Pclass
- Most passengers embarked from Southampton (S)

### 2.3 Create a GCS Bucket for Data Storage

**Why use GCS:** Centralized storage accessible by all GCP services, similar to S3 in AWS.

**Create the bucket:**
- Navigate to "Cloud Storage" > "Buckets" in GCP Console
- Click "Create Bucket"
- Name your bucket (must be globally unique): `titanic-ml-data-YOUR_PROJECT_ID`
- Choose location type: Region
- Select your default region (e.g., us-central1)
- Choose Standard storage class
- Set access control to "Uniform"
- Leave encryption as Google-managed
- Click "Create"

**Bucket structure to create:**
- Create folders: `data/raw/`, `data/processed/`, `models/`, `predictions/`
- Folders in GCS are logical constructs (prefix-based)

**Upload raw data:**
- Save your DataFrame as CSV
- Upload to `gs://YOUR_BUCKET/data/raw/titanic_train.csv`
- Upload test data to `gs://YOUR_BUCKET/data/raw/titanic_test.csv`

**Set up local GCS access:**
- Use the google-cloud-storage Python library
- Create a storage client with your credentials
- Test by listing bucket contents

**AWS Comparison:** GCS buckets are like S3 buckets. Path format is `gs://` instead of `s3://`. gsutil CLI is equivalent to aws s3 CLI.

---

## Step 3: Data Preprocessing and Feature Engineering

### 3.1 Handle Missing Values

**Strategy for Age:**
- Impute missing ages using median or mean
- Consider grouping by Pclass and Sex for more accurate imputation
- Alternative: Use machine learning imputation based on other features
- Create an `Age_Missing` binary indicator feature

**Strategy for Cabin:**
- Cabin has too many missing values for simple imputation
- Extract cabin deck letter (first character: A, B, C, etc.)
- Create a binary feature `Has_Cabin` (0 if missing, 1 if present)
- Drop the original Cabin column

**Strategy for Embarked:**
- Only 2 missing values
- Impute with mode (most common value = 'S')
- Alternatively, drop these rows

**Strategy for Fare:**
- Test set may have missing Fare values
- Impute with median fare for the corresponding Pclass

### 3.2 Feature Engineering

**Create new derived features:**

**Family size features:**
- Create `Family_Size` = SibSp + Parch + 1 (self)
- Create `Is_Alone` = 1 if Family_Size == 1, else 0
- Research shows both very small and very large families had lower survival

**Title extraction from Name:**
- Extract titles (Mr., Mrs., Miss., Master., Dr., Rev., etc.)
- Group rare titles into "Rare" category
- Convert to categorical feature
- Title correlates with age, sex, and social status

**Fare bins:**
- Create quartile-based fare bins (cheap, medium, expensive, very expensive)
- Helps model capture non-linear relationships

**Age bins:**
- Create age groups: Child (0-16), Young Adult (17-32), Adult (33-48), Senior (49+)
- Captures non-linear age effects on survival

**Deck information:**
- If Cabin present, extract deck letter
- Missing cabins get "Unknown" deck
- Deck correlates with Pclass and survival

### 3.3 Encode Categorical Variables

**Binary encoding for Sex:**
- Convert 'male' to 1, 'female' to 0
- Or use 0 and 1 arbitrarily but stay consistent

**One-hot encoding for Embarked:**
- Create three binary columns: Embarked_C, Embarked_Q, Embarked_S
- Drop one column to avoid multicollinearity (drop-first approach)

**Ordinal encoding for Pclass:**
- Pclass is already numeric (1, 2, 3)
- Ensure it's treated as integer not float

**One-hot encoding for Title:**
- Convert extracted titles to binary columns
- Keep only significant titles, group rare ones

**Label encoding for other categoricals:**
- Use label encoding for any remaining categorical features
- Document the encoding scheme for later use

### 3.4 Feature Scaling

**Normalize numerical features:**
- Use StandardScaler from scikit-learn
- Fit on training data only (avoid data leakage)
- Transform both train and test sets
- Features to scale: Age, Fare, Family_Size
- Save the scaler object for later use in production

**Why scaling matters:**
- XGBoost doesn't strictly require scaling, but it can improve convergence
- Essential if you later experiment with neural networks or SVM
- Helps with model interpretability

### 3.5 Create Final Feature Set

**Select features for modeling:**
- Numerical: Age, Fare, Family_Size, Pclass
- Binary: Sex, Is_Alone, Has_Cabin, Age_Missing
- One-hot encoded: Embarked, Title, optional Deck
- Drop: PassengerId, Name, Ticket, Cabin (already processed)

**Create X_train and y_train:**
- X_train contains all selected features
- y_train contains the Survived target column
- Ensure no data leakage from test set

**Verify data quality:**
- Check for any remaining null values
- Verify all features are numeric
- Confirm shapes match expectations
- Calculate basic statistics to ensure no anomalies

### 3.6 Save Processed Data

**Save to local files:**
- Save processed training data as CSV and/or pickle
- Save test data similarly
- Save the scaler and encoder objects using joblib

**Upload to GCS:**
- Upload processed data to `gs://YOUR_BUCKET/data/processed/`
- This makes data accessible to Vertex AI jobs
- Verify uploads completed successfully

---

## Step 4: Vertex AI Feature Store Setup

### 4.1 Understanding Feature Store Concepts

**What is a Feature Store:**
- Centralized repository for ML features
- Provides feature versioning, serving, and monitoring
- Enables feature reuse across projects
- Handles online (low-latency) and offline (batch) serving

**AWS Comparison:** Similar to AWS SageMaker Feature Store, but GCP's implementation is more tightly integrated with Vertex AI.

**Key concepts:**
- **Featurestore:** Top-level container (like a database)
- **Entity Type:** Represents a business entity (like "passenger")
- **Feature:** Individual data column (like "age" or "sex")
- **Feature Values:** Actual data points with timestamps

### 4.2 Create a Feature Store

**Navigate to Vertex AI Feature Store:**
- In GCP Console, go to Vertex AI > Feature Store
- Click "Create Feature Store"

**Configuration:**
- Name: `titanic_featurestore`
- Region: Use same region as your bucket (e.g., us-central1)
- Online serving: Enable (for low-latency predictions)
- Offline serving: Enable (for batch training)
- Encryption: Use Google-managed keys (default)
- Click "Create"

**Creation time:** Feature Store creation takes 10-15 minutes. You can proceed with other tasks while waiting.

### 4.3 Create an Entity Type

**What is an Entity Type:**
- Represents the entity being predicted (passengers in this case)
- Acts like a table in a traditional database
- Contains multiple features

**Create Entity Type:**
- In your Feature Store, click "Create Entity Type"
- Name: `passenger`
- Description: "Titanic passenger features"
- Monitoring: Enable to track feature distributions over time
- Click "Create"

### 4.4 Define and Register Features

**Create features for each column:**

For each feature in your processed dataset, register it with the Feature Store:

**Numerical features:**
- Age (DOUBLE type, description: "Passenger age in years")
- Fare (DOUBLE type, description: "Ticket fare paid")
- Family_Size (INT64 type, description: "Number of family members aboard")

**Categorical/Binary features:**
- Pclass (INT64 type, description: "Passenger class: 1, 2, or 3")
- Sex (INT64 type, description: "0=female, 1=male")
- Embarked_C, Embarked_Q, Embarked_S (BOOL type)
- Is_Alone (BOOL type, description: "Traveling alone")
- Has_Cabin (BOOL type, description: "Cabin information available")

**Target variable (optional):**
- Survived (INT64 type, description: "Survived: 0=No, 1=Yes")

**Feature creation process:**
- Click "Create Feature" in your Entity Type
- Enter name, type, and description for each
- Set value type appropriately (DOUBLE, INT64, BOOL, STRING)
- Enable monitoring for drift detection
- Repeat for all features

### 4.5 Ingest Feature Data

**Prepare data for ingestion:**
- Feature Store requires specific format with timestamp and entity ID
- Add a timestamp column (use current timestamp or original event time)
- Ensure PassengerId is the entity ID column
- Format as CSV or Avro

**Batch ingestion from GCS:**
- Navigate to your Entity Type
- Click "Import Feature Values"
- Choose "Batch import from Cloud Storage"
- Specify source GCS path: `gs://YOUR_BUCKET/data/processed/features.csv`
- Map columns to features (entity_id = PassengerId)
- Set worker count (start with 5)
- Start import job

**Monitor ingestion:**
- Ingestion can take 15-30 minutes
- Monitor progress in the Vertex AI > Feature Store > Import Jobs
- Check for errors in Cloud Logging if import fails

**Verify ingestion:**
- Once complete, navigate to your Entity Type
- Click on a feature to view its statistics
- Verify record count matches expectations
- Check feature distributions

### 4.6 Test Feature Serving

**Online serving test:**
- Use the Vertex AI SDK to fetch features for specific entity IDs
- Request features for a sample passenger (by PassengerId)
- Verify returned values match expected features
- Check latency (should be < 50ms)

**Offline serving test:**
- Create a batch export job
- Export features for multiple passengers to GCS
- Load exported data and verify completeness

**Why this matters:** Ensures the Feature Store is working correctly before building models that depend on it.

---

## Step 5: Model Training with XGBoost

### 5.1 Choose Training Approach

**Two options for Vertex AI training:**

**Option A: Custom Training (Recommended for learning)**
- Full control over training code
- Use any framework (XGBoost, scikit-learn, TensorFlow, PyTorch)
- Package code in a container
- More flexible for hyperparameter tuning

**Option B: AutoML Tables**
- Fully managed, no code required
- Automatically tries multiple algorithms
- Built-in hyperparameter tuning
- Higher cost, less control

**For this project:** Use Custom Training with XGBoost to learn the full workflow.

### 5.2 Prepare Training Script

**Create src/models/trainer.py:**

This script should:
- Load preprocessed data from GCS or Feature Store
- Split into train/validation sets (80/20 split)
- Define XGBoost classifier parameters
- Train the model with cross-validation
- Evaluate on validation set
- Save model artifacts to GCS
- Log metrics to Cloud Logging

**XGBoost parameters to configure:**
- objective: 'binary:logistic' (for classification)
- max_depth: Start with 5
- learning_rate: Start with 0.1
- n_estimators: Start with 100
- subsample: 0.8
- colsample_bytree: 0.8
- random_state: 42 (for reproducibility)

**Enable early stopping:**
- Monitor validation AUC or logloss
- Stop if no improvement for 10 rounds
- Prevents overfitting

### 5.3 Package Training Code

**Create a Docker container:**

Create `deployment/docker/Dockerfile.training`:
- Base image: Use a Vertex AI pre-built image or python:3.12-slim
- Install dependencies from requirements.txt (using uv for faster builds)
- Copy training script to /app/
- Set entrypoint to run trainer.py

**Build container locally:**
- Build Docker image with appropriate tag
- Test locally with sample data
- Ensure all dependencies are included

**Push to Artifact Registry:**
- Create a repository in Artifact Registry (like ECR in AWS)
- Authenticate Docker with Artifact Registry
- Tag image with registry path
- Push image to registry

### 5.4 Create Vertex AI Training Job

**Navigate to Vertex AI Training:**
- Go to Vertex AI > Training
- Click "Create Training Job"

**Configure job:**
- Display name: `titanic-xgboost-training-v1`
- Region: Your default region
- Training method: Custom container
- Container image: Path to your Artifact Registry image

**Compute resources:**
- Machine type: n1-standard-4 (4 vCPUs, 15GB RAM)
- Accelerator: None needed for small Titanic dataset
- Disk: 100GB boot disk

**Input data:**
- Specify GCS paths for training and validation data
- Pass as command-line arguments to your script

**Output location:**
- Set model output path: `gs://YOUR_BUCKET/models/xgboost/v1/`
- Vertex AI will save artifacts here

**Service account:**
- Use your titanic-ml-sa service account
- Ensures job has proper GCS permissions

### 5.5 Monitor Training

**Training execution:**
- Click "Start Training"
- Job will provision compute resources (takes 3-5 minutes)
- Training begins and logs appear in real-time

**View logs:**
- Click on the training job
- Go to "Logs" tab
- Monitor training progress, epochs, and metrics
- Check for errors or warnings

**Typical training time:**
- For Titanic dataset: 2-5 minutes of actual training
- Total job time including setup: 10-15 minutes

### 5.6 Evaluate Model Performance

**After training completes:**
- Download model artifacts from GCS
- Load the saved model locally
- Evaluate on test set (not validation set)

**Calculate key metrics:**
- Accuracy: Overall correct predictions
- Precision: Of predicted survivors, how many actually survived
- Recall: Of actual survivors, how many did we predict
- F1 Score: Harmonic mean of precision and recall
- AUC-ROC: Area under the ROC curve (most important for binary classification)
- Confusion Matrix: Visualize true/false positives/negatives

**Expected performance:**
- Good Titanic models achieve 78-82% accuracy
- AUC-ROC should be 0.85-0.90
- If much worse, check for data leakage or preprocessing errors

**Save evaluation results:**
- Save metrics to a JSON file
- Upload to GCS for tracking
- Create visualizations (ROC curve, confusion matrix)

---

## Step 6: Model Deployment to Vertex AI Endpoint

### 6.1 Register Model in Vertex AI

**Upload model to Model Registry:**
- Navigate to Vertex AI > Model Registry
- Click "Import Model"
- Model name: `titanic-xgboost`
- Version: `v1`
- Description: "XGBoost model for Titanic survival prediction"
- Region: Your default region

**Model source:**
- Import from: Container image
- Container image URI: Use a prediction container
  - Pre-built XGBoost container from Vertex AI
  - Or custom container with prediction logic
- Model artifact location: `gs://YOUR_BUCKET/models/xgboost/v1/`

**Model settings:**
- Prediction schema: Define input/output format
- Instance schema: JSON with feature names
- Prediction schema: JSON with 'survived' and 'probability'

### 6.2 Create Prediction Container (if using custom approach)

**Create src/models/predictor.py:**

This script should:
- Load the trained XGBoost model
- Accept input via HTTP POST (JSON format)
- Preprocess input features (apply same transformations as training)
- Make predictions
- Return results as JSON with prediction and probability

**Create deployment/docker/Dockerfile.prediction:**
- Base image: python:3.12-slim
- Install minimal dependencies (XGBoost, scikit-learn, Flask/FastAPI)
- Consider using uv in Docker for faster image builds
- Copy prediction script and model
- Expose port 8080 (Vertex AI requirement)
- Set entrypoint to run prediction server

**Build and test locally:**
- Build Docker image
- Run container locally
- Send test POST requests with sample passenger data
- Verify predictions are correct

**Push to Artifact Registry:**
- Tag with prediction-specific name
- Push to your Artifact Registry repository

### 6.3 Deploy Model to Endpoint

**Create an endpoint:**
- Navigate to Vertex AI > Endpoints
- Click "Create Endpoint"
- Name: `titanic-prediction-endpoint`
- Region: Your default region (must match model region)
- Click "Create"

**Deploy model to endpoint:**
- Select your newly created endpoint
- Click "Deploy Model"
- Select your `titanic-xgboost` model (v1)
- Choose deployment settings:
  - Model display name: `titanic-xgboost-v1`
  - Traffic split: 100% (first deployment)
  - Machine type: n1-standard-2 (2 vCPUs, 7.5GB RAM)
  - Min replicas: 1 (for consistent availability)
  - Max replicas: 3 (for autoscaling)
  - Accelerator: None

**Deployment time:**
- Takes 10-20 minutes to deploy
- Endpoint status will show "Deploying" then "Deployed"
- Can monitor progress in the Endpoints console

**Cost consideration:**
- Endpoints charge per node hour (~$0.50-1/hour)
- Even with 0 traffic, minimum replica incurs charges
- Delete endpoint when not actively using

### 6.4 Test the Endpoint

**Using gcloud CLI:**
- Use gcloud ai endpoints predict command
- Pass JSON input with feature values
- Verify response format and predictions

**Using Vertex AI SDK:**
- Create an endpoint client with your project and endpoint ID
- Prepare test instances (list of feature dictionaries)
- Call predict() method
- Parse and display results

**Test cases to try:**
- Female, 1st class passenger (expect high survival probability)
- Male, 3rd class passenger (expect low survival probability)
- Child under 10 years old (expect higher survival)
- Adult male with large family (expect lower survival)

**Latency testing:**
- Measure prediction latency (should be 50-200ms)
- Test with multiple concurrent requests
- Verify autoscaling works under load

### 6.5 Set Up Monitoring

**Enable endpoint monitoring:**
- Vertex AI automatically collects endpoint metrics
- Navigate to your endpoint and click "Monitoring" tab

**Key metrics to monitor:**
- Request count: Total API calls
- Request latency: Prediction response time (p50, p95, p99)
- Error rate: Failed predictions
- CPU/Memory utilization: Resource usage

**Create custom dashboard:**
- Go to Cloud Monitoring > Dashboards
- Create new dashboard: "Titanic ML Monitoring"
- Add charts for each key metric
- Set refresh interval to 1 minute

**Set up alerts:**
- Create alert policy for high error rate (>5%)
- Create alert policy for high latency (>500ms)
- Configure notification channels (email, Slack, PagerDuty)

---

## Step 7: Build Streamlit UI for Inference

### 7.1 Design the User Interface

**Purpose:**
- Allow users to input passenger characteristics
- Make real-time predictions via Vertex AI endpoint
- Display prediction results with confidence
- Show model explanations (feature importance)

**UI components needed:**
- Sidebar for input controls
- Main area for results display
- Visualization section for explanations
- Header with project description

### 7.2 Create Streamlit Application Structure

**Create src/app/streamlit_app.py:**

**Application layout:**

**Header section:**
- Title: "Titanic Survival Prediction"
- Brief description of the model and purpose
- Link to methodology or documentation

**Sidebar inputs (left panel):**
- Personal information section:
  - Name input (optional, for display)
  - Age slider (0-80 years)
  - Sex radio buttons (Male/Female)
- Travel details section:
  - Passenger class dropdown (1st, 2nd, 3rd)
  - Embarkation port dropdown (Southampton, Cherbourg, Queenstown)
  - Fare input (0-500)
- Family information section:
  - Number of siblings/spouses slider (0-8)
  - Number of parents/children slider (0-6)
- Additional features section:
  - Cabin checkbox (had cabin assignment?)
  - Title dropdown (Mr., Mrs., Miss., Master., etc.)

**Main area:**
- Large prediction result box
  - "Survived" or "Did Not Survive" with icon
  - Confidence percentage
  - Color-coded (green for survived, red for not survived)
- Prediction explanation section
  - Feature importance bar chart
  - Top 5 most influential features
  - Brief text explaining the prediction

**Footer:**
- Model version information
- Last updated date
- Links to code repository

### 7.3 Implement Endpoint Integration

**Create src/app/inference.py helper module:**

**Purpose:** Encapsulate all Vertex AI endpoint communication logic

**Key functions:**

**Initialize endpoint connection:**
- Load credentials from environment or service account key
- Store endpoint details (project ID, region, endpoint ID)
- Create reusable endpoint client

**Preprocess user inputs:**
- Convert UI inputs to model features
- Apply same transformations as training (encoding, scaling)
- Handle derived features (Family_Size, Is_Alone, etc.)
- Create feature dictionary matching model schema

**Make prediction request:**
- Format features as JSON instances
- Send request to Vertex AI endpoint
- Handle timeouts and errors gracefully
- Parse prediction response
- Extract survival prediction and probability

**Error handling:**
- Catch network errors (show user-friendly message)
- Handle invalid inputs (validate before sending)
- Log errors for debugging
- Provide fallback behavior if endpoint unavailable

### 7.4 Add Model Explanations with SHAP

**Purpose:** Help users understand why the model made its prediction

**SHAP (SHapley Additive exPlanations):**
- Industry-standard method for model interpretability
- Calculates each feature's contribution to prediction
- Works with any ML model including XGBoost

**Implementation approach:**

**Create src/models/explainer.py:**
- Load trained XGBoost model
- Initialize SHAP TreeExplainer (optimized for tree models)
- Compute SHAP values for given input

**Generate explanations:**
- For each prediction, calculate SHAP values
- SHAP values show how each feature pushed prediction higher or lower
- Positive SHAP = increased survival probability
- Negative SHAP = decreased survival probability

**Visualization in Streamlit:**
- Create horizontal bar chart of SHAP values
- Show top 5 most impactful features
- Use colors: positive contributions in green, negative in red
- Display actual feature values alongside contributions

**Example interpretation:**
- "Sex (Female): +0.25 - Being female strongly increased survival chance"
- "Pclass (3): -0.15 - Third class reduced survival chance"
- "Age (28): -0.05 - Age had minor negative impact"

**Alternative: Feature Importance:**
- If SHAP is too complex initially, start with model feature importance
- XGBoost provides built-in feature_importances_ attribute
- Shows global importance, not specific to current prediction
- Easier to implement but less informative

### 7.5 Implement Caching and Performance Optimization

**Streamlit caching:**
- Use @st.cache_data decorator for data loading
- Use @st.cache_resource for endpoint client initialization
- Prevents recreating connections on every UI interaction

**Optimize prediction latency:**
- Minimize preprocessing in prediction path
- Pre-compute any possible values
- Use batch prediction if supporting multiple inputs
- Keep endpoint warm with health checks

**User experience enhancements:**
- Show loading spinner during prediction
- Display estimated wait time
- Provide immediate feedback on user actions
- Disable predict button until all inputs are valid

### 7.6 Add Data Validation

**Input validation:**
- Check age is between 0-80
- Ensure fare is non-negative
- Validate family counts are reasonable
- Prevent invalid combinations

**User feedback:**
- Show warnings for unusual inputs
- Highlight required fields
- Provide helpful error messages
- Suggest corrections for invalid inputs

### 7.7 Style and Polish

**Apply custom CSS:**
- Use Streamlit's theming system
- Create custom color scheme (blue/white for maritime theme)
- Style prediction result box prominently
- Ensure mobile responsiveness

**Add visual elements:**
- Include ship emoji or icon 🚢
- Use status icons for survived/not survived
- Add progress bars for probability visualization
- Include charts and graphs for data exploration

**Improve user experience:**
- Add "Reset" button to clear all inputs
- Provide example passengers to try
- Include tooltips explaining each feature
- Add FAQ or help section

### 7.8 Test Locally

**Run Streamlit application:**
- Execute: `streamlit run src/app/streamlit_app.py`
- App opens in browser at localhost:8501
- Test all input combinations
- Verify predictions match endpoint responses
- Check explanation visualizations

**Test scenarios:**
- Test with known Titanic passengers (if available)
- Try edge cases (very old passenger, very high fare)
- Verify error handling (disconnect endpoint temporarily)
- Test on different screen sizes

**Performance testing:**
- Measure time from input to prediction
- Should be under 2 seconds total
- Most time is endpoint latency (100-300ms)
- UI should remain responsive

---

## Step 8: Integrate Vertex AI Explainable AI Features

### 8.1 Understanding Vertex AI Explainable AI

**What it provides:**
- Built-in model explanation capabilities
- Feature attributions for each prediction
- Global feature importance
- Multiple explanation methods (sampled Shapley, XRAI, integrated gradients)

**Comparison to SHAP:**
- Vertex AI Explainable AI is GCP's managed service
- SHAP is open-source and more flexible
- Both use Shapley value principles
- Vertex AI integrates better with GCP ecosystem

**When to use each:**
- Use Vertex AI Explainable AI if you want fully managed solution
- Use SHAP for more control and customization
- Can use both for comparison

### 8.2 Configure Model for Explainability

**Enable explanations during model upload:**
- When registering model in Model Registry, enable explanations
- Choose explanation method: Sampled Shapley (best for tabular data)
- Specify baseline: Median values of training features
- Set number of paths: 50 (balance accuracy vs speed)

**Explanation parameters:**
- Path count: Higher = more accurate but slower (50 is good default)
- Baseline: What to compare predictions against (use training data median/mean)
- Output indices: For multi-class, specify which class to explain

### 8.3 Request Explanations from Endpoint

**Modify prediction requests:**
- Add explanations parameter to predict() call
- Request feature attributions
- Receive predictions with attribution values

**Parse explanation response:**
- Response includes predictions and attributions
- Attributions show contribution of each feature
- Values similar to SHAP values

**Display in Streamlit UI:**
- Create explanation visualization similar to SHAP
- Show feature contributions as bar chart
- Highlight most important features
- Provide interpretation text

### 8.4 Compare Explanation Methods

**Side-by-side comparison (optional enhancement):**
- Show SHAP values on left
- Show Vertex AI attributions on right
- Highlight differences and similarities
- Helps understand model behavior better

**Benefits of having both:**
- Validation: Similar results increase confidence
- Debugging: Divergent results indicate issues
- Learning: Understand different explanation approaches

---

## Step 9: Testing and Validation

### 9.1 End-to-End Testing

**Test complete workflow:**
- Start with raw Titanic data
- Run preprocessing pipeline
- Verify features in Feature Store
- Train model successfully
- Deploy to endpoint
- Make predictions via API
- Use Streamlit UI for interactive predictions

**Validation checkpoints:**
- Data integrity: Check for data leaks between train/test
- Feature consistency: Ensure training and serving features match
- Prediction accuracy: Verify test set performance matches expectations
- Latency: Confirm predictions are fast enough (<500ms)
- Explanations: Check that attributions make logical sense

### 9.2 Test Known Cases

**Historical Titanic passengers:**
- Test with known survivors (e.g., first-class women)
- Test with known non-survivors (e.g., third-class men)
- Verify predictions align with historical outcomes

**Edge cases:**
- Very young children (should have higher survival)
- Elderly passengers
- Passengers with very high fares
- Large families vs solo travelers

### 9.3 Model Sanity Checks

**Feature importance validation:**
- Verify Sex is the most important feature (historically accurate)
- Pclass should be second most important
- Age and Fare should have moderate importance
- Ticket and Name should have minimal importance

**Prediction distribution:**
- Check that ~38% of test set is predicted as survived (matches training data)
- Ensure no systematic bias
- Verify probability calibration (predicted probabilities match actual frequencies)

### 9.4 UI Testing

**Usability testing:**
- Have someone unfamiliar with the project use the UI
- Note any confusion or difficulty
- Gather feedback on clarity and intuitiveness

**Browser compatibility:**
- Test in Chrome, Firefox, Safari
- Verify mobile responsiveness
- Check on different screen resolutions

**Error handling:**
- Intentionally cause errors (disconnect endpoint, invalid inputs)
- Verify graceful error messages
- Ensure app doesn't crash

---

## Step 10: Documentation and Cleanup

### 10.1 Document Your Work

**Create README.md for Phase 1:**
- Overview of what was built
- Architecture diagram showing GCP components
- Instructions to reproduce your work
- List of GCP resources created
- Cost breakdown
- Known issues and limitations

**Document key decisions:**
- Why XGBoost was chosen
- Feature engineering rationale
- Hyperparameter choices
- Deployment configuration decisions

**Create runbook:**
- How to retrain the model
- How to update the deployed model
- How to scale the endpoint
- Troubleshooting common issues

### 10.2 Cost Management

**Review costs incurred:**
- Check GCP Billing dashboard
- Identify most expensive resources
- Compare to estimate

**Optimize for cost:**
- Consider reducing min_replicas on endpoint to 0 (cold start trade-off)
- Delete resources not actively needed
- Set up budget alerts

**When pausing work:**
- Delete or stop the endpoint (major cost)
- Can keep Feature Store (minimal cost)
- Keep models in GCS (storage is cheap)
- Delete training jobs history if desired

### 10.3 Prepare for Phase 2

**What's next:**
- Phase 2 will add interactive retraining
- Optuna hyperparameter optimization
- MLflow experiment tracking
- Enhanced metrics dashboard

**Ensure Phase 1 is solid:**
- All components working end-to-end
- Code is clean and organized
- Configurations are parameterized
- Ready to extend with new features

---

## Troubleshooting Common Issues

### Issue: Feature Store ingestion fails

**Possible causes:**
- Incorrect schema mapping
- Missing timestamp column
- Invalid entity IDs
- Data type mismatches

**Solutions:**
- Check Cloud Logging for detailed error messages
- Verify CSV format matches expected schema
- Ensure entity_id column exists and is unique
- Check for null values in required fields

### Issue: Endpoint deployment fails

**Possible causes:**
- Container image issues
- Insufficient permissions
- Resource quota exceeded
- Health check failures

**Solutions:**
- Test container locally first
- Verify service account has required roles
- Request quota increase if needed
- Check container logs for startup errors

### Issue: Predictions are incorrect

**Possible causes:**
- Feature preprocessing mismatch
- Model version mismatch
- Data leakage in training
- Wrong model loaded

**Solutions:**
- Verify preprocessing in serving matches training
- Check model artifact version
- Review training code for data leaks
- Test model locally before deploying

### Issue: High latency

**Possible causes:**
- Cold starts (first request after idle)
- Insufficient resources
- Network latency
- Complex preprocessing

**Solutions:**
- Increase min_replicas to keep endpoint warm
- Use larger machine type
- Move preprocessing to training time
- Implement caching

### Issue: Streamlit app won't connect to endpoint

**Possible causes:**
- Authentication issues
- Wrong endpoint ID
- Network/firewall blocks
- Permissions missing

**Solutions:**
- Verify GOOGLE_APPLICATION_CREDENTIALS is set
- Double-check project ID and endpoint ID
- Test endpoint with gcloud CLI first
- Add required IAM roles to service account

---

## Success Criteria for Phase 1

You have successfully completed Phase 1 when:

✅ GCP project is set up with all necessary APIs enabled
✅ Titanic dataset is loaded, preprocessed, and uploaded to GCS
✅ Feature Store is created and populated with features
✅ XGBoost model is trained and achieves 78%+ accuracy
✅ Model is deployed to a Vertex AI endpoint
✅ Endpoint responds to prediction requests in <500ms
✅ Streamlit UI allows interactive predictions
✅ Model explanations (SHAP or Vertex AI) are displayed
✅ All code is organized and documented
✅ You understand how each GCP component works

---

## Next Steps

Proceed to [implementation_2.md](implementation_2.md) for Phase 2: Interactive Retraining with MLOps Tools.

Phase 2 will build on this foundation by adding:
- Feature selection UI
- Automated retraining workflows
- Optuna hyperparameter optimization
- MLflow experiment tracking
- Enhanced metrics visualization

