# Titanic ML GCP - Progress Tracker

## Phase 1: Core ML Pipeline with Vertex AI and Feature Store

### ✅ Completed Steps

#### Step 1: GCP Project Setup and Authentication (COMPLETE)
- [x] Created GCP project `titanic-ml-gcp`
- [x] Enabled all 8 required APIs
- [x] Configured gcloud CLI
- [x] Created service account with appropriate IAM roles
- [x] Set up Python 3.12 virtual environment
- [x] Installed all dependencies via uv
- [x] Created configuration management system (.env + config.py)
- [x] Created GCP client utilities

**Key Files Created:**
- `src/config.py` - Configuration management
- `src/utils/gcp_client.py` - GCP authentication utilities
- `SERVICE_ACCOUNT_SETUP.md` - Authentication documentation
- `todo_1.md` - Phase 1 checklist
- `.env` - Environment variables

---

#### Step 2.1: Load Titanic Dataset (COMPLETE)
- [x] Created reusable `TitanicDataLoader` class
- [x] Loaded dataset from HuggingFace (direct CSV method)
- [x] Converted to pandas DataFrames
- [x] Saved data locally to `data/raw/`

**Key Files Created:**
- `src/data/loader.py` - Reusable TitanicDataLoader class
- `scripts/01_load_and_save_data.py` - Data loading pipeline script

---

#### Step 2.2: Exploratory Data Analysis (COMPLETE)
- [x] Created comprehensive EDA notebook `notebooks/01_eda.ipynb`
- [x] Analyzed missing data patterns
- [x] Visualized feature distributions
- [x] Identified key insights:
    - Sex is strongest predictor (74% vs 19% survival)
    - Pclass is 2nd strongest (1st >> 3rd)
    - Fare is positively correlated
    - Age: Children have higher survival

**Key Files Created:**
- `notebooks/01_eda.ipynb` - Executed notebook with visualizations

---

#### Step 2.3: Create GCS Bucket for Data Storage (COMPLETE)
- [x] Created bucket `titanic-ml-data-titanic-ml-gcp`
- [x] Set location: us-central1 (region)
- [x] Configured: Standard storage class, Uniform bucket-level access
- [x] Created folder structure: `data/raw/`, `data/processed/`, `models/`, `predictions/`
- [x] Uploaded training and test data to GCS

**Key Files Created:**
- `src/utils/gcs_utils.py` - GCS operations manager (comprehensive utilities)

---

#### Step 3: Data Preprocessing and Feature Engineering (COMPLETE)
- [x] 3.1: Handle Missing Values (Age, Cabin, Embarked, Fare)
- [x] 3.2: Feature Engineering (Title, Family_Size, Is_Alone, Bins)
- [x] 3.3: Encode Categorical Variables (Sex, Embarked, Title)
- [x] 3.4: Feature Scaling (StandardScaler)
- [x] 3.5: Create Final Feature Set
- [x] 3.6: Save Processed Data (Local + GCS)

**Key Files Created:**
- `src/features/preprocess.py` - Modular `TitanicPreprocessor` class
- `scripts/02_preprocess_data.py` - Preprocessing pipeline script
- `tests/test_preprocess.py` - Unit tests for preprocessing logic

---

#### Step 4: Vertex AI Feature Store Setup (COMPLETE)
- [x] 4.1: Review Concepts
- [x] 4.2: Create Feature Store (`titanic_featurestore`)
- [x] 4.3: Create Entity Type (`passenger`)
- [x] 4.4: Define and Register Features (15 features)
- [x] 4.5: Ingest Feature Data (From GCS CSV)
- [x] 4.6: Test Feature Serving (Verified connection)

**Key Files Created:**
- `scripts/03_prepare_feature_store_data.py` - Data prep for ingestion
- `scripts/03_setup_feature_store.py` - Infrastructure creation and ingestion
- `scripts/check_feature_store_status.py` - Status verification
- `scripts/test_feature_serving.py` - Online serving test

**Infrastructure:**
- **Feature Store:** `titanic_featurestore` (us-central1)
- **Entity Type:** `passenger`
- **Features:** 15 features registered (all lowercase IDs)

---

### 📝 Next Steps

#### Step 5: Model Training with XGBoost
- [ ] 5.1: Choose Training Approach
- [ ] 5.2: Prepare Training Script
- [ ] 5.3: Package Training Code
- [ ] 5.4: Create Vertex AI Training Job
- [ ] 5.5: Monitor Training
- [ ] 5.6: Evaluate Model Performance

---

## Technical Decisions & Notes

### Architecture Patterns
- **Feature Store IDs:** Feature Store 1.0 requires **lowercase** alphanumeric IDs. Our scripts enforce this mapping (e.g., CSV headers lowercased before ingestion).
- **Infrastructure as Code:** We used Python scripts (`scripts/03_setup_feature_store.py`) using the Vertex AI SDK to create resources idempotently (checking for existence before creating).
- **Ingestion:** Data is ingested via GCS batch import. The process is asynchronous; `scripts/test_feature_serving.py` can be used to verify when data becomes available.

---

**Last Updated:** 2025-11-21
**Current Phase:** Phase 1 - Step 4 (Complete)
**Next Milestone:** Model Training (Step 5)
