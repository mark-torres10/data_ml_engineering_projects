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

**Preprocessing Logic:**
- **Imputation:**
    - Age: Median by (Pclass, Sex)
    - Fare: Median by Pclass
    - Embarked: Mode
- **Feature Engineering:**
    - `Family_Size` = SibSp + Parch + 1
    - `Is_Alone` = 1 if Family_Size == 1
    - `Has_Cabin` = 1 if Cabin is not null
    - `Title`: Extracted from Name, grouped rare titles
- **Encoding:**
    - Sex: Binary (male=1, female=0)
    - Embarked, Title: One-Hot Encoding
- **Scaling:**
    - StandardScaler applied to all features

---

### 📝 Next Steps

#### Step 4: Vertex AI Feature Store Setup
- [ ] 4.1: Understanding Feature Store Concepts
- [ ] 4.2: Create a Feature Store
- [ ] 4.3: Create an Entity Type
- [ ] 4.4: Define and Register Features
- [ ] 4.5: Ingest Feature Data
- [ ] 4.6: Test Feature Serving

---

## Technical Decisions & Notes

### Architecture Patterns
- **Modular Preprocessing:** The `TitanicPreprocessor` class encapsulates all transformation logic. It follows the sklearn Transformer API (`fit`, `transform`), making it easy to integrate into pipelines or save/load as an artifact (`joblib`).
- **Testing:** Added unit tests (`pytest`) to ensure preprocessing logic (imputation, engineering, encoding) works correctly before moving to modeling.
- **Artifact Management:** Scalers and preprocessors are saved as artifacts to ensure training-serving skew is minimized (same logic applied at inference time).

---

**Last Updated:** 2025-11-21
**Current Phase:** Phase 1 - Step 3 (Complete)
**Next Milestone:** Vertex AI Feature Store Setup (Step 4)
