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

**Dataset Info:**
- Training samples: 891
- Test samples: 418
- Features: 12 (PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked)
- Survival rate: 38.38%
- Missing data: Age (19.9%), Cabin (77.1%), Embarked (0.2%)

**Key Files Created:**
- `src/data/__init__.py` - Data module initialization
- `src/data/loader.py` - Reusable TitanicDataLoader class
- `scripts/01_load_and_save_data.py` - Data loading pipeline script
- `data/raw/titanic_train.csv` - Training data (local)
- `data/raw/titanic_test.csv` - Test data (local)

**Features:**
- Fallback mechanisms for multiple dataset sources
- Support for HuggingFace datasets library and direct CSV loading
- Comprehensive metadata and statistics
- Convenience functions for quick operations
- Proper logging and error handling

---

#### Step 2.3: Create GCS Bucket for Data Storage (COMPLETE)
- [x] Created bucket `titanic-ml-data-titanic-ml-gcp`
- [x] Set location: us-central1 (region)
- [x] Configured: Standard storage class, Uniform bucket-level access
- [x] Created folder structure: `data/raw/`, `data/processed/`, `models/`, `predictions/`
- [x] Uploaded training and test data to GCS

**Bucket Structure:**
```
gs://titanic-ml-data-titanic-ml-gcp/
├── data/
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       ├── titanic_test.csv
│       └── titanic_train.csv
├── models/
│   └── .gitkeep
└── predictions/
    └── .gitkeep
```

**Key Files Created:**
- `src/utils/gcs_utils.py` - GCS operations manager (comprehensive utilities)

**GCS Manager Features:**
- Upload/download files
- List files with prefix filtering
- File existence checks
- Get file metadata
- Upload entire directories
- Generate signed URLs
- Convenience functions for common operations

---

### 🔄 Skipped (For Now)

#### Step 2.2: Exploratory Data Analysis (SKIPPED)
- Will be completed later as needed
- Dataset statistics already available via `TitanicDataLoader.display_info()`

---

### 📝 Next Steps

#### Step 3: Data Preprocessing and Feature Engineering
- [ ] 3.1: Handle Missing Values
- [ ] 3.2: Feature Engineering
- [ ] 3.3: Encode Categorical Variables
- [ ] 3.4: Feature Scaling
- [ ] 3.5: Create Final Feature Set
- [ ] 3.6: Save Processed Data

#### Step 4: Vertex AI Feature Store Setup
- [ ] 4.1: Understanding Feature Store Concepts
- [ ] 4.2: Create a Feature Store
- [ ] 4.3: Create an Entity Type
- [ ] 4.4: Define and Register Features
- [ ] 4.5: Ingest Feature Data
- [ ] 4.6: Test Feature Serving

---

## Technical Decisions & Notes

### Authentication Approach
- Using **Application Default Credentials (ADC)** for local development
- No service account JSON keys (per organizational policy)
- GCS operations work via `gcloud` CLI
- Python SDK credential refresh needed for some operations (non-blocking)

### Data Loading Strategy
- Direct CSV loading from HuggingFace URLs (primary method)
- Fallback to HuggingFace datasets library if needed
- Handles missing target column in test set gracefully

### Architecture Patterns
- Modular, reusable classes (`TitanicDataLoader`, `GCSManager`)
- Configuration centralized in `src/config.py`
- Comprehensive logging throughout
- Type hints and docstrings for all public APIs
- Error handling with graceful fallbacks

---

## Quick Reference Commands

### Load Data
```bash
python scripts/01_load_and_save_data.py
```

### Test Data Loader
```python
from src.data.loader import TitanicDataLoader
loader = TitanicDataLoader()
train_df, test_df = loader.load_data()
loader.display_info()
```

### GCS Operations (gcloud CLI)
```bash
# List bucket contents
gcloud storage ls -r gs://titanic-ml-data-titanic-ml-gcp/

# Upload file
gcloud storage cp local_file.csv gs://titanic-ml-data-titanic-ml-gcp/path/

# Download file
gcloud storage cp gs://titanic-ml-data-titanic-ml-gcp/path/file.csv ./
```

### Configuration
```python
from src.config import config
config.display_config()  # Show all settings
```

---

**Last Updated:** 2025-11-21  
**Current Phase:** Phase 1 - Step 2 (partially complete)  
**Next Milestone:** Complete data preprocessing (Step 3)

