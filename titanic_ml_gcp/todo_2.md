# Phase 2 Implementation Checklist (Local Focus)

**Project:** Titanic ML Pipeline - Interactive Retraining with MLflow & Optuna (Local)
**Time Estimate:** 2-3 days
**Cost Estimate:** N/A (Local execution)

---

## Step 1: Understanding MLOps Tools
- [ ] Review MLflow concepts (Experiments, Runs, Registry)
- [ ] Review Optuna concepts (Study, Trial, Objective, Sampler)
- [ ] Understand integration strategy (Optuna driving search, MLflow tracking results)

## Step 2: Set Up MLflow (Local)
### 2.1 Install and Configure
- [ ] Add `mlflow` to `requirements.txt` or `pyproject.toml`
- [ ] Install dependencies
- [ ] Create `mlflow/` directory in project root
- [ ] Set up local SQLite backend (default for local dev)
- [ ] Test `mlflow ui` command locally

### 2.2 Create MLflow Helper Module
- [ ] Create `src/utils/mlflow_utils.py`
- [ ] Implement `init_mlflow` (set tracking URI to local)
- [ ] Implement `start_run` context manager
- [ ] Implement logging functions (params, metrics, artifacts)
- [ ] Implement model registration (local registry)

### 2.3 MLflow Security (Local Best Practices)
- [ ] Pin MLflow version in dependencies
- [ ] Implement basic logging/audit trails via standard logging

---

## Step 3: Set Up Optuna (Local)
### 3.1 Install and Configure
- [ ] Add `optuna` and `optuna-integration` to dependencies
- [ ] Install dependencies
- [ ] Configure local SQLite storage for Optuna studies (`optuna.db`)

### 3.2 Define Search Space & Objective
- [ ] Define XGBoost hyperparameter search space (max_depth, learning_rate, etc.)
- [ ] Create `src/models/optuna_objective.py`
- [ ] Implement objective function (takes trial, returns metric)
- [ ] Integrate MLflow logging inside objective function (log trials as nested runs)

---

## Step 4: Enhance Streamlit UI for Feature Selection
### 4.1 Design Interface
- [ ] Add "Feature Selection" section to Streamlit sidebar
- [ ] Group features by category (Demographics, Travel, Family)
- [ ] Implement "Select All" / "Deselect All" / Presets

### 4.2 Implement Logic
- [ ] Store selected features in `st.session_state`
- [ ] Validate selection (min features, target variable handling)
- [ ] Update prediction logic to use dynamic feature set

---

## Step 5: Implement Model Retraining Workflow
### 5.1 Retraining Interface
- [ ] Add "Model Training" tab to Streamlit UI
- [ ] Add configuration for Optuna (num trials, timeout)
- [ ] Add "Train New Model" button

### 5.2 Retraining Pipeline Module
- [ ] Create `src/app/retraining.py`
- [ ] Implement data preparation with selected features
- [ ] Implement `run_retraining` function (handles Optuna loop)
- [ ] Ensure proper MLflow nesting (Parent run for experiment, Child runs for trials)

### 5.3 Progress Monitoring
- [ ] Implement progress bar in Streamlit
- [ ] Display best metric found so far
- [ ] Link to local MLflow UI for details

---

## Step 6: Optuna Hyperparameter Optimization
### 6.1 Run Optimization
- [ ] Execute optimization study locally
- [ ] Verify trials are saved to SQLite
- [ ] Verify results are logged to MLflow

### 6.2 Analysis
- [ ] Access best trial parameters
- [ ] Visualize optimization history (using Optuna's plotting or in MLflow)

---

## Step 7: Display Comprehensive Metrics
### 7.1 Calculate Metrics
- [ ] Compute Accuracy, Precision, Recall, F1, AUC-ROC
- [ ] Generate Confusion Matrix
- [ ] Calculate Feature Importance

### 7.2 Metrics Dashboard
- [ ] Display key metrics in Streamlit "Model Training" tab
- [ ] Show Confusion Matrix plot
- [ ] Show ROC Curve plot
- [ ] Show Feature Importance chart

---

## Step 8: MLflow Model Registry (Local)
### 8.1 Register Models
- [ ] Register best model from optimization to local MLflow Registry
- [ ] Use semantic versioning (tags or descriptions)
- [ ] Verify model shows up in MLflow UI "Models" tab

### 8.2 Model Loading
- [ ] Update inference code to load specific model versions from MLflow
- [ ] Test loading "Production" vs "Staging" models (using aliases/tags)

---

## Step 9: Store Models (Local Artifacts)
### 9.1 Local Storage
- [ ] Ensure MLflow stores artifacts in `./mlruns` (default) or specific local directory
- [ ] Verify model binaries (XGBoost), scaler, and metadata are saved

### 9.2 Version Control
- [ ] Commit code changes
- [ ] (Optional) Update `models/metadata.json` if tracking manually outside MLflow

---

## Step 10: Integration and Testing
### 10.1 End-to-End Test
- [ ] Run Streamlit app
- [ ] Select features -> Train -> Optimize
- [ ] Verify new model is created, registered, and usable for inference
- [ ] Check MLflow UI for complete trace

### 10.2 Documentation
- [ ] Update README with local MLOps instructions
- [ ] Document how to run MLflow and Streamlit locally


