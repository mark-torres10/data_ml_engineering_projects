# Phase 2: Interactive Retraining with MLOps Tools

## Overview

This phase enhances your ML pipeline with interactive retraining capabilities, hyperparameter optimization, and experiment tracking. You'll learn to use Optuna for automated hyperparameter search and MLflow for managing ML experiments and model artifacts.

**Prerequisites:** Complete Phase 1 successfully

**Time Estimate:** 2-3 days

**Cost Estimate:** $3-5 (mostly for retraining iterations)

---

## Step 1: Understanding MLOps Tools

### 1.1 What is MLflow?

**Purpose:**
- Track experiments: Log parameters, metrics, and artifacts
- Model registry: Version and manage models
- Model serving: Deploy models (though we'll use Vertex AI for this)
- Reproducibility: Ensure experiments can be recreated

**Core concepts:**
- **Experiment:** Logical grouping of related runs (e.g., "titanic-xgboost-experiments")
- **Run:** Single execution of your training code with specific parameters
- **Parameters:** Input values like learning_rate, max_depth
- **Metrics:** Output values like accuracy, AUC, F1
- **Artifacts:** Files like models, plots, datasets
- **Tags:** Metadata for organization and search

**AWS Comparison:** Similar to AWS SageMaker Experiments, but MLflow is open-source and cloud-agnostic.

### 1.2 What is Optuna?

**Purpose:**
- Automated hyperparameter optimization
- More efficient than grid search or random search
- Uses advanced algorithms (TPE, CMA-ES) to find optimal parameters
- Supports pruning to stop unpromising trials early

**Key concepts:**
- **Study:** Optimization session for a specific objective
- **Trial:** Single evaluation of a parameter combination
- **Objective function:** What to minimize/maximize (e.g., maximize AUC)
- **Search space:** Range of possible values for each hyperparameter
- **Sampler:** Algorithm for selecting next parameters to try (TPE is default)
- **Pruner:** Mechanism to stop bad trials early

**How it works:**
- Define search space (e.g., max_depth between 3-10)
- Define objective (e.g., maximize validation AUC)
- Optuna intelligently samples parameters
- Evaluates each combination
- Learns from results to suggest better parameters
- Typically finds good parameters in 50-100 trials

### 1.3 Integration Strategy

**How MLflow and Optuna work together:**
- Optuna drives the hyperparameter search
- Each Optuna trial is logged as an MLflow run
- MLflow tracks all parameters tested and results achieved
- Best model is registered in both Optuna and MLflow
- MLflow provides visualization and comparison tools

---

## Step 2: Set Up MLflow

### 2.1 Install and Configure MLflow

**Installation:**
- Add mlflow to your requirements.txt
- Install with pip
- Verify installation by checking MLflow version

**MLflow tracking server options:**

**Option A: Local SQLite backend (Simplest for learning)**
- MLflow stores data in local SQLite database
- Fast to set up, good for single-user development
- Limited scalability and no collaboration features
- Tracking data in `./mlruns` directory

**Option B: GCS-backed tracking (Recommended for team projects)**
- Tracking data stored in GCS bucket
- Artifacts (models, plots) also in GCS
- Accessible from anywhere
- Better for production use

**Option C: Managed MLflow on GCP (Most robust)**
- Run MLflow tracking server on Compute Engine VM or Cloud Run
- Use Cloud SQL for backend database
- Use GCS for artifact storage
- Requires more setup but most scalable

**For this project:** Start with Option A (local), can migrate to Option B later.

### 2.2 Set Up Local MLflow Tracking

**Create mlflow directory structure:**
- In project root, create `mlflow/` directory
- This will contain the `mlruns/` subdirectory
- Set environment variable MLFLOW_TRACKING_URI to point here

**Configure MLflow:**
- Set tracking URI to local directory or remote server
- Set experiment name (e.g., "titanic-xgboost-optimization")
- Set artifact location to GCS bucket for persistence

**Test MLflow setup:**
- Start MLflow UI server with `mlflow ui` command
- Access at http://localhost:5000
- Verify empty experiments page loads
- Create a test experiment

**MLflow UI features:**
- Experiments list: See all experiments
- Runs table: Compare runs side-by-side
- Run details: View parameters, metrics, artifacts
- Charts: Visualize metric trends
- Compare: Select multiple runs to compare

### 2.3 Create MLflow Helper Module

**Create src/utils/mlflow_utils.py:**

**Purpose:** Centralize MLflow operations and configuration

**Key functions to implement:**

**Initialize MLflow:**
- Set tracking URI
- Create or get experiment
- Enable automatic logging for XGBoost
- Configure artifact storage location

**Start run:**
- Begin new MLflow run
- Set run name with timestamp or identifier
- Add tags (e.g., "phase:2", "optimization:optuna")
- Return run context manager

**Log parameters:**
- Log all hyperparameters used
- Log data configuration (train size, features used)
- Log model configuration

**Log metrics:**
- Log training metrics (loss, accuracy per epoch)
- Log validation metrics (AUC, F1, precision, recall)
- Log confusion matrix
- Log feature importance

**Log artifacts:**
- Save and log trained model
- Save and log plots (ROC curve, confusion matrix)
- Save and log feature importance charts
- Save and log preprocessing objects (scaler, encoders)

**Register model:**
- Register best model in MLflow Model Registry
- Tag with version number
- Add description with key metrics
- Set stage (Staging, Production, Archived)

### 2.4 Set Up GCS Integration for Artifacts

**Create GCS bucket for MLflow:**
- Create bucket: `gs://mlflow-artifacts-YOUR_PROJECT_ID`
- Set lifecycle policy to delete artifacts older than 90 days (optional)
- Ensure your service account has write access

**Configure MLflow to use GCS:**
- Set artifact URI to GCS bucket
- Install google-cloud-storage if not already present
- Verify MLflow can write to bucket

**Benefits of GCS storage:**
- Artifacts persist even if local machine is wiped
- Team members can access same artifacts
- Integrates well with other GCP services
- Supports large model files

---

## Step 3: Set Up Optuna

### 3.1 Install and Configure Optuna

**Installation:**
- Add optuna to requirements.txt
- Install with pip
- Install optional integration: optuna-integration[mlflow]
- This enables automatic MLflow logging from Optuna

**Optuna storage options:**

**Option A: In-memory (Simplest)**
- Study stored only in RAM
- Lost when process ends
- Good for quick experiments

**Option B: SQLite database (Recommended for local)**
- Persistent storage
- Can resume studies
- Lightweight and local

**Option C: PostgreSQL/MySQL (For production)**
- Supports distributed optimization
- Multiple processes can work on same study
- Requires database setup

**For this project:** Use SQLite storage with database file in project directory.

### 3.2 Define Hyperparameter Search Space

**XGBoost hyperparameters to tune:**

**Tree structure:**
- max_depth: Controls tree depth (suggest_int: 3 to 10)
- min_child_weight: Minimum sum of instance weight in a child (suggest_int: 1 to 10)
- gamma: Minimum loss reduction for split (suggest_float: 0 to 1)

**Training process:**
- learning_rate (eta): Step size shrinkage (suggest_float: 0.01 to 0.3, log=True)
- n_estimators: Number of boosting rounds (suggest_int: 50 to 500)

**Sampling:**
- subsample: Fraction of samples for each tree (suggest_float: 0.5 to 1.0)
- colsample_bytree: Fraction of features for each tree (suggest_float: 0.5 to 1.0)

**Regularization:**
- reg_alpha: L1 regularization (suggest_float: 0 to 1)
- reg_lambda: L2 regularization (suggest_float: 0 to 1)

**Fixed parameters (not tuned):**
- objective: binary:logistic (classification task)
- eval_metric: auc (for early stopping)
- random_state: 42 (reproducibility)
- tree_method: auto (let XGBoost choose)

**Search space considerations:**
- Larger search space = more thorough but slower
- Start with most impactful parameters (learning_rate, max_depth, n_estimators)
- Add more parameters in later iterations
- Use logarithmic scales for parameters that span multiple orders of magnitude

### 3.3 Create Objective Function

**Purpose:** Define what Optuna should optimize

**Create src/models/optuna_objective.py:**

**Objective function structure:**

**Function receives:**
- trial: Optuna trial object for suggesting parameters
- X_train, y_train: Training data
- X_val, y_val: Validation data
- fixed_params: Any parameters not being tuned

**Function suggests parameters:**
- Use trial.suggest_int() for integer parameters
- Use trial.suggest_float() for continuous parameters
- Use trial.suggest_categorical() for categorical choices

**Function trains model:**
- Create XGBoost classifier with suggested parameters
- Train on training set with early stopping
- Evaluate on validation set

**Function returns:**
- Single metric to optimize (e.g., validation AUC)
- Higher is better for AUC (Optuna maximizes by default)
- Or use study with direction='maximize'

**Add MLflow logging:**
- Start MLflow run within objective function
- Log all suggested parameters
- Log training and validation metrics
- Log trained model if performance is good
- Log trial number and study name

**Add pruning callback:**
- Report intermediate values to Optuna
- Enable early stopping of unpromising trials
- Saves computation time
- Use MedianPruner or HyperbandPruner

### 3.4 Configure Optuna Study

**Create study:**
- Use optuna.create_study()
- Set direction='maximize' (for AUC)
- Set study name for identification
- Set storage to SQLite database path
- Enable load_if_exists to resume interrupted studies

**Configure sampler:**
- Use TPESampler (Tree-structured Parzen Estimator) - default and recommended
- Alternative: RandomSampler for baseline
- Alternative: CmaEsSampler for continuous spaces
- Alternative: GridSampler for exhaustive search

**Configure pruner:**
- Use MedianPruner to stop trials performing worse than median
- Set n_startup_trials: Don't prune first N trials
- Set n_warmup_steps: Don't prune first N steps of trial
- Alternative: HyperbandPruner for more aggressive pruning

**Set optimization parameters:**
- n_trials: Number of parameter combinations to try (start with 50)
- timeout: Maximum time for optimization (e.g., 3600 seconds)
- n_jobs: Parallel trials (1 for single machine, >1 if multiple GPUs/cores)

---

## Step 4: Enhance Streamlit UI for Feature Selection

### 4.1 Design Feature Selection Interface

**Purpose:** Allow users to choose which features to include in model training

**UI components to add:**

**Feature selection section in sidebar:**
- Add expandable section titled "Feature Selection"
- List all available features with checkboxes
- Group features by category (demographics, family, travel details)
- Select/deselect all buttons for convenience
- Show count of selected features

**Feature categories to display:**

**Demographics:**
- Age (checkbox)
- Sex (checkbox)
- Title extracted from name (checkbox)

**Travel details:**
- Passenger class (checkbox)
- Fare (checkbox)
- Embarked port (checkbox)
- Cabin information (checkbox)

**Family information:**
- Number of siblings/spouses (checkbox)
- Number of parents/children (checkbox)
- Family size (derived) (checkbox)
- Is alone flag (derived) (checkbox)

**Default selections:**
- By default, check all features
- Store selections in Streamlit session state
- Persist across retraining runs

### 4.2 Feature Selection Logic

**Track selected features:**
- Store selection state in st.session_state
- Create list of selected feature names
- Validate minimum features selected (e.g., at least 3)
- Warn if important features are excluded

**Feature dependencies:**
- If user deselects SibSp and Parch, automatically deselect Family_Size
- If Age is deselected, deselect age-related derived features
- Show warnings about feature dependencies

**Feature importance display:**
- Show current model's feature importance next to each checkbox
- Help users make informed decisions
- Highlight most important features

**Presets:**
- Add quick select buttons:
  - "Top 5 Features" - select only most important
  - "Demographic Only" - age, sex, pclass
  - "All Features" - select everything
  - "Reset to Default" - return to recommended set

### 4.3 Validate Feature Selection

**Validation rules:**
- At least 3 features must be selected
- Must include at least one demographic feature
- Must include target variable (Survived) - don't show as option
- Warn if correlated features are selected together

**User feedback:**
- Display error messages for invalid selections
- Disable "Retrain Model" button until valid selection
- Show estimated impact on model performance
- Suggest adding/removing features based on correlations

---

## Step 5: Implement Model Retraining Workflow

### 5.1 Design Retraining Interface

**Add to Streamlit UI:**

**New tab or section for retraining:**
- Create "Model Training" tab (separate from "Predictions" tab)
- Show current model information (version, accuracy, features used)
- Provide interface to train new model

**Retraining configuration:**
- Feature selection (from Step 4)
- Hyperparameter optimization settings:
  - Enable/disable Optuna optimization
  - Number of Optuna trials (slider: 10-200)
  - Optimization timeout (minutes)
- Training options:
  - Train/validation split ratio
  - Cross-validation folds
  - Random seed

**Training trigger:**
- Large "Train New Model" button
- Confirmation dialog if major changes detected
- Estimate training time based on configuration
- Show warnings about costs (if applicable)

### 5.2 Implement Training Pipeline

**Create src/app/retraining.py module:**

**Key functions:**

**Prepare data with selected features:**
- Load full dataset from GCS or Feature Store
- Filter to include only selected features
- Apply preprocessing (same as training)
- Split into train/validation sets
- Return prepared data

**Run training without optimization:**
- Use default or last-known-good hyperparameters
- Train single XGBoost model
- Log to MLflow
- Return trained model and metrics

**Run training with Optuna optimization:**
- Create Optuna study
- Run optimization for specified trials
- Log each trial to MLflow
- Track best trial
- Train final model with best parameters
- Return best model and comprehensive metrics

**Update model registry:**
- Save model to GCS
- Register in MLflow Model Registry
- Tag with feature set used
- Add performance metrics to model description
- Update deployed endpoint (optional)

### 5.3 Progress Monitoring

**Real-time feedback in UI:**
- Show progress bar for optimization
- Display current trial number / total trials
- Show best metric found so far
- Stream log messages to UI
- Display estimated time remaining

**Use Streamlit components:**
- st.progress() for progress bar
- st.status() for current operation
- st.empty() container for updating metrics
- st.spinner() for long operations

**MLflow tracking URL:**
- Provide link to MLflow UI to view detailed progress
- Open in new tab
- Update in real-time

### 5.4 Handle Training Errors

**Error scenarios:**
- Insufficient data for selected features
- Optuna optimization fails
- Out of memory errors
- GCS access errors
- MLflow logging failures

**Error handling:**
- Catch all exceptions gracefully
- Display user-friendly error messages
- Log detailed errors to Cloud Logging
- Provide suggestions for resolution
- Allow retry with adjusted settings

**Validation before training:**
- Check data availability
- Verify feature names match dataset
- Ensure GCS bucket is accessible
- Confirm MLflow tracking is working
- Validate hyperparameter ranges

---

## Step 6: Optuna Hyperparameter Optimization

### 6.1 Run Optimization Study

**Optimization workflow:**

**Initialize study:**
- Create or load existing study from SQLite
- Set study name (e.g., "titanic-xgb-2024-01-15")
- Configure for maximizing AUC
- Set up pruner and sampler

**Run optimization:**
- Call study.optimize() with objective function
- Set n_trials based on user selection
- Set timeout if desired
- Enable parallel execution if supported (n_jobs)
- Show progress in UI

**Monitor progress:**
- Optuna provides callbacks for progress updates
- Log after each trial completes
- Update UI with current best parameters
- Display optimization history plot

**Typical timing:**
- Each trial: 30 seconds to 2 minutes (depending on n_estimators)
- 50 trials: 25-100 minutes total
- Can stop early if good results found

### 6.2 Analyze Optimization Results

**Access best trial:**
- study.best_trial contains best parameters and value
- study.best_params is dictionary of best parameters
- study.best_value is best metric achieved

**Analyze all trials:**
- study.trials_dataframe() returns pandas DataFrame
- Contains all parameters and metrics for each trial
- Useful for detailed analysis

**Visualization:**
- Optimization history: Metric over trial number
- Parameter importance: Which parameters matter most
- Parallel coordinate plot: See parameter relationships
- Slice plot: Effect of single parameter
- Contour plot: Effect of parameter pairs

**Generate plots:**
- Use optuna.visualization functions
- Save plots to artifacts directory
- Display in Streamlit UI
- Upload to MLflow as artifacts

### 6.3 Optuna-MLflow Integration

**Automatic logging:**
- Install optuna-integration package
- Use MLflowCallback in study.optimize()
- Automatically logs each trial to MLflow
- Creates nested runs structure

**Manual logging enhancement:**
- Add custom tags to MLflow runs
- Log additional context (feature set, data version)
- Log Optuna visualizations as artifacts
- Link related runs together

**Best practices:**
- Use consistent naming conventions
- Tag production-worthy runs
- Archive old experiments periodically
- Document significant changes

---

## Step 7: Display Comprehensive Metrics

### 7.1 Calculate All Relevant Metrics

**Classification metrics to compute:**

**Accuracy-based:**
- Overall accuracy: Correct predictions / Total predictions
- Balanced accuracy: Average of recall for each class
- Top-k accuracy: Not applicable for binary classification

**Precision and Recall:**
- Precision: True Positives / (True Positives + False Positives)
- Recall (Sensitivity): True Positives / (True Positives + False Negatives)
- Specificity: True Negatives / (True Negatives + False Positives)
- F1 Score: Harmonic mean of precision and recall
- F-beta scores: Weighted variants for different precision/recall balance

**Probability-based:**
- AUC-ROC: Area under ROC curve (most important for binary classification)
- AUC-PR: Area under precision-recall curve (better for imbalanced data)
- Log Loss: Penalizes confident wrong predictions
- Brier Score: Mean squared difference between predicted probabilities and actual outcomes

**Confusion Matrix:**
- True Positives, True Negatives, False Positives, False Negatives
- Visualize as heatmap
- Calculate all derived metrics from this

**Performance metrics:**
- Training time: Total seconds to train model
- Prediction latency: Average milliseconds per prediction
- Model size: Disk space used by saved model
- Memory usage: RAM required during inference

### 7.2 Create Metrics Dashboard

**Design Streamlit metrics display:**

**Summary statistics (top of page):**
- Use st.metric() for key metrics with delta from previous run
- Display in columns:
  - Column 1: AUC-ROC (primary metric)
  - Column 2: Accuracy
  - Column 3: F1 Score
  - Column 4: Training time
- Show comparison to baseline or previous model

**Detailed metrics table:**
- Create DataFrame with all metrics
- Display using st.dataframe() with styling
- Highlight metrics that improved
- Highlight metrics that degraded
- Include comparison column to previous model

**Confusion matrix:**
- Create heatmap visualization with seaborn or plotly
- Annotate with counts
- Show percentages in each cell
- Display below metrics table

**ROC and PR curves:**
- Plot ROC curve with AUC score
- Plot Precision-Recall curve with AP score
- Display side by side
- Include diagonal reference line for ROC
- Mark optimal threshold points

**Feature importance:**
- Bar chart of top 10 most important features
- Show both gain and weight importance
- Compare to previous model if available

### 7.3 Metrics Comparison

**Compare across models:**
- Create comparison table showing multiple models
- Columns: Model version, AUC, Accuracy, F1, Date trained
- Allow sorting by any metric
- Highlight current production model

**Trends over time:**
- Line chart showing metric evolution across versions
- X-axis: Model version or date
- Y-axis: Metric value
- Multiple lines for different metrics
- Show when model was deployed to production

**A/B test simulation:**
- Show predicted outcomes if new model deployed
- Estimate impact on success metrics
- Calculate statistical significance of improvement

### 7.4 Export Metrics

**Save metrics to files:**
- Save as JSON for programmatic access
- Save as CSV for spreadsheet analysis
- Include timestamp and model version
- Upload to GCS for persistence

**Integrate with monitoring:**
- Send metrics to Cloud Monitoring
- Create custom dashboards
- Set up alerts for metric degradation
- Track metrics over time

---

## Step 8: MLflow Model Registry

### 8.1 Register Best Model

**After optimization completes:**

**Register model in MLflow:**
- Use mlflow.xgboost.log_model() to log the trained model
- Provide model artifact path
- Include signature (input/output schema)
- Add conda or pip environment for reproducibility

**Register in Model Registry:**
- Use mlflow.register_model() to promote to registry
- Provide model name: "titanic-xgboost"
- Model version is auto-incremented
- Add description with key metrics and feature set

**Model metadata to include:**
- Training date and time
- Features used (list)
- Hyperparameters (all final values)
- Performance metrics (AUC, accuracy, F1)
- Training data version or hash
- Git commit SHA if using version control

### 8.2 Model Versioning Strategy

**Version numbering:**
- MLflow auto-assigns version numbers (1, 2, 3, ...)
- Use tags for semantic versioning (v1.0.0, v1.1.0, etc.)
- Tag major changes: "major-update", "feature-change", "hotfix"

**Model stages:**
- None: Newly registered models
- Staging: Models being tested
- Production: Currently deployed model
- Archived: Old models kept for reference

**Transitioning between stages:**
- Manually promote from None to Staging
- Test thoroughly in staging
- Promote to Production when validated
- Archive previous production model

**Best practices:**
- Always test in Staging before Production
- Keep only latest 2-3 models in Staging
- Archive models older than 6 months
- Document reason for each stage transition

### 8.3 Model Comparison

**Compare models in registry:**
- View all registered versions
- Compare metrics side-by-side
- Identify best performing version
- Analyze performance trends

**Selection criteria:**
- Not always just highest AUC
- Consider: Model size, latency, interpretability
- Regulatory requirements may favor simpler models
- Production constraints (memory, CPU)

### 8.4 Model Deployment from Registry

**Load model for deployment:**
- Use mlflow.pyfunc.load_model() with model URI
- URI format: "models:/titanic-xgboost/Production"
- Always references current production model
- Simplifies deployment code

**Deploy to Vertex AI:**
- Export model from MLflow
- Upload to GCS in Vertex AI compatible format
- Update existing endpoint
- Or create new endpoint for A/B testing

**Deployment validation:**
- Test predictions match MLflow predictions
- Verify latency is acceptable
- Check resource usage
- Monitor for errors

---

## Step 9: Store Models in GCS

### 9.1 Configure GCS Storage

**Create GCS bucket for models:**
- Create bucket: `gs://titanic-models-YOUR_PROJECT_ID`
- Set retention policy (optional)
- Enable versioning for accidental deletion protection
- Set lifecycle rule to archive old versions

**Organize bucket structure:**
- models/xgboost/v1/, v2/, etc. - versioned models
- models/xgboost/production/ - symlink to current production
- models/xgboost/staging/ - models being tested
- models/artifacts/ - feature importance, plots, etc.
- models/metadata/ - JSON files with model metadata

### 9.2 Save Models to GCS

**After training:**
- Save model using XGBoost's save_model()
- Save as binary format (.xgb or .json)
- Save preprocessing objects (scaler, encoders) as pickle
- Save feature list as JSON
- Upload all to versioned GCS path

**Model package contents:**
- model.xgb - trained XGBoost model
- scaler.pkl - fitted StandardScaler
- encoders.pkl - fitted label encoders
- features.json - list of feature names in order
- metadata.json - hyperparameters, metrics, date
- requirements.txt - package versions used

**Upload to GCS:**
- Use google-cloud-storage library
- Upload each file individually
- Set appropriate content types
- Add custom metadata tags

### 9.3 Version Control

**Semantic versioning:**
- Use format: vMAJOR.MINOR.PATCH
- Increment MAJOR for breaking changes (feature set change)
- Increment MINOR for new capabilities (new features added)
- Increment PATCH for bug fixes and retraining

**Git integration:**
- Tag Git commits with model version
- Include Git SHA in model metadata
- Link to commit in model description
- Enable reproducibility

**Metadata tracking:**
- Create manifest file listing all model versions
- Include path, version, date, metrics, status
- Update after each new model
- Store in GCS bucket

### 9.4 Load Models from GCS

**Loading for inference:**
- Download model files from GCS
- Load into memory
- Cache for performance
- Refresh when new version detected

**Version pinning:**
- Specify exact version to load
- Or always load "production" version
- Implement graceful fallback to previous version
- Handle missing models elegantly

---

## Step 10: Integration and Testing

### 10.1 End-to-End Workflow Test

**Complete workflow:**
- Select subset of features in UI
- Enable Optuna optimization
- Set number of trials (use 10 for quick test)
- Click "Train New Model"
- Monitor progress in UI and MLflow
- Wait for completion
- Review metrics displayed
- Check model registered in MLflow
- Verify model saved to GCS

**Validation points:**
- MLflow run appears with correct parameters
- All metrics are logged correctly
- Model artifact is saved
- GCS bucket contains new model version
- Streamlit UI updates with new results
- Can load and use new model for predictions

### 10.2 Test Different Scenarios

**Feature selection variations:**
- Test with minimal features (3-4)
- Test with all features
- Test with only demographic features
- Test with derived features excluded

**Optimization settings:**
- Test without Optuna (single training run)
- Test with 10 Optuna trials (quick)
- Test with 50 trials (thorough)
- Test with different pruners

**Edge cases:**
- Test with invalid feature combination
- Test with no features selected (should fail gracefully)
- Test canceling mid-training
- Test with very small training set

### 10.3 Performance Validation

**Compare to baseline:**
- Baseline model from Phase 1
- New model with default features
- New model with optimized hyperparameters
- Verify optimization improves performance

**Expected improvements:**
- Optuna optimization should improve AUC by 1-3%
- Better hyperparameters should reduce overfitting
- Feature selection may improve or hurt depending on choices

**Verify no regression:**
- New model should not be significantly worse than baseline
- If worse, investigate (data issues, bugs, etc.)
- May need to adjust optimization search space

### 10.4 UI/UX Testing

**Usability:**
- Is the interface intuitive?
- Are instructions clear?
- Is feedback adequate during long operations?
- Are errors handled gracefully?

**Performance:**
- UI remains responsive during training
- Progress updates regularly
- No crashes or freezes
- Plots render quickly

**Mobile/responsive:**
- Test on different screen sizes
- Ensure readability
- Check that interactive elements work

---

## Step 11: Documentation and Best Practices

### 11.1 Document New Capabilities

**Update README:**
- Add section on retraining workflow
- Document Optuna usage
- Explain MLflow organization
- List available metrics

**Create user guide:**
- How to select features
- How to start retraining
- How to interpret results
- How to choose between models

**Create developer guide:**
- How to add new features
- How to modify hyperparameter search space
- How to customize objective function
- How to extend metrics

### 11.2 MLOps Best Practices

**Experiment tracking:**
- Always log to MLflow, never skip
- Use consistent naming conventions
- Tag experiments appropriately
- Document experiment goals

**Model versioning:**
- Never overwrite existing models
- Always increment versions
- Keep metadata synchronized
- Archive old models systematically

**Reproducibility:**
- Pin all package versions
- Set random seeds consistently
- Save preprocessing objects
- Document data versions used

**Monitoring:**
- Track model performance over time
- Monitor for metric drift
- Alert on significant degradation
- Regular retraining schedule

### 11.3 Cost Optimization

**Training costs:**
- Use appropriate machine types (don't over-provision)
- Delete training jobs after completion
- Use preemptible instances for Optuna trials
- Limit optimization trials to what's necessary

**Storage costs:**
- Archive old models periodically
- Compress artifacts before storing
- Set lifecycle policies on GCS buckets
- Delete temporary files

**Endpoint costs:**
- Delete endpoints when not in use (Phase 1)
- Use model from GCS without endpoint for retraining
- Only deploy after validation

---

## Step 12: Advanced Features (Optional)

### 12.1 Multi-objective Optimization

**Optimize for multiple objectives:**
- Maximize AUC and minimize model size
- Maximize accuracy and minimize latency
- Use Optuna multi-objective optimization
- Get Pareto front of solutions

**Implementation:**
- Return tuple from objective function
- Create study with multiple directions
- Analyze trade-offs
- Let user choose preferred point on Pareto front

### 12.2 Automated Retraining

**Schedule periodic retraining:**
- Use Cloud Scheduler to trigger retraining
- Run weekly or monthly
- Compare new model to production
- Auto-deploy if improvement threshold met

**Monitoring-triggered retraining:**
- Monitor production model performance
- Trigger retraining if metrics degrade
- Concept drift detection
- Automated response to data changes

### 12.3 A/B Testing Framework

**Compare models in production:**
- Deploy two model versions simultaneously
- Split traffic between them
- Collect metrics for each
- Determine statistical significance
- Promote winner to 100% traffic

**Implementation considerations:**
- Need unique identifiers for each prediction
- Log model version used for each prediction
- Collect ground truth labels when available
- Calculate online metrics

### 12.4 Advanced Visualizations

**Interactive dashboards:**
- Use Plotly for interactive charts
- Allow drill-down into metrics
- Filter by date range, model version, etc.
- Export to PDF for reports

**Custom visualizations:**
- Learning curves (training vs validation loss)
- Calibration plots (predicted probabilities vs actual)
- Lift curves (for marketing applications)
- Cost-benefit analysis based on confusion matrix

---

## Troubleshooting

### Issue: Optuna optimization runs forever

**Causes:**
- Too many trials configured
- No timeout set
- Slow objective function

**Solutions:**
- Reduce n_trials to 50 or less initially
- Set timeout in seconds
- Profile objective function for bottlenecks
- Enable pruning to stop bad trials early

### Issue: MLflow not logging runs

**Causes:**
- Tracking URI not set
- Experiment doesn't exist
- Network issues (if remote tracking server)
- Permissions issues (if GCS backend)

**Solutions:**
- Verify MLFLOW_TRACKING_URI environment variable
- Create experiment before first run
- Check network connectivity
- Verify service account permissions

### Issue: Model performance worse after optimization

**Causes:**
- Overfitting to validation set
- Bug in objective function
- Inappropriate search space
- Data leakage

**Solutions:**
- Use cross-validation instead of single validation split
- Review objective function logic
- Narrow search space around sensible defaults
- Check for data leakage carefully

### Issue: GCS upload fails

**Causes:**
- Service account lacks write permissions
- Bucket doesn't exist
- Network timeout
- File too large

**Solutions:**
- Add Storage Object Creator role to service account
- Verify bucket exists and name is correct
- Implement retry logic with backoff
- Compress large files before upload

---

## Success Criteria for Phase 2

You have successfully completed Phase 2 when:

✅ MLflow is set up and tracking experiments
✅ Optuna can optimize XGBoost hyperparameters
✅ Streamlit UI allows feature selection
✅ Retraining workflow is functional and reliable
✅ Comprehensive metrics are calculated and displayed
✅ Models are versioned and registered in MLflow
✅ Models are saved to GCS with metadata
✅ Can compare multiple model versions
✅ Optimization improves model performance
✅ All code is well-documented and tested

---

## Next Steps

Proceed to [implementation_3.md](implementation_3.md) for Phase 3: Kubernetes Deployment with GKE Autopilot.

Phase 3 will focus on:
- Containerizing the application with Docker
- Deploying to Google Kubernetes Engine
- Implementing auto-scaling
- Production-grade orchestration
- Monitoring and logging at scale

