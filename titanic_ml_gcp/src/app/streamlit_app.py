import streamlit as st
import pandas as pd
import mlflow
import time
import sys
import os

# Add project root to path if needed
sys.path.append(os.getcwd())

from src.app.retraining import run_retraining, load_local_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb

st.set_page_config(page_title="Titanic ML Platform", page_icon="🚢", layout="wide")

st.title("🚢 Titanic ML Platform (Local Dev)")

# Tabs
tab_inference, tab_training = st.tabs(["🔮 Inference", "⚙️ Model Training"])

# --- SIDEBAR: Feature Selection (Step 4) ---
st.sidebar.header("Feature Selection")

# Define available features (excluding target)
# This should ideally be dynamic, but hardcoding for now based on known features
ALL_FEATURES = [
    "Pclass", "Age", "Fare", "Family_Size", "Is_Alone", "Has_Cabin",
    "Sex_male", "Embarked_Q", "Embarked_S",
    "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Officer", "Title_Royalty"
]

# Feature categories for better UI
categories = {
    "Demographics": ["Age", "Sex_male", "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Officer", "Title_Royalty"],
    "Travel": ["Pclass", "Fare", "Embarked_Q", "Embarked_S", "Has_Cabin"],
    "Family": ["Family_Size", "Is_Alone"]
}

selected_features = []

# Create expandable sections for categories
with st.sidebar.expander("Features", expanded=True):
    # Select all/Deselect all buttons
    col_sel, col_desel = st.columns(2)
    if col_sel.button("Select All"):
        for f in ALL_FEATURES:
            st.session_state[f"feat_{f}"] = True
    
    if col_desel.button("Deselect All"):
        for f in ALL_FEATURES:
            st.session_state[f"feat_{f}"] = False

    for cat, features in categories.items():
        st.subheader(cat)
        for f in features:
            # Initialize session state if not set
            if f"feat_{f}" not in st.session_state:
                st.session_state[f"feat_{f}"] = True
            
            if st.checkbox(f, key=f"feat_{f}"):
                selected_features.append(f)

st.sidebar.metric("Selected Features", len(selected_features))

# --- TAB: INFERENCE ---
with tab_inference:
    st.header("Prediction Interface")
    st.info("Load a model from the registry to make predictions.")
    
    # List models from local registry
    try:
        # Set tracking URI explicitly if needed
        # mlflow.set_tracking_uri("sqlite:///mlflow/optuna.db") # Or rely on environment/default
        client = mlflow.tracking.MlflowClient()
        
        # Attempt to list models
        model_name = "titanic_xgboost_local"
        # This might fail if registry is not set up or DB is locked/empty
        # But we'll wrap in try/except
        
        # Note: search_registered_models might return empty list if none found
        models = client.search_registered_models(filter_string=f"name='{model_name}'")
        
        if models:
            st.success(f"Model Registry found: {model_name}")
            # Get versions
            versions = client.search_model_versions(f"name='{model_name}'")
            if versions:
                # Sort by version number desc
                versions.sort(key=lambda x: int(x.version), reverse=True)
                
                selected_version_obj = st.selectbox(
                    "Select Model Version", 
                    versions,
                    format_func=lambda v: f"v{v.version} (Stage: {v.current_stage})"
                )
                
                if selected_version_obj:
                    st.write(f"Selected Version: {selected_version_obj.version}")
                    st.write(f"Source: {selected_version_obj.source}")
                    # Here we would load the model and predict
            else:
                st.warning("Model registered but no versions found.")
        else:
            st.warning(f"No models found registered as '{model_name}'. Train a model first!")
            
    except Exception as e:
        st.warning(f"Could not connect to Model Registry: {e}")
        st.caption("Ensure MLflow is set up and models are registered. Try running training first.")

# --- TAB: TRAINING ---
with tab_training:
    st.header("Model Retraining Workflow")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configuration")
        n_trials = st.slider("Number of Optuna Trials", min_value=1, max_value=50, value=5)
        timeout = st.number_input("Timeout (seconds)", min_value=30, value=300)
        experiment_name = st.text_input("Experiment Name", value="titanic_retraining_local")
        
    with col2:
        st.subheader("Action")
        st.markdown(f"Ready to train with **{len(selected_features)}** selected features.")
        
        if len(selected_features) < 3:
            st.error("Please select at least 3 features.")
            train_btn = st.button("Train New Model", disabled=True)
        else:
            train_btn = st.button("Train New Model", type="primary")
            
    if train_btn:
        status_container = st.status("Running Optimization...", expanded=True)
        
        try:
            status_container.write("Initializing...")
            progress_bar = status_container.progress(0)
            
            # Run retraining
            status_container.write("Running Optuna trials (check terminal for logs)...")
            
            # We need to capture the return values
            with st.spinner("Training in progress..."):
                final_model, best_trial = run_retraining(
                    selected_features=selected_features,
                    n_trials=n_trials,
                    timeout=timeout,
                    experiment_name=experiment_name
                )
            
            progress_bar.progress(100)
            status_container.update(label="Training Complete!", state="complete", expanded=True)
            
            st.success(f"Training finished! Best AUC: {best_trial.value:.4f}")
            
            # Re-load data to compute validation metrics for display
            X, y = load_local_data()
            # Filter features
            valid_features = [f for f in selected_features if f in X.columns]
            X = X[valid_features]
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Predict on validation set
            y_pred = final_model.predict(X_val)
            
            # Metrics
            acc = accuracy_score(y_val, y_pred)
            conf_mat = confusion_matrix(y_val, y_pred)
            
            st.divider()
            st.subheader("Validation Metrics")
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Best AUC", f"{best_trial.value:.4f}")
            col_m2.metric("Accuracy", f"{acc:.4f}")
            
            col_p1, col_p2 = st.columns(2)
            
            with col_p1:
                st.subheader("Confusion Matrix")
                fig_cm, ax_cm = plt.subplots()
                sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_xlabel('Predicted')
                ax_cm.set_ylabel('Actual')
                st.pyplot(fig_cm)
                
            with col_p2:
                st.subheader("Feature Importance")
                # XGBoost plot_importance
                fig_imp, ax_imp = plt.subplots()
                xgb.plot_importance(final_model, max_num_features=10, ax=ax_imp, height=0.8)
                st.pyplot(fig_imp)

            st.subheader("Best Hyperparameters")
            st.json(best_trial.params)
            
            st.balloons()

            
        except Exception as e:
            status_container.update(label="Training Failed", state="error")
            st.error(f"Error during training: {str(e)}")
            # Print traceback to console
            import traceback
            traceback.print_exc()

