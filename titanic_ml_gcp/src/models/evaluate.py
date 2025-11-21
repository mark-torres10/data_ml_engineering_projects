"""
Model evaluation module for Titanic ML project.

Provides comprehensive model evaluation including:
- Classification metrics (accuracy, precision, recall, F1, AUC-ROC)
- Confusion matrix
- ROC curve and precision-recall curve
- Feature importance analysis
- Visualization and reporting
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    average_precision_score
)

from src.models.model_config import EvaluationConfig
from src.models.model_utils import (
    ModelArtifactManager,
    load_training_artifacts,
    get_feature_importance
)
from src.utils.gcs_utils import load_dataframe_from_gcs

logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class ModelEvaluator:
    """
    Comprehensive model evaluator for classification tasks.
    
    Handles model evaluation, metrics calculation, and visualization.
    """
    
    def __init__(
        self,
        model: Optional[XGBClassifier] = None,
        eval_config: Optional[EvaluationConfig] = None
    ):
        """
        Initialize model evaluator.
        
        Args:
            model: Trained model to evaluate
            eval_config: Evaluation configuration
        """
        self.model = model
        self.eval_config = eval_config or EvaluationConfig()
        self.artifact_manager = ModelArtifactManager()
        
        # Evaluation results
        self.metrics: Dict[str, float] = {}
        self.predictions: Optional[np.ndarray] = None
        self.prediction_probas: Optional[np.ndarray] = None
        
        logger.info("Initialized ModelEvaluator")
    
    def load_model(
        self,
        model_dir: Path,
        load_format: str = "json"
    ) -> XGBClassifier:
        """
        Load trained model from directory.
        
        Args:
            model_dir: Directory containing model artifacts
            load_format: Model file format
            
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {model_dir}")
        self.model, metadata, feature_names = load_training_artifacts(
            model_dir, load_format
        )
        logger.info(f"Model loaded successfully with {len(feature_names)} features")
        return self.model
    
    def predict(
        self,
        X: pd.DataFrame,
        return_proba: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions on data.
        
        Args:
            X: Feature matrix
            return_proba: Whether to return probabilities
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        predictions = self.model.predict(X)
        
        probabilities = None
        if return_proba:
            probabilities = self.model.predict_proba(X)[:, 1]
        
        self.predictions = predictions
        self.prediction_probas = probabilities
        
        return predictions, probabilities
    
    def calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: Optional[np.ndarray] = None,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        y_pred = y_pred if y_pred is not None else self.predictions
        y_pred_proba = y_pred_proba if y_pred_proba is not None else self.prediction_probas
        
        if y_pred is None:
            raise ValueError("No predictions available")
        
        metrics = {}
        
        # Basic classification metrics
        if self.eval_config.calculate_accuracy:
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        
        if self.eval_config.calculate_precision:
            metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        
        if self.eval_config.calculate_recall:
            metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        
        if self.eval_config.calculate_f1:
            metrics["f1_score"] = float(f1_score(y_true, y_pred, zero_division=0))
        
        # Probability-based metrics
        if y_pred_proba is not None:
            if self.eval_config.calculate_auc_roc:
                metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred_proba))
            
            metrics["average_precision"] = float(average_precision_score(y_true, y_pred_proba))
        
        # Store metrics
        self.metrics = metrics
        
        # Log metrics
        logger.info("Calculated metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        return metrics
    
    def get_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        y_pred = y_pred if y_pred is not None else self.predictions
        
        if y_pred is None:
            raise ValueError("No predictions available")
        
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")
        
        return cm
    
    def plot_confusion_matrix(
        self,
        y_true: pd.Series,
        y_pred: Optional[np.ndarray] = None,
        output_path: Optional[Path] = None,
        normalize: bool = False
    ) -> plt.Figure:
        """
        Plot confusion matrix as heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_path: Path to save plot
            normalize: Whether to normalize counts
            
        Returns:
            Matplotlib figure
        """
        cm = self.get_confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'],
            ax=ax
        )
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {output_path}")
        
        return fig
    
    def plot_roc_curve(
        self,
        y_true: pd.Series,
        y_pred_proba: Optional[np.ndarray] = None,
        output_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            output_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        y_pred_proba = y_pred_proba if y_pred_proba is not None else self.prediction_probas
        
        if y_pred_proba is None:
            raise ValueError("No prediction probabilities available")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {output_path}")
        
        return fig
    
    def plot_precision_recall_curve(
        self,
        y_true: pd.Series,
        y_pred_proba: Optional[np.ndarray] = None,
        output_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            output_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        y_pred_proba = y_pred_proba if y_pred_proba is not None else self.prediction_probas
        
        if y_pred_proba is None:
            raise ValueError("No prediction probabilities available")
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.3f})', linewidth=2)
        ax.axhline(y=y_true.mean(), color='k', linestyle='--', 
                   label=f'Baseline (y={y_true.mean():.3f})', linewidth=1)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-recall curve saved to {output_path}")
        
        return fig
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        top_n: int = 20,
        output_path: Optional[Path] = None,
        importance_type: str = "weight"
    ) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to show
            output_path: Path to save plot
            importance_type: Type of importance ("weight", "gain", "cover")
            
        Returns:
            Matplotlib figure
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        # Get feature importance
        importance_df = get_feature_importance(
            self.model, feature_names, importance_type
        )
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()
        ax.set_xlabel(f'Importance ({importance_type})')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {output_path}")
        
        return fig
    
    def generate_classification_report(
        self,
        y_true: pd.Series,
        y_pred: Optional[np.ndarray] = None,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_path: Path to save report
            
        Returns:
            Classification report as string
        """
        y_pred = y_pred if y_pred is not None else self.predictions
        
        if y_pred is None:
            raise ValueError("No predictions available")
        
        report = classification_report(
            y_true, y_pred,
            target_names=['Not Survived', 'Survived'],
            digits=4
        )
        
        logger.info(f"Classification Report:\n{report}")
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Classification report saved to {output_path}")
        
        return report
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Perform complete evaluation on test data.
        
        Args:
            X: Feature matrix
            y: True labels
            feature_names: List of feature names
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with all evaluation results
        """
        if self.model is None:
            raise ValueError("No model loaded")
        
        logger.info("Starting model evaluation...")
        
        # Make predictions
        y_pred, y_pred_proba = self.predict(X)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y, y_pred, y_pred_proba)
        
        # Get confusion matrix
        cm = self.get_confusion_matrix(y, y_pred)
        
        # Generate classification report
        report = self.generate_classification_report(y, y_pred)
        
        # Set up output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot confusion matrix
            if self.eval_config.plot_confusion_matrix:
                self.plot_confusion_matrix(
                    y, y_pred,
                    output_path=output_dir / "confusion_matrix.png"
                )
            
            # Plot ROC curve
            if self.eval_config.plot_roc_curve and y_pred_proba is not None:
                self.plot_roc_curve(
                    y, y_pred_proba,
                    output_path=output_dir / "roc_curve.png"
                )
            
            # Plot precision-recall curve
            if self.eval_config.plot_precision_recall and y_pred_proba is not None:
                self.plot_precision_recall_curve(
                    y, y_pred_proba,
                    output_path=output_dir / "precision_recall_curve.png"
                )
            
            # Plot feature importance
            if self.eval_config.plot_feature_importance and feature_names:
                self.plot_feature_importance(
                    feature_names,
                    output_path=output_dir / "feature_importance.png"
                )
            
            # Save classification report
            report_path = output_dir / "classification_report.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            
            # Save metrics
            if self.eval_config.save_metrics:
                metrics_path = output_dir / "metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                logger.info(f"Metrics saved to {metrics_path}")
        
        results = {
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "n_samples": len(y),
            "n_correct": int((y_pred == y).sum()),
            "evaluated_at": datetime.now().isoformat()
        }
        
        logger.info("Evaluation completed successfully")
        
        return results


def main():
    """Main function for running evaluation from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate XGBoost model for Titanic prediction")
    parser.add_argument("--model-dir", type=str, required=True, help="Directory with model artifacts")
    parser.add_argument("--test-path", type=str, required=True, help="GCS path to test data")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation", help="Output directory")
    parser.add_argument("--target-column", type=str, default="Survived", help="Target column name")
    parser.add_argument("--load-format", type=str, default="json", help="Model load format")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load test data
    logger.info(f"Loading test data from {args.test_path}")
    test_df = load_dataframe_from_gcs(args.test_path)
    
    # Prepare features and target
    y_test = test_df[args.target_column]
    X_test = test_df.drop(columns=[args.target_column])
    feature_names = X_test.columns.tolist()
    
    # Create evaluator and load model
    evaluator = ModelEvaluator()
    evaluator.load_model(Path(args.model_dir), args.load_format)
    
    # Run evaluation
    results = evaluator.evaluate(
        X_test, y_test,
        feature_names=feature_names,
        output_dir=Path(args.output_dir)
    )
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"Test Samples: {results['n_samples']}")
    print(f"Correct Predictions: {results['n_correct']}")
    print("\nMetrics:")
    for metric_name, value in results['metrics'].items():
        print(f"  {metric_name}: {value:.4f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()

