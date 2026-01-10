"""
Models package for Titanic ML GCP project.

This package contains modules for training, evaluating, and serving ML models.
"""

from src.models.trainer import XGBoostTrainer
from src.models.evaluate import ModelEvaluator

__all__ = ["XGBoostTrainer", "ModelEvaluator"]

