"""
Category Classification Module

This module provides LightGBM + Text Embeddings based transaction category classification.

Classes:
    CategoryLightGBMTrainer: Train category classification models
    CategoryLightGBMPredictor: Load and use trained models for prediction

Functions:
    train_category_model: Convenience function to train from CSV file
"""

from .lightgbm_trainer import (
    CategoryLightGBMTrainer,
    train_category_model
)
from .lightgbm_predictor import CategoryLightGBMPredictor

__all__ = [
    'CategoryLightGBMTrainer',
    'CategoryLightGBMPredictor',
    'train_category_model'
]

