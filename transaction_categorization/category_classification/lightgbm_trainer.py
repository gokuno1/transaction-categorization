"""
LightGBM + Text Embeddings Trainer for Transaction Category Classification

This module provides functionality to train a hybrid model combining:
- Sentence Transformer embeddings for text features (normalized_description, merchant_name)
- LightGBM classifier for multiclass category prediction

Features used:
- normalized_description (text -> embeddings)
- merchant_name (text -> embeddings)
- amount (numerical)
- dr_cr (categorical)
- transaction_type (categorical)

Target: category
"""

import os
import re
import pickle
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score

import lightgbm as lgb
from sentence_transformers import SentenceTransformer


def normalize(text: str) -> str:
    """
    Normalize transaction text for better matching.
    
    Args:
        text: Raw transaction description text.
        
    Returns:
        Normalized uppercase text with standardized spacing.
    """
    text = text if isinstance(text, str) else ""
    text = text.upper()
    text = re.sub(r"[\/\-\_\*]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class CategoryLightGBMTrainer:
    """
    Trainer class for LightGBM + Text Embeddings based category classification.
    
    Uses SentenceTransformer for text embeddings and LightGBM for classification.
    """
    
    def __init__(
        self,
        embedding_model_name: str = 'all-MiniLM-L6-v2',
        output_dir: str = 'models/category',
        test_size: float = 0.2,
        random_state: int = 42,
        lgbm_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Category LightGBM trainer.
        
        Args:
            embedding_model_name: SentenceTransformer model name for text embeddings.
            output_dir: Directory to save the trained models.
            test_size: Fraction of data to use for validation.
            random_state: Random seed for reproducibility.
            lgbm_params: Custom LightGBM parameters (optional).
        """
        self.embedding_model_name = embedding_model_name
        self.output_dir = output_dir
        self.test_size = test_size
        self.random_state = random_state
        
        # Default LightGBM parameters for multiclass classification
        self.lgbm_params = lgbm_params or {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': random_state,
            'n_jobs': -1
        }
        
        self.embedding_model: Optional[SentenceTransformer] = None
        self.lgbm_model: Optional[lgb.Booster] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.dr_cr_encoder: Optional[LabelEncoder] = None
        self.transaction_type_encoder: Optional[LabelEncoder] = None
        self.feature_names: List[str] = []
        self.embedding_dim: int = 0
        self.default_transaction_type: str = 'UPI'  # Default fallback for unseen types
    
    def _load_embedding_model(self) -> None:
        """Load the SentenceTransformer embedding model."""
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def _encode_text_features(
        self, 
        descriptions: List[str], 
        merchant_names: List[str]
    ) -> np.ndarray:
        """
        Encode text features using SentenceTransformer.
        
        Args:
            descriptions: List of normalized descriptions.
            merchant_names: List of merchant names.
            
        Returns:
            Combined embeddings array of shape (n_samples, 2 * embedding_dim).
        """
        print("Generating text embeddings...")
        
        # Encode descriptions
        desc_embeddings = self.embedding_model.encode(
            descriptions,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        # Encode merchant names
        merchant_embeddings = self.embedding_model.encode(
            merchant_names,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        # Concatenate embeddings
        text_features = np.concatenate([desc_embeddings, merchant_embeddings], axis=1)
        print(f"Text features shape: {text_features.shape}")
        
        return text_features
    
    def _encode_categorical_features(
        self,
        dr_cr: pd.Series,
        transaction_type: pd.Series,
        fit: bool = True
    ) -> np.ndarray:
        """
        Encode categorical features using LabelEncoder.
        
        Args:
            dr_cr: Series of DR/CR values.
            transaction_type: Series of transaction types.
            fit: Whether to fit the encoders (True for training, False for prediction).
            
        Returns:
            Encoded categorical features array of shape (n_samples, 2).
        """
        if fit:
            self.dr_cr_encoder = LabelEncoder()
            self.transaction_type_encoder = LabelEncoder()
            
            txn_type_filled = transaction_type.fillna('UNKNOWN')
            
            # Store the most common transaction_type as default for unseen values
            self.default_transaction_type = txn_type_filled.mode().iloc[0]
            print(f"Default transaction type (most common): {self.default_transaction_type}")
            
            dr_cr_encoded = self.dr_cr_encoder.fit_transform(dr_cr.fillna('UNKNOWN'))
            txn_type_encoded = self.transaction_type_encoder.fit_transform(txn_type_filled)
        else:
            # Handle unseen categories gracefully
            dr_cr_filled = dr_cr.fillna('UNKNOWN')
            txn_type_filled = transaction_type.fillna('UNKNOWN')
            
            # Map unseen categories to -1
            dr_cr_encoded = np.array([
                self.dr_cr_encoder.transform([val])[0] 
                if val in self.dr_cr_encoder.classes_ else -1
                for val in dr_cr_filled
            ])
            txn_type_encoded = np.array([
                self.transaction_type_encoder.transform([val])[0]
                if val in self.transaction_type_encoder.classes_ else -1
                for val in txn_type_filled
            ])
        
        categorical_features = np.column_stack([dr_cr_encoded, txn_type_encoded])
        return categorical_features
    
    def _prepare_features(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare all features for training or prediction.
        
        Args:
            df: DataFrame with required columns.
            fit: Whether to fit encoders and extract labels.
            
        Returns:
            Tuple of (features array, labels array or None).
        """
        # Ensure normalized_description exists
        if 'normalized_description' not in df.columns:
            df = df.copy()
            df['normalized_description'] = df['description'].apply(normalize)
        
        # Fill missing values
        df_filled = df.copy()
        df_filled['normalized_description'] = df_filled['normalized_description'].fillna('')
        df_filled['merchant_name'] = df_filled['merchant_name'].fillna('')
        df_filled['amount'] = df_filled['amount'].fillna(0)
        
        # Get text features
        descriptions = df_filled['normalized_description'].tolist()
        merchant_names = df_filled['merchant_name'].tolist()
        text_features = self._encode_text_features(descriptions, merchant_names)
        
        # Get numerical features
        amount_features = df_filled['amount'].values.reshape(-1, 1)
        
        # Get categorical features
        categorical_features = self._encode_categorical_features(
            df_filled['dr_cr'],
            df_filled['transaction_type'],
            fit=fit
        )
        
        # Combine all features
        all_features = np.concatenate([
            text_features,
            amount_features,
            categorical_features
        ], axis=1)
        
        # Build feature names
        self.feature_names = (
            [f'desc_emb_{i}' for i in range(self.embedding_dim)] +
            [f'merchant_emb_{i}' for i in range(self.embedding_dim)] +
            ['amount', 'dr_cr_encoded', 'transaction_type_encoded']
        )
        
        print(f"Total features: {all_features.shape[1]}")
        
        # Get labels if fitting
        labels = None
        if fit and 'category' in df.columns:
            self.label_encoder = LabelEncoder()
            labels = self.label_encoder.fit_transform(df['category'].fillna('Unknown'))
            self.lgbm_params['num_class'] = len(self.label_encoder.classes_)
            print(f"Number of categories: {len(self.label_encoder.classes_)}")
            print(f"Categories: {list(self.label_encoder.classes_)}")
        
        return all_features, labels
    
    def prepare_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[lgb.Dataset, lgb.Dataset, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for LightGBM training.
        
        Args:
            df: DataFrame with required columns.
            
        Returns:
            Tuple of (train_dataset, val_dataset, X_val, y_train, y_val).
        """
        print("Preparing data for LightGBM training...")
        
        # Prepare features and labels
        X, y = self._prepare_features(df, fit=True)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        
        # Create LightGBM datasets
        train_dataset = lgb.Dataset(
            X_train, 
            label=y_train,
            feature_name=self.feature_names,
            categorical_feature=['dr_cr_encoded', 'transaction_type_encoded']
        )
        val_dataset = lgb.Dataset(
            X_val, 
            label=y_val,
            reference=train_dataset,
            feature_name=self.feature_names,
            categorical_feature=['dr_cr_encoded', 'transaction_type_encoded']
        )
        
        return train_dataset, val_dataset, X_val, y_train, y_val
    
    def train(
        self, 
        df: pd.DataFrame,
        num_boost_round: int = 500,
        early_stopping_rounds: int = 50
    ) -> Dict[str, float]:
        """
        Train the LightGBM model.
        
        Args:
            df: DataFrame with required columns.
            num_boost_round: Maximum number of boosting rounds.
            early_stopping_rounds: Early stopping patience.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        # Load embedding model
        self._load_embedding_model()
        
        # Prepare data
        train_dataset, val_dataset, X_val, y_train, y_val = self.prepare_data(df)
        
        # Train model
        print("\nStarting LightGBM training...")
        callbacks = [
            lgb.early_stopping(early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=50)
        ]
        
        self.lgbm_model = lgb.train(
            self.lgbm_params,
            train_dataset,
            num_boost_round=num_boost_round,
            valid_sets=[train_dataset, val_dataset],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
        
        # Evaluate on validation set
        print("\nEvaluating model...")
        val_preds_proba = self.lgbm_model.predict(
            X_val,
            num_iteration=self.lgbm_model.best_iteration
        )
        val_preds = np.argmax(val_preds_proba, axis=1)
        
        accuracy = accuracy_score(y_val, val_preds)
        f1_weighted = f1_score(y_val, val_preds, average='weighted')
        f1_macro = f1_score(y_val, val_preds, average='macro')
        
        print(f"\nValidation Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (weighted): {f1_weighted:.4f}")
        print(f"  F1 (macro): {f1_macro:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(
            y_val, 
            val_preds,
            target_names=self.label_encoder.classes_
        ))
        
        # Save model
        self.save_model()
        
        metrics = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'best_iteration': self.lgbm_model.best_iteration
        }
        
        return metrics
    
    def save_model(self) -> None:
        """Save all model components."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"\nSaving models to: {self.output_dir}")
        
        # Save LightGBM model
        lgbm_path = os.path.join(self.output_dir, 'lightgbm_model.txt')
        self.lgbm_model.save_model(lgbm_path)
        print(f"  - LightGBM model saved: {lgbm_path}")
        
        # Save label encoder
        label_encoder_path = os.path.join(self.output_dir, 'label_encoder.pkl')
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"  - Label encoder saved: {label_encoder_path}")
        
        # Save categorical encoders
        encoders_path = os.path.join(self.output_dir, 'categorical_encoders.pkl')
        with open(encoders_path, 'wb') as f:
            pickle.dump({
                'dr_cr_encoder': self.dr_cr_encoder,
                'transaction_type_encoder': self.transaction_type_encoder
            }, f)
        print(f"  - Categorical encoders saved: {encoders_path}")
        
        # Save model config
        config = {
            'embedding_model_name': self.embedding_model_name,
            'embedding_dim': self.embedding_dim,
            'feature_names': self.feature_names,
            'categories': list(self.label_encoder.classes_),
            'dr_cr_classes': list(self.dr_cr_encoder.classes_),
            'transaction_type_classes': list(self.transaction_type_encoder.classes_),
            'default_transaction_type': self.default_transaction_type,
            'lgbm_params': self.lgbm_params
        }
        config_path = os.path.join(self.output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  - Config saved: {config_path}")
        
        print("Category classification model saved successfully!")


def train_category_model(
    data_path: str,
    output_dir: str = 'models/category',
    embedding_model_name: str = 'all-MiniLM-L6-v2',
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50,
    **kwargs
) -> Tuple[CategoryLightGBMTrainer, Dict[str, float]]:
    """
    Convenience function to train category classification model from a CSV file.
    
    Args:
        data_path: Path to CSV file with required columns.
        output_dir: Directory to save the trained models.
        embedding_model_name: SentenceTransformer model name.
        num_boost_round: Maximum number of boosting rounds.
        early_stopping_rounds: Early stopping patience.
        **kwargs: Additional arguments for CategoryLightGBMTrainer.
        
    Returns:
        Tuple of (trained trainer instance, metrics dictionary).
    """
    df = pd.read_csv(data_path)
    
    # Check required columns
    required_cols = ['normalized_description', 'merchant_name', 'amount', 
                     'dr_cr', 'transaction_type', 'category']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    trainer = CategoryLightGBMTrainer(
        embedding_model_name=embedding_model_name,
        output_dir=output_dir,
        **kwargs
    )
    
    metrics = trainer.train(
        df,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds
    )
    
    return trainer, metrics

