"""
LightGBM + Text Embeddings Predictor for Transaction Category Classification

This module provides functionality to load trained models and make predictions
for transaction category classification using the hybrid LightGBM + Text Embeddings approach.
"""

import os
import pickle
import json
import re
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import pandas as pd
import numpy as np

import lightgbm as lgb
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder


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


class CategoryLightGBMPredictor:
    """
    Predictor class for LightGBM + Text Embeddings based category classification.
    
    Loads pre-trained models and provides inference capabilities.
    """
    
    def __init__(self, model_path: str = 'models/category'):
        """
        Initialize the Category LightGBM predictor.
        
        Args:
            model_path: Path to the directory containing trained model files.
        """
        self.model_path = model_path
        
        self.embedding_model: Optional[SentenceTransformer] = None
        self.lgbm_model: Optional[lgb.Booster] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.dr_cr_encoder: Optional[LabelEncoder] = None
        self.transaction_type_encoder: Optional[LabelEncoder] = None
        self.config: Dict[str, Any] = {}
        self.default_transaction_type: str = 'UPI'  # Fallback for unseen transaction types
        
        self._load_models()
    
    def _load_models(self) -> None:
        """Load all model components from disk."""
        print(f"Loading category classification models from: {self.model_path}")
        
        # Load config
        config_path = os.path.join(self.model_path, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load embedding model
        print(f"  Loading embedding model: {self.config['embedding_model_name']}")
        self.embedding_model = SentenceTransformer(self.config['embedding_model_name'])
        
        # Load LightGBM model
        lgbm_path = os.path.join(self.model_path, 'lightgbm_model.txt')
        if not os.path.exists(lgbm_path):
            raise FileNotFoundError(f"LightGBM model not found: {lgbm_path}")
        
        print(f"  Loading LightGBM model...")
        self.lgbm_model = lgb.Booster(model_file=lgbm_path)
        
        # Load label encoder
        label_encoder_path = os.path.join(self.model_path, 'label_encoder.pkl')
        if not os.path.exists(label_encoder_path):
            raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")
        
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load categorical encoders
        encoders_path = os.path.join(self.model_path, 'categorical_encoders.pkl')
        if not os.path.exists(encoders_path):
            raise FileNotFoundError(f"Categorical encoders not found: {encoders_path}")
        
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
            self.dr_cr_encoder = encoders['dr_cr_encoder']
            self.transaction_type_encoder = encoders['transaction_type_encoder']
        
        # Load default transaction type for unseen values
        self.default_transaction_type = self.config.get('default_transaction_type', 'UPI')
        
        print(f"  Loaded {len(self.config['categories'])} categories")
        print(f"  Default transaction type for unseen values: {self.default_transaction_type}")
        print("Category classification models loaded successfully!")
    
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
            Combined embeddings array.
        """
        # Encode descriptions
        desc_embeddings = self.embedding_model.encode(
            descriptions,
            show_progress_bar=False,
            batch_size=32,
            convert_to_numpy=True
        )
        
        # Encode merchant names
        merchant_embeddings = self.embedding_model.encode(
            merchant_names,
            show_progress_bar=False,
            batch_size=32,
            convert_to_numpy=True
        )
        
        # Concatenate embeddings
        text_features = np.concatenate([desc_embeddings, merchant_embeddings], axis=1)
        return text_features
    
    def _encode_categorical_features(
        self,
        dr_cr: List[str],
        transaction_type: List[str]
    ) -> np.ndarray:
        """
        Encode categorical features using pre-trained encoders.
        
        For unseen transaction_type values, uses the default_transaction_type 
        (most common type from training).
        
        Args:
            dr_cr: List of DR/CR values.
            transaction_type: List of transaction types.
            
        Returns:
            Encoded categorical features array.
        """
        # Get default transaction type encoding
        default_txn_type_encoded = self.transaction_type_encoder.transform(
            [self.default_transaction_type]
        )[0]
        
        # Encode dr_cr (only DR/CR values, simple encoding)
        dr_cr_encoded = np.array([
            self.dr_cr_encoder.transform([val])[0] 
            if val in self.dr_cr_encoder.classes_ else 0  # Default to first class
            for val in dr_cr
        ])
        
        # Encode transaction_type - use default for unseen values
        txn_type_encoded = np.array([
            self.transaction_type_encoder.transform([val])[0]
            if val in self.transaction_type_encoder.classes_ 
            else default_txn_type_encoded  # Use default (most common) for unseen types
            for val in transaction_type
        ])
        
        categorical_features = np.column_stack([dr_cr_encoded, txn_type_encoded])
        return categorical_features
    
    def _prepare_features(
        self,
        descriptions: List[str],
        merchant_names: List[str],
        amounts: List[float],
        dr_crs: List[str],
        transaction_types: List[str]
    ) -> np.ndarray:
        """
        Prepare all features for prediction.
        
        Args:
            descriptions: List of normalized descriptions.
            merchant_names: List of merchant names.
            amounts: List of amounts.
            dr_crs: List of DR/CR values.
            transaction_types: List of transaction types.
            
        Returns:
            Features array ready for prediction.
        """
        # Get text features
        text_features = self._encode_text_features(descriptions, merchant_names)
        
        # Get numerical features
        amount_features = np.array(amounts).reshape(-1, 1)
        
        # Get categorical features
        categorical_features = self._encode_categorical_features(dr_crs, transaction_types)
        
        # Combine all features
        all_features = np.concatenate([
            text_features,
            amount_features,
            categorical_features
        ], axis=1)
        
        return all_features
    
    def predict(
        self,
        normalized_description: str,
        merchant_name: str,
        amount: float,
        dr_cr: str,
        transaction_type: str
    ) -> Dict[str, Any]:
        """
        Predict category for a single transaction.
        
        Args:
            normalized_description: Normalized transaction description.
            merchant_name: Extracted merchant name.
            amount: Transaction amount.
            dr_cr: Debit/Credit indicator.
            transaction_type: Transaction type (UPI, NEFT, etc.).
            
        Returns:
            Dictionary with predicted category and confidence.
        """
        # Prepare features
        features = self._prepare_features(
            descriptions=[normalized_description or ''],
            merchant_names=[merchant_name or ''],
            amounts=[amount or 0],
            dr_crs=[dr_cr or 'UNKNOWN'],
            transaction_types=[transaction_type or 'UNKNOWN']
        )
        
        # Predict
        proba = self.lgbm_model.predict(features)[0]
        predicted_idx = np.argmax(proba)
        predicted_category = self.label_encoder.inverse_transform([predicted_idx])[0]
        confidence = float(proba[predicted_idx])
        
        return {
            'category': predicted_category,
            'confidence': confidence,
            'probabilities': {
                cat: float(p) 
                for cat, p in zip(self.label_encoder.classes_, proba)
            }
        }
    
    def predict_batch(
        self,
        normalized_descriptions: List[str],
        merchant_names: List[str],
        amounts: List[float],
        dr_crs: List[str],
        transaction_types: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Predict categories for a batch of transactions.
        
        Args:
            normalized_descriptions: List of normalized descriptions.
            merchant_names: List of merchant names.
            amounts: List of amounts.
            dr_crs: List of DR/CR values.
            transaction_types: List of transaction types.
            
        Returns:
            List of prediction dictionaries.
        """
        # Fill missing values
        n = len(normalized_descriptions)
        descriptions = [d or '' for d in normalized_descriptions]
        merchants = [m or '' for m in merchant_names]
        amts = [a or 0 for a in amounts]
        drs = [d or 'UNKNOWN' for d in dr_crs]
        txn_types = [t or 'UNKNOWN' for t in transaction_types]
        
        # Prepare features
        features = self._prepare_features(descriptions, merchants, amts, drs, txn_types)
        
        # Predict
        probas = self.lgbm_model.predict(features)
        predicted_indices = np.argmax(probas, axis=1)
        predicted_categories = self.label_encoder.inverse_transform(predicted_indices)
        
        results = []
        for i in range(n):
            results.append({
                'category': predicted_categories[i],
                'confidence': float(probas[i][predicted_indices[i]]),
                'probabilities': {
                    cat: float(p) 
                    for cat, p in zip(self.label_encoder.classes_, probas[i])
                }
            })
        
        return results
    
    def predict_dataframe(
        self,
        df: pd.DataFrame,
        description_col: str = 'normalized_description',
        merchant_col: str = 'merchant_name',
        amount_col: str = 'amount',
        dr_cr_col: str = 'dr_cr',
        txn_type_col: str = 'transaction_type',
        include_confidence: bool = True
    ) -> pd.DataFrame:
        """
        Predict categories for a DataFrame of transactions.
        
        Args:
            df: Input DataFrame with required columns.
            description_col: Column name for normalized descriptions.
            merchant_col: Column name for merchant names.
            amount_col: Column name for amounts.
            dr_cr_col: Column name for DR/CR values.
            txn_type_col: Column name for transaction types.
            include_confidence: Whether to include confidence scores.
            
        Returns:
            DataFrame with added prediction columns.
        """
        result_df = df.copy()
        
        # Get predictions
        predictions = self.predict_batch(
            normalized_descriptions=df[description_col].tolist(),
            merchant_names=df[merchant_col].tolist(),
            amounts=df[amount_col].tolist(),
            dr_crs=df[dr_cr_col].tolist(),
            transaction_types=df[txn_type_col].tolist()
        )
        
        # Add predictions to DataFrame
        result_df['predicted_category'] = [p['category'] for p in predictions]
        
        if include_confidence:
            result_df['category_confidence'] = [p['confidence'] for p in predictions]
        
        return result_df
    
    def get_category(
        self,
        normalized_description: str,
        merchant_name: str,
        amount: float,
        dr_cr: str,
        transaction_type: str
    ) -> str:
        """
        Get the predicted category for a single transaction.
        
        Args:
            normalized_description: Normalized transaction description.
            merchant_name: Extracted merchant name.
            amount: Transaction amount.
            dr_cr: Debit/Credit indicator.
            transaction_type: Transaction type.
            
        Returns:
            Predicted category string.
        """
        result = self.predict(
            normalized_description=normalized_description,
            merchant_name=merchant_name,
            amount=amount,
            dr_cr=dr_cr,
            transaction_type=transaction_type
        )
        return result['category']
    
    @property
    def categories(self) -> List[str]:
        """Get list of all supported categories."""
        return list(self.label_encoder.classes_)

