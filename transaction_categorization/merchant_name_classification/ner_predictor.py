"""
NER Model Predictor for Merchant Name Extraction

This module provides functionality to load a trained NER model
and extract merchant names from transaction descriptions.
"""

import os
import re
from typing import Dict, List, Optional
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


# BIO Labels for merchant entity
LABELS = ["O", "B-MERCHANT", "I-MERCHANT"]
ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}


def normalize(text: str) -> str:
    """
    Normalize transaction text for better matching.
    
    Args:
        text: Raw transaction description text.
        
    Returns:
        Normalized uppercase text with standardized spacing.
    """
    text = text or ""
    text = text.upper()
    text = re.sub(r"[\/\-\_\*]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class NERMerchantPredictor:
    """
    Predictor class for NER-based merchant name extraction.
    
    Loads a trained BERT model and extracts merchant names
    from transaction descriptions using BIO tagging.
    """
    
    def __init__(self, model_path: str = 'models/merchant_name/ner_model'):
        """
        Initialize the NER predictor.
        
        Args:
            model_path: Path to the trained model directory.
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._is_loaded = False
    
    def load_model(self) -> None:
        """Load the trained model and tokenizer."""
        if self._is_loaded:
            return
            
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please train the model first using train.py"
            )
        
        print(f"Loading NER model from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True
        print("NER model loaded successfully!")
    
    def _extract_merchant_from_predictions(
        self,
        original_text: str,
        predictions: torch.Tensor,
        offset_mapping: List[tuple]
    ) -> str:
        """
        Extract merchant name from model predictions using character offsets.
        
        Args:
            original_text: Original normalized text.
            predictions: Model predictions tensor.
            offset_mapping: Token to character offset mapping.
            
        Returns:
            Extracted merchant name string.
        """
        predicted_labels = [ID_TO_LABEL[p.item()] for p in predictions.squeeze()]
        
        extracted_chars = []
        in_merchant_entity = False
        
        for token_idx in range(len(predicted_labels)):
            label = predicted_labels[token_idx]
            
            if token_idx >= len(offset_mapping):
                continue
                
            start_char, end_char = offset_mapping[token_idx]
            
            # Skip special tokens
            if start_char == 0 and end_char == 0:
                if token_idx == 0 or token_idx == len(predicted_labels) - 1:
                    continue
            
            if label == 'B-MERCHANT':
                if extracted_chars and extracted_chars[-1] != ' ':
                    if start_char > 0 and original_text[start_char - 1] == ' ':
                        extracted_chars.append(' ')
                extracted_chars.extend(list(original_text[start_char:end_char]))
                in_merchant_entity = True
                
            elif label == 'I-MERCHANT':
                if not in_merchant_entity:
                    if extracted_chars and extracted_chars[-1] != ' ':
                        if start_char > 0 and original_text[start_char - 1] == ' ':
                            extracted_chars.append(' ')
                elif extracted_chars:
                    if start_char > 0 and original_text[start_char - 1] == ' ':
                        if extracted_chars[-1] != ' ':
                            extracted_chars.append(' ')
                
                extracted_chars.extend(list(original_text[start_char:end_char]))
                in_merchant_entity = True
            else:
                in_merchant_entity = False
        
        return "".join(extracted_chars).strip()
    
    def predict(self, text: str) -> Dict:
        """
        Extract merchant name from a transaction description.
        
        Args:
            text: Raw transaction description text.
            
        Returns:
            Dictionary with normalized_description and merchant_name.
        """
        if not self._is_loaded:
            self.load_model()
        
        # Normalize text
        normalized_text = normalize(text)
        
        # Tokenize
        tokenized = self.tokenizer(
            normalized_text,
            return_tensors="pt",
            truncation=True,
            return_offsets_mapping=True
        )
        
        offset_mapping = tokenized['offset_mapping'].squeeze().tolist()
        
        # Move to device
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Extract merchant name
        merchant_name = self._extract_merchant_from_predictions(
            normalized_text, predictions, offset_mapping
        )
        
        return {
            'normalized_description': normalized_text,
            'merchant_name': merchant_name
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Extract merchant names from multiple transaction descriptions.
        
        Args:
            texts: List of raw transaction description texts.
            
        Returns:
            List of dictionaries with normalized_description and merchant_name.
        """
        return [self.predict(text) for text in texts]
    
    def get_merchant_name(self, text: str) -> str:
        """
        Get just the merchant name for a transaction description.
        
        Args:
            text: Raw transaction description text.
            
        Returns:
            Extracted merchant name string.
        """
        return self.predict(text)['merchant_name']

