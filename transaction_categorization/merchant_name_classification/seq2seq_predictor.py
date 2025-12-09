"""
Seq2Seq Model Predictor for Merchant Name Extraction

This module provides functionality to load a trained Seq2Seq model
and generate merchant names from transaction descriptions.
"""

import os
import re
from typing import Dict, List, Optional
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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


class Seq2SeqMerchantPredictor:
    """
    Predictor class for Seq2Seq-based merchant name extraction.
    
    Loads a trained T5 model and generates merchant names
    from transaction descriptions.
    """
    
    def __init__(
        self,
        model_path: str = 'models/merchant_name/seq2seq_model',
        max_input_length: int = 128,
        max_output_length: int = 32,
        num_beams: int = 4
    ):
        """
        Initialize the Seq2Seq predictor.
        
        Args:
            model_path: Path to the trained model directory.
            max_input_length: Maximum input sequence length.
            max_output_length: Maximum output sequence length.
            num_beams: Number of beams for beam search.
        """
        self.model_path = model_path
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_beams = num_beams
        
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
        
        print(f"Loading Seq2Seq model from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        self._is_loaded = True
        print("Seq2Seq model loaded successfully!")
    
    def _create_input_text(
        self,
        normalized_description: str,
        amount: Optional[float] = None,
        dr_cr: Optional[str] = None
    ) -> str:
        """
        Create formatted input text for the model.
        
        Args:
            normalized_description: Normalized transaction description.
            amount: Transaction amount (optional).
            dr_cr: Debit/Credit indicator (optional).
            
        Returns:
            Formatted input string.
        """
        amt = amount if amount is not None else 0.0
        txn_type = dr_cr if dr_cr else "UNKNOWN"
        return f"extract_merchant: DESC: {normalized_description} | AMT: {amt} | TYPE: {txn_type}"
    
    def predict(
        self,
        text: str,
        amount: Optional[float] = None,
        dr_cr: Optional[str] = None
    ) -> Dict:
        """
        Extract merchant name from a transaction description.
        
        Args:
            text: Raw transaction description text.
            amount: Transaction amount (optional, improves accuracy).
            dr_cr: Debit/Credit indicator (optional, improves accuracy).
            
        Returns:
            Dictionary with normalized_description and merchant_name.
        """
        if not self._is_loaded:
            self.load_model()
        
        # Normalize text
        normalized_text = normalize(text)
        
        # Create input text
        input_text = self._create_input_text(normalized_text, amount, dr_cr)
        
        # Tokenize
        tokenized = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_output_length,
                num_beams=self.num_beams,
                early_stopping=True
            )
        
        # Decode
        merchant_name = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        
        return {
            'normalized_description': normalized_text,
            'merchant_name': merchant_name
        }
    
    def predict_batch(
        self,
        texts: List[str],
        amounts: Optional[List[float]] = None,
        dr_crs: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Extract merchant names from multiple transaction descriptions.
        
        Args:
            texts: List of raw transaction description texts.
            amounts: List of transaction amounts (optional).
            dr_crs: List of Debit/Credit indicators (optional).
            
        Returns:
            List of dictionaries with normalized_description and merchant_name.
        """
        if not self._is_loaded:
            self.load_model()
        
        # Normalize texts and create inputs
        normalized_texts = [normalize(text) for text in texts]
        
        if amounts is None:
            amounts = [None] * len(texts)
        if dr_crs is None:
            dr_crs = [None] * len(texts)
        
        input_texts = [
            self._create_input_text(norm_text, amt, drcr)
            for norm_text, amt, drcr in zip(normalized_texts, amounts, dr_crs)
        ]
        
        # Tokenize batch
        tokenized = self.tokenizer(
            input_texts,
            max_length=self.max_input_length,
            truncation=True,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_output_length,
                num_beams=self.num_beams,
                early_stopping=True
            )
        
        # Decode
        merchant_names = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return [
            {
                'normalized_description': norm_text,
                'merchant_name': merchant_name
            }
            for norm_text, merchant_name in zip(normalized_texts, merchant_names)
        ]
    
    def get_merchant_name(
        self,
        text: str,
        amount: Optional[float] = None,
        dr_cr: Optional[str] = None
    ) -> str:
        """
        Get just the merchant name for a transaction description.
        
        Args:
            text: Raw transaction description text.
            amount: Transaction amount (optional).
            dr_cr: Debit/Credit indicator (optional).
            
        Returns:
            Extracted merchant name string.
        """
        return self.predict(text, amount, dr_cr)['merchant_name']

