"""
Seq2Seq Model Trainer for Merchant Name Extraction

This module provides functionality to train a T5-based Seq2Seq model
for generating merchant names from transaction descriptions.
"""

import os
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset


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


def create_input_text(row: pd.Series) -> str:
    """
    Create formatted input text for the Seq2Seq model.
    
    Args:
        row: DataFrame row with transaction details.
        
    Returns:
        Formatted input string for the model.
    """
    desc = row.get('normalized_description', '')
    amt = row.get('amount', '')
    drcr = row.get('dr_cr', '')
    return f"extract_merchant: DESC: {desc} | AMT: {amt} | TYPE: {drcr}"


class Seq2SeqMerchantTrainer:
    """
    Trainer class for Seq2Seq-based merchant name extraction model.
    
    Uses T5-small for generating merchant names from transaction descriptions.
    """
    
    def __init__(
        self,
        model_name: str = 't5-small',
        output_dir: str = 'models/merchant_name/seq2seq_model',
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        max_input_length: int = 128,
        max_target_length: int = 32,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize the Seq2Seq trainer.
        
        Args:
            model_name: Pre-trained model name from HuggingFace.
            output_dir: Directory to save the trained model.
            num_epochs: Number of training epochs.
            batch_size: Training and evaluation batch size.
            learning_rate: Learning rate for optimization.
            warmup_steps: Number of warmup steps.
            weight_decay: Weight decay for regularization.
            max_input_length: Maximum input sequence length.
            max_target_length: Maximum target sequence length.
            test_size: Fraction of data to use for validation.
            random_state: Random seed for reproducibility.
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.test_size = test_size
        self.random_state = random_state
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
    
    def _tokenize_examples(self, examples: Dict) -> Dict:
        """
        Tokenize examples for Seq2Seq training.
        
        Args:
            examples: Dictionary with 'input_text' and 'target_text' keys.
            
        Returns:
            Dictionary with tokenized inputs and labels.
        """
        model_inputs = self.tokenizer(
            examples['input_text'],
            max_length=self.max_input_length,
            truncation=True
        )
        
        # Clean target texts
        target_texts_cleaned = [
            str(t) if t is not None else "" 
            for t in examples['target_text']
        ]
        
        labels = self.tokenizer(
            text_target=target_texts_cleaned,
            max_length=self.max_target_length,
            truncation=True
        )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[Dataset, Dataset]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with transaction data and merchant names.
            
        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        print("Preparing data for Seq2Seq training...")
        
        # Normalize descriptions
        df = df.copy()
        df['normalized_description'] = df['description'].apply(normalize)
        
        # Create input and target texts
        df['input_text'] = df.apply(create_input_text, axis=1)
        df['target_text'] = df['merchant_name']
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_pandas(df[['input_text', 'target_text']].reset_index(drop=True))
        
        # Split data
        split_dataset = dataset.train_test_split(
            test_size=self.test_size, 
            seed=self.random_state
        )
        
        # Tokenize datasets
        # When we call .map() on the dataset, the _tokenize_examples function converts the raw text columns (input_text, target_text)
        # into tokenized formats (input_ids, labels, etc.).
        # Keeping them would -Waste memory due to duplicate data and cause issues with DataLoader-the trainer expects numerical tensors,
        # not string columns
        columns_to_remove = ['input_text', 'target_text']
        
        train_dataset = split_dataset['train'].map(
            self._tokenize_examples,
            batched=True,
            remove_columns=columns_to_remove
        )
        
        val_dataset = split_dataset['test'].map(
            self._tokenize_examples,
            batched=True,
            remove_columns=columns_to_remove
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train(self, df: pd.DataFrame) -> None:
        """
        Train the Seq2Seq model.
        
        Args:
            df: DataFrame with transaction data and merchant names.
        """
        # Prepare data
        train_dataset, val_dataset = self.prepare_data(df)
        
        # Load model
        print(f"Loading model: {self.model_name}")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=10,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            predict_with_generate=True,
            report_to=['none']
        )
        
        # Initialize trainer
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator
        )
        
        # Train
        print("Starting Seq2Seq model training...")
        self.trainer.train()
        
        # Save model
        self.save_model()
    
    def save_model(self) -> None:
        """Save the trained model and tokenizer."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Saving Seq2Seq model to: {self.output_dir}")
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print("Seq2Seq model saved successfully!")


def train_seq2seq_model(
    data_path: str,
    output_dir: str = 'models/merchant_name/seq2seq_model',
    **kwargs
) -> Seq2SeqMerchantTrainer:
    """
    Convenience function to train Seq2Seq model from a CSV file.
    
    Args:
        data_path: Path to CSV file with transaction data.
        output_dir: Directory to save the trained model.
        **kwargs: Additional arguments for Seq2SeqMerchantTrainer.
        
    Returns:
        Trained Seq2SeqMerchantTrainer instance.
    """
    df = pd.read_csv(data_path)
    
    # Check required columns
    required_cols = ['description', 'merchant_name']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    trainer = Seq2SeqMerchantTrainer(output_dir=output_dir, **kwargs)
    trainer.train(df)
    
    return trainer

