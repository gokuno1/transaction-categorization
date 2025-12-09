"""
NER Model Trainer for Merchant Name Extraction

This module provides functionality to train a BERT-based NER model
for identifying merchant names in transaction descriptions using BIO tagging.
"""

import os
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from datasets import Dataset


# BIO Labels for merchant entity
LABELS = ["O", "B-MERCHANT", "I-MERCHANT"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
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


def find_merchant_positions(row: pd.Series) -> Tuple[int, int]:
    """
    Find the character positions of the merchant name in the normalized description.
    
    Args:
        row: DataFrame row containing 'normalized_description' and 'merchant_name'.
        
    Returns:
        Tuple of (start_pos, end_pos) or (-1, -1) if not found.
    """
    description = row.get('normalized_description', '')
    merchant_name = row.get('merchant_name', '')
    
    if not isinstance(description, str) or not isinstance(merchant_name, str):
        return -1, -1
    
    normalized_merchant_name = normalize(merchant_name)
    start_pos = description.find(normalized_merchant_name)
    
    if start_pos != -1:
        end_pos = start_pos + len(normalized_merchant_name)
        return start_pos, end_pos
    return -1, -1


def tokenize_and_align_labels(
    row: pd.Series,
    tokenizer,
    label_to_id: Dict[str, int]
) -> Dict:
    """
    Tokenize text and align BIO labels with tokenized output.
    
    Args:
        row: DataFrame row with description and merchant position info.
        tokenizer: HuggingFace tokenizer.
        label_to_id: Mapping from label string to ID.
        
    Returns:
        Dictionary with input_ids and labels.
    """
    description = row['normalized_description']
    merchant_start_char = row['merchant_start_pos']
    merchant_end_char = row['merchant_end_pos']
    
    tokenized_input = tokenizer(
        description,
        truncation=True,
        return_offsets_mapping=True,
        return_attention_mask=False
    )
    
    word_ids = tokenized_input.word_ids()
    offset_mapping = tokenized_input['offset_mapping']
    input_ids = tokenized_input['input_ids']
    
    labels = []
    previous_word_idx = None
    
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            labels.append(-100)
        else:
            start_char, end_char = offset_mapping[token_idx]
            
            is_token_in_merchant_span = False
            if merchant_start_char != -1 and merchant_end_char != -1:
                if max(start_char, merchant_start_char) < min(end_char, merchant_end_char):
                    is_token_in_merchant_span = True
            
            if is_token_in_merchant_span:
                if word_idx != previous_word_idx:
                    labels.append(label_to_id['B-MERCHANT'])
                else:
                    labels.append(label_to_id['I-MERCHANT'])
            else:
                labels.append(label_to_id['O'])
        
        previous_word_idx = word_idx
    
    return {'input_ids': input_ids, 'labels': labels}


class NERMerchantTrainer:
    """
    Trainer class for NER-based merchant name extraction model.
    
    Uses BERT-base-cased for token classification with BIO tagging scheme.
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-cased',
        output_dir: str = 'models/merchant_name/ner_model',
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        test_size: float = 0.2,
        random_state: int = 42
    ):
        """
        Initialize the NER trainer.
        
        Args:
            model_name: Pre-trained model name from HuggingFace.
            output_dir: Directory to save the trained model.
            num_epochs: Number of training epochs.
            batch_size: Training and evaluation batch size.
            learning_rate: Learning rate for optimization.
            warmup_steps: Number of warmup steps.
            weight_decay: Weight decay for regularization.
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
        self.test_size = test_size
        self.random_state = random_state
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[Dataset, Dataset]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with 'description' and 'merchant_name' columns.
            
        Returns:
            Tuple of (train_dataset, val_dataset).
        """
        print("Preparing data for NER training...")
        
        # Normalize descriptions
        df = df.copy()
        df['normalized_description'] = df['description'].apply(normalize)
        
        # Find merchant positions
        df[['merchant_start_pos', 'merchant_end_pos']] = df.apply(
            find_merchant_positions, axis=1, result_type='expand'
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Tokenize and align labels
        tokenized_data = df.apply(
            lambda row: tokenize_and_align_labels(row, self.tokenizer, LABEL_TO_ID),
            axis=1
        )
        
        df['input_ids'] = tokenized_data.apply(lambda x: x['input_ids'])
        df['labels'] = tokenized_data.apply(lambda x: x['labels'])
        
        # Split data
        train_df, val_df = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state
        )
        
        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df[['input_ids', 'labels']].reset_index(drop=True))
        val_dataset = Dataset.from_pandas(val_df[['input_ids', 'labels']].reset_index(drop=True))
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train(self, df: pd.DataFrame) -> None:
        """
        Train the NER model.
        
        Args:
            df: DataFrame with 'description' and 'merchant_name' columns.
        """
        # Prepare data
        train_dataset, val_dataset = self.prepare_data(df)
        
        # Load model
        print(f"Loading model: {self.model_name}")
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(LABELS),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
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
            report_to=['none']
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator
        )
        
        # Train
        print("Starting NER model training...")
        self.trainer.train()
        
        # Save model
        self.save_model()
    
    def save_model(self) -> None:
        """Save the trained model and tokenizer."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Saving NER model to: {self.output_dir}")
        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print("NER model saved successfully!")


def train_ner_model(
    data_path: str,
    output_dir: str = 'models/merchant_name/ner_model',
    **kwargs
) -> NERMerchantTrainer:
    """
    Convenience function to train NER model from a CSV file.
    
    Args:
        data_path: Path to CSV file with 'description' and 'merchant_name' columns.
        output_dir: Directory to save the trained model.
        **kwargs: Additional arguments for NERMerchantTrainer.
        
    Returns:
        Trained NERMerchantTrainer instance.
    """
    df = pd.read_csv(data_path)
    
    # Check required columns
    required_cols = ['description', 'merchant_name']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    trainer = NERMerchantTrainer(output_dir=output_dir, **kwargs)
    trainer.train(df)
    
    return trainer

