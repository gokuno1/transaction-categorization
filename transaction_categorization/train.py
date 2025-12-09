"""
Model Training Script for Transaction Categorization

This script trains the merchant name extraction models (NER and Seq2Seq)
and category classification model (LightGBM + Text Embeddings).
Run this script to train models before using main.py for inference.

Usage:
    python -m transaction_categorization.train --model ner --data path/to/data.csv
    python -m transaction_categorization.train --model seq2seq --data path/to/data.csv
    python -m transaction_categorization.train --model category --data path/to/data.csv
    python -m transaction_categorization.train --model all --data path/to/data.csv
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from transaction_categorization_v2.merchant_name_classification.ner_trainer import (
    NERMerchantTrainer,
    train_ner_model
)
from transaction_categorization_v2.merchant_name_classification.seq2seq_trainer import (
    Seq2SeqMerchantTrainer,
    train_seq2seq_model
)
from transaction_categorization_v2.category_classification import (
    CategoryLightGBMTrainer,
    train_category_model
)


# Default model output directories
DEFAULT_NER_MODEL_DIR = 'models/merchant_name/ner_model'
DEFAULT_SEQ2SEQ_MODEL_DIR = 'models/merchant_name/seq2seq_model'
DEFAULT_CATEGORY_MODEL_DIR = 'models/category'


def train_ner(
    data_path: str,
    output_dir: str = DEFAULT_NER_MODEL_DIR,
    num_epochs: int = 3,
    batch_size: int = 16
) -> None:
    """
    Train the NER model for merchant name extraction.
    
    Args:
        data_path: Path to training data CSV.
        output_dir: Directory to save the trained model.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
    """
    print("=" * 60)
    print("TRAINING NER MODEL")
    print("=" * 60)
    
    trainer = train_ner_model(
        data_path=data_path,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    print("\nNER Model training completed!")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)


def train_seq2seq(
    data_path: str,
    output_dir: str = DEFAULT_SEQ2SEQ_MODEL_DIR,
    num_epochs: int = 3,
    batch_size: int = 8
) -> None:
    """
    Train the Seq2Seq model for merchant name extraction.
    
    Args:
        data_path: Path to training data CSV.
        output_dir: Directory to save the trained model.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
    """
    print("=" * 60)
    print("TRAINING SEQ2SEQ MODEL")
    print("=" * 60)
    
    trainer = train_seq2seq_model(
        data_path=data_path,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size
    )
    
    print("\nSeq2Seq Model training completed!")
    print(f"Model saved to: {output_dir}")
    print("=" * 60)


def train_category(
    data_path: str,
    output_dir: str = DEFAULT_CATEGORY_MODEL_DIR,
    embedding_model: str = 'all-MiniLM-L6-v2',
    num_boost_round: int = 500,
    early_stopping_rounds: int = 50
) -> None:
    """
    Train the LightGBM + Text Embeddings model for category classification.
    
    Args:
        data_path: Path to training data CSV.
        output_dir: Directory to save the trained model.
        embedding_model: SentenceTransformer model name for text embeddings.
        num_boost_round: Maximum number of boosting rounds.
        early_stopping_rounds: Early stopping patience.
    """
    print("=" * 60)
    print("TRAINING CATEGORY CLASSIFICATION MODEL")
    print("=" * 60)
    
    trainer, metrics = train_category_model(
        data_path=data_path,
        output_dir=output_dir,
        embedding_model_name=embedding_model,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds
    )
    
    print("\nCategory Classification Model training completed!")
    print(f"Model saved to: {output_dir}")
    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation F1 (weighted): {metrics['f1_weighted']:.4f}")
    print("=" * 60)


def train_all(
    data_path: str,
    ner_output_dir: str = DEFAULT_NER_MODEL_DIR,
    seq2seq_output_dir: str = DEFAULT_SEQ2SEQ_MODEL_DIR,
    category_output_dir: str = DEFAULT_CATEGORY_MODEL_DIR,
    num_epochs: int = 3
) -> None:
    """
    Train all models: NER, Seq2Seq, and Category Classification.
    
    Args:
        data_path: Path to training data CSV.
        ner_output_dir: Directory to save the NER model.
        seq2seq_output_dir: Directory to save the Seq2Seq model.
        category_output_dir: Directory to save the category model.
        num_epochs: Number of training epochs.
    """
    print("\n" + "=" * 60)
    print("TRAINING ALL MODELS")
    print("=" * 60 + "\n")
    
    # Train NER model
    train_ner(data_path, ner_output_dir, num_epochs=num_epochs)
    
    print("\n")
    
    # Train Seq2Seq model
    train_seq2seq(data_path, seq2seq_output_dir, num_epochs=num_epochs)
    
    print("\n")
    
    # Train Category Classification model
    train_category(data_path, category_output_dir)
    
    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nNER Model: {ner_output_dir}")
    print(f"Seq2Seq Model: {seq2seq_output_dir}")
    print(f"Category Model: {category_output_dir}")


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description='Train transaction categorization models'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        choices=['ner', 'seq2seq', 'category', 'all'],
        default='all',
        help='Model to train: ner, seq2seq, category, or all (default: all)'
    )
    parser.add_argument(
        '-d', '--data',
        type=str,
        required=True,
        help='Path to training data CSV'
    )
    parser.add_argument(
        '--ner-output',
        type=str,
        default=DEFAULT_NER_MODEL_DIR,
        help=f'Output directory for NER model (default: {DEFAULT_NER_MODEL_DIR})'
    )
    parser.add_argument(
        '--seq2seq-output',
        type=str,
        default=DEFAULT_SEQ2SEQ_MODEL_DIR,
        help=f'Output directory for Seq2Seq model (default: {DEFAULT_SEQ2SEQ_MODEL_DIR})'
    )
    parser.add_argument(
        '--category-output',
        type=str,
        default=DEFAULT_CATEGORY_MODEL_DIR,
        help=f'Output directory for category model (default: {DEFAULT_CATEGORY_MODEL_DIR})'
    )
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=3,
        help='Number of training epochs for NER/Seq2Seq (default: 3)'
    )
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=None,
        help='Training batch size (default: 16 for NER, 8 for Seq2Seq)'
    )
    parser.add_argument(
        '--num-boost-round',
        type=int,
        default=500,
        help='Max boosting rounds for LightGBM (default: 500)'
    )
    parser.add_argument(
        '--early-stopping',
        type=int,
        default=50,
        help='Early stopping rounds for LightGBM (default: 50)'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='SentenceTransformer model for text embeddings (default: all-MiniLM-L6-v2)'
    )
    
    args = parser.parse_args()
    
    # Resolve data path
    data_path = args.data
    if not os.path.isabs(data_path):
        script_dir = Path(__file__).parent.parent
        candidate_path = script_dir / data_path
        if candidate_path.exists():
            data_path = str(candidate_path)
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    
    # Resolve output paths
    ner_output = args.ner_output
    seq2seq_output = args.seq2seq_output
    category_output = args.category_output
    
    script_dir = Path(__file__).parent.parent
    
    if not os.path.isabs(ner_output):
        ner_output = str(script_dir / ner_output)
    
    if not os.path.isabs(seq2seq_output):
        seq2seq_output = str(script_dir / seq2seq_output)
    
    if not os.path.isabs(category_output):
        category_output = str(script_dir / category_output)
    
    # Create output directories
    os.makedirs(ner_output, exist_ok=True)
    os.makedirs(seq2seq_output, exist_ok=True)
    os.makedirs(category_output, exist_ok=True)
    
    print(f"\nData file: {data_path}")
    print(f"NER output: {ner_output}")
    print(f"Seq2Seq output: {seq2seq_output}")
    print(f"Category output: {category_output}")
    print(f"Epochs: {args.epochs}")
    print()
    
    # Train models
    if args.model == 'ner':
        batch_size = args.batch_size if args.batch_size else 16
        train_ner(data_path, ner_output, num_epochs=args.epochs, batch_size=batch_size)
    elif args.model == 'seq2seq':
        batch_size = args.batch_size if args.batch_size else 8
        train_seq2seq(data_path, seq2seq_output, num_epochs=args.epochs, batch_size=batch_size)
    elif args.model == 'category':
        train_category(
            data_path, 
            category_output,
            embedding_model=args.embedding_model,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping
        )
    else:  # all
        train_all(data_path, ner_output, seq2seq_output, category_output, num_epochs=args.epochs)


if __name__ == '__main__':
    main()

