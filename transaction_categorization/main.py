"""
Transaction Categorization - Main

This module serves as the main entry point for processing transaction data.
Implements:
- Transaction type identification (UPI, CARD, ACH, IMPS, NEFT, etc.)
- Merchant name identification (using NER or Seq2Seq models)
- Category classification (using LightGBM + Text Embeddings)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Literal

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from transaction_categorization_v2.transaction_type_classifier import (
    TransactionTypeClassifier,
    ClassificationResult
)
from transaction_categorization_v2.merchant_name_classification import (
    NERMerchantPredictor,
    Seq2SeqMerchantPredictor
)
from transaction_categorization_v2.category_classification import (
    CategoryLightGBMPredictor
)


# Default model paths
DEFAULT_NER_MODEL_PATH = 'models/merchant_name/ner_model'
DEFAULT_SEQ2SEQ_MODEL_PATH = 'models/merchant_name/seq2seq_model'
DEFAULT_CATEGORY_MODEL_PATH = 'models/category'


class TransactionProcessor:
    """
    Main processor for transaction categorization pipeline.
    
    This class orchestrates the complete transaction processing workflow
    including type identification, merchant extraction, and categorization.
    """
    
    def __init__(
        self,
        fuzzy_threshold: int = 75,
        merchant_model: Optional[Literal['ner', 'seq2seq']] = None,
        enable_category: bool = False,
        ner_model_path: str = DEFAULT_NER_MODEL_PATH,
        seq2seq_model_path: str = DEFAULT_SEQ2SEQ_MODEL_PATH,
        category_model_path: str = DEFAULT_CATEGORY_MODEL_PATH
    ):
        """
        Initialize the TransactionProcessor.
        
        Args:
            fuzzy_threshold: Minimum score for fuzzy matching in type classification.
            merchant_model: Model to use for merchant extraction ('ner', 'seq2seq', or None).
            enable_category: Whether to enable category classification.
            ner_model_path: Path to the trained NER model.
            seq2seq_model_path: Path to the trained Seq2Seq model.
            category_model_path: Path to the trained category classification model.
        """
        self.type_classifier = TransactionTypeClassifier(fuzzy_threshold=fuzzy_threshold)
        
        self.merchant_model = merchant_model
        self.merchant_predictor = None
        self.enable_category = enable_category
        self.category_predictor = None
        
        # Resolve model paths
        script_dir = Path(__file__).parent.parent
        
        if not os.path.isabs(ner_model_path):
            ner_model_path = str(script_dir / ner_model_path)
        if not os.path.isabs(seq2seq_model_path):
            seq2seq_model_path = str(script_dir / seq2seq_model_path)
        if not os.path.isabs(category_model_path):
            category_model_path = str(script_dir / category_model_path)
        
        self.ner_model_path = ner_model_path
        self.seq2seq_model_path = seq2seq_model_path
        self.category_model_path = category_model_path
        
        # Initialize merchant predictor if specified
        if merchant_model:
            self._init_merchant_predictor()
        
        # Initialize category predictor if enabled
        if enable_category:
            self._init_category_predictor()
    
    def _init_merchant_predictor(self) -> None:
        """Initialize the merchant name predictor based on selected model."""
        if self.merchant_model == 'ner':
            self.merchant_predictor = NERMerchantPredictor(model_path=self.ner_model_path)
        elif self.merchant_model == 'seq2seq':
            self.merchant_predictor = Seq2SeqMerchantPredictor(model_path=self.seq2seq_model_path)
    
    def _init_category_predictor(self) -> None:
        """Initialize the category classification predictor."""
        self.category_predictor = CategoryLightGBMPredictor(model_path=self.category_model_path)
    
    def process_single_transaction(
        self,
        description: str,
        amount: Optional[float] = None,
        dr_cr: Optional[str] = None,
        merchant_name: Optional[str] = None
    ) -> dict:
        """
        Process a single transaction description.
        
        Args:
            description: Raw transaction description text.
            amount: Transaction amount (used for seq2seq model and category prediction).
            dr_cr: Debit/Credit indicator (used for seq2seq model and category prediction).
            merchant_name: Pre-extracted merchant name (optional, used for category prediction).
            
        Returns:
            Dictionary containing:
                - normalized_description: Cleaned/normalized text
                - transaction_type: Identified transaction type
                - merchant_name: Extracted merchant name (if model is configured)
                - category: Predicted category (if enabled)
        """
        # Normalize the description
        normalized = self.type_classifier.normalize(description)
        
        # Classify transaction type
        classification = self.type_classifier.classify(description)
        
        result = {
            'normalized_description': normalized,
            'transaction_type': classification.transaction_type,
            'type_matched': classification.matched,
            'type_score': classification.score,
            'type_rule': classification.rule
        }
        
        # Extract merchant name if model is configured
        extracted_merchant = merchant_name
        if self.merchant_predictor:
            if self.merchant_model == 'seq2seq':
                merchant_result = self.merchant_predictor.predict(description, amount, dr_cr)
            else:
                merchant_result = self.merchant_predictor.predict(description)
            extracted_merchant = merchant_result['merchant_name']
            result['merchant_name'] = extracted_merchant
        
        # Predict category if enabled
        if self.category_predictor:
            category_result = self.category_predictor.predict(
                normalized_description=normalized,
                merchant_name=extracted_merchant or '',
                amount=amount or 0,
                dr_cr=dr_cr or 'UNKNOWN',
                transaction_type=classification.transaction_type
            )
            result['category'] = category_result['category']
            result['category_confidence'] = category_result['confidence']
        
        return result
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        description_column: str = 'description',
        amount_column: Optional[str] = 'amount',
        dr_cr_column: Optional[str] = 'dr_cr',
        merchant_column: Optional[str] = None,
        include_details: bool = False,
        include_category_confidence: bool = True
    ) -> pd.DataFrame:
        """
        Process a DataFrame of transactions.
        
        Args:
            df: Input DataFrame containing transaction data.
            description_column: Name of the column containing transaction descriptions.
            amount_column: Name of the amount column (for seq2seq model and category prediction).
            dr_cr_column: Name of the dr/cr column (for seq2seq model and category prediction).
            merchant_column: Name of column with pre-extracted merchant names (optional).
            include_details: If True, include match details (matched text, score, rule).
            include_category_confidence: If True, include category confidence scores.
            
        Returns:
            DataFrame with added columns for normalized description, transaction type,
            merchant name (if model is configured), and category (if enabled).
        """
        result_df = df.copy()
        
        # Add normalized description
        result_df['normalized_description'] = result_df[description_column].apply(
            self.type_classifier.normalize
        )
        
        # Classify transaction types
        classifications = result_df[description_column].apply(self.type_classifier.classify)
        
        result_df['transaction_type'] = classifications.apply(lambda x: x.transaction_type)
        
        if include_details:
            result_df['type_matched'] = classifications.apply(lambda x: x.matched)
            result_df['type_score'] = classifications.apply(lambda x: x.score)
            result_df['type_rule'] = classifications.apply(lambda x: x.rule)
        
        # Extract merchant names if model is configured
        if self.merchant_predictor:
            print("Extracting merchant names...")
            
            if self.merchant_model == 'seq2seq':
                # Get amounts and dr_cr if available
                amounts = None
                dr_crs = None
                
                if amount_column and amount_column in result_df.columns:
                    amounts = result_df[amount_column].tolist()
                if dr_cr_column and dr_cr_column in result_df.columns:
                    dr_crs = result_df[dr_cr_column].tolist()
                
                # Process in batches for efficiency
                descriptions = result_df[description_column].tolist()
                merchant_results = self.merchant_predictor.predict_batch(
                    descriptions, amounts, dr_crs
                )
                result_df['merchant_name'] = [r['merchant_name'] for r in merchant_results]
            else:
                # NER model - process each transaction
                result_df['merchant_name'] = result_df[description_column].apply(
                    lambda x: self.merchant_predictor.get_merchant_name(x)
                )
        
        # Predict categories if enabled
        if self.category_predictor:
            print("Predicting categories...")
            
            # Determine merchant name column
            merchant_col = 'merchant_name' if 'merchant_name' in result_df.columns else merchant_column
            
            # Get required features
            normalized_descriptions = result_df['normalized_description'].tolist()
            
            if merchant_col and merchant_col in result_df.columns:
                merchant_names = result_df[merchant_col].tolist()
            else:
                merchant_names = [''] * len(result_df)
            
            if amount_column and amount_column in result_df.columns:
                amounts = result_df[amount_column].tolist()
            else:
                amounts = [0] * len(result_df)
            
            if dr_cr_column and dr_cr_column in result_df.columns:
                dr_crs = result_df[dr_cr_column].tolist()
            else:
                dr_crs = ['UNKNOWN'] * len(result_df)
            
            transaction_types = result_df['transaction_type'].tolist()
            
            # Predict categories
            category_results = self.category_predictor.predict_batch(
                normalized_descriptions=normalized_descriptions,
                merchant_names=merchant_names,
                amounts=amounts,
                dr_crs=dr_crs,
                transaction_types=transaction_types
            )
            
            result_df['predicted_category'] = [r['category'] for r in category_results]
            
            if include_category_confidence:
                result_df['category_confidence'] = [r['confidence'] for r in category_results]
        
        return result_df
    
    def process_csv(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        description_column: str = 'description',
        amount_column: Optional[str] = 'amount',
        dr_cr_column: Optional[str] = 'dr_cr',
        merchant_column: Optional[str] = None,
        include_details: bool = False,
        include_category_confidence: bool = True
    ) -> pd.DataFrame:
        """
        Process a CSV file of transactions.
        
        Args:
            input_path: Path to input CSV file.
            output_path: Path to save output CSV (optional).
            description_column: Name of the column containing transaction descriptions.
            amount_column: Name of the amount column (for seq2seq model and category).
            dr_cr_column: Name of the dr/cr column (for seq2seq model and category).
            merchant_column: Name of column with pre-extracted merchant names (optional).
            include_details: If True, include match details.
            include_category_confidence: If True, include category confidence scores.
            
        Returns:
            Processed DataFrame.
        """
        print(f"Reading data from: {input_path}")
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} transactions")
        
        print("Processing transactions...")
        result_df = self.process_dataframe(
            df,
            description_column=description_column,
            amount_column=amount_column,
            dr_cr_column=dr_cr_column,
            merchant_column=merchant_column,
            include_details=include_details,
            include_category_confidence=include_category_confidence
        )
        
        if output_path:
            result_df.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")
        
        return result_df


def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description='Transaction Categorization V2 - Process and classify transactions'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default='data/raw-merchant-name-transactions.csv',
        help='Path to input CSV file (default: data/raw-merchant-name-transactions.csv)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Path to output CSV file (optional)'
    )
    parser.add_argument(
        '-c', '--column',
        type=str,
        default='description',
        help='Name of the description column (default: description)'
    )
    parser.add_argument(
        '-d', '--details',
        action='store_true',
        help='Include detailed matching information in output'
    )
    parser.add_argument(
        '--fuzzy-threshold',
        type=int,
        default=75,
        help='Fuzzy matching threshold 0-100 (default: 75)'
    )
    parser.add_argument(
        '-m', '--merchant-model',
        type=str,
        choices=['ner', 'seq2seq', 'none'],
        default='none',
        help='Merchant extraction model to use (default: none)'
    )
    parser.add_argument(
        '--ner-model-path',
        type=str,
        default=DEFAULT_NER_MODEL_PATH,
        help=f'Path to NER model (default: {DEFAULT_NER_MODEL_PATH})'
    )
    parser.add_argument(
        '--seq2seq-model-path',
        type=str,
        default=DEFAULT_SEQ2SEQ_MODEL_PATH,
        help=f'Path to Seq2Seq model (default: {DEFAULT_SEQ2SEQ_MODEL_PATH})'
    )
    parser.add_argument(
        '--enable-category',
        action='store_true',
        help='Enable category classification using LightGBM + Text Embeddings'
    )
    parser.add_argument(
        '--category-model-path',
        type=str,
        default=DEFAULT_CATEGORY_MODEL_PATH,
        help=f'Path to category model (default: {DEFAULT_CATEGORY_MODEL_PATH})'
    )
    parser.add_argument(
        '--merchant-column',
        type=str,
        default=None,
        help='Name of column with pre-extracted merchant names (optional)'
    )
    parser.add_argument(
        '--no-category-confidence',
        action='store_true',
        help='Disable category confidence scores in output'
    )
    
    args = parser.parse_args()
    
    # Resolve input path relative to project root if needed
    input_path = args.input
    if not os.path.isabs(input_path):
        script_dir = Path(__file__).parent.parent
        candidate_path = script_dir / input_path
        if candidate_path.exists():
            input_path = str(candidate_path)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine merchant model
    merchant_model = None if args.merchant_model == 'none' else args.merchant_model
    
    # Initialize processor and process data
    processor = TransactionProcessor(
        fuzzy_threshold=args.fuzzy_threshold,
        merchant_model=merchant_model,
        enable_category=args.enable_category,
        ner_model_path=args.ner_model_path,
        seq2seq_model_path=args.seq2seq_model_path,
        category_model_path=args.category_model_path
    )
    
    result_df = processor.process_csv(
        input_path=input_path,
        output_path=args.output,
        description_column=args.column,
        merchant_column=args.merchant_column,
        include_details=args.details,
        include_category_confidence=not args.no_category_confidence
    )
    
    return result_df


if __name__ == '__main__':
    main()
