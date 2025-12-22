"""
Transaction Type Classifier Module

This module provides functionality to classify transaction descriptions
into different types such as UPI, CARD, ACH, IMPS, NEFT, etc.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


@dataclass
class ClassificationResult:
    """Data class to hold classification result."""
    transaction_type: str
    matched: Optional[str]
    score: float
    rule: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'type': self.transaction_type,
            'matched': self.matched,
            'score': self.score,
            'rule': self.rule
        }


class TransactionTypeClassifier:    
    # Canonical transaction types
    TRANSACTION_TYPES = [
        "UPI",
        "ACH-DR",       # Automated Clearing House - Debit
        "ACH-CR",       # ACH Credit
        "NACH",         # National Automated Clearing House (India)
        "IMPS",
        "NEFT",
        "RTGS",
        "CARD",
        "CASH-CR",
        "CASH-DR",
        "EMI",
        "SALARY",
        "TAX",
        "UNKNOWN"
    ]
    
    # Strong regex patterns (first pass, prioritized). Case-insensitive matching.
    TYPE_REGEX = {
        "UPI": [
            r"\bUPI\b",
            r"\bP2M\b",         # P2M appears in UPI contexts
            r"\bPAYTM\b|\bGPAI\b",
            r"\bPHONEPE\b",
            r"\bBHIM\b",
            r"\bONLINE\b"
        ],
        "ACH-DR": [
            r"\bACH[-\s]?DR\b",
            r"\bAUTOMATED CLEARING HOUSE\b",
            r"\bACH DEBIT\b",
            r"\bACH-?DEBIT\b",
        ],
        "ACH-CR": [r"\bACH[-\s]?CR\b", r"\bACH CREDIT\b"],
        "NACH": [r"\bNACH\b", r"\bRACPC\b", r"\bECS\b"],
        "EMI": [r"\bEMI\b", r"\bEMI PAYMENT\b", r"\bEMI-?\b"],
        "IMPS": [r"\bIMPS\b"],
        "NEFT": [r"\bNEFT\b"],
        "RTGS": [r"\bRTGS\b"],
        "CARD": [
            r"\bPOS\b", 
            r"\bCARD\b", 
            r"\bDEBIT CARD\b", 
            r"\bCREDIT CARD\b", 
            r"\bATM WITHDRAWAL\b", 
            r"\bATM-CASH \b"
        ],
        "SALARY": [r"\bSALARY\b", r"\bSAL\b"],
        "TAX": [r"\bTAX\b", r"\bGST\b", r"\bINCOME TAX\b"],
        "CASH-CR": [r"\bCASH IN\b", r"\bCASHDEPOSIT\b"],
        "CASH-DR": [r"\bCASH OUT\b", r"\bCASH WITHDRAWAL\b"]
    }
    
    # Keyword lists as secondary checks (tokens)
    TYPE_KEYWORDS = {
        "UPI": ["upi", "p2m", "paytm", "gpay", "googlepay", "phonepe", "bhim", "online"],
        "ACH-DR": ["ach-dr", "ach dr", "achdebit", "ach-debit", "automated clearing house", "ach"],
        "NACH": ["nach", "racpc", "ecs", "nac", "nach-debit"],
        "EMI": ["emi", "equated", "monthly installment", "emi-payment"],
        "IMPS": ["imps"],
        "NEFT": ["neft"],
        "RTGS": ["rtgs"],
        "CARD": ["pos", "card", "swipe", "debit", "credit", "netbanking", "atmwithdrawal", "atm-cash"],
        "SALARY": ["salary", "payroll"],
        "TAX": ["tax", "gst", "tds"],
        "CASH-CR": ["cashin", "cashdeposit"],
        "CASH-DR": ["cashout", "cashwithdrawal"]
    }
    
    def __init__(self, fuzzy_threshold: int = 75):
        """
        Initialize the TransactionTypeClassifier.
        
        Args:
            fuzzy_threshold: Minimum score (0-100) for fuzzy matching to be considered valid.
        """
        self.fuzzy_threshold = fuzzy_threshold
        self._build_keyword_lookup()
    
    def _build_keyword_lookup(self) -> None:
        """Build the keyword lookup list for fuzzy matching."""
        self._all_keywords: List[Tuple[str, str]] = []
        for txn_type, keywords in self.TYPE_KEYWORDS.items():
            for keyword in keywords:
                self._all_keywords.append((keyword, txn_type))
    
    @staticmethod
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
        # Replace common separators with a space, keep alphanumerics and basic punct
        text = re.sub(r"[\/\-\_\*]+", " ", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def _regex_match(self, text: str) -> Optional[ClassificationResult]:
        """
        Attempt to classify using regex patterns (highest priority).
        
        Args:
            text: Normalized transaction text.
            
        Returns:
            ClassificationResult if a match is found, None otherwise.
        """
        for txn_type, patterns in self.TYPE_REGEX.items():
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    return ClassificationResult(
                        transaction_type=txn_type,
                        matched=match.group(0),
                        score=1.0,
                        rule="regex"
                    )
        return None
    
    def _keyword_match(self, text: str) -> Optional[ClassificationResult]:
        """
        Attempt to classify using keyword matching (medium priority).
        
        Args:
            text: Normalized transaction text.
            
        Returns:
            ClassificationResult if a match is found, None otherwise.
        """
        text_lower = text.lower()
        for txn_type, keywords in self.TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return ClassificationResult(
                        transaction_type=txn_type,
                        matched=keyword,
                        score=0.95,
                        rule="keyword"
                    )
        return None
    
    def _fuzzy_fallback(self, text: str) -> Optional[ClassificationResult]:
        """
        Attempt to classify using fuzzy matching (lowest priority).
        
        Args:
            text: Normalized transaction text.
            
        Returns:
            ClassificationResult if a match is found above threshold, None otherwise.
        """
        if not RAPIDFUZZ_AVAILABLE:
            return None
            
        text_lower = text.lower()
        candidates = [k for k, _ in self._all_keywords]
        
        result = process.extractOne(text_lower, candidates, scorer=fuzz.partial_ratio)
        if result is None:
            return None
            
        match, score, idx = result
        if score >= self.fuzzy_threshold:
            matched_keyword, mapped_type = self._all_keywords[idx]
            return ClassificationResult(
                transaction_type=mapped_type,
                matched=matched_keyword,
                score=score / 100.0,
                rule="fuzzy"
            )
        return None
    
    def _heuristic_fallback(self, text: str) -> ClassificationResult:
        """
        Apply heuristic rules as final fallback.
        
        Args:
            text: Normalized transaction text.
            
        Returns:
            ClassificationResult based on heuristic rules or UNKNOWN.
        """
        # Detect "DEBIT" or "CR" words
        if re.search(r"\bDEBIT\b", text):
            return ClassificationResult(
                transaction_type="CASH-DR",
                matched="DEBIT",
                score=0.6,
                rule="heuristic"
            )
        if re.search(r"\bCREDIT\b|\bCR\b", text):
            return ClassificationResult(
                transaction_type="CASH-CR",
                matched="CREDIT",
                score=0.6,
                rule="heuristic"
            )
        
        # No match found
        return ClassificationResult(
            transaction_type="UNKNOWN",
            matched=None,
            score=0.0,
            rule="none"
        )
    
    def classify(self, raw_text: str) -> ClassificationResult:
        """
        Classify a transaction description into a transaction type.
        
        The classification follows this priority order:
        1. Regex pattern matching (highest confidence)
        2. Keyword matching (high confidence)
        3. Fuzzy matching (medium confidence)
        4. Heuristic rules (low confidence)
        5. UNKNOWN (no match)
        
        Args:
            raw_text: Raw transaction description text.
            
        Returns:
            ClassificationResult containing the transaction type and metadata.
        """
        # Normalize the text first
        text = self.normalize(raw_text)
        
        # 1. Try regex strong signals (highest priority)
        result = self._regex_match(text)
        if result:
            return result
        
        # 2. Try keyword matching
        result = self._keyword_match(text)
        if result:
            return result
        
        # 3. Try fuzzy fallback
        result = self._fuzzy_fallback(text)
        if result:
            return result
        
        # 4. Apply heuristic fallback
        return self._heuristic_fallback(text)
    
    def classify_to_dict(self, raw_text: str) -> Dict:
        """
        Classify a transaction and return result as dictionary.
        
        Args:
            raw_text: Raw transaction description text.
            
        Returns:
            Dictionary containing type, matched, score, and rule.
        """
        return self.classify(raw_text).to_dict()
    
    def get_transaction_type(self, raw_text: str) -> str:
        """
        Get just the transaction type for a description.
        
        Args:
            raw_text: Raw transaction description text.
            
        Returns:
            Transaction type string.
        """
        return self.classify(raw_text).transaction_type

