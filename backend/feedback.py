# ============================================================
# FILE: feedback.py
# Feedback Calculation Logic
# ============================================================

import pandas as pd
from typing import Dict, List, Set

def clean_types(type_set: Set) -> Set:
    """Clean type set by removing NaN and None values"""
    cleaned = {t for t in type_set if t is not None and not (isinstance(t, float) and pd.isna(t))}
    return cleaned

def get_feedback(secret: pd.Series, guess: pd.Series, 
                attributes: List[str], numeric_attrs: List[str] = ['Height', 'Weight']) -> Dict[str, str]:
    """
    Calculate feedback for a guess compared to secret Pokemon.
    
    Returns:
        Dict with feedback for each attribute:
        - 'green': Exact match
        - 'yellow': Type exists but in wrong position
        - 'gray': Does not match
        - 'higher': Guess is lower than secret
        - 'lower': Guess is higher than secret
    """
    feedback = {}
    
    # Get Pokemon types
    secret_types = clean_types({secret.get('Type1'), secret.get('Type2')})
    guess_types = clean_types({guess.get('Type1'), guess.get('Type2')})
    
    for attr in attributes:
        if attr == 'image_url':
            continue
        
        secret_val = secret[attr]
        guess_val = guess[attr]
        
        # Handle Type attributes specially
        if attr in ['Type1', 'Type2']:
            if pd.isna(guess_val) and pd.isna(secret_val):
                feedback[attr] = 'green'
            elif pd.isna(guess_val) or pd.isna(secret_val):
                feedback[attr] = 'gray'
            elif guess_val == secret_val:
                feedback[attr] = 'green'
            elif guess_val in secret_types:
                feedback[attr] = 'yellow'
            else:
                feedback[attr] = 'gray'
        
        # Handle missing values
        elif pd.isna(secret_val) or pd.isna(guess_val):
            feedback[attr] = 'gray'
        
        # Exact match
        elif secret_val == guess_val:
            feedback[attr] = 'green'
        
        # Numeric attributes
        elif attr in numeric_attrs:
            if guess_val < secret_val:
                feedback[attr] = 'higher'
            else:
                feedback[attr] = 'lower'
        
        # Categorical attributes
        else:
            feedback[attr] = 'gray'
    
    return feedback

