# ============================================================
# FILE: feedback.py
# Feedback Calculation Logic - FIXED
# ============================================================

import pandas as pd
from typing import Dict, List, Set

def clean_types(type_set: Set) -> Set:
    """Clean type set by removing NaN and None values"""
    if type_set is None:
        return set()
    cleaned = set()
    for t in type_set:
        if t is not None and not (isinstance(t, float) and pd.isna(t)):
            cleaned.add(t)
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
    
    # Get Pokemon types - safely handle None/NaN
    secret_type1 = secret.get('Type1')
    secret_type2 = secret.get('Type2')
    guess_type1 = guess.get('Type1')
    guess_type2 = guess.get('Type2')
    
    secret_types = clean_types({secret_type1, secret_type2})
    guess_types = clean_types({guess_type1, guess_type2})
    
    for attr in attributes:
        if attr == 'image_url':
            continue
        
        secret_val = secret.get(attr)
        guess_val = guess.get(attr)
        
        # Handle Type attributes specially
        if attr in ['Type1', 'Type2']:
            # Both are None/NaN
            if pd.isna(guess_val) and pd.isna(secret_val):
                feedback[attr] = 'green'
            # One is None/NaN
            elif pd.isna(guess_val) or pd.isna(secret_val):
                feedback[attr] = 'gray'
            # Exact match
            elif guess_val == secret_val:
                feedback[attr] = 'green'
            # Type exists but wrong position
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
            try:
                secret_num = float(secret_val)
                guess_num = float(guess_val)
                if guess_num < secret_num:
                    feedback[attr] = 'higher'
                else:
                    feedback[attr] = 'lower'
            except (ValueError, TypeError):
                feedback[attr] = 'gray'
        
        # Categorical attributes
        else:
            feedback[attr] = 'gray'
    
    return feedback

def is_complete_match(feedback: Dict[str, str]) -> bool:
    """
    Check if all feedback values indicate a complete match.
    
    Args:
        feedback: Feedback dictionary
        
    Returns:
        True if all attributes are 'green', False otherwise
    """
    non_image_feedback = {k: v for k, v in feedback.items() if k != 'image_url'}
    return all(v == 'green' for v in non_image_feedback.values())

def calculate_feedback_score(feedback: Dict[str, str]) -> float:
    """
    Calculate a numerical score from feedback.
    Higher score = closer to solution.
    
    Args:
        feedback: Feedback dictionary
        
    Returns:
        Feedback score (0-1)
    """
    if not feedback:
        return 0.0
    
    score = 0
    total = 0
    
    for attr, status in feedback.items():
        if attr == 'image_url':
            continue
        
        total += 1
        
        if status == 'green':
            score += 1.0
        elif status == 'yellow':
            score += 0.5
        elif status in ['higher', 'lower']:
            # Numeric feedback gives some information
            score += 0.3
        # 'gray' gives 0
    
    return score / total if total > 0 else 0.0