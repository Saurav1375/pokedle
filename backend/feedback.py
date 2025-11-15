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
    
    IMPROVED VERSION with better type handling and edge cases.
    
    Returns:
        Dict with feedback for each attribute:
        - 'green': Exact match
        - 'yellow': Type exists but in wrong position (Type1/Type2 only)
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
    
    # Clean NaN values
    if isinstance(secret_type2, float) and pd.isna(secret_type2):
        secret_type2 = None
    if isinstance(guess_type2, float) and pd.isna(guess_type2):
        guess_type2 = None
    
    secret_types = clean_types({secret_type1, secret_type2})
    guess_types = clean_types({guess_type1, guess_type2})
    
    for attr in attributes:
        if attr == 'image_url':
            continue
        
        secret_val = secret.get(attr)
        guess_val = guess.get(attr)
        
        # Handle Type attributes specially
        if attr == 'Type1':
            # Type1 must match exactly for green
            if guess_type1 == secret_type1:
                feedback[attr] = 'green'
            # Yellow: guess type exists in secret but wrong slot
            elif guess_type1 is not None and guess_type1 in secret_types:
                feedback[attr] = 'yellow'
            else:
                feedback[attr] = 'gray'
        
        elif attr == 'Type2':
            # Both None/missing
            if guess_type2 is None and secret_type2 is None:
                feedback[attr] = 'green'
            # Exact match
            elif guess_type2 == secret_type2:
                feedback[attr] = 'green'
            # Yellow: guess type exists in secret but wrong slot
            elif guess_type2 is not None and guess_type2 in secret_types:
                feedback[attr] = 'yellow'
            else:
                feedback[attr] = 'gray'
        
        # Handle missing values for non-type attributes
        elif pd.isna(secret_val) or pd.isna(guess_val):
            # Both missing = match
            if pd.isna(secret_val) and pd.isna(guess_val):
                feedback[attr] = 'green'
            else:
                feedback[attr] = 'gray'
        
        # Exact match
        elif secret_val == guess_val:
            feedback[attr] = 'green'
        
        # Numeric attributes with directional feedback
        elif attr in numeric_attrs:
            try:
                secret_num = float(secret_val)
                guess_num = float(guess_val)
                
                # Add tolerance for floating point comparison
                if abs(secret_num - guess_num) < 0.01:
                    feedback[attr] = 'green'
                elif guess_num < secret_num:
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
    """
    non_image_feedback = {k: v for k, v in feedback.items() if k != 'image_url'}
    return all(v == 'green' for v in non_image_feedback.values())

def calculate_feedback_score(feedback: Dict[str, str]) -> float:
    """
    Calculate a numerical score from feedback.
    Higher score = closer to solution.
    
    IMPROVED: Better weighting for different feedback types.
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
            # Yellow is better than numeric partial matches
            score += 0.6
        elif status in ['higher', 'lower']:
            # Numeric feedback gives some information
            score += 0.4
        # 'gray' gives 0
    
    return score / total if total > 0 else 0.0

def validate_feedback_consistency(feedback_history: List[tuple]) -> bool:
    """
    NEW FUNCTION: Validate that feedback history is logically consistent.
    
    This helps catch bugs in the solving algorithms.
    
    Returns:
        True if consistent, False if contradictions found
    """
    # Check for contradictions in feedback
    green_constraints = {}  # attr -> value that must be green
    
    for guess_idx, feedback in feedback_history:
        for attr, status in feedback.items():
            if attr == 'image_url':
                continue
            
            if status == 'green':
                # If we previously saw green for this attr with different value, contradiction
                if attr in green_constraints and green_constraints[attr] != guess_idx:
                    return False
                green_constraints[attr] = guess_idx
    
    return True

def get_constraint_implications(feedback: Dict[str, str], guess: pd.Series) -> Dict[str, List]:
    """
    NEW FUNCTION: Extract explicit constraints from feedback.
    
    Returns:
        Dict mapping attributes to list of constraints
    """
    constraints = {}
    
    for attr, status in feedback.items():
        if attr == 'image_url':
            continue
        
        value = guess.get(attr)
        if pd.isna(value):
            continue
        
        if attr not in constraints:
            constraints[attr] = []
        
        if status == 'green':
            constraints[attr].append(('must_equal', value))
        elif status == 'gray':
            if attr in ['Type1', 'Type2']:
                constraints[attr].append(('type_not_in', value))
            else:
                constraints[attr].append(('not_equal', value))
        elif status == 'yellow':
            constraints[attr].append(('type_in_other_slot', value))
        elif status == 'higher':
            constraints[attr].append(('greater_than', value))
        elif status == 'lower':
            constraints[attr].append(('less_than', value))
    
    return constraints