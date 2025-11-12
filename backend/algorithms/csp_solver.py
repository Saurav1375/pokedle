# ============================================================
# FILE: algorithms/csp_solver.py
# CSP Solver Implementation
# ============================================================

import pandas as pd
import math
from typing import Dict, Tuple, Any, List, Set
from algorithms.base import BaseSolver
from heuristics.csp_heuristics import CSPHeuristics

class EnhancedPokedleCSP(BaseSolver):
    """Enhanced Constraint Satisfaction Problem solver for Pokedle"""
    
    def __init__(self, dataframe: pd.DataFrame, attributes: list, heuristic: str = 'random'):
        super().__init__(dataframe, attributes)
        self.numeric_attrs = ['Height', 'Weight']
        self.heuristic = heuristic
        self.constraints = {col: [] for col in self.attributes}
        self.type_must_have = set()
        self.candidates = dataframe.copy()
        self.heuristics = CSPHeuristics()
        
    def apply_feedback(self, guess: pd.Series, feedback: Dict[str, str]):
        """Apply feedback to update constraints"""
        for attr, status in feedback.items():
            if attr == 'image_url':
                continue
                
            value = guess.get(attr)
            
            # Skip if value is None or NaN
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            
            if attr in ['Type1', 'Type2']:
                if status == "green":
                    self.constraints[attr].append(("==", value))
                elif status == "yellow":
                    self.constraints[attr].append(("!=", value))
                    self.type_must_have.add(value)
                elif status == "gray":
                    self.constraints[attr].append(("!=", value))
            elif attr in self.numeric_attrs:
                if status == "green":
                    self.constraints[attr].append(("==", value))
                elif status == "higher":
                    self.constraints[attr].append((">", value))
                elif status == "lower":
                    self.constraints[attr].append(("<", value))
            else:
                if status == "green":
                    self.constraints[attr].append(("==", value))
                elif status == "gray":
                    self.constraints[attr].append(("!=", value))
    
    def apply_numeric_feedback(self, attr: str, guess_value: float, secret_value: float):
        """Apply numeric feedback for Height/Weight"""
        if attr not in self.numeric_attrs:
            return
            
        if guess_value == secret_value:
            self.constraints[attr].append(("==", secret_value))
        elif guess_value < secret_value:
            self.constraints[attr].append((">", guess_value))
        else:
            self.constraints[attr].append(("<", guess_value))
    
    def filter_candidates(self) -> pd.DataFrame:
        """Filter candidates based on current constraints"""
        candidates = self.df.copy()
        
        for attr, conds in self.constraints.items():
            if attr == 'image_url':
                continue
                
            for op, val in conds:
                # Skip None/NaN values
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    continue
                    
                if op == "==":
                    candidates = candidates[candidates[attr] == val]
                elif op == "!=":
                    candidates = candidates[candidates[attr] != val]
                elif op == ">":
                    # Only apply to numeric attributes
                    if attr in self.numeric_attrs:
                        candidates = candidates[candidates[attr] > val]
                elif op == "<":
                    # Only apply to numeric attributes
                    if attr in self.numeric_attrs:
                        candidates = candidates[candidates[attr] < val]
        
        # Apply type must-have constraints
        if self.type_must_have:
            def has_required_types(row):
                pokemon_types = {row.get('Type1'), row.get('Type2')}
                # Clean the types
                pokemon_types = {t for t in pokemon_types if t is not None and not (isinstance(t, float) and pd.isna(t))}
                return self.type_must_have.issubset(pokemon_types)
            
            candidates = candidates[candidates.apply(has_required_types, axis=1)]
        
        self.candidates = candidates
        return candidates
    
    def next_guess(self) -> Tuple[pd.Series, Dict[str, Any]]:
        """Generate next guess using selected heuristic"""
        candidates = self.filter_candidates()
        
        if len(candidates) == 0:
            return None, {}
        
        # Select heuristic method
        if self.heuristic == 'mrv':
            return self.heuristics.mrv(candidates, self.attributes)
        elif self.heuristic == 'lcv':
            return self.heuristics.lcv(candidates, self.attributes)
        elif self.heuristic == 'entropy':
            return self.heuristics.entropy(candidates, self.attributes)
        elif self.heuristic == 'degree':
            return self.heuristics.degree(candidates, self.attributes, self.constraints)
        elif self.heuristic == 'forward_checking':
            return self.heuristics.forward_checking(candidates, self.attributes, self.constraints)
        elif self.heuristic == 'domain_wipeout':
            return self.heuristics.domain_wipeout(candidates, self.attributes)
        elif self.heuristic == 'mac':
            # MAC is similar to forward checking with arc consistency
            return self.heuristics.forward_checking(candidates, self.attributes, self.constraints)
        else:  # random
            return self.heuristics.random(candidates, self.attributes)
    
    def update_feedback(self, guess: pd.Series, feedback: Dict[str, str]):
        """Update solver with new feedback"""
        guess_idx = guess.name
        self.add_feedback(guess_idx, feedback)
        self.apply_feedback(guess, feedback)
        self.filter_candidates()
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information"""
        return {
            "algorithm": "CSP",
            "heuristic": self.heuristic,
            "candidates_remaining": len(self.candidates),
            "constraints": {attr: len(cons) for attr, cons in self.constraints.items()},
            "type_requirements": list(self.type_must_have)
        }