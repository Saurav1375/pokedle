# ============================================================
# FILE: algorithms/base.py
# Abstract Base Solver Class
# ============================================================

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Tuple, Any

class BaseSolver(ABC):
    """Abstract base class for all solving algorithms"""
    
    def __init__(self, dataframe: pd.DataFrame, attributes: list):
        self.df = dataframe.copy()
        self.attributes = attributes
        self.feedback_history = []
    
    @abstractmethod
    def next_guess(self) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Generate next guess.
        
        Returns:
            Tuple of (pokemon_series, info_dict)
        """
        pass
    
    @abstractmethod
    def update_feedback(self, guess: pd.Series, feedback: Dict[str, str]):
        """Update solver state with feedback from guess"""
        pass
    
    @abstractmethod
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information for debugging/display"""
        pass
    
