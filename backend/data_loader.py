# ============================================================
# FILE: data_loader.py
# Dataset Loading and Preprocessing
# ============================================================

import pandas as pd
from typing import Optional

class DataLoader:
    _instance = None
    _df = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataLoader, cls).__new__(cls)
        return cls._instance
    
    def load_data(self, filepath: str = "03_cleaned_with_images_and_evolutionary_stages.csv"):
        """Load Pokemon dataset"""
        if self._df is None:
            self._df = pd.read_csv(filepath)
            self._preprocess()
        return self._df
    
    def _preprocess(self):
        """Preprocess data"""
        # Convert numeric columns
        numeric_cols = ['Height', 'Weight', 'Generation']
        for col in numeric_cols:
            if col in self._df.columns:
                self._df[col] = pd.to_numeric(self._df[col], errors='coerce')
        
        # Handle missing values
        self._df['Type2'] = self._df['Type2'].fillna('None')
        
    def get_pokemon_by_name(self, name: str) -> Optional[pd.Series]:
        """Get Pokemon by name"""
        matches = self._df[self._df['Original_Name'] == name]
        return matches.iloc[0] if not matches.empty else None
    
    def get_random_pokemon(self) -> pd.Series:
        """Get random Pokemon"""
        return self._df.sample(1).iloc[0]
    
    def get_dataframe(self) -> pd.DataFrame:
        """Get full dataframe"""
        return self._df.copy()
    
