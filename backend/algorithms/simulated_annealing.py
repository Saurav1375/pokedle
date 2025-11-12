# ============================================================
# FILE: algorithms/simulated_annealing.py - FIXED
# ============================================================

import pandas as pd
import math
from typing import Dict, Tuple, Any, List, Set
from algorithms.base import BaseSolver
from heuristics.csp_heuristics import CSPHeuristics
import random

class SimulatedAnnealingSolver(BaseSolver):
    """Simulated Annealing algorithm for Pokedle"""
    
    def __init__(self, dataframe: pd.DataFrame, attributes: list, config: dict):
        super().__init__(dataframe, attributes)
        self.initial_temp = config.get('initial_temp', 100.0)
        self.cooling_rate = config.get('cooling_rate', 0.95)
        self.min_temp = config.get('min_temp', 0.01)
        self.iterations_per_temp = config.get('iterations_per_temp', 50)
        self.reheat_threshold = config.get('reheat_threshold', 0.1)
        
        self.current_temp = self.initial_temp
        self.current_solution = None
        self.best_solution = None
        self.best_energy = float('inf')
        self.iteration = 0
        self.no_improvement_count = 0
    
    def _safe_value_check(self, val1, val2) -> bool:
        """Safely check if two values are equal"""
        if val1 is None or val2 is None:
            return val1 == val2
        if isinstance(val1, float) and pd.isna(val1):
            return isinstance(val2, float) and pd.isna(val2)
        if isinstance(val2, float) and pd.isna(val2):
            return False
        return val1 == val2
    
    def _get_pokemon_types(self, pokemon) -> set:
        """Safely get Pokemon types"""
        types = set()
        type1 = pokemon.get('Type1') if isinstance(pokemon, pd.Series) else pokemon['Type1']
        type2 = pokemon.get('Type2') if isinstance(pokemon, pd.Series) else pokemon['Type2']
        
        if type1 is not None and not (isinstance(type1, float) and pd.isna(type1)):
            types.add(type1)
        if type2 is not None and not (isinstance(type2, float) and pd.isna(type2)):
            types.add(type2)
        
        return types
    
    def energy(self, pokemon_idx: int) -> float:
        """Calculate energy (lower is better)"""
        pokemon = self.df.loc[pokemon_idx]
        
        if not self.feedback_history:
            return self._diversity_energy(pokemon)
        
        violations = 0
        satisfied = 0
        penalty_multiplier = 1.0
        
        for guess_idx, feedback in self.feedback_history:
            guess = self.df.loc[guess_idx]
            
            for attr, status in feedback.items():
                if attr == 'image_url':
                    continue
                
                pokemon_val = pokemon.get(attr)
                guess_val = guess.get(attr)
                
                if status == 'green':
                    if self._safe_value_check(pokemon_val, guess_val):
                        satisfied += 1
                    else:
                        violations += 3 * penalty_multiplier
                        penalty_multiplier *= 1.1
                
                elif status == 'gray':
                    if attr in ['Type1', 'Type2']:
                        pokemon_types = self._get_pokemon_types(pokemon)
                        if guess_val is not None and not (isinstance(guess_val, float) and pd.isna(guess_val)):
                            if guess_val in pokemon_types:
                                violations += 2
                    else:
                        if self._safe_value_check(pokemon_val, guess_val):
                            violations += 2
                
                elif status == 'yellow':
                    pokemon_types = self._get_pokemon_types(pokemon)
                    if guess_val is None or (isinstance(guess_val, float) and pd.isna(guess_val)):
                        violations += 2
                    elif guess_val not in pokemon_types:
                        violations += 2
                    elif self._safe_value_check(pokemon_val, guess_val):
                        violations += 1
                
                elif status == 'higher':
                    try:
                        if pokemon_val is not None and guess_val is not None:
                            if not (isinstance(pokemon_val, float) and pd.isna(pokemon_val)):
                                if not (isinstance(guess_val, float) and pd.isna(guess_val)):
                                    if float(pokemon_val) <= float(guess_val):
                                        violations += 2
                    except (ValueError, TypeError):
                        pass
                
                elif status == 'lower':
                    try:
                        if pokemon_val is not None and guess_val is not None:
                            if not (isinstance(pokemon_val, float) and pd.isna(pokemon_val)):
                                if not (isinstance(guess_val, float) and pd.isna(guess_val)):
                                    if float(pokemon_val) >= float(guess_val):
                                        violations += 2
                    except (ValueError, TypeError):
                        pass
        
        base_energy = violations - satisfied
        diversity_penalty = self._diversity_energy(pokemon) * 0.1
        
        return max(0, base_energy + diversity_penalty)
    
    def _diversity_energy(self, pokemon: pd.Series) -> float:
        """Energy based on how common the attributes are"""
        energy = 0
        
        for attr in self.attributes:
            if attr == 'image_url':
                continue
            
            value = pokemon.get(attr)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                energy += 0.5
                continue
            
            count = (self.df[attr] == value).sum()
            ratio = count / len(self.df)
            energy += ratio
        
        return energy
    
    def acceptance_probability(self, current_energy: float, new_energy: float) -> float:
        """Calculate probability of accepting worse solution"""
        if new_energy < current_energy:
            return 1.0
        
        if self.current_temp == 0:
            return 0.0
        
        delta = new_energy - current_energy
        return math.exp(-delta / self.current_temp)
    
    def get_neighbor(self, pokemon_idx: int) -> int:
        """Generate neighbor solution"""
        current_pokemon = self.df.loc[pokemon_idx]
        
        if self.current_temp > self.initial_temp * 0.5:
            return self.df.sample(1).index[0]
        else:
            candidates = self.df.copy()
            similarity_scores = []
            
            sample_size = min(100, len(self.df))
            sample = self.df.sample(sample_size)
            
            for idx, row in sample.iterrows():
                similarity = 0
                for attr in self.attributes:
                    if attr == 'image_url':
                        continue
                    
                    row_val = row.get(attr)
                    curr_val = current_pokemon.get(attr)
                    
                    if row_val is None or (isinstance(row_val, float) and pd.isna(row_val)):
                        continue
                    if curr_val is None or (isinstance(curr_val, float) and pd.isna(curr_val)):
                        continue
                    
                    if self._safe_value_check(row_val, curr_val):
                        similarity += 1
                    elif attr in ['Height', 'Weight']:
                        try:
                            diff = abs(float(row_val) - float(curr_val))
                            max_diff = self.df[attr].max() - self.df[attr].min()
                            if max_diff > 0:
                                similarity += 1 - (diff / max_diff)
                        except (ValueError, TypeError):
                            pass
                
                similarity_scores.append((idx, similarity))
            
            similarity_scores.sort(key=lambda x: x[1], reverse=True)
            top_candidates = similarity_scores[:20]
            
            if top_candidates:
                weights = [s for _, s in top_candidates]
                total_weight = sum(weights) + 0.01
                probs = [w / total_weight for w in weights]
                selected = random.choices([idx for idx, _ in top_candidates], weights=probs)[0]
                return selected
            
            return self.df.sample(1).index[0]
    
    def next_guess(self) -> Tuple[pd.Series, Dict[str, Any]]:
        """Generate next guess using simulated annealing"""
        
        if self.current_solution is None:
            self.current_solution = self.df.sample(1).index[0]
            self.best_solution = self.current_solution
            self.best_energy = self.energy(self.current_solution)
        
        for _ in range(self.iterations_per_temp):
            neighbor = self.get_neighbor(self.current_solution)
            
            current_energy = self.energy(self.current_solution)
            neighbor_energy = self.energy(neighbor)
            
            if random.random() < self.acceptance_probability(current_energy, neighbor_energy):
                self.current_solution = neighbor
                
                if neighbor_energy < self.best_energy:
                    self.best_solution = neighbor
                    self.best_energy = neighbor_energy
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
            
            self.iteration += 1
        
        self.current_temp *= self.cooling_rate
        
        if self.no_improvement_count > 100:
            self.current_temp = self.initial_temp * self.reheat_threshold
            self.no_improvement_count = 0
        
        if self.current_temp < self.min_temp:
            self.current_temp = self.min_temp
        
        pokemon = self.df.loc[self.best_solution]
        
        info = {
            "algorithm": "simulated_annealing",
            "temperature": round(self.current_temp, 3),
            "current_energy": round(self.energy(self.current_solution), 3),
            "best_energy": round(self.best_energy, 3),
            "iteration": self.iteration,
            "no_improvement": self.no_improvement_count
        }
        
        return pokemon, info
    
    def update_feedback(self, guess: pd.Series, feedback: Dict[str, str]):
        """Update solver with new feedback"""
        guess_idx = guess.name
        self.add_feedback(guess_idx, feedback)
        
        current_energy = self.energy(self.current_solution)
        best_energy = self.energy(self.best_solution)
        
        if abs(current_energy - self.best_energy) > 10:
            self.current_temp = self.initial_temp * 0.5
            self.no_improvement_count = 0