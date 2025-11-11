# ============================================================
# FILE: algorithms/simulated_annealing.py
# Simulated Annealing Algorithm Implementation
# ============================================================

import pandas as pd
import random
import math
from typing import Dict, Tuple, Any
from algorithms.base import BaseSolver

class SimulatedAnnealingSolver(BaseSolver):
    """
    Simulated Annealing algorithm for Pokedle.
    Uses temperature-based acceptance probability to escape local optima.
    """
    
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
        
    def energy(self, pokemon_idx: int) -> float:
        """
        Calculate energy (lower is better).
        Based on constraint violations from feedback history.
        """
        pokemon = self.df.loc[pokemon_idx]
        
        if not self.feedback_history:
            # Initial energy based on attribute diversity
            return self._diversity_energy(pokemon)
        
        violations = 0
        satisfied = 0
        penalty_multiplier = 1.0
        
        for guess_idx, feedback in self.feedback_history:
            guess = self.df.loc[guess_idx]
            
            for attr, status in feedback.items():
                if attr == 'image_url':
                    continue
                
                if status == 'green':
                    if pokemon[attr] == guess[attr]:
                        satisfied += 1
                    else:
                        violations += 3 * penalty_multiplier
                        penalty_multiplier *= 1.1
                
                elif status == 'gray':
                    if attr in ['Type1', 'Type2']:
                        pokemon_types = {pokemon['Type1'], pokemon['Type2']}
                        pokemon_types = {t for t in pokemon_types if not pd.isna(t)}
                        if guess[attr] in pokemon_types:
                            violations += 2
                    else:
                        if pokemon[attr] == guess[attr]:
                            violations += 2
                
                elif status == 'yellow':
                    pokemon_types = {pokemon['Type1'], pokemon['Type2']}
                    pokemon_types = {t for t in pokemon_types if not pd.isna(t)}
                    if guess[attr] not in pokemon_types:
                        violations += 2
                    elif pokemon[attr] == guess[attr]:
                        violations += 1
                
                elif status == 'higher':
                    if pokemon[attr] <= guess[attr]:
                        violations += 2
                
                elif status == 'lower':
                    if pokemon[attr] >= guess[attr]:
                        violations += 2
        
        # Lower energy for more satisfied constraints
        base_energy = violations - satisfied
        
        # Add diversity bonus to avoid getting stuck
        diversity_penalty = self._diversity_energy(pokemon) * 0.1
        
        return max(0, base_energy + diversity_penalty)
    
    def _diversity_energy(self, pokemon: pd.Series) -> float:
        """Energy based on how common the attributes are"""
        energy = 0
        
        for attr in self.attributes:
            if attr == 'image_url':
                continue
            
            value = pokemon[attr]
            if pd.isna(value):
                energy += 0.5
                continue
            
            # Penalize very common values
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
        """
        Generate neighbor solution.
        Prefer solutions similar to current but with variation.
        """
        current_pokemon = self.df.loc[pokemon_idx]
        
        # High temperature: explore more randomly
        # Low temperature: exploit local neighborhood
        if self.current_temp > self.initial_temp * 0.5:
            # Exploration: random pokemon
            return self.df.sample(1).index[0]
        else:
            # Exploitation: similar pokemon
            # Find pokemon with similar attributes
            candidates = self.df.copy()
            similarity_scores = []
            
            sample_size = min(100, len(self.df))
            sample = self.df.sample(sample_size)
            
            for idx, row in sample.iterrows():
                similarity = 0
                for attr in self.attributes:
                    if attr == 'image_url':
                        continue
                    if not pd.isna(row[attr]) and not pd.isna(current_pokemon[attr]):
                        if row[attr] == current_pokemon[attr]:
                            similarity += 1
                        elif attr in ['Height', 'Weight']:
                            diff = abs(row[attr] - current_pokemon[attr])
                            max_diff = self.df[attr].max() - self.df[attr].min()
                            if max_diff > 0:
                                similarity += 1 - (diff / max_diff)
                
                similarity_scores.append((idx, similarity))
            
            # Select with probability proportional to similarity
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
        
        # Initialize current solution
        if self.current_solution is None:
            self.current_solution = self.df.sample(1).index[0]
            self.best_solution = self.current_solution
            self.best_energy = self.energy(self.current_solution)
        
        # Run iterations at current temperature
        for _ in range(self.iterations_per_temp):
            # Get neighbor
            neighbor = self.get_neighbor(self.current_solution)
            
            # Calculate energies
            current_energy = self.energy(self.current_solution)
            neighbor_energy = self.energy(neighbor)
            
            # Accept or reject
            if random.random() < self.acceptance_probability(current_energy, neighbor_energy):
                self.current_solution = neighbor
                
                # Update best solution
                if neighbor_energy < self.best_energy:
                    self.best_solution = neighbor
                    self.best_energy = neighbor_energy
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
            
            self.iteration += 1
        
        # Cool down
        self.current_temp *= self.cooling_rate
        
        # Reheat if stuck
        if self.no_improvement_count > 100:
            self.current_temp = self.initial_temp * self.reheat_threshold
            self.no_improvement_count = 0
        
        # Ensure minimum temperature
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
        
        # Recalculate energy for current and best solutions
        current_energy = self.energy(self.current_solution)
        best_energy = self.energy(self.best_solution)
        
        # Reset if feedback changes energy landscape significantly
        if abs(current_energy - self.best_energy) > 10:
            self.current_temp = self.initial_temp * 0.5
            self.no_improvement_count = 0
