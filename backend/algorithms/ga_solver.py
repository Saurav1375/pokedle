# ============================================================
# FILE: algorithms/ga_solver.py
# Genetic Algorithm Solver Implementation - FIXED
# ============================================================

import pandas as pd
import random
from typing import Dict, Tuple, Any, List
from algorithms.base import BaseSolver

class EnhancedPokedleGA(BaseSolver):
    """Enhanced Genetic Algorithm solver for Pokedle"""
    
    def __init__(self, dataframe: pd.DataFrame, attributes: list, config: dict):
        super().__init__(dataframe, attributes)
        self.pop_size = config.get('pop_size', 100)
        self.elite_size = config.get('elite_size', 20)
        self.mutation_rate = config.get('mutation_rate', 0.15)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        self.tournament_size = config.get('tournament_size', 7)
        self.crossover_strategy = config.get('crossover_strategy', 'attribute_blend')
        self.generations_per_guess = config.get('generations_per_guess', 30)
        
        self.population = self.df.sample(min(self.pop_size, len(self.df))).index.tolist()
        self.generation = 0
        self.fitness_cache = {}
        
    def _safe_value_check(self, val1, val2) -> bool:
        """Safely check if two values are equal, handling None/NaN"""
        if val1 is None or val2 is None:
            return val1 == val2
        if isinstance(val1, float) and pd.isna(val1):
            return isinstance(val2, float) and pd.isna(val2)
        if isinstance(val2, float) and pd.isna(val2):
            return False
        return val1 == val2
    
    def _get_pokemon_types(self, pokemon: pd.Series) -> set:
        """Safely get Pokemon types as a set"""
        types = set()
        type1 = pokemon.get('Type1')
        type2 = pokemon.get('Type2')
        
        if type1 is not None and not (isinstance(type1, float) and pd.isna(type1)):
            types.add(type1)
        if type2 is not None and not (isinstance(type2, float) and pd.isna(type2)):
            types.add(type2)
        
        return types
    
    def fitness(self, pokemon_idx: int) -> float:
        """Calculate fitness score for a Pokemon"""
        # Cache fitness calculations
        if pokemon_idx in self.fitness_cache:
            return self.fitness_cache[pokemon_idx]
            
        pokemon = self.df.loc[pokemon_idx]
        score = 0
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
                        score += 15 * penalty_multiplier
                    else:
                        score -= 30 * penalty_multiplier
                        penalty_multiplier *= 1.2
                
                elif status == 'gray':
                    if attr in ['Type1', 'Type2']:
                        pokemon_types = self._get_pokemon_types(pokemon)
                        if guess_val is None or (isinstance(guess_val, float) and pd.isna(guess_val)):
                            score += 8
                        elif guess_val not in pokemon_types:
                            score += 8
                        else:
                            score -= 15
                    else:
                        if not self._safe_value_check(pokemon_val, guess_val):
                            score += 8
                        else:
                            score -= 15
                
                elif status == 'yellow':
                    pokemon_types = self._get_pokemon_types(pokemon)
                    if guess_val is None or (isinstance(guess_val, float) and pd.isna(guess_val)):
                        score -= 15
                    elif guess_val in pokemon_types and not self._safe_value_check(pokemon_val, guess_val):
                        score += 12
                    else:
                        score -= 15
                
                elif status == 'higher':
                    try:
                        if pokemon_val is not None and guess_val is not None:
                            if not (isinstance(pokemon_val, float) and pd.isna(pokemon_val)):
                                if not (isinstance(guess_val, float) and pd.isna(guess_val)):
                                    if float(pokemon_val) > float(guess_val):
                                        score += 12
                                    else:
                                        score -= 20
                    except (ValueError, TypeError):
                        pass
                
                elif status == 'lower':
                    try:
                        if pokemon_val is not None and guess_val is not None:
                            if not (isinstance(pokemon_val, float) and pd.isna(pokemon_val)):
                                if not (isinstance(guess_val, float) and pd.isna(guess_val)):
                                    if float(pokemon_val) < float(guess_val):
                                        score += 12
                                    else:
                                        score -= 20
                    except (ValueError, TypeError):
                        pass
        
        result = max(0, score)
        self.fitness_cache[pokemon_idx] = result
        return result
    
    def tournament_selection(self) -> int:
        """Select parent using tournament selection"""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        tournament_fitness = [(idx, self.fitness(idx)) for idx in tournament]
        tournament_fitness.sort(key=lambda x: x[1], reverse=True)
        return tournament_fitness[0][0]
    
    def find_best_match(self, target_attrs: Dict) -> int:
        """Find Pokemon that best matches target attributes"""
        best_match_idx = None
        best_match_score = -1
        
        sample_size = min(150, len(self.df))
        candidates = self.df.sample(sample_size)
        
        for idx, row in candidates.iterrows():
            match_score = 0
            for attr in self.attributes:
                if attr == 'image_url':
                    continue
                
                target_val = target_attrs.get(attr)
                row_val = row.get(attr)
                
                # Skip None/NaN comparisons
                if target_val is None or (isinstance(target_val, float) and pd.isna(target_val)):
                    continue
                if row_val is None or (isinstance(row_val, float) and pd.isna(row_val)):
                    continue
                
                if self._safe_value_check(row_val, target_val):
                    match_score += 1
                elif attr in ['Height', 'Weight']:
                    try:
                        diff = abs(float(row_val) - float(target_val))
                        max_diff = self.df[attr].max() - self.df[attr].min()
                        if max_diff > 0:
                            match_score += 1 - (diff / max_diff)
                    except (ValueError, TypeError):
                        pass
            
            if match_score > best_match_score:
                best_match_score = match_score
                best_match_idx = idx
        
        return best_match_idx if best_match_idx is not None else candidates.sample(1).index[0]
    
    def crossover_attribute_blend(self, parent1_idx: int, parent2_idx: int) -> int:
        """Blend attributes from both parents based on fitness"""
        parent1 = self.df.loc[parent1_idx]
        parent2 = self.df.loc[parent2_idx]
        
        target_attrs = {}
        p1_fitness = self.fitness(parent1_idx)
        p2_fitness = self.fitness(parent2_idx)
        
        for attr in self.attributes:
            if attr == 'image_url':
                continue
            p1_val = parent1.get(attr)
            p2_val = parent2.get(attr)
            
            if p1_fitness > p2_fitness:
                target_attrs[attr] = p1_val if random.random() < 0.7 else p2_val
            else:
                target_attrs[attr] = p2_val if random.random() < 0.7 else p1_val
        
        return self.find_best_match(target_attrs)
    
    def crossover_uniform(self, parent1_idx: int, parent2_idx: int) -> int:
        """Uniform crossover - each attribute has 50% chance from either parent"""
        parent1 = self.df.loc[parent1_idx]
        parent2 = self.df.loc[parent2_idx]
        
        target_attrs = {}
        for attr in self.attributes:
            if attr == 'image_url':
                continue
            p1_val = parent1.get(attr)
            p2_val = parent2.get(attr)
            target_attrs[attr] = p1_val if random.random() < 0.5 else p2_val
        
        return self.find_best_match(target_attrs)
    
    def crossover_single_point(self, parent1_idx: int, parent2_idx: int) -> int:
        """Single-point crossover"""
        parent1 = self.df.loc[parent1_idx]
        parent2 = self.df.loc[parent2_idx]
        
        crossover_point = random.randint(1, len(self.attributes) - 1)
        target_attrs = {}
        
        for i, attr in enumerate(self.attributes):
            if attr == 'image_url':
                continue
            target_attrs[attr] = parent1.get(attr) if i < crossover_point else parent2.get(attr)
        
        return self.find_best_match(target_attrs)
    
    def crossover_two_point(self, parent1_idx: int, parent2_idx: int) -> int:
        """Two-point crossover"""
        parent1 = self.df.loc[parent1_idx]
        parent2 = self.df.loc[parent2_idx]
        
        points = sorted(random.sample(range(1, len(self.attributes)), 2))
        target_attrs = {}
        
        for i, attr in enumerate(self.attributes):
            if attr == 'image_url':
                continue
            if points[0] <= i < points[1]:
                target_attrs[attr] = parent2.get(attr)
            else:
                target_attrs[attr] = parent1.get(attr)
        
        return self.find_best_match(target_attrs)
    
    def crossover_fitness_weighted(self, parent1_idx: int, parent2_idx: int) -> int:
        """Weighted by fitness - higher fitness parent contributes more"""
        parent1 = self.df.loc[parent1_idx]
        parent2 = self.df.loc[parent2_idx]
        
        p1_fitness = self.fitness(parent1_idx)
        p2_fitness = self.fitness(parent2_idx)
        total_fitness = p1_fitness + p2_fitness + 0.01  # Avoid division by zero
        
        p1_weight = p1_fitness / total_fitness
        
        target_attrs = {}
        for attr in self.attributes:
            if attr == 'image_url':
                continue
            target_attrs[attr] = parent1.get(attr) if random.random() < p1_weight else parent2.get(attr)
        
        return self.find_best_match(target_attrs)
    
    def crossover_adaptive(self, parent1_idx: int, parent2_idx: int) -> int:
        """Adaptive crossover based on generation"""
        # Early generations: more exploration (uniform)
        # Later generations: more exploitation (fitness-weighted)
        exploration_ratio = max(0.2, 1.0 - (self.generation / 100))
        
        if random.random() < exploration_ratio:
            return self.crossover_uniform(parent1_idx, parent2_idx)
        else:
            return self.crossover_fitness_weighted(parent1_idx, parent2_idx)
    
    def crossover(self, parent1_idx: int, parent2_idx: int) -> int:
        """Perform crossover based on selected strategy"""
        if random.random() > self.crossover_rate:
            return parent1_idx if random.random() < 0.5 else parent2_idx
        
        strategy = self.crossover_strategy
        
        if strategy == 'uniform':
            return self.crossover_uniform(parent1_idx, parent2_idx)
        elif strategy == 'single_point':
            return self.crossover_single_point(parent1_idx, parent2_idx)
        elif strategy == 'two_point':
            return self.crossover_two_point(parent1_idx, parent2_idx)
        elif strategy == 'fitness_weighted':
            return self.crossover_fitness_weighted(parent1_idx, parent2_idx)
        elif strategy == 'adaptive':
            return self.crossover_adaptive(parent1_idx, parent2_idx)
        else:  # attribute_blend
            return self.crossover_attribute_blend(parent1_idx, parent2_idx)
    
    def mutate(self, pokemon_idx: int) -> int:
        """Adaptive mutation"""
        if random.random() < self.mutation_rate:
            fitness = self.fitness(pokemon_idx)
            # Higher mutation chance for low fitness
            if fitness < 10 or random.random() < 0.3:
                return self.df.sample(1).index[0]
        return pokemon_idx
    
    def evolve(self):
        """Run one generation"""
        # Clear fitness cache for new generation
        self.fitness_cache.clear()
        
        # Calculate fitness
        fitness_scores = [(idx, self.fitness(idx)) for idx in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Elite selection
        new_population = [idx for idx, _ in fitness_scores[:self.elite_size]]
        
        # Diversity preservation
        diversity_size = max(5, int(self.pop_size * 0.1))
        new_population.extend(self.df.sample(min(diversity_size, len(self.df))).index.tolist())
        
        # Generate offspring
        while len(new_population) < self.pop_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        
        self.population = new_population[:self.pop_size]
        self.generation += 1
    
    def next_guess(self) -> Tuple[pd.Series, Dict[str, Any]]:
        """Generate next guess using genetic algorithm"""
        # Evolve population
        for _ in range(self.generations_per_guess):
            self.evolve()
        
        # Get best individual
        fitness_scores = [(idx, self.fitness(idx)) for idx in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        best_idx = fitness_scores[0][0]
        
        pokemon = self.df.loc[best_idx]
        info = self.get_population_stats()
        
        return pokemon, info
    
    def update_feedback(self, guess: pd.Series, feedback: Dict[str, str]):
        """Update solver with new feedback"""
        guess_idx = guess.name
        self.add_feedback(guess_idx, feedback)
        self.fitness_cache.clear()  # Clear cache when new feedback is added
    
    def get_population_stats(self) -> Dict[str, Any]:
        """Get population statistics"""
        unique_pokemon = len(set(self.population))
        fitness_scores = [self.fitness(idx) for idx in self.population]
        avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0
        max_fitness = max(fitness_scores) if fitness_scores else 0
        min_fitness = min(fitness_scores) if fitness_scores else 0
        
        fitness_variance = sum((f - avg_fitness) ** 2 for f in fitness_scores) / len(fitness_scores) if fitness_scores else 0
        
        return {
            "algorithm": "GA",
            "generation": self.generation,
            "unique_pokemon": unique_pokemon,
            "candidates": unique_pokemon,
            "avg_fitness": round(avg_fitness, 2),
            "max_fitness": round(max_fitness, 2),
            "min_fitness": round(min_fitness, 2),
            "fitness_variance": round(fitness_variance, 2),
            "crossover_strategy": self.crossover_strategy,
            "population_diversity": round(unique_pokemon / self.pop_size * 100, 1)
        }
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information"""
        return self.get_population_stats()