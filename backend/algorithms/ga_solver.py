import pandas as pd
import random
from typing import Dict, Tuple, Any, List, Set, Optional, Callable
from algorithms.base import BaseSolver
import numpy as np

class GASolver(BaseSolver):
    """
    Genetic Algorithm solver for Pokedle with detailed generation tracking.
    """
    
    def __init__(self, dataframe: pd.DataFrame, attributes: list, config: dict, progress_callback: Optional[Callable] = None):
        super().__init__(dataframe, attributes)
        print(f"[DEBUG] GASolver init - generations_per_guess: {config.get('generations_per_guess')}")
        print(f"[DEBUG] GASolver init - pop_size: {config.get('pop_size')}")
        print(f"[DEBUG] GASolver init - has callback: {progress_callback is not None}")
    
        
        self.progress_callback = progress_callback
    
        # Use new defaults
        self.pop_size = config.get('pop_size', 50)  # Was 100
        self.generations_per_guess = config.get('generations_per_guess', 10)  # Was 30
        
    
        # GA parameters
        self.pop_size = config.get('pop_size', 50)              # OPTIMIZED: Was 100
        self.elite_size = config.get('elite_size', 10)          # OPTIMIZED: Was 20
        self.mutation_rate = config.get('mutation_rate', 0.2)   # OPTIMIZED: Was 0.15
        self.crossover_rate = config.get('crossover_rate', 0.7) # OPTIMIZED: Was 0.8
        self.tournament_size = config.get('tournament_size', 3) # OPTIMIZED: Was 5
        self.generations_per_guess = config.get('generations_per_guess', 10)  # OPTIMIZED: Was 30
        

        
        print(f"[DEBUG] Final generations_per_guess: {self.generations_per_guess}")

        # Population: indices of Pokemon
        self.population = []
        self.initialize_population()
        
        # Track best individual
        self.best_individual = None
        self.best_fitness = -float('inf')
        self.fitness_cache = {}
        
        # Generation counter
        self.generation = 0
        
        # Constraint tracking for fitness evaluation
        self.hard_constraints = []
        self.soft_constraints = []
        self.numeric_constraints = []
        
        # NEW: Detailed generation history for visualization
        self.generation_history = []
        self.current_generation_details = None
    
    def initialize_population(self):
        """Initialize population with diverse Pokemon"""
        if len(self.df) <= self.pop_size:
            self.population = list(self.df.index)
        else:
            self.population = self.df.sample(self.pop_size).index.tolist()
    
    def fitness(self, pokemon_idx: int) -> float:
        """
        Calculate fitness of a Pokemon.
        
        Fitness starts at 0 and increases for satisfying constraints.
        Maximum fitness is 100 (perfect match).
        """
        # Check cache first
        cache_key = (pokemon_idx, len(self.feedback_history))
        if cache_key in self.fitness_cache:
            return self.fitness_cache[cache_key]
        
        pokemon = self.df.loc[pokemon_idx]
        
        # NEW APPROACH: Start at 0, add points for satisfaction
        fitness = 0.0
        total_constraints = 0
        satisfied_constraints = 0
        
        # Hard constraints (worth 30 points each when satisfied)
        for constraint_type, var, value in self.hard_constraints:
            total_constraints += 1
            pokemon_val = pokemon.get(var)
            if pd.isna(pokemon_val):
                pokemon_val = None
            
            if constraint_type == 'must_equal':
                if pokemon_val == value:
                    satisfied_constraints += 1
                    fitness += 30
            elif constraint_type == 'must_not_equal':
                if pokemon_val != value:
                    satisfied_constraints += 1
                    fitness += 30
        
        # Soft constraints (worth 15 points each when satisfied)
        for constraint_type, var, value in self.soft_constraints:
            total_constraints += 1
            pokemon_val = pokemon.get(var)
            if pd.isna(pokemon_val):
                pokemon_val = None
            
            if constraint_type == 'not_equal':
                if pokemon_val != value:
                    satisfied_constraints += 1
                    fitness += 15
            elif constraint_type == 'type_not_in_any':
                type1 = pokemon.get('Type1')
                type2 = pokemon.get('Type2')
                if value not in [type1, type2]:
                    satisfied_constraints += 1
                    fitness += 15
        
        # Numeric constraints (worth 20 points each when satisfied)
        for constraint_type, var, value in self.numeric_constraints:
            total_constraints += 1
            pokemon_val = pokemon.get(var)
            
            if pokemon_val is None or pd.isna(pokemon_val):
                continue
            
            try:
                pokemon_num = float(pokemon_val)
                value_num = float(value)
                
                if constraint_type == 'greater_than':
                    if pokemon_num > value_num:
                        satisfied_constraints += 1
                        fitness += 20
                
                elif constraint_type == 'less_than':
                    if pokemon_num < value_num:
                        satisfied_constraints += 1
                        fitness += 20
                
            except (ValueError, TypeError):
                pass
        
        # If no constraints yet, use diversity score
        if total_constraints == 0:
            uniqueness = self.calculate_uniqueness(pokemon_idx)
            fitness = uniqueness * 50  # Base exploration fitness
        
        # Normalize to 0-100 scale if we have constraints
        if total_constraints > 0:
            # Perfect satisfaction = 100
            # No satisfaction = 0
            fitness = (satisfied_constraints / total_constraints) * 100
        
        # Cache result (already 0-100, no need to cap)
        self.fitness_cache[cache_key] = fitness
        return fitness
    
    def calculate_uniqueness(self, pokemon_idx: int) -> float:
        """Calculate how unique this Pokemon is compared to population."""
        if not self.population:
            return 0
        
        pokemon = self.df.loc[pokemon_idx]
        uniqueness = 0
        
        sample_size = min(20, len(self.population))
        sample = random.sample(self.population, sample_size)
        
        for other_idx in sample:
            if other_idx == pokemon_idx:
                continue
            
            other = self.df.loc[other_idx]
            differences = 0
            
            for attr in self.attributes:
                if attr == 'image_url':
                    continue
                
                val1 = pokemon.get(attr)
                val2 = other.get(attr)
                
                if pd.isna(val1):
                    val1 = None
                if pd.isna(val2):
                    val2 = None
                
                if val1 != val2:
                    differences += 1
            
            uniqueness += differences
        
        return uniqueness / (sample_size * len(self.attributes))
    
    def update_constraints_from_feedback(self, guess: pd.Series, feedback: Dict[str, str]):
        """Convert feedback into constraints for fitness evaluation."""
        for var, status in feedback.items():
            if var not in self.attributes or var == 'image_url':
                continue
            
            value = guess.get(var)
            if pd.isna(value):
                value = None
            
            if status == 'green':
                self.hard_constraints.append(('must_equal', var, value))
            elif status == 'gray':
                if var in ['Type1', 'Type2']:
                    self.soft_constraints.append(('type_not_in_any', var, value))
                else:
                    self.soft_constraints.append(('not_equal', var, value))
            elif status == 'yellow':
                self.soft_constraints.append(('not_equal', var, value))
                other_type = 'Type2' if var == 'Type1' else 'Type1'
                self.hard_constraints.append(('must_equal', other_type, value))
            elif status == 'higher':
                self.numeric_constraints.append(('greater_than', var, value))
            elif status == 'lower':
                self.numeric_constraints.append(('less_than', var, value))
    
    def tournament_selection(self) -> int:
        """Select parent using tournament selection."""
        if not self.population:
            return None
        
        tournament_size = min(self.tournament_size, len(self.population))
        tournament = random.sample(self.population, tournament_size)
        best = max(tournament, key=lambda idx: self.fitness(idx))
        return best
    
    def crossover(self, parent1_idx: int, parent2_idx: int) -> int:
        """Perform crossover to create offspring."""
        if random.random() > self.crossover_rate:
            return parent1_idx if random.random() < 0.5 else parent2_idx
        
        parent1 = self.df.loc[parent1_idx]
        parent2 = self.df.loc[parent2_idx]
        
        fitness1 = self.fitness(parent1_idx)
        fitness2 = self.fitness(parent2_idx)
        total_fitness = fitness1 + fitness2
        
        if total_fitness == 0:
            weight1 = 0.5
        else:
            weight1 = fitness1 / total_fitness
        
        target_attributes = {}
        for attr in self.attributes:
            if attr == 'image_url':
                continue
            
            if random.random() < weight1:
                target_attributes[attr] = parent1.get(attr)
            else:
                target_attributes[attr] = parent2.get(attr)
        
        best_match = self.find_closest_pokemon(target_attributes)
        return best_match if best_match is not None else parent1_idx
    
    def find_closest_pokemon(self, target_attributes: Dict, search_pool: List[int] = None) -> int:
        """Find Pokemon that best matches target attribute profile."""
        if search_pool is None:
            search_pool = self.get_valid_candidates()
        
        if not search_pool:
            search_pool = self.population
        
        if not search_pool:
            return self.df.sample(1).index[0]
        
        if len(search_pool) > 100:
            search_pool = random.sample(search_pool, 100)
        
        best_match = None
        best_score = -1
        
        for idx in search_pool:
            pokemon = self.df.loc[idx]
            score = 0
            
            for attr, target_val in target_attributes.items():
                pokemon_val = pokemon.get(attr)
                
                if pd.isna(pokemon_val):
                    pokemon_val = None
                if pd.isna(target_val):
                    target_val = None
                
                if pokemon_val == target_val:
                    score += 1
                elif attr in ['Height', 'Weight'] and pokemon_val is not None and target_val is not None:
                    try:
                        diff = abs(float(pokemon_val) - float(target_val))
                        max_diff = self.df[attr].max() - self.df[attr].min()
                        if max_diff > 0:
                            score += 1 - (diff / max_diff)
                    except (ValueError, TypeError):
                        pass
            
            if score > best_score:
                best_score = score
                best_match = idx
        
        return best_match if best_match is not None else search_pool[0]
    
    def get_valid_candidates(self) -> List[int]:
        """Get Pokemon indices that satisfy all known constraints."""
        valid = []
        check_pool = list(range(len(self.df)))
        if len(check_pool) > 500:
            check_pool = random.sample(check_pool, 500)
        
        for idx in check_pool:
            if self.satisfies_constraints(idx):
                valid.append(idx)
        
        return valid if valid else list(range(len(self.df)))
    
    def satisfies_constraints(self, pokemon_idx: int) -> bool:
        """Quick check if Pokemon satisfies hard constraints."""
        pokemon = self.df.loc[pokemon_idx]
        
        for constraint_type, var, value in self.hard_constraints:
            pokemon_val = pokemon.get(var)
            if pd.isna(pokemon_val):
                pokemon_val = None
            
            if constraint_type == 'must_equal':
                if pokemon_val != value:
                    return False
            elif constraint_type == 'must_not_equal':
                if pokemon_val == value:
                    return False
        
        return True
    
    def mutate(self, pokemon_idx: int) -> int:
        """Mutate an individual."""
        if random.random() > self.mutation_rate:
            return pokemon_idx
        
        pokemon = self.df.loc[pokemon_idx]
        
        if random.random() < 0.3:
            return self.df.sample(1).index[0]
        else:
            target_attributes = {}
            for attr in self.attributes:
                if attr == 'image_url':
                    continue
                
                if random.random() < 0.5:
                    target_attributes[attr] = pokemon.get(attr)
            
            mutant = self.find_closest_pokemon(target_attributes)
            return mutant if mutant is not None else pokemon_idx
    
    def get_pokemon_info(self, idx: int) -> Dict[str, Any]:
        """Get detailed Pokemon information for visualization."""
        pokemon = self.df.loc[idx]
        return {
            "index": int(idx),
            "name": pokemon['Original_Name'],
            "fitness": round(self.fitness(idx), 2),
            "attributes": {
                attr: str(pokemon.get(attr, 'N/A')) 
                for attr in self.attributes if attr != 'image_url'
            },
            "image_url": pokemon.get('image_url', '')
        }
    
    def evolve_generation(self):
        """Evolve population for one generation with detailed tracking."""
        # NEW: Initialize generation details
        generation_details = {
            "generation_number": self.generation,
            "initial_population": [],
            "selection_pairs": [],
            "crossover_results": [],
            "mutation_results": [],
            "elite_preserved": [],
            "new_population": [],
            "fitness_stats": {}
        }
        print(f"[DEBUG] Generation {self.generation} starting...")  # ADD THIS
        # Evaluate fitness
        fitness_scores = [(idx, self.fitness(idx)) for idx in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Track initial population
        generation_details["initial_population"] = [
            self.get_pokemon_info(idx) for idx, _ in fitness_scores[:10]  # Top 10
        ]
        
        # Track best individual
        if fitness_scores[0][1] > self.best_fitness:
            self.best_fitness = fitness_scores[0][1]
            self.best_individual = fitness_scores[0][0]
        print(f"[DEBUG] Best fitness: {self.best_fitness}")  # ADD THIS

        if self.best_fitness >= 100:
            print("[DEBUG] Perfect fitness reached!")  # ADD THIS    
            return True  # Signal to stop
        
        # ADD THIS: Progress callback
        if self.progress_callback:
            print("[DEBUG] Calling progress callback") 
            avg_fitness = sum(f for _, f in fitness_scores) / len(fitness_scores)
            self.progress_callback({
                'type': 'generation',
                'generation': self.generation,
                'best_fitness': round(self.best_fitness, 2),
                'avg_fitness': round(avg_fitness, 2),
            })
        else:
            print("[DEBUG] No progress callback!")  # ADD THIS
            
        # Elitism: keep best individuals
        new_population = [idx for idx, _ in fitness_scores[:self.elite_size]]
        generation_details["elite_preserved"] = [
            self.get_pokemon_info(idx) for idx in new_population[:5]  # Top 5 elite
        ]
        
        # Generate offspring
        offspring_count = 0
        max_tracked = 5  # Track first 5 crossovers
        
        while len(new_population) < self.pop_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Track selection (only first few)
            if offspring_count < max_tracked:
                generation_details["selection_pairs"].append({
                    "parent1": self.get_pokemon_info(parent1),
                    "parent2": self.get_pokemon_info(parent2)
                })
            
            # Crossover
            offspring = self.crossover(parent1, parent2)
            
            # Track crossover result
            if offspring_count < max_tracked:
                generation_details["crossover_results"].append({
                    "offspring": self.get_pokemon_info(offspring),
                    "is_new": offspring not in [parent1, parent2]
                })
            
            # Mutation
            mutated = self.mutate(offspring)
            
            # Track mutation result
            if offspring_count < max_tracked:
                generation_details["mutation_results"].append({
                    "before": self.get_pokemon_info(offspring),
                    "after": self.get_pokemon_info(mutated),
                    "mutated": mutated != offspring
                })
            
            new_population.append(mutated)
            offspring_count += 1
        
        # Diversity injection
        diversity_size = max(5, int(self.pop_size * 0.05))
        random_individuals = self.df.sample(diversity_size).index.tolist()
        new_population[-diversity_size:] = random_individuals
        
        # Calculate fitness stats for new population
        new_fitness_scores = [self.fitness(idx) for idx in new_population]
        generation_details["fitness_stats"] = {
            "min": round(min(new_fitness_scores), 2),
            "max": round(max(new_fitness_scores), 2),
            "avg": round(sum(new_fitness_scores) / len(new_fitness_scores), 2),
            "median": round(sorted(new_fitness_scores)[len(new_fitness_scores) // 2], 2)
        }
        
        # Track new population (sample)
        generation_details["new_population"] = [
            self.get_pokemon_info(idx) for idx in new_population[:10]
        ]
        
        self.population = new_population
        self.generation += 1
        
        # Store generation details
        self.current_generation_details = generation_details
        self.generation_history.append(generation_details)
        
        return False
    


    def next_guess(self) -> Tuple[pd.Series, Dict[str, Any]]:
        """Generate next guess using GA with detailed tracking."""
        # Clear generation history for this guess
        self.generation_history = []
        
        # Evolve for specified number of generations
        for _ in range(self.generations_per_guess):
            should_stop = self.evolve_generation()
            if should_stop:
                break
        
        # Return best individual
        if self.best_individual is None:
            best_idx = random.choice(self.population)
        else:
            best_idx = self.best_individual
        
        pokemon = self.df.loc[best_idx]
        
        # Calculate population statistics
        fitness_scores = [self.fitness(idx) for idx in self.population]
        avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0
        unique_individuals = len(set(self.population))
        
        serializable_history = _convert_to_serializable(self.generation_history)

        info = {
            "algorithm": "GA",
            "generation": self.generation,
            "best_fitness": round(self.best_fitness, 2),
            "avg_fitness": round(avg_fitness, 2),
            "population_diversity": round(unique_individuals / self.pop_size * 100, 1),
            "population_size": len(self.population),
            "num_constraints": len(self.hard_constraints) + len(self.soft_constraints) + len(self.numeric_constraints),
            "generation_history": serializable_history
        }
        
        return pokemon, info
    
    def update_feedback(self, guess: pd.Series, feedback: Dict[str, str]):
        """Update solver with new feedback"""
        guess_idx = guess.name
        self.add_feedback(guess_idx, feedback)
        
        self.update_constraints_from_feedback(guess, feedback)
        
        # CRITICAL: Clear fitness cache when constraints change
        self.fitness_cache = {}
        
        # Re-evaluate best fitness with new constraints
        self.best_fitness = -float('inf')
        for idx in self.population:
            fitness = self.fitness(idx)
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = idx
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information"""
        return {
            "algorithm": "GA",
            "generation": self.generation,
            "population_size": len(self.population),
            "best_fitness": round(self.best_fitness, 2),
            "hard_constraints": len(self.hard_constraints),
            "soft_constraints": len(self.soft_constraints),
            "numeric_constraints": len(self.numeric_constraints),
            "generation_history": self.generation_history
        }
        
def _convert_to_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj