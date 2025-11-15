import pandas as pd
import random
from typing import Dict, Tuple, Any, List, Set
from algorithms.base import BaseSolver

class GASolver(BaseSolver):
    """
    Genetic Algorithm solver for Pokedle.
    
    CORRECTED VERSION with proper GA principles:
    - Individual: A Pokemon (index into dataframe)
    - Population: Set of candidate Pokemon
    - Fitness: How well Pokemon satisfies known constraints from feedback
    - Selection: Tournament selection based on fitness
    - Crossover: Combine attributes from two parents to create offspring
    - Mutation: Random changes to explore search space
    
    Key correctness improvements:
    1. Fitness function properly evaluates constraint satisfaction
    2. Crossover creates valid Pokemon (not arbitrary attribute combinations)
    3. Elitism preserves best solutions
    4. Diversity maintenance prevents premature convergence
    """
    
    def __init__(self, dataframe: pd.DataFrame, attributes: list, config: dict):
        super().__init__(dataframe, attributes)
        
        # GA parameters
        self.pop_size = config.get('pop_size', 100)
        self.elite_size = config.get('elite_size', 20)
        self.mutation_rate = config.get('mutation_rate', 0.15)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        self.tournament_size = config.get('tournament_size', 5)
        self.generations_per_guess = config.get('generations_per_guess', 30)
        
        # Population: indices of Pokemon
        self.population = []
        self.initialize_population()
        
        # Track best individual
        self.best_individual = None
        self.best_fitness = -float('inf')
        
        # Generation counter
        self.generation = 0
        
        # Constraint tracking for fitness evaluation
        self.hard_constraints = []  # Must satisfy (green feedback)
        self.soft_constraints = []  # Should avoid (gray feedback)
        self.numeric_constraints = []  # Numeric bounds
    
    def initialize_population(self):
        """Initialize population with diverse Pokemon"""
        if len(self.df) <= self.pop_size:
            self.population = list(self.df.index)
        else:
            # Random sampling for diversity
            self.population = self.df.sample(self.pop_size).index.tolist()
    
    def fitness(self, pokemon_idx: int) -> float:
        """
        Calculate fitness of a Pokemon.
        
        CORRECTED: Fitness should measure constraint satisfaction, not arbitrary scoring.
        
        Fitness components:
        1. Hard constraints (green feedback): MUST be satisfied (high penalty if not)
        2. Soft constraints (gray feedback): Should be satisfied (medium penalty if not)
        3. Numeric constraints (higher/lower): Distance-based penalty
        4. Type constraints (yellow): Special handling for type swaps
        
        Returns: Float where higher = better fit
        """
        pokemon = self.df.loc[pokemon_idx]
        fitness = 100.0  # Start with base fitness
        
        # Hard constraints (from green feedback)
        for constraint_type, var, value in self.hard_constraints:
            pokemon_val = pokemon.get(var)
            if pd.isna(pokemon_val):
                pokemon_val = None
            
            if constraint_type == 'must_equal':
                if pokemon_val != value:
                    fitness -= 50  # Huge penalty for violating hard constraint
            elif constraint_type == 'must_not_equal':
                if pokemon_val == value:
                    fitness -= 50
        
        # Soft constraints (from gray feedback)
        for constraint_type, var, value in self.soft_constraints:
            pokemon_val = pokemon.get(var)
            if pd.isna(pokemon_val):
                pokemon_val = None
            
            if constraint_type == 'not_equal':
                if pokemon_val == value:
                    fitness -= 15  # Medium penalty
            elif constraint_type == 'type_not_in_any':
                # Value should not appear in either type
                type1 = pokemon.get('Type1')
                type2 = pokemon.get('Type2')
                if value in [type1, type2]:
                    fitness -= 15
        
        # Numeric constraints (from higher/lower feedback)
        for constraint_type, var, value in self.numeric_constraints:
            pokemon_val = pokemon.get(var)
            
            if pokemon_val is None or pd.isna(pokemon_val):
                fitness -= 10
                continue
            
            try:
                pokemon_num = float(pokemon_val)
                value_num = float(value)
                
                if constraint_type == 'greater_than':
                    if pokemon_num <= value_num:
                        # Penalty proportional to violation
                        violation = value_num - pokemon_num
                        fitness -= min(20, violation)
                    else:
                        # Reward for satisfaction
                        fitness += 5
                
                elif constraint_type == 'less_than':
                    if pokemon_num >= value_num:
                        violation = pokemon_num - value_num
                        fitness -= min(20, violation)
                    else:
                        fitness += 5
                
            except (ValueError, TypeError):
                fitness -= 10
        
        # Bonus for having diverse attributes (exploratory bonus)
        uniqueness_bonus = self.calculate_uniqueness(pokemon_idx)
        fitness += uniqueness_bonus * 2
        
        return max(0, fitness)
    
    def calculate_uniqueness(self, pokemon_idx: int) -> float:
        """
        Calculate how unique this Pokemon is compared to population.
        
        This helps maintain diversity and avoid premature convergence.
        """
        if not self.population:
            return 0
        
        pokemon = self.df.loc[pokemon_idx]
        uniqueness = 0
        
        # Sample some individuals to compare
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
        """
        Convert feedback into constraints for fitness evaluation.
        
        CORRECTED: Properly categorize constraints by type.
        """
        for var, status in feedback.items():
            if var not in self.attributes or var == 'image_url':
                continue
            
            value = guess.get(var)
            if pd.isna(value):
                value = None
            
            if status == 'green':
                # Hard constraint: must equal
                self.hard_constraints.append(('must_equal', var, value))
            
            elif status == 'gray':
                if var in ['Type1', 'Type2']:
                    # Type doesn't exist anywhere
                    self.soft_constraints.append(('type_not_in_any', var, value))
                else:
                    # Not equal
                    self.soft_constraints.append(('not_equal', var, value))
            
            elif status == 'yellow':
                # Type in wrong position
                # Current variable must not be this value
                self.soft_constraints.append(('not_equal', var, value))
                # But the other type must be this value
                other_type = 'Type2' if var == 'Type1' else 'Type1'
                self.hard_constraints.append(('must_equal', other_type, value))
            
            elif status == 'higher':
                self.numeric_constraints.append(('greater_than', var, value))
            
            elif status == 'lower':
                self.numeric_constraints.append(('less_than', var, value))
    
    def tournament_selection(self) -> int:
        """
        Select parent using tournament selection.
        
        CORRECTED: Proper tournament with fitness comparison.
        """
        if not self.population:
            return None
        
        tournament_size = min(self.tournament_size, len(self.population))
        tournament = random.sample(self.population, tournament_size)
        
        # Select best from tournament
        best = max(tournament, key=lambda idx: self.fitness(idx))
        return best
    
    def crossover(self, parent1_idx: int, parent2_idx: int) -> int:
        """
        Perform crossover to create offspring.
        
        CORRECTED: Returns actual Pokemon index (valid individual).
        
        Strategy: Find Pokemon that combines attributes from both parents.
        Uses attribute-based matching to find similar Pokemon in dataset.
        """
        if random.random() > self.crossover_rate:
            # No crossover - return one parent
            return parent1_idx if random.random() < 0.5 else parent2_idx
        
        parent1 = self.df.loc[parent1_idx]
        parent2 = self.df.loc[parent2_idx]
        
        # Weight parents by fitness
        fitness1 = self.fitness(parent1_idx)
        fitness2 = self.fitness(parent2_idx)
        total_fitness = fitness1 + fitness2
        
        if total_fitness == 0:
            weight1 = 0.5
        else:
            weight1 = fitness1 / total_fitness
        
        # Create target attribute profile
        target_attributes = {}
        for attr in self.attributes:
            if attr == 'image_url':
                continue
            
            # Probabilistically choose from parent1 or parent2
            if random.random() < weight1:
                target_attributes[attr] = parent1.get(attr)
            else:
                target_attributes[attr] = parent2.get(attr)
        
        # Find Pokemon that best matches target profile
        best_match = self.find_closest_pokemon(target_attributes)
        
        return best_match if best_match is not None else parent1_idx
    
    def find_closest_pokemon(self, target_attributes: Dict, search_pool: List[int] = None) -> int:
        """
        Find Pokemon that best matches target attribute profile.
        
        OPTIMIZED: Search in constrained candidate pool, not entire dataset.
        
        Args:
            target_attributes: Target attribute profile
            search_pool: Specific Pokemon indices to search (defaults to valid candidates)
        """
        # Search in valid candidates first (much smaller pool)
        if search_pool is None:
            search_pool = self.get_valid_candidates()
        
        if not search_pool:
            # Fallback to population if no valid candidates
            search_pool = self.population
        
        if not search_pool:
            # Last resort: random from dataset
            return self.df.sample(1).index[0]
        
        # Limit search size for performance
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
                    # Partial credit for numeric proximity
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
        """
        Get Pokemon indices that satisfy all known constraints.
        
        This dramatically reduces search space for crossover/mutation.
        """
        valid = []
        
        # Quick check: sample if dataset is large
        check_pool = list(range(len(self.df)))
        if len(check_pool) > 500:
            check_pool = random.sample(check_pool, 500)
        
        for idx in check_pool:
            if self.satisfies_constraints(idx):
                valid.append(idx)
        
        return valid if valid else list(range(len(self.df)))
    
    def satisfies_constraints(self, pokemon_idx: int) -> bool:
        """
        Quick check if Pokemon satisfies hard constraints.
        
        Used to filter candidate pool before expensive operations.
        """
        pokemon = self.df.loc[pokemon_idx]
        
        # Check hard constraints (must satisfy)
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
        """
        Mutate an individual.
        
        CORRECTED: Returns actual Pokemon (not invalid combination).
        
        Strategy: With mutation probability, replace with similar Pokemon.
        """
        if random.random() > self.mutation_rate:
            return pokemon_idx
        
        # Mutation: find a nearby Pokemon (similar in some attributes)
        pokemon = self.df.loc[pokemon_idx]
        
        # Determine mutation strength
        if random.random() < 0.3:
            # Strong mutation: random Pokemon
            return self.df.sample(1).index[0]
        else:
            # Weak mutation: find similar Pokemon
            # Keep some attributes, randomize others
            target_attributes = {}
            for attr in self.attributes:
                if attr == 'image_url':
                    continue
                
                # 50% chance to keep attribute
                if random.random() < 0.5:
                    target_attributes[attr] = pokemon.get(attr)
                # else: leave unspecified for diversity
            
            mutant = self.find_closest_pokemon(target_attributes)
            return mutant if mutant is not None else pokemon_idx
    
    def evolve_generation(self):
        """
        Evolve population for one generation.
        
        CORRECTED: Proper GA evolutionary cycle.
        
        Steps:
        1. Evaluate fitness of all individuals
        2. Select elite individuals (best performers)
        3. Create offspring through selection, crossover, mutation
        4. Replace old population with new generation
        """
        # Evaluate fitness
        fitness_scores = [(idx, self.fitness(idx)) for idx in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Track best individual
        if fitness_scores[0][1] > self.best_fitness:
            self.best_fitness = fitness_scores[0][1]
            self.best_individual = fitness_scores[0][0]
        
        # Elitism: keep best individuals
        new_population = [idx for idx, _ in fitness_scores[:self.elite_size]]
        
        # Generate offspring
        while len(new_population) < self.pop_size:
            # Selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # Crossover
            offspring = self.crossover(parent1, parent2)
            
            # Mutation
            offspring = self.mutate(offspring)
            
            new_population.append(offspring)
        
        # Diversity injection: add some random individuals
        diversity_size = max(5, int(self.pop_size * 0.05))
        random_individuals = self.df.sample(diversity_size).index.tolist()
        
        # Replace worst individuals with random ones
        new_population[-diversity_size:] = random_individuals
        
        self.population = new_population
        self.generation += 1
    
    def next_guess(self) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Generate next guess using GA.
        
        Strategy: Evolve population, return best individual.
        """
        # Evolve for specified number of generations
        for _ in range(self.generations_per_guess):
            self.evolve_generation()
        
        # Return best individual
        if self.best_individual is None:
            # Fallback: random from population
            best_idx = random.choice(self.population)
        else:
            best_idx = self.best_individual
        
        pokemon = self.df.loc[best_idx]
        
        # Calculate population statistics
        fitness_scores = [self.fitness(idx) for idx in self.population]
        avg_fitness = sum(fitness_scores) / len(fitness_scores) if fitness_scores else 0
        unique_individuals = len(set(self.population))
        
        info = {
            "algorithm": "GA",
            "generation": self.generation,
            "best_fitness": round(self.best_fitness, 2),
            "avg_fitness": round(avg_fitness, 2),
            "population_diversity": round(unique_individuals / self.pop_size * 100, 1),
            "population_size": len(self.population),
            "num_constraints": len(self.hard_constraints) + len(self.soft_constraints) + len(self.numeric_constraints)
        }
        
        return pokemon, info
    
    def update_feedback(self, guess: pd.Series, feedback: Dict[str, str]):
        """Update solver with new feedback"""
        guess_idx = guess.name
        self.add_feedback(guess_idx, feedback)
        
        # Update constraints
        self.update_constraints_from_feedback(guess, feedback)
        
        # Reset best individual (constraints changed)
        # Let evolution find new best
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
            "numeric_constraints": len(self.numeric_constraints)
        }