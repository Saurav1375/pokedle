# CSV file path
CSV_PATH = "03_cleaned_with_images_and_evolutionary_stages.csv"

AVAILABLE_ATTRIBUTES = [
    'Generation', 'Height', 'Weight', 
    'Type1', 'Type2', 'Color', 'evolutionary_stage'
]

NUMERIC_ATTRIBUTES = ['Height', 'Weight']

AVAILABLE_ALGORITHMS = ['CSP', 'GA', 'ASTAR', 'SA']

# CSP Heuristics - Split into two categories
VARIABLE_ORDERING_HEURISTICS = [
    'mrv',              # Minimum Remaining Values
    'degree',           # Degree heuristic (most constrained)
    'mrv_degree',       # MRV with degree tiebreaker
    'none',             # No heuristic (first available)
]

VALUE_ORDERING_HEURISTICS = [
    'lcv',              # Least Constraining Value
    'most_common',      # Most frequently occurring value
    'none',             # No heuristic (first available)
]

# Legacy support - for backward compatibility
AVAILABLE_HEURISTICS = VARIABLE_ORDERING_HEURISTICS

AVAILABLE_CROSSOVER_STRATEGIES = [
    'attribute_blend',  # No longer used - GA now uses proper crossover
    'uniform',
    'single_point',
    'two_point',
    'fitness_weighted',
    'adaptive'
]

VARIABLE_HEURISTIC_DESCRIPTIONS = {
    "mrv": "Minimum Remaining Values - choose attribute with smallest domain (fail-fast)",
    "degree": "Degree heuristic - choose attribute with most constraints",
    "mrv_degree": "MRV with degree as tiebreaker - best of both worlds",
    "none": "No heuristic - choose first available variable"
}

VALUE_HEURISTIC_DESCRIPTIONS = {
    "lcv": "Least Constraining Value - choose value that rules out fewest options",
    "most_common": "Most common value - choose most frequently occurring value in candidates",
    "none": "No heuristic - choose first available value"
}

ALGORITHM_DESCRIPTIONS = {
    "CSP": "Constraint Satisfaction Problem solver with AC-3 propagation and dual heuristics",
    "GA": "Genetic Algorithm with population-based evolution and valid Pokemon crossover",
    "ASTAR": "A* Search algorithm with admissible heuristic guaranteeing optimal solution",
    "SA": "Simulated Annealing with temperature-based optimization and energy minimization"
}

# GA Configuration - OPTIMIZED for speed
DEFAULT_GA_CONFIG = {
    'pop_size': 50,              # Reduced
    'elite_size': 10,            # Reduced
    'mutation_rate': 0.2,
    'crossover_rate': 0.7,
    'tournament_size': 3,
    'generations_per_guess': 10  # KEY FIX: 30 → 10
}

# SA Configuration
DEFAULT_SA_CONFIG = {
    'initial_temp': 100.0,
    'cooling_rate': 0.95,
    'min_temp': 0.01,
    'iterations_per_temp': 50,
    'reheat_threshold': 0.1
}

# A* Configuration
DEFAULT_ASTAR_CONFIG = {
    'beam_width': 100,
    'heuristic_weight': 1.0  # 1.0 ensures admissibility
}

# CSP Configuration
DEFAULT_CSP_CONFIG = {
    'variable_heuristic': 'mrv',
    'value_heuristic': 'lcv',
    'use_ac3': True
}