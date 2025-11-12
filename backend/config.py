# ============================================================
# FILE: config.py
# Configuration and Constants
# ============================================================

# CSV file path
CSV_PATH = "03_cleaned_with_images_and_evolutionary_stages.csv"

AVAILABLE_ATTRIBUTES = [
    'Generation', 'Height', 'Weight', 
    'Type1', 'Type2', 'Color', 'evolutionary_stage'
]

NUMERIC_ATTRIBUTES = ['Height', 'Weight']

AVAILABLE_ALGORITHMS = ['CSP', 'GA', 'ASTAR', 'SA']

AVAILABLE_HEURISTICS = [
    'random',           # Random selection
    'mrv',              # Minimum Remaining Values
    'lcv',              # Least Constraining Value
    'entropy',          # Maximum Information Gain
    'degree',           # Degree heuristic
    'mac',              # Maintaining Arc Consistency
    'forward_checking', # Forward checking
    'domain_wipeout',   # Domain wipeout prevention
]

AVAILABLE_CROSSOVER_STRATEGIES = [
    'attribute_blend',
    'uniform',
    'single_point',
    'two_point',
    'fitness_weighted',
    'adaptive'
]

HEURISTIC_DESCRIPTIONS = {
    "random": "Random selection from remaining candidates",
    "mrv": "Minimum Remaining Values - choose most constrained attribute",
    "lcv": "Least Constraining Value - minimize future constraint",
    "entropy": "Maximum information gain - highest uncertainty reduction",
    "degree": "Choose variable involved in most constraints",
    "mac": "Maintaining Arc Consistency - propagate constraints",
    "forward_checking": "Check future variable domains after assignment",
    "domain_wipeout": "Avoid assignments that cause domain wipeout"
}

ALGORITHM_DESCRIPTIONS = {
    "CSP": "Constraint Satisfaction Problem solver with various heuristics",
    "GA": "Genetic Algorithm with population-based evolution",
    "ASTAR": "A* Search algorithm with admissible heuristics",
    "SA": "Simulated Annealing with temperature-based optimization"
}

CROSSOVER_DESCRIPTIONS = {
    "attribute_blend": "Blend attributes based on parent fitness",
    "uniform": "50-50 chance for each attribute from either parent",
    "single_point": "Single crossover point splits attributes",
    "two_point": "Two crossover points create three segments",
    "fitness_weighted": "Higher fitness parent contributes more",
    "adaptive": "Adapts strategy based on generation"
}

# GA Configuration
DEFAULT_GA_CONFIG = {
    'pop_size': 100,
    'elite_size': 20,
    'mutation_rate': 0.15,
    'crossover_rate': 0.8,
    'tournament_size': 7,
    'crossover_strategy': 'attribute_blend',
    'generations_per_guess': 30
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
    'max_open_set': 1000,
    'beam_width': 100,
    'heuristic_weight': 1.0
}