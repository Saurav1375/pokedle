# ============================================================
# FILE: main.py
# Enhanced FastAPI Application
# ============================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time
import pandas as pd
from typing import List, Optional

# Import configurations and models
from config import *
from models import *
from data_loader import DataLoader
from feedback import get_feedback, is_complete_match

# Import algorithms
from algorithms.csp_solver import EnhancedPokedleCSP
from algorithms.ga_solver import EnhancedPokedleGA
from algorithms.astar_solver import AStarSolver
from algorithms.simulated_annealing import SimulatedAnnealingSolver

# Import utilities
from utils.metrics import calculate_metrics
from utils.validators import validate_config

app = FastAPI(
    title="Enhanced Pokedle Solver API",
    version="4.0",
    description="AI-powered Pokedle solver with multiple algorithms and heuristics"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data loader
data_loader = DataLoader()
data_loader.load_data(CSV_PATH)

# ============ Helper Functions ============

def create_solver(config: SolverConfig):
    """Factory function to create appropriate solver"""
    df = data_loader.get_dataframe()
    
    if config.algorithm == 'CSP':
        return EnhancedPokedleCSP(df, config.attributes, config.heuristic)
    
    elif config.algorithm == 'GA':
        ga_config = config.ga_config or GAConfig()
        return EnhancedPokedleGA(df, config.attributes, ga_config.dict())
    
    elif config.algorithm == 'ASTAR':
        astar_config = config.astar_config or AStarConfig()
        return AStarSolver(df, config.attributes, astar_config.dict())
    
    elif config.algorithm == 'SA':
        sa_config = config.sa_config or SAConfig()
        return SimulatedAnnealingSolver(df, config.attributes, sa_config.dict())
    
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

# ============ API Endpoints ============

@app.get("/")
def root():
    return {
        "message": "Enhanced Pokedle Solver API",
        "version": "4.0",
        "features": [
            "Multiple algorithms: CSP, GA, A*, Simulated Annealing",
            "Enhanced heuristics: MRV, LCV, Entropy, Degree, Forward Checking, Domain Wipeout",
            "Multiple crossover strategies for GA",
            "Adaptive temperature control for SA",
            "Beam search for A*"
        ]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "pokemon_loaded": data_loader.pokemon_count,
        "timestamp": time.time()
    }

@app.get("/pokemon")
def get_pokemon_list():
    """Get list of all Pokemon"""
    df = data_loader.get_dataframe()
    pokemon_list = []
    
    for _, row in df.iterrows():
        pokemon_list.append({
            "name": row['Original_Name'],
            "image_url": row.get('image_url', ''),
            "generation": int(row.get('Generation', 0)) if not pd.isna(row.get('Generation')) else None,
            "type1": row.get('Type1'),
            "type2": row.get('Type2') if not pd.isna(row.get('Type2')) else None
        })
    
    return {
        "pokemon": pokemon_list,
        "count": len(pokemon_list)
    }

@app.get("/config")
def get_config():
    """Get available configuration options"""
    return {
        "attributes": AVAILABLE_ATTRIBUTES,
        "algorithms": AVAILABLE_ALGORITHMS,
        "algorithm_descriptions": ALGORITHM_DESCRIPTIONS,
        "heuristics": AVAILABLE_HEURISTICS,
        "heuristic_descriptions": HEURISTIC_DESCRIPTIONS,
        "crossover_strategies": AVAILABLE_CROSSOVER_STRATEGIES,
        "crossover_descriptions": CROSSOVER_DESCRIPTIONS,
        "default_configs": {
            "ga": DEFAULT_GA_CONFIG,
            "sa": DEFAULT_SA_CONFIG,
            "astar": DEFAULT_ASTAR_CONFIG
        }
    }

@app.get("/algorithms/{algorithm}")
def get_algorithm_info(algorithm: str):
    """Get detailed information about a specific algorithm"""
    if algorithm.upper() not in AVAILABLE_ALGORITHMS:
        raise HTTPException(404, f"Algorithm {algorithm} not found")
    
    algo = algorithm.upper()
    
    info = {
        "name": algo,
        "description": ALGORITHM_DESCRIPTIONS.get(algo),
        "compatible_heuristics": [],
        "config_options": {}
    }
    
    if algo == 'CSP':
        info["compatible_heuristics"] = AVAILABLE_HEURISTICS
        info["config_options"] = {"heuristic": "Choose search heuristic"}
    
    elif algo == 'GA':
        info["compatible_heuristics"] = ["fitness-based"]
        info["config_options"] = DEFAULT_GA_CONFIG
    
    elif algo == 'ASTAR':
        info["compatible_heuristics"] = ["admissible distance-based"]
        info["config_options"] = DEFAULT_ASTAR_CONFIG
    
    elif algo == 'SA':
        info["compatible_heuristics"] = ["energy-based"]
        info["config_options"] = DEFAULT_SA_CONFIG
    
    return info

@app.post("/solve")
def solve(config: SolverConfig):
    """Main solving endpoint"""
    start_time = time.time()
    
    # Validate configuration
    try:
        validate_config(config.dict())
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(400, str(e))
    
    # Get secret Pokemon
    if config.secret_pokemon:
        secret = data_loader.get_pokemon_by_name(config.secret_pokemon)
        if secret is None:
            raise HTTPException(400, f"Pokemon '{config.secret_pokemon}' not found")
    else:
        secret = data_loader.get_random_pokemon()
    
    # Create solver
    try:
        solver = create_solver(config)
    except Exception as e:
        raise HTTPException(500, f"Failed to create solver: {str(e)}")
    
    # Solving loop
    steps = []
    success = False
    
    for attempt in range(1, config.max_attempts + 1):
        # Get next guess
        try:
            guess, heuristic_info = solver.next_guess()
        except Exception as e:
            raise HTTPException(500, f"Solver error at attempt {attempt}: {str(e)}")
        
        if guess is None:
            break
        
        # Calculate feedback
        feedback = get_feedback(secret, guess, config.attributes, NUMERIC_ATTRIBUTES)
        
        # Create step
        step = SolverStep(
            attempt=attempt,
            guess_name=guess['Original_Name'],
            guess_data={attr: str(guess.get(attr, 'N/A')) for attr in config.attributes},
            feedback=feedback,
            remaining_candidates=heuristic_info.get('candidates', 0),
            timestamp=time.time() - start_time,
            image_url=guess.get('image_url', ''),
            heuristic_info=heuristic_info,
            algorithm_state=solver.get_state_info() if hasattr(solver, 'get_state_info') else None
        )
        steps.append(step)
        
        # Check if solved
        if is_complete_match(feedback):
            success = True
            break
        
        # Update solver with feedback
        try:
            solver.update_feedback(guess, feedback)
        except Exception as e:
            raise HTTPException(500, f"Failed to update solver: {str(e)}")
    
    execution_time = time.time() - start_time
    
    # Calculate performance metrics
    metrics = calculate_metrics(steps, execution_time, success)
    
    return SolverResult(
        secret_name=secret['Original_Name'],
        secret_image=secret.get('image_url', ''),
        success=success,
        total_attempts=len(steps),
        steps=steps,
        execution_time=round(execution_time, 3),
        algorithm=config.algorithm,
        heuristic=config.heuristic,
        performance_metrics=metrics.to_dict()
    )

@app.post("/compare")
def compare_algorithms(
    algorithms: List[str],
    attributes: List[str],
    secret_pokemon: Optional[str] = None,
    max_attempts: int = 10
):
    """Compare multiple algorithms on the same Pokemon"""
    
    results = {}
    
    # Get secret Pokemon once
    if secret_pokemon:
        secret = data_loader.get_pokemon_by_name(secret_pokemon)
        if secret is None:
            raise HTTPException(400, f"Pokemon '{secret_pokemon}' not found")
    else:
        secret = data_loader.get_random_pokemon()
    
    secret_name = secret['Original_Name']
    
    for algo in algorithms:
        if algo.upper() not in AVAILABLE_ALGORITHMS:
            continue
        
        # Create config for this algorithm
        config = SolverConfig(
            algorithm=algo.upper(),
            attributes=attributes,
            secret_pokemon=secret_name,
            max_attempts=max_attempts,
            heuristic='entropy' if algo.upper() == 'CSP' else 'random'
        )
        
        try:
            result = solve(config)
            results[algo] = {
                "success": result.success,
                "attempts": result.total_attempts,
                "time": result.execution_time,
                "metrics": result.performance_metrics
            }
        except Exception as e:
            results[algo] = {"error": str(e)}
    
    # Determine winner
    winner = None
    if results:
        valid_results = [(k, v) for k, v in results.items() if "error" not in v and v.get("success")]
        if valid_results:
            winner = min(valid_results, key=lambda x: x[1]["attempts"])[0]
    
    return {
        "secret_pokemon": secret_name,
        "results": results,
        "winner": winner
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)