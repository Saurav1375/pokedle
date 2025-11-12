# ============================================================
# FILE: models.py
# Pydantic Models
# ============================================================

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class GAConfig(BaseModel):
    """Genetic Algorithm Configuration"""
    pop_size: int = Field(default=100, ge=10, le=500, description="Population size")
    elite_size: int = Field(default=20, ge=5, le=100, description="Number of elite individuals")
    mutation_rate: float = Field(default=0.15, ge=0.0, le=1.0, description="Mutation probability")
    crossover_rate: float = Field(default=0.8, ge=0.0, le=1.0, description="Crossover probability")
    tournament_size: int = Field(default=7, ge=2, le=20, description="Tournament selection size")
    crossover_strategy: str = Field(default='attribute_blend', description="Crossover strategy")
    generations_per_guess: int = Field(default=30, ge=1, le=200, description="Generations per guess")

class SAConfig(BaseModel):
    """Simulated Annealing Configuration"""
    initial_temp: float = Field(default=100.0, gt=0, description="Initial temperature")
    cooling_rate: float = Field(default=0.95, gt=0, lt=1, description="Temperature cooling rate")
    min_temp: float = Field(default=0.01, gt=0, description="Minimum temperature")
    iterations_per_temp: int = Field(default=50, ge=1, description="Iterations per temperature")
    reheat_threshold: float = Field(default=0.1, ge=0, le=1, description="Reheat threshold")

class AStarConfig(BaseModel):
    """A* Search Configuration"""
    max_open_set: int = Field(default=1000, ge=10, description="Maximum open set size")
    beam_width: int = Field(default=100, ge=1, description="Beam search width")
    heuristic_weight: float = Field(default=1.0, ge=0, description="Heuristic weight factor")

class SolverConfig(BaseModel):
    """Main Solver Configuration"""
    algorithm: str = Field(description="Algorithm to use (CSP, GA, ASTAR, SA)")
    attributes: List[str] = Field(description="List of attributes to use")
    heuristic: str = Field(default='random', description="Heuristic for CSP")
    secret_pokemon: Optional[str] = Field(default=None, description="Secret Pokemon name (random if None)")
    max_attempts: int = Field(default=10, ge=1, le=50, description="Maximum number of guesses")
    ga_config: Optional[GAConfig] = Field(default=None, description="GA configuration")
    sa_config: Optional[SAConfig] = Field(default=None, description="SA configuration")
    astar_config: Optional[AStarConfig] = Field(default=None, description="A* configuration")

class SolverStep(BaseModel):
    """Single step in the solving process"""
    attempt: int = Field(description="Attempt number")
    guess_name: str = Field(description="Guessed Pokemon name")
    guess_data: Dict[str, Any] = Field(description="Attribute values of guess")
    feedback: Dict[str, str] = Field(description="Feedback for each attribute")
    remaining_candidates: int = Field(description="Number of remaining candidates")
    timestamp: float = Field(description="Time elapsed since start")
    image_url: Optional[str] = Field(default=None, description="Pokemon image URL")
    heuristic_info: Optional[Dict[str, Any]] = Field(default=None, description="Heuristic-specific info")
    algorithm_state: Optional[Dict[str, Any]] = Field(default=None, description="Algorithm state info")

class SolverResult(BaseModel):
    """Complete solving result"""
    secret_name: str = Field(description="Name of secret Pokemon")
    secret_image: str = Field(description="Image URL of secret Pokemon")
    success: bool = Field(description="Whether solver succeeded")
    total_attempts: int = Field(description="Total number of attempts")
    steps: List[SolverStep] = Field(description="List of solving steps")
    execution_time: float = Field(description="Total execution time in seconds")
    algorithm: str = Field(description="Algorithm used")
    heuristic: str = Field(description="Heuristic used")
    performance_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Performance metrics")

class PokemonInfo(BaseModel):
    """Basic Pokemon information"""
    name: str = Field(description="Pokemon name")
    image_url: str = Field(description="Image URL")
    generation: Optional[int] = Field(default=None, description="Generation number")
    type1: Optional[str] = Field(default=None, description="Primary type")
    type2: Optional[str] = Field(default=None, description="Secondary type")

class ComparisonRequest(BaseModel):
    """Request for algorithm comparison"""
    algorithms: List[str] = Field(description="Algorithms to compare")
    attributes: List[str] = Field(description="Attributes to use")
    secret_pokemon: Optional[str] = Field(default=None, description="Secret Pokemon name")
    max_attempts: int = Field(default=10, ge=1, le=50, description="Max attempts per algorithm")
    num_runs: int = Field(default=1, ge=1, le=10, description="Number of runs per algorithm")