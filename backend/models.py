# ============================================================
# FILE: models.py
# Pydantic Models
# ============================================================

from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class GAConfig(BaseModel):
    pop_size: int = Field(default=100, ge=10, le=500)
    elite_size: int = Field(default=20, ge=5, le=100)
    mutation_rate: float = Field(default=0.15, ge=0.0, le=1.0)
    crossover_rate: float = Field(default=0.8, ge=0.0, le=1.0)
    tournament_size: int = Field(default=7, ge=2, le=20)
    crossover_strategy: str = 'attribute_blend'
    generations_per_guess: int = Field(default=30, ge=1, le=200)

class SAConfig(BaseModel):
    initial_temp: float = Field(default=100.0, gt=0)
    cooling_rate: float = Field(default=0.95, gt=0, lt=1)
    min_temp: float = Field(default=0.01, gt=0)
    iterations_per_temp: int = Field(default=50, ge=1)
    reheat_threshold: float = Field(default=0.1, ge=0, le=1)

class AStarConfig(BaseModel):
    max_open_set: int = Field(default=1000, ge=10)
    beam_width: int = Field(default=100, ge=1)
    heuristic_weight: float = Field(default=1.0, ge=0)

class SolverConfig(BaseModel):
    algorithm: str
    attributes: List[str]
    heuristic: str = 'random'
    secret_pokemon: Optional[str] = None
    max_attempts: int = Field(default=10, ge=1, le=50)
    ga_config: Optional[GAConfig] = None
    sa_config: Optional[SAConfig] = None
    astar_config: Optional[AStarConfig] = None

class SolverStep(BaseModel):
    attempt: int
    guess_name: str
    guess_data: Dict
    feedback: Dict
    remaining_candidates: int
    timestamp: float
    image_url: Optional[str] = None
    heuristic_info: Optional[Dict] = None
    algorithm_state: Optional[Dict] = None

