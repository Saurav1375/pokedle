# ============================================================
# FILE: algorithms/astar_solver.py - FIXED
# ============================================================

import pandas as pd
import heapq
import math
from typing import Dict, List, Tuple, Any
from algorithms.base import BaseSolver

class Node:
    """Node for A* search tree"""
    def __init__(self, pokemon_idx: int, g_cost: float, h_cost: float, parent=None):
        self.pokemon_idx = pokemon_idx
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost
        self.parent = parent
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost

class AStarSolver(BaseSolver):
    """A* Search algorithm for Pokedle"""
    
    def __init__(self, dataframe: pd.DataFrame, attributes: list, config: dict):
        super().__init__(dataframe, attributes)
        self.max_open_set = config.get('max_open_set', 1000)
        self.beam_width = config.get('beam_width', 100)
        self.heuristic_weight = config.get('heuristic_weight', 1.0)
        self.open_set = []
        self.closed_set = set()
        self.candidates = set(dataframe.index)
        self.constraints = {attr: [] for attr in attributes}
    
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
    
    def heuristic_distance(self, pokemon_idx: int) -> float:
        """Admissible heuristic: estimate minimum remaining guesses"""
        pokemon = self.df.loc[pokemon_idx]
        
        if not self.feedback_history:
            return self._diversity_heuristic(pokemon)
        
        violations = 0
        satisfied = 0
        
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
                        violations += 2
                
                elif status == 'gray':
                    if attr in ['Type1', 'Type2']:
                        pokemon_types = self._get_pokemon_types(pokemon)
                        if guess_val is not None and not (isinstance(guess_val, float) and pd.isna(guess_val)):
                            if guess_val in pokemon_types:
                                violations += 1
                    else:
                        if self._safe_value_check(pokemon_val, guess_val):
                            violations += 1
                
                elif status == 'yellow':
                    pokemon_types = self._get_pokemon_types(pokemon)
                    if guess_val is None or (isinstance(guess_val, float) and pd.isna(guess_val)):
                        violations += 1
                    elif guess_val not in pokemon_types:
                        violations += 1
                
                elif status == 'higher':
                    try:
                        if pokemon_val is not None and guess_val is not None:
                            if not (isinstance(pokemon_val, float) and pd.isna(pokemon_val)):
                                if not (isinstance(guess_val, float) and pd.isna(guess_val)):
                                    if float(pokemon_val) <= float(guess_val):
                                        violations += 1
                    except (ValueError, TypeError):
                        pass
                
                elif status == 'lower':
                    try:
                        if pokemon_val is not None and guess_val is not None:
                            if not (isinstance(pokemon_val, float) and pd.isna(pokemon_val)):
                                if not (isinstance(guess_val, float) and pd.isna(guess_val)):
                                    if float(pokemon_val) >= float(guess_val):
                                        violations += 1
                    except (ValueError, TypeError):
                        pass
        
        if satisfied == len(self.attributes) * len(self.feedback_history):
            return 0
        
        return violations / (satisfied + 1)
    
    def _diversity_heuristic(self, pokemon: pd.Series) -> float:
        """Heuristic based on attribute diversity"""
        score = 0
        
        for attr in self.attributes:
            if attr == 'image_url':
                continue
            
            value = pokemon.get(attr)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                score += 0.5
                continue
            
            matching = (self.df.loc[list(self.candidates)][attr] == value).sum()
            total = len(self.candidates)
            
            ratio = matching / total if total > 0 else 0
            score += abs(0.5 - ratio)
        
        return score
    
    def next_guess(self) -> Tuple[pd.Series, Dict[str, Any]]:
        """Generate next guess using A* search"""
        
        if not self.open_set and not self.closed_set:
            initial_candidates = self._select_initial_candidates()
            for idx in initial_candidates:
                h_cost = self.heuristic_distance(idx) * self.heuristic_weight
                node = Node(idx, 0, h_cost)
                heapq.heappush(self.open_set, node)
        
        if len(self.open_set) > self.beam_width:
            self.open_set = heapq.nsmallest(self.beam_width, self.open_set)
            heapq.heapify(self.open_set)
        
        if not self.open_set:
            if self.candidates:
                idx = list(self.candidates)[0]
                return self.df.loc[idx], {"algorithm": "astar", "fallback": True}
            return None, {}
        
        current_node = heapq.heappop(self.open_set)
        self.closed_set.add(current_node.pokemon_idx)
        
        pokemon = self.df.loc[current_node.pokemon_idx]
        
        info = {
            "algorithm": "astar",
            "g_cost": round(current_node.g_cost, 3),
            "h_cost": round(current_node.h_cost, 3),
            "f_cost": round(current_node.f_cost, 3),
            "open_set_size": len(self.open_set),
            "closed_set_size": len(self.closed_set),
            "candidates": len(self.candidates)
        }
        
        return pokemon, info
    
    def _select_initial_candidates(self, n: int = 50) -> List[int]:
        """Select diverse initial candidates"""
        if len(self.candidates) <= n:
            return list(self.candidates)
        
        sample = self.df.loc[list(self.candidates)].sample(min(n, len(self.candidates)))
        return sample.index.tolist()
    
    def update_feedback(self, guess: pd.Series, feedback: Dict[str, str]):
        """Update search state with new feedback"""
        guess_idx = guess.name
        self.add_feedback(guess_idx, feedback)
        
        self._update_constraints(guess, feedback)
        self._filter_candidates()
        self._rebuild_open_set()
    
    def _update_constraints(self, guess: pd.Series, feedback: Dict[str, str]):
        """Update constraints based on feedback"""
        for attr, status in feedback.items():
            if attr == 'image_url':
                continue
            
            value = guess.get(attr)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            
            if status == 'green':
                self.constraints[attr].append(('==', value))
            elif status in ['gray', 'yellow']:
                self.constraints[attr].append(('!=', value))
            elif status == 'higher':
                self.constraints[attr].append(('>', value))
            elif status == 'lower':
                self.constraints[attr].append(('<', value))
    
    def _filter_candidates(self):
        """Filter candidates based on constraints"""
        valid_candidates = set()
        
        for idx in self.candidates:
            pokemon = self.df.loc[idx]
            valid = True
            
            for attr, constraints in self.constraints.items():
                if attr == 'image_url':
                    continue
                
                pokemon_val = pokemon.get(attr)
                
                for op, val in constraints:
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        continue
                    if pokemon_val is None or (isinstance(pokemon_val, float) and pd.isna(pokemon_val)):
                        continue
                    
                    try:
                        if op == '==' and pokemon_val != val:
                            valid = False
                            break
                        elif op == '!=' and pokemon_val == val:
                            valid = False
                            break
                        elif op == '>' and not float(pokemon_val) > float(val):
                            valid = False
                            break
                        elif op == '<' and not float(pokemon_val) < float(val):
                            valid = False
                            break
                    except (ValueError, TypeError):
                        pass
                
                if not valid:
                    break
            
            if valid:
                valid_candidates.add(idx)
        
        self.candidates = valid_candidates
    
    def _rebuild_open_set(self):
        """Rebuild open set with updated heuristics"""
        new_open_set = []
        
        for node in self.open_set:
            if node.pokemon_idx in self.candidates:
                h_cost = self.heuristic_distance(node.pokemon_idx) * self.heuristic_weight
                new_node = Node(node.pokemon_idx, node.g_cost + 1, h_cost, node.parent)
                heapq.heappush(new_open_set, new_node)
        
        for idx in self.candidates:
            if idx not in self.closed_set and idx not in [n.pokemon_idx for n in new_open_set]:
                h_cost = self.heuristic_distance(idx) * self.heuristic_weight
                node = Node(idx, len(self.feedback_history), h_cost)
                heapq.heappush(new_open_set, node)
        
        self.open_set = new_open_set