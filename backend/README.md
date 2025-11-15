# Pokedle Solver API v5.0 - Logically Correct Version

## 🎯 What Changed in v5.0?

This version fixes **fundamental logical flaws** in the AI algorithm implementations. All algorithms now follow proper theoretical foundations.

### Major Improvements

#### 1. CSP (Constraint Satisfaction Problem)
**Before:** ❌ Treated Pokemon as variables  
**Now:** ✅ Properly treats **attributes** as variables with domains

**New Features:**
- ✅ **AC-3 Constraint Propagation** - Automatically reduces domains
- ✅ **Two-Level Heuristics:**
  - **Variable Ordering:** Which attribute to constrain next (MRV, Degree, etc.)
  - **Value Ordering:** Which value to try first (LCV, Most Common, etc.)
- ✅ Proper constraint modeling from feedback

#### 2. GA (Genetic Algorithm)
**Before:** ❌ Created invalid Pokemon through arbitrary attribute combinations  
**Now:** ✅ All individuals are **valid Pokemon**

**New Features:**
- ✅ Crossover finds real Pokemon matching parent attributes
- ✅ Fitness properly measures constraint satisfaction
- ✅ Diversity maintenance prevents premature convergence

#### 3. A* Search
**Before:** ❌ Non-admissible heuristic (could overestimate)  
**Now:** ✅ **Admissible heuristic** guaranteeing optimal solution

**New Features:**
- ✅ Heuristic never overestimates remaining cost
- ✅ Properly tracks search path
- ✅ Guarantees shortest solution

#### 4. Simulated Annealing
**Already Correct:** ✅ But improved energy function and constraint handling

---

## 📡 API Endpoints

### Core Endpoints

#### `POST /solve` - Solve Pokedle
```json
{
  "algorithm": "CSP",
  "attributes": ["Type1", "Type2", "Generation"],
  "secret_pokemon": "Charizard",  // Optional
  "max_attempts": 10,
  "csp_config": {
    "variable_heuristic": "mrv",
    "value_heuristic": "lcv",
    "use_ac3": true
  }
}
```

**Response:**
```json
{
  "secret_name": "Charizard",
  "success": true,
  "total_attempts": 4,
  "steps": [...],
  "algorithm_config": {
    "variable_heuristic": "mrv",
    "value_heuristic": "lcv",
    "use_ac3": true
  }
}
```

#### `GET /config` - Get Configuration Options
Returns all available heuristics, algorithms, and configurations.

**Response includes:**
```json
{
  "csp_heuristics": {
    "variable_ordering": {
      "options": ["mrv", "degree", "mrv_degree", "none"],
      "descriptions": {...}
    },
    "value_ordering": {
      "options": ["lcv", "most_common", "none"],
      "descriptions": {...}
    }
  }
}
```

#### `POST /compare` - Compare Algorithms
```json
{
  "algorithms": ["CSP", "GA", "ASTAR", "SA"],
  "attributes": ["Type1", "Type2", "Generation"],
  "secret_pokemon": "Pikachu",
  "max_attempts": 10
}
```

#### `POST /test/csp-heuristics` - Test CSP Heuristic Combinations
Tests all combinations of variable and value ordering heuristics.

```json
{
  "attributes": ["Type1", "Type2", "Generation"],
  "max_attempts": 10
}
```

**Response:**
```json
{
  "results": {
    "mrv+lcv": {"success": true, "attempts": 3},
    "mrv+most_common": {"success": true, "attempts": 4},
    "degree+lcv": {"success": true, "attempts": 5},
    ...
  },
  "best_combination": "mrv+lcv"
}
```

#### `GET /algorithm-theory/{algorithm}` - Get Algorithm Theory
Returns theoretical background and correctness properties.

---

## 🎓 CSP Configuration

### Variable Ordering Heuristics
Choose which attribute to constrain next:

| Heuristic | Description | When to Use |
|-----------|-------------|-------------|
| `mrv` | Minimum Remaining Values - smallest domain | Default, fail-fast strategy |
| `degree` | Most constrained attribute | When constraints are complex |
| `mrv_degree` | MRV with degree tiebreaker | Best of both worlds |
| `none` | No heuristic | Baseline comparison |

### Value Ordering Heuristics
Choose which value to try for the selected attribute:

| Heuristic | Description | When to Use |
|-----------|-------------|-------------|
| `lcv` | Least Constraining Value | Default, keeps options open |
| `most_common` | Most frequent value | When exploring likelihood |
| `none` | No heuristic | Baseline comparison |

### Example Configurations

**Aggressive (Fast Failure):**
```json
{
  "variable_heuristic": "mrv",
  "value_heuristic": "lcv",
  "use_ac3": true
}
```

**Conservative (Explore Options):**
```json
{
  "variable_heuristic": "degree",
  "value_heuristic": "most_common",
  "use_ac3": true
}
```

**Balanced:**
```json
{
  "variable_heuristic": "mrv_degree",
  "value_heuristic": "lcv",
  "use_ac3": true
}
```

---

## 🧬 GA Configuration

```json
{
  "pop_size": 100,        // Population size (10-500)
  "elite_size": 20,       // Best individuals preserved (5-100)
  "mutation_rate": 0.15,  // Mutation probability (0.0-1.0)
  "crossover_rate": 0.8,  // Crossover probability (0.0-1.0)
  "tournament_size": 5,   // Tournament selection size (2-20)
  "generations_per_guess": 30  // Generations to evolve (1-200)
}
```

**Key Point:** All individuals are now **valid Pokemon** - no arbitrary combinations!

---

## 🔍 A* Configuration

```json
{
  "beam_width": 100,        // Beam search width (1+)
  "heuristic_weight": 1.0   // Heuristic weight (1.0 = admissible)
}
```

**Key Point:** `heuristic_weight = 1.0` ensures optimality. Values > 1.0 trade optimality for speed (Weighted A*).

---

## 🌡️ SA Configuration

```json
{
  "initial_temp": 100.0,        // Starting temperature (> 0)
  "cooling_rate": 0.95,         // Cooling factor (0-1)
  "min_temp": 0.01,            // Minimum temperature (> 0)
  "iterations_per_temp": 50,   // Iterations per temperature (≥ 1)
  "reheat_threshold": 0.1      // When to reheat (0-1)
}
```

---

## 📊 Example Usage

### Python Client

```python
import requests

# Solve with CSP
response = requests.post('http://localhost:8000/solve', json={
    "algorithm": "CSP",
    "attributes": ["Type1", "Type2", "Generation", "Height"],
    "secret_pokemon": "Charizard",
    "max_attempts": 10,
    "csp_config": {
        "variable_heuristic": "mrv",
        "value_heuristic": "lcv",
        "use_ac3": True
    }
})

result = response.json()
print(f"Success: {result['success']}")
print(f"Attempts: {result['total_attempts']}")
print(f"Time: {result['execution_time']}s")

# View algorithm state at each step
for step in result['steps']:
    print(f"\nAttempt {step['attempt']}: {step['guess_name']}")
    print(f"Remaining candidates: {step['remaining_candidates']}")
    print(f"Algorithm state: {step['algorithm_state']}")
```

### Compare All Algorithms

```python
response = requests.post('http://localhost:8000/compare', json={
    "algorithms": ["CSP", "GA", "ASTAR", "SA"],
    "attributes": ["Type1", "Type2", "Generation"],
    "secret_pokemon": "Pikachu",
    "max_attempts": 10
})

results = response.json()
print(f"Winner: {results['winner']}")
for algo, data in results['results'].items():
    print(f"{algo}: {data['attempts']} attempts, {data['time']}s")
```

### Test CSP Heuristics

```python
response = requests.post('http://localhost:8000/test/csp-heuristics', json={
    "attributes": ["Type1", "Type2", "Generation"],
    "max_attempts": 10
})

results = response.json()
print(f"Best combination: {results['best_combination']}")

# View all results
for combo, data in results['results'].items():
    if 'error' not in data:
        print(f"{combo}: {data['attempts']} attempts")
```

### Get Algorithm Theory

```python
response = requests.get('http://localhost:8000/algorithm-theory/CSP')
theory = response.json()

print(f"Formulation: {theory['formulation']}")
print(f"Properties: {theory['properties']}")
print(f"Correctness: {theory['correctness']}")
```

---

## 🔬 Verification & Testing

### Test CSP Correctness

```bash
# Test that CSP properly reduces domains with AC-3
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "CSP",
    "attributes": ["Type1", "Type2"],
    "secret_pokemon": "Charizard",
    "max_attempts": 5,
    "csp_config": {
      "variable_heuristic": "mrv",
      "value_heuristic": "lcv",
      "use_ac3": true
    }
  }'
```

Check that:
- Domain sizes decrease after each guess
- AC-3 propagates constraints
- Assignment dictionary grows

### Test GA Valid Individuals

```bash
# Verify all guesses are valid Pokemon
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "GA",
    "attributes": ["Type1", "Type2", "Generation"],
    "max_attempts": 10,
    "ga_config": {
      "pop_size": 50,
      "generations_per_guess": 20
    }
  }'
```

Check that:
- All `guess_name` values are real Pokemon
- Fitness scores are based on constraints
- Population diversity is maintained

### Test A* Admissibility

```bash
# Verify heuristic never overestimates
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "ASTAR",
    "attributes": ["Type1", "Type2"],
    "max_attempts": 10,
    "astar_config": {
      "heuristic_weight": 1.0
    }
  }'
```

Check that:
- `h_cost` decreases with feedback
- `f_cost` guides search correctly
- Solution is optimal (minimal attempts)

---

## 📈 Performance Comparison

Expected performance characteristics:

| Algorithm | Attempts | Time | Optimality | Use Case |
|-----------|----------|------|------------|----------|
| **CSP (MRV+LCV)** | 3-5 | Fast | High | Systematic solving |
| **GA** | 5-8 | Medium | Medium | Exploration |
| **A*** | 3-4 | Medium | **Optimal** | Shortest path |
| **SA** | 4-7 | Fast | Medium | Quick approximation |

---

## 🐛 Debugging

### Enable Detailed Logging

Each algorithm returns detailed state in `algorithm_state`:

**CSP:**
```json
{
  "algorithm": "CSP",
  "variable_heuristic": "mrv",
  "value_heuristic": "lcv",
  "candidates": 45,
  "assignment": {"Type1": "Fire"},
  "domain_sizes": {"Type1": 1, "Type2": 5, "Generation": 3},
  "selected_variable": "Type2",
  "selected_value": "Flying"
}
```

**GA:**
```json
{
  "algorithm": "GA",
  "generation": 30,
  "best_fitness": 85.5,
  "avg_fitness": 62.3,
  "population_diversity": 78.5,
  "hard_constraints": 2,
  "soft_constraints": 3
}
```

**A*:**
```json
{
  "algorithm": "astar",
  "g_cost": 2,
  "h_cost": 1.3,
  "f_cost": 3.3,
  "open_set_size": 45,
  "candidates": 67
}
```

---

## 🎯 Best Practices

1. **Start with CSP (MRV+LCV)** - Most systematic and reliable
2. **Use A* for optimality** - Guarantees shortest path
3. **Try GA for exploration** - Good for diverse search spaces
4. **Use SA for quick results** - Fast approximation

### Attribute Selection
- **Start simple:** 2-3 attributes (Type1, Type2, Generation)
- **Add complexity:** Include Height, Weight for numeric constraints
- **Full challenge:** All 7 attributes

### Troubleshooting

**CSP finds no solution:**
- Check if constraints are contradictory
- Try `use_ac3: false` to debug
- Review domain sizes in algorithm state

**GA not converging:**
- Increase `generations_per_guess`
- Adjust `mutation_rate` (higher = more exploration)
- Check `population_diversity` metric

**A* too slow:**
- Reduce `beam_width`
- Increase `heuristic_weight` (trades optimality for speed)

**SA stuck in local optimum:**
- Increase `initial_temp`
- Adjust `cooling_rate` (slower = more exploration)
- Enable reheating

---

## 📚 References

- CSP: Russell & Norvig, "Artificial Intelligence: A Modern Approach"
- GA: Goldberg, "Genetic Algorithms in Search, Optimization, and Machine Learning"
- A*: Hart, Nilsson, Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths"
- SA: Kirkpatrick et al., "Optimization by Simulated Annealing"

---

## 🚀 Running the API

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
python main.py

# Or with uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

---

## Version History

- **v5.0**: Logically correct AI algorithms with proper formulations
- **v4.0**: Multiple algorithms and heuristics (deprecated - had logical flaws)
- **v3.0**: Enhanced CSP solver
- **v2.0**: Basic GA implementation
- **v1.0**: Initial release