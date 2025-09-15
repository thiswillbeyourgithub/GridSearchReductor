# GridSearchReductor

> **⚠️ Disclaimer**: This project was almost purely vibecoded with assistance from [aider.chat](https://github.com/Aider-AI/aider/). While it includes comprehensive pytest tests, the author doesn't have the mathematical expertise to verify that the Latin Hypercube Sampling implementation is mathematically sound. Use at your own discretion for production workloads.

A Python package for optimizing hyperparameter search using Latin Hypercube Sampling principles.

Inspired by [NightHawkInLight's video on Taguchi arrays](https://www.youtube.com/watch?v=5oULEuOoRd0&pp=ygUPdGFndXNoaSBhcnJhYXlz).


**Do fewer experiments than grid search, but do the right ones using Latin Hypercube Sampling!**

## Why use GridSearchReductor?

This library is designed to work seamlessly with scikit-learn's ParameterGrid, providing a drop-in replacement that can significantly reduce your hyperparameter search space.

When tuning machine learning models, traditional grid search can require an exponentially large number of experiments. GridSearchReductor helps reduce the number of experiments needed while still effectively exploring the parameter space.

Instead of testing every possible combination of parameters (which can be computationally expensive), this package uses Latin Hypercube Sampling principles to:
1. Reduce the number of experiments needed
2. Maintain excellent coverage of the parameter space through stratified sampling
3. Ensure each parameter dimension is sampled uniformly
4. Provide better space-filling properties than random sampling
5. **Generate deterministic results by default** - the same parameter grid will always produce the same reduced combinations

## Getting started

### Installation

* From PyPI:
    * Via uv: `uv pip install GridSearchReductor`
    * Via pip: `pip install GridSearchReductor`
* From GitHub:
    * Clone this repo then `pip install .`

### Basic Usage

```python
from sklearn.model_selection import ParameterGrid
from GridSearchReductor import GridSearchReductor

grid_converter = GridSearchReductor()

sample_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'verbose': [True],  # also handles length 1 lists for fixed params
}

full_grid = ParameterGrid(sample_grid)

reduced_grid = grid_converter.fit_transform(sample_grid)
# Alternative way:
# reduced_grid = grid_converter.fit_transform(full_grid)

# Use the reduced grid in your experiments
for params in reduced_grid:
    # Your training/evaluation code here
    print(params)
```

The reduced experiments list will be significantly smaller than the full grid while maintaining good parameter space coverage through Latin Hypercube Sampling.

The full experiments list would have been 18 combinations (3×3×2×1), but the reduced grid provides effective coverage with fewer experiments!

## Advanced Usage

### Reproducible Results

GridSearchReductor is **deterministic by default** (using `random_state=42`). The same parameter grid will always produce the same reduced combinations.

```python
# Default behavior - deterministic results
grid_converter = GridSearchReductor()
reduced_grid = grid_converter.fit_transform(sample_grid)

# Use a different random_state if needed
grid_converter = GridSearchReductor(random_state=123)
reduced_grid = grid_converter.fit_transform(sample_grid)

# Use global random state (non-deterministic)
grid_converter = GridSearchReductor(random_state=None)
reduced_grid = grid_converter.fit_transform(sample_grid)
```

### Verbose Logging

```python
# Enable verbose logging to see the sampling process
grid_converter = GridSearchReductor(verbose=True)
reduced_grid = grid_converter.fit_transform(sample_grid)
```

## How it works

The converter takes a parameter grid (similar to scikit-learn's ParameterGrid) and:
1. Separates fixed parameters (single values) from variable parameters
2. Determines the number of levels for each variable parameter
3. Generates Latin Hypercube Samples in normalized [0,1] space
4. Maps these samples to discrete parameter indices
5. Creates a reduced set ensuring uniform coverage across all parameter dimensions
6. Removes duplicate combinations and ensures the result is smaller than the full grid

### Latin Hypercube Sampling Benefits

Latin Hypercube Sampling (LHS) provides superior space-filling properties compared to random sampling:
- **Stratified sampling**: Each parameter dimension is divided into equally probable intervals
- **Uniform coverage**: Exactly one sample per interval ensures no clustering
- **Better convergence**: More efficient exploration of the parameter space
- **Reproducible**: When using a fixed random_state

This approach is particularly useful when:
- You have limited computational resources
- You need comprehensive parameter space exploration with fewer experiments
- You want better coverage than random search
- You need reproducible hyperparameter optimization results

## Dependencies

- numpy
- scikit-learn
- joblib

---

*This project was almost purely vibecoded with assistance from [aider.chat](https://github.com/Aider-AI/aider/).*

