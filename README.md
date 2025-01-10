
# TaguchiGridSearchConverter

A Python package for optimizing hyperparameter search using Taguchi array principles. Inspired by [NightHawkInLight's video on Taguchi arrays](https://www.youtube.com/watch?v=5oULEuOoRd0&pp=ygUPdGFndXNoaSBhcnJhYXlz).

## Why use TaguchiGridSearchConverter?

When tuning machine learning models, traditional grid search can require an exponentially large number of experiments. TaguchiGridSearchConverter helps reduce the number of experiments needed while still effectively exploring the parameter space.

Instead of testing every possible combination of parameters (which can be computationally expensive), this package uses Taguchi array principles to:
1. Reduce the number of experiments needed
2. Maintain good coverage of the parameter space
3. Identify significant parameters efficiently

## Getting started

### Installation

* From PyPI:
    * Via uv: `uv pip install TaguchiGridSearchConverter`
    * Via pip: `pip install TaguchiGridSearchConverter`
* From GitHub:
    * Clone this repo then `pip install .`

### Basic Usage

```python
from TaguchiGridSearchConverter import TaguchiGridSearchConverter

# Define your parameter grid
param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}

# Create converter instance
converter = TaguchiGridSearchConverter()

# Get reduced parameter combinations
reduced_grid = converter.convert(param_grid)

# Use the reduced grid in your experiments
for params in reduced_grid:
    # Your training/evaluation code here
    print(params)
```

## How it works

The converter takes a parameter grid (similar to scikit-learn's ParameterGrid) and:
1. Determines the number of levels for each parameter
2. Calculates the minimum number of experiments needed
3. Creates a reduced set of parameter combinations using modulo cycling
4. Ensures the reduced set is smaller than the full grid

This approach is particularly useful when:
- You have limited computational resources
- You need quick insights into parameter importance
- You want to reduce experiment time without sacrificing too much accuracy
