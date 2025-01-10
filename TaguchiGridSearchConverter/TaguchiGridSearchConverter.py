from beartype import beartype
from typing import Dict, List, Any
from sklearn.model_selection import ParameterGrid
import numpy as np

@beartype  # this will apply to all methods
class TaguchiGridSearchConverter:
    __VERSION__: str = "0.0.1"

    def __init__(self) -> None:
        """
        Initializes a Taguchi Grid Search Converter.
        This class helps optimize hyperparameter search using Taguchi arrays.
        """
        pass

    def convert(self, param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Converts a full parameter grid into a reduced set using Taguchi array principles.
        
        Args:
            param_grid: Dictionary with parameters names (str) as keys and lists of
                       parameter settings to try as values.
        
        Returns:
            List of dictionaries with reduced parameter combinations to test.
        
        Raises:
            ValueError: If the parameter grid is empty or contains empty lists
        """
        # Validate input
        if not param_grid:
            raise ValueError("Parameter grid cannot be empty")
        
        for param, values in param_grid.items():
            if not values:
                raise ValueError(f"Parameter '{param}' has an empty list of values")
        
        # Get the number of parameters and their levels
        param_names = list(param_grid.keys())
        levels = [len(values) for values in param_grid.values()]
        
        # Determine the minimum number of experiments needed
        # Using the maximum number of levels as the base
        num_experiments = max(levels)
        
        # Create the reduced parameter combinations
        reduced_grid = []
        for i in range(num_experiments):
            combination = {}
            for param, values in param_grid.items():
                # Cycle through values using modulo to ensure we stay within bounds
                idx = i % len(values)
                combination[param] = values[idx]
            reduced_grid.append(combination)
        
        # Calculate full grid size
        full_grid_size = len(list(ParameterGrid(param_grid)))
        
        # Assert that our reduced grid is indeed smaller
        assert len(reduced_grid) < full_grid_size, \
            f"Reduced grid size {len(reduced_grid)} not smaller than full grid size {full_grid_size}"
            
        return reduced_grid

if __name__ == "__main__":
    # Example usage without requiring sklearn as a dependency
    sample_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto']
    }
    
    converter = TaguchiGridSearchConverter()
    reduced = converter.convert(sample_grid)
    
    print("Reduced parameter combinations:")
    for i, params in enumerate(reduced, 1):
        print(f"Combination {i}: {params}")
