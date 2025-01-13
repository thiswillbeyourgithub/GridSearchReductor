from typing import Dict, List, Any, Union
from sklearn.model_selection import ParameterGrid
import numpy as np

class TaguchiGridSearchConverter:
    __VERSION__: str = "0.2.4"

    def __init__(self) -> None:
        """
        Initializes a Taguchi Grid Search Converter.
        This class helps optimize hyperparameter search using Taguchi arrays.
        """
        pass

    def fit_transform(self, param_grid: Union[Dict[str, List[Any]], ParameterGrid]) -> List[Dict[str, Any]]:
        """
        Converts a full parameter grid into a reduced set using Taguchi array principles.
        
        Args:
            param_grid: Either a dictionary with parameters names (str) as keys and lists of
                       parameter settings to try as values, or a ParameterGrid object.
        
        Returns:
            List of dictionaries with reduced parameter combinations to test.
        
        Raises:
            ValueError: If the parameter grid is empty or contains empty lists
            TypeError: If the input is neither a dict nor ParameterGrid
        """
        # Convert ParameterGrid to dict if needed
        if isinstance(param_grid, ParameterGrid):
            param_grid = param_grid.param_grid
            assert isinstance(param_grid, list) and len(param_grid) == 1, (
                f"Unexpected param_grid attribute structure: {param_grid}"
            )
            param_grid = param_grid[0]
            
        # Validate input type
        if not isinstance(param_grid, dict):
            raise TypeError("Input must be either a dictionary or ParameterGrid")
            
        # Separate fixed parameters (non-list values) from variable parameters
        fixed_params = {}
        variable_params = {}
        
        for param, values in param_grid.items():
            # Convert single values to single-item lists
            if not isinstance(values, (list, tuple)):
                values = [values]
                
            # If it's a single value, treat as fixed parameter
            if len(values) == 1:
                fixed_params[param] = values[0]
            else:
                variable_params[param] = values
                
        # Validate we have at least one variable parameter
        if not variable_params:
            raise ValueError("Parameter grid must contain at least one parameter with multiple values")
            
        # Validate variable parameters
        for param, values in variable_params.items():
            if not values:
                raise ValueError(f"Parameter '{param}' has an empty list of values")
        
        # Get the number of parameters and their levels
        param_names = list(variable_params.keys())
        levels = [len(values) for values in variable_params.values()]
        
        # Determine the minimum number of experiments needed
        # Using the maximum number of levels as the base
        num_experiments = max(levels)
        
        # Create the reduced parameter combinations
        reduced_grid = []
        for i in range(num_experiments):
            combination = fixed_params.copy()  # Start with fixed parameters
            for param, values in variable_params.items():
                # Cycle through values using modulo to ensure we stay within bounds
                idx = i % len(values)
                combination[param] = values[idx]
            reduced_grid.append(combination)
        
        # Calculate full grid size using only variable parameters
        full_grid_size = len(list(ParameterGrid(variable_params)))
        
        # Assert that our reduced grid is indeed smaller
        assert len(reduced_grid) < full_grid_size, \
            f"Reduced grid size {len(reduced_grid)} not smaller than full grid size {full_grid_size}"
            
        return reduced_grid

if __name__ == "__main__":
    # Example usage without requiring sklearn as a dependency
    sample_grid = {
        'kernel': ['linear', 'rbf', 'poly'],
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'verbose': [True],  # also handles length 1 lists for fixed params
    }
    
    converter = TaguchiGridSearchConverter()
    reduced = converter.fit_transform(sample_grid)

    # or similarly:
    from sklearn.model_selection import ParameterGrid
    grid = ParameterGrid(sample_grid)
    reduced2 = converter.fit_transform(grid)

    assert reduced2 == reduced

    print("Reduced parameter combinations:")
    for i, params in enumerate(reduced, 1):
        print(f"Combination {i}: {params}")
