from typing import Dict, List, Any, Union
from sklearn.model_selection import ParameterGrid
import numpy as np
import logging
import joblib


class TaguchiGridSearchConverter:
    __VERSION__: str = "0.2.6"

    def __init__(self, verbose: bool = False) -> None:
        """
        Initializes a Taguchi Grid Search Converter.
        This class helps optimize hyperparameter search using Taguchi arrays.

        Args:
            verbose: If True, enables debug logging
        """
        self.logger = logging.getLogger(__name__)
        if verbose:
            logging.basicConfig(level=logging.DEBUG)

    def fit_transform(
        self, param_grid: Union[Dict[str, List[Any]], ParameterGrid]
    ) -> List[Dict[str, Any]]:
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
            assert (
                isinstance(param_grid, list) and len(param_grid) == 1
            ), f"Unexpected param_grid attribute structure: {param_grid}"
            param_grid = param_grid[0]

        # Validate input type
        if not isinstance(param_grid, dict):
            self.logger.error("Input must be either a dictionary or ParameterGrid")
            raise TypeError("Input must be either a dictionary or ParameterGrid")

        # Validate parameter names are strings
        for param in param_grid.keys():
            if not isinstance(param, str):
                self.logger.error(f"Parameter name '{param}' is not a string")
                raise TypeError(f"Parameter name '{param}' must be a string")

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
            raise ValueError(
                "Parameter grid must contain at least one parameter with multiple values"
            )

        # Validate variable parameters
        for param, values in variable_params.items():
            if not values:
                self.logger.error(f"Parameter '{param}' has an empty list of values")
                raise ValueError(f"Parameter '{param}' has an empty list of values")
            if not isinstance(values, (list, tuple)):
                self.logger.error(f"Parameter '{param}' values must be a list or tuple")
                raise TypeError(f"Parameter '{param}' values must be a list or tuple")

        # Get the number of parameters and their levels
        param_names = list(variable_params.keys())
        levels = [len(values) for values in variable_params.values()]

        # Determine the minimum number of experiments needed
        # For Taguchi reduction, use a strategy that ensures actual reduction
        max_levels = max(levels)
        if len(levels) == 1:  # Only one variable parameter
            # For single parameter, use approximately half the levels (minimum 2)
            num_experiments = max(2, max_levels // 2)
        else:
            # For multiple parameters, use the maximum levels as base
            num_experiments = max_levels

        # Create the reduced parameter combinations
        reduced_grid = []
        self.logger.debug(f"Creating reduced grid with {num_experiments} experiments")

        # Create set to track seen combinations
        seen_combinations = set()

        # Generate full ParameterGrid for validation
        full_grid = list(ParameterGrid(variable_params))

        for i in range(num_experiments):
            combination = fixed_params.copy()
            self.logger.debug(f"Creating combination {i+1}")

            for param, values in variable_params.items():
                idx = i % len(values)
                combination[param] = values[idx]
                self.logger.debug(f"  {param} = {values[idx]} (index {idx})")

            # Create hashable version of combination for duplicate checking
            combination_hash = joblib.hash(combination)

            # Validate combination exists in full grid and isn't a duplicate
            if combination_hash not in seen_combinations:
                # Extract only variable parameters for validation against full grid
                variable_combination = {
                    k: v for k, v in combination.items() if k in variable_params
                }
                if variable_combination in full_grid:
                    reduced_grid.append(combination)
                    seen_combinations.add(combination_hash)
                    self.logger.debug(f"Combination {i+1} complete: {combination}")
                else:
                    self.logger.warning(f"Invalid combination skipped: {combination}")
            else:
                self.logger.debug(f"Duplicate combination skipped: {combination}")

        # Calculate full grid size using only variable parameters
        full_grid_size = len(list(ParameterGrid(variable_params)))

        # Assert that our reduced grid is indeed smaller
        assert (
            len(reduced_grid) < full_grid_size
        ), f"Reduced grid size {len(reduced_grid)} not smaller than full grid size {full_grid_size}"

        return reduced_grid


if __name__ == "__main__":
    # Example usage without requiring sklearn as a dependency
    sample_grid = {
        "kernel": ["linear", "rbf", "poly"],
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
        "verbose": [True],  # also handles length 1 lists for fixed params
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
