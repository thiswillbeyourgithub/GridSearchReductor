from typing import Dict, List, Any, Union
from sklearn.model_selection import ParameterGrid
import numpy as np
import logging
import joblib


class GridSearchReductor:
    __VERSION__: str = "0.3.2"

    def __init__(
        self, verbose: bool = False, random_state: Union[int, None] = 42
    ) -> None:
        """
        Initializes a Grid Search Reductor.
        This class helps optimize hyperparameter search using Latin Hypercube Sampling.

        Args:
            verbose: If True, enables debug logging
            random_state: Random seed for reproducible results. Defaults to 42 for deterministic behavior. If None, uses global random state.
        """
        self.logger = logging.getLogger(__name__)
        if verbose:
            logging.basicConfig(level=logging.DEBUG)

        # Initialize random number generator for reproducible sampling
        self.random_state = np.random.RandomState(random_state)

    def _generate_latin_hypercube_samples(
        self, num_dimensions: int, num_samples: int
    ) -> np.ndarray:
        """
        Generate Latin Hypercube Samples without scipy dependency.

        LHS ensures each parameter dimension is divided into equally probable intervals
        with one sample per interval, providing better coverage than random sampling.

        Args:
            num_dimensions: Number of parameter dimensions
            num_samples: Number of samples to generate

        Returns:
            Array of shape (num_samples, num_dimensions) with values in [0, 1]
        """
        # Create equally spaced intervals for each dimension
        samples = np.zeros((num_samples, num_dimensions))

        for dim in range(num_dimensions):
            # Create intervals [0, 1/n, 2/n, ..., (n-1)/n] and add random jitter
            intervals = np.arange(num_samples) / num_samples
            jitter = self.random_state.random(num_samples) / num_samples
            dimension_samples = intervals + jitter

            # Shuffle to ensure Latin Hypercube property
            self.random_state.shuffle(dimension_samples)
            samples[:, dim] = dimension_samples

        return samples

    def fit_transform(
        self, param_grid: Union[Dict[str, List[Any]], ParameterGrid]
    ) -> List[Dict[str, Any]]:
        """
        Converts a full parameter grid into a reduced set using Latin Hypercube Sampling.

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

        # Determine the number of experiments using LHS strategy for good coverage
        full_grid_size = np.prod(levels)

        # For effective reduction, use approximately square root of full grid size
        target_samples = max(2, int(np.sqrt(full_grid_size)))

        # Ensure we have reasonable coverage - at least a few samples per variable parameter
        min_samples = 2 * len(variable_params)
        target_samples = max(target_samples, min_samples)

        # Ensure we actually reduce the grid size (leave at least one combination out)
        num_experiments = min(target_samples, full_grid_size - 1)

        self.logger.debug(
            f"Creating LHS reduced grid with {num_experiments} experiments"
        )
        self.logger.debug(
            f"Parameter levels: {levels}, full grid size: {full_grid_size}"
        )
        self.logger.debug(
            f"Target samples: {target_samples}, min samples: {min_samples}"
        )

        # Generate Latin Hypercube Samples in [0,1] space
        num_dimensions = len(param_names)
        lhs_samples = self._generate_latin_hypercube_samples(
            num_dimensions, num_experiments
        )

        # Convert LHS samples to discrete parameter indices
        reduced_grid = []
        seen_combinations = set()

        for i, sample in enumerate(lhs_samples):
            combination = fixed_params.copy()
            self.logger.debug(f"Creating LHS combination {i+1}")

            for j, param in enumerate(param_names):
                values = variable_params[param]
                # Map [0,1] sample to discrete parameter index
                idx = int(sample[j] * len(values))
                # Ensure idx is within bounds (handle edge case where sample == 1.0)
                idx = min(idx, len(values) - 1)
                combination[param] = values[idx]
                self.logger.debug(f"  {param} = {values[idx]} (LHS index {idx})")

            # Check for duplicates using hash
            combination_hash = joblib.hash(combination)
            if combination_hash not in seen_combinations:
                reduced_grid.append(combination)
                seen_combinations.add(combination_hash)
                self.logger.debug(f"LHS combination {i+1} complete: {combination}")
            else:
                self.logger.debug(f"Duplicate LHS combination skipped: {combination}")

        # Calculate full grid size using only variable parameters for validation
        full_grid_size_check = len(list(ParameterGrid(variable_params)))

        # Assert that our reduced grid is indeed smaller than the full grid
        assert (
            len(reduced_grid) < full_grid_size_check
        ), f"LHS reduced grid size {len(reduced_grid)} not smaller than full grid size {full_grid_size_check}"

        return reduced_grid


if __name__ == "__main__":
    # Example usage without requiring sklearn as a dependency
    sample_grid = {
        "kernel": ["linear", "rbf", "poly"],
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
        "verbose": [True],  # also handles length 1 lists for fixed params
    }

    converter = GridSearchReductor()
    reduced = converter.fit_transform(sample_grid)

    # or similarly:
    from sklearn.model_selection import ParameterGrid

    grid = ParameterGrid(sample_grid)
    reduced2 = converter.fit_transform(grid)

    assert reduced2 == reduced

    print("Reduced parameter combinations:")
    for i, params in enumerate(reduced, 1):
        print(f"Combination {i}: {params}")
