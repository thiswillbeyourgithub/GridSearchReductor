from typing import Dict, List, Any, Union
from sklearn.model_selection import ParameterGrid
import numpy as np
import logging
import joblib


class GridSearchReductor:
    __VERSION__: str = "1.0.0"

    def __init__(
        self,
        verbose: bool = False,
        random_state: Union[int, None] = 42,
        reduction_factor: float = 0.2,
    ) -> None:
        """
        Initializes a Grid Search Reductor.
        This class helps optimize hyperparameter search using stratified random sampling.

        Args:
            verbose: If True, enables debug logging
            random_state: Random seed for reproducible results. Defaults to 42 for deterministic behavior. If None, uses global random state.
            reduction_factor: Fraction of the full parameter grid to sample. Must be > 0 and < 1, and must result in actual reduction.
        """
        # Validate reduction_factor
        if reduction_factor <= 0 or reduction_factor >= 1:
            raise ValueError(
                f"reduction_factor must be between 0 and 1 (exclusive), got {reduction_factor}"
            )

        self.reduction_factor = reduction_factor
        self.logger = logging.getLogger(__name__)
        if verbose:
            # Set the logger level directly instead of using basicConfig
            # which might not work if logging is already configured
            self.logger.setLevel(logging.DEBUG)
            # Ensure there's a handler if none exists
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                handler.setLevel(logging.DEBUG)
                formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)

        # Initialize random number generator for reproducible sampling
        self.random_state = np.random.RandomState(random_state)

    def _generate_stratified_samples(
        self, num_dimensions: int, num_samples: int
    ) -> np.ndarray:
        """
        Generate stratified random samples.

        Stratified sampling divides each parameter dimension into strata and samples
        proportionally from each stratum, ensuring better coverage across the parameter space
        while being more flexible than Latin Hypercube Sampling.

        Args:
            num_dimensions: Number of parameter dimensions
            num_samples: Number of samples to generate

        Returns:
            Array of shape (num_samples, num_dimensions) with values in [0, 1]
        """
        samples = np.zeros((num_samples, num_dimensions))

        # Determine number of strata - balance between coverage and flexibility
        num_strata = max(2, min(num_samples, int(np.sqrt(num_samples * 2))))

        for dim in range(num_dimensions):
            # Create strata boundaries
            strata_boundaries = np.linspace(0, 1, num_strata + 1)

            # Calculate samples per stratum
            base_samples_per_stratum = num_samples // num_strata
            extra_samples = num_samples % num_strata

            dimension_samples = []

            for i in range(num_strata):
                stratum_start = strata_boundaries[i]
                stratum_end = strata_boundaries[i + 1]

                # Number of samples for this stratum
                stratum_sample_count = base_samples_per_stratum + (
                    1 if i < extra_samples else 0
                )

                # Generate random samples within this stratum
                if stratum_sample_count > 0:
                    stratum_width = stratum_end - stratum_start
                    random_offsets = self.random_state.random(stratum_sample_count)
                    stratum_samples = stratum_start + random_offsets * stratum_width
                    dimension_samples.extend(stratum_samples)

            # Shuffle to remove any ordering bias
            self.random_state.shuffle(dimension_samples)
            samples[:, dim] = np.array(dimension_samples)

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

        # Determine the number of experiments using reduction_factor
        full_grid_size = np.prod(levels)

        # Calculate target samples based on reduction_factor
        target_samples = max(2, int(full_grid_size * self.reduction_factor))

        # Ensure we have reasonable coverage - at least a few samples per variable parameter
        min_samples = 2 * len(variable_params)
        target_samples = max(target_samples, min_samples)

        # Handle case where meaningful reduction isn't possible
        if target_samples >= full_grid_size:
            self.logger.debug(
                f"Parameter grid is too small ({full_grid_size} combinations) for meaningful reduction "
                f"with reduction_factor {self.reduction_factor}. Returning full grid."
            )
            # Return the full parameter grid when reduction isn't beneficial
            return list(
                ParameterGrid(
                    variable_params
                    if not fixed_params
                    else {
                        **variable_params,
                        **{k: [v] for k, v in fixed_params.items()},
                    }
                )
            )

        num_experiments = target_samples

        self.logger.debug(
            f"Creating stratified reduced grid with {num_experiments} experiments"
        )
        self.logger.debug(
            f"Parameter levels: {levels}, full grid size: {full_grid_size}"
        )
        self.logger.debug(
            f"Target samples: {target_samples}, min samples: {min_samples}"
        )

        # Generate stratified samples in [0,1] space
        num_dimensions = len(param_names)
        stratified_samples = self._generate_stratified_samples(
            num_dimensions, num_experiments
        )

        # Convert stratified samples to discrete parameter indices
        reduced_grid = []
        seen_combinations = set()

        for i, sample in enumerate(stratified_samples):
            combination = fixed_params.copy()
            self.logger.debug(f"Creating stratified combination {i+1}")

            for j, param in enumerate(param_names):
                values = variable_params[param]
                # Map [0,1] sample to discrete parameter index
                idx = int(sample[j] * len(values))
                # Ensure idx is within bounds (handle edge case where sample == 1.0)
                idx = min(idx, len(values) - 1)
                combination[param] = values[idx]
                self.logger.debug(f"  {param} = {values[idx]} (stratified index {idx})")

            # Check for duplicates using hash
            combination_hash = joblib.hash(combination)
            if combination_hash not in seen_combinations:
                reduced_grid.append(combination)
                seen_combinations.add(combination_hash)
                self.logger.debug(
                    f"Stratified combination {i+1} complete: {combination}"
                )
            else:
                self.logger.debug(
                    f"Duplicate stratified combination skipped: {combination}"
                )

        # Calculate full grid size using only variable parameters for validation
        full_grid_size_check = len(list(ParameterGrid(variable_params)))

        # Assert that our reduced grid is indeed smaller than the full grid
        assert (
            len(reduced_grid) < full_grid_size_check
        ), f"Stratified reduced grid size {len(reduced_grid)} not smaller than full grid size {full_grid_size_check}"

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
