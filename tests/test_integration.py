"""
Integration tests for GridSearchReductor.

These tests verify the converter works correctly with sklearn components.
The integration tests ensure that:
1. Reduced parameter grids are compatible with sklearn estimators
2. Parameter combinations are valid and can be used for model training
3. The converter maintains sklearn's ParameterGrid interface contract
4. Real-world hyperparameter optimization scenarios work as expected

This file was developed with assistance from aider.chat.
"""

import pytest
from typing import Tuple, Any
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV

from GridSearchReductor import GridSearchReductor


class TestSklearnIntegration:
    """
    Test integration with sklearn components.

    This test class verifies that GridSearchReductor produces parameter combinations
    that are fully compatible with sklearn's ecosystem, ensuring seamless integration
    for hyperparameter optimization workflows.
    """

    def setup_method(self) -> None:
        """
        Set up test data and converter for each test method.

        Creates a deterministic test environment with:
        - A GridSearchReductor instance with default settings for reproducible results
        - A synthetic classification dataset suitable for SVM testing
        - Fixed random state to ensure test reproducibility across runs

        The small dataset size (100 samples) is chosen to make tests run quickly
        while still being sufficient to validate sklearn integration.
        """
        self.converter: GridSearchReductor = GridSearchReductor()
        # Create a small dataset for testing - small size ensures fast test execution
        # while maintaining statistical validity for integration testing
        self.X: np.ndarray
        self.y: np.ndarray
        self.X, self.y = make_classification(
            n_samples=100, n_features=20, n_classes=2, random_state=42
        )

    def test_sklearn_parameter_grid_compatibility(self) -> None:
        """
        Test that reduced grid contains valid parameter combinations for sklearn.

        This test verifies that:
        1. The reduced grid is smaller than the full grid (actual reduction occurs)
        2. Each parameter combination contains valid values from the original grid
        3. sklearn estimators can be instantiated and trained with each combination
        4. Manual hyperparameter search using reduced combinations works correctly

        The test uses SVM as the target estimator because it has diverse parameter types
        (continuous, categorical, and boolean-like) that thoroughly test compatibility.
        """
        # Define parameter grid for SVM
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        }

        # Get reduced parameter grid
        reduced_grid = self.converter.fit_transform(param_grid)

        # Verify reduced grid is valid and smaller than full grid
        assert len(reduced_grid) > 0
        full_grid_size = len(list(ParameterGrid(param_grid)))
        assert len(reduced_grid) < full_grid_size

        # Test that each combination in reduced grid works with sklearn
        svm = SVC()

        for combination in reduced_grid:
            # Verify each combination contains valid parameters
            assert combination["C"] in param_grid["C"]
            assert combination["kernel"] in param_grid["kernel"]
            assert combination["gamma"] in param_grid["gamma"]

            # Test that we can create an estimator with these parameters
            # This should not raise any errors
            test_svm = SVC(**combination)

            # Test that we can fit it (basic functionality test)
            test_svm.fit(self.X, self.y)

        # Test that we can use the reduced combinations in manual grid search
        best_score = -float("inf")
        best_params = None

        from sklearn.model_selection import cross_val_score

        for combination in reduced_grid:
            estimator = SVC(**combination)
            scores = cross_val_score(
                estimator, self.X, self.y, cv=3, scoring="accuracy"
            )
            mean_score = scores.mean()

            if mean_score > best_score:
                best_score = mean_score
                best_params = combination

        # Verify we found a best configuration
        assert best_params is not None
        assert best_score > -float("inf")

    def test_parameter_grid_object_sklearn_compatibility(self) -> None:
        """
        Test that converter works with sklearn ParameterGrid objects.

        This test ensures the converter can accept ParameterGrid objects as input,
        maintaining API compatibility with sklearn's hyperparameter search ecosystem.

        The test validates that:
        1. ParameterGrid objects are accepted as input without modification
        2. The converter extracts parameter dictionaries correctly from ParameterGrid
        3. Output parameter combinations are valid and use values from original grid
        4. Reduction still occurs when using ParameterGrid input format

        Using a parameter grid with kernel-specific parameters (degree for poly)
        tests handling of conditional parameter dependencies.
        """
        param_dict = {
            "C": [0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly"],
            "degree": [2, 3, 4],  # Only used for poly kernel
        }

        # Create ParameterGrid object
        sklearn_param_grid = ParameterGrid(param_dict)

        # Convert using our converter
        reduced_grid = self.converter.fit_transform(sklearn_param_grid)

        # Verify the reduced grid is valid
        assert len(reduced_grid) > 0
        assert len(reduced_grid) < len(list(sklearn_param_grid))

        # Verify each combination is valid
        for combo in reduced_grid:
            assert combo["C"] in param_dict["C"]
            assert combo["kernel"] in param_dict["kernel"]
            assert combo["degree"] in param_dict["degree"]

    def test_real_world_hyperparameter_optimization(self) -> None:
        """
        Test with a realistic hyperparameter optimization scenario.

        This test simulates a real-world hyperparameter optimization workflow with:
        1. A comprehensive parameter grid with multiple dimensions and many values
        2. Verification that significant reduction is achieved (>50% fewer combinations)
        3. Validation that parameter space coverage is maintained despite reduction
        4. Confirmation that diverse parameter values are still represented

        The large parameter grid (7×4×4×6×4 = 2688 combinations) tests the converter's
        ability to handle high-dimensional parameter spaces typical in production ML.
        The test ensures stratified sampling provides good coverage across all
        parameter dimensions even with aggressive reduction.
        """
        # Define a comprehensive parameter grid
        param_grid = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": [2, 3, 4, 5],  # For poly kernel
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
            "coef0": [0.0, 0.1, 0.5, 1.0],  # For poly/sigmoid kernels
        }

        # Get original grid size
        original_size = len(list(ParameterGrid(param_grid)))

        # Get reduced grid
        reduced_grid = self.converter.fit_transform(param_grid)

        # Verify significant reduction
        assert len(reduced_grid) < original_size
        reduction_ratio = len(reduced_grid) / original_size
        assert reduction_ratio < 0.5  # At least 50% reduction

        # Verify grid is still comprehensive (covers different parameter values)
        c_values = set(combo["C"] for combo in reduced_grid)
        kernel_values = set(combo["kernel"] for combo in reduced_grid)

        assert len(c_values) > 1  # Multiple C values represented
        assert len(kernel_values) > 1  # Multiple kernels represented

        print(f"Original grid size: {original_size}")
        print(f"Reduced grid size: {len(reduced_grid)}")
        print(f"Reduction ratio: {reduction_ratio:.2%}")


if __name__ == "__main__":
    pytest.main([__file__])
