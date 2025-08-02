"""
Integration tests for TaguchiGridSearchConverter.
These tests verify the converter works correctly with sklearn components.
"""

import pytest
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV

from TaguchiGridSearchConverter import TaguchiGridSearchConverter


class TestSklearnIntegration:
    """Test integration with sklearn components."""

    def setup_method(self):
        """Set up test data and converter."""
        self.converter = TaguchiGridSearchConverter()
        # Create a small dataset for testing
        self.X, self.y = make_classification(
            n_samples=100, n_features=20, n_classes=2, random_state=42
        )

    def test_sklearn_parameter_grid_compatibility(self):
        """Test that reduced grid contains valid parameter combinations for sklearn."""
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

    def test_parameter_grid_object_sklearn_compatibility(self):
        """Test that converter works with sklearn ParameterGrid objects."""
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

    def test_real_world_hyperparameter_optimization(self):
        """Test with a realistic hyperparameter optimization scenario."""
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
