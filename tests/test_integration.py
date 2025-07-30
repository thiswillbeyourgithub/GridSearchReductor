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
        """Test that reduced grid works with sklearn's GridSearchCV."""
        # Define parameter grid for SVM
        param_grid = {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"],
        }

        # Get reduced parameter grid
        reduced_grid = self.converter.fit_transform(param_grid)

        # Verify reduced grid works with GridSearchCV
        svm = SVC()

        # This should not raise any errors
        grid_search = GridSearchCV(
            estimator=svm, param_grid=reduced_grid, cv=3, scoring="accuracy"
        )

        # Verify we can fit the grid search
        grid_search.fit(self.X, self.y)

        # Verify results
        assert hasattr(grid_search, "best_params_")
        assert hasattr(grid_search, "best_score_")
        assert len(grid_search.cv_results_["params"]) == len(reduced_grid)

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
