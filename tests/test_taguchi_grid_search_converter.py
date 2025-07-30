import pytest
import logging
from typing import Dict, List, Any
from sklearn.model_selection import ParameterGrid

from TaguchiGridSearchConverter import TaguchiGridSearchConverter


class TestTaguchiGridSearchConverter:
    """Test suite for TaguchiGridSearchConverter class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.converter = TaguchiGridSearchConverter(verbose=False)
        self.converter_verbose = TaguchiGridSearchConverter(verbose=True)

    def test_init_default(self):
        """Test initialization with default parameters."""
        converter = TaguchiGridSearchConverter()
        assert hasattr(converter, "logger")
        assert (
            converter.logger.name
            == "TaguchiGridSearchConverter.TaguchiGridSearchConverter"
        )

    def test_init_verbose(self):
        """Test initialization with verbose=True."""
        converter = TaguchiGridSearchConverter(verbose=True)
        assert hasattr(converter, "logger")
        # Verify that verbose initialization completed successfully
        # Note: logging.basicConfig() may not change root logger level if already configured
        assert converter.logger is not None

    def test_version_attribute(self):
        """Test that version attribute is accessible."""
        assert hasattr(TaguchiGridSearchConverter, "__VERSION__")
        assert isinstance(TaguchiGridSearchConverter.__VERSION__, str)
        assert TaguchiGridSearchConverter.__VERSION__ == "0.2.5"

    def test_basic_parameter_grid_dict(self):
        """Test fit_transform with basic dictionary parameter grid."""
        param_grid = {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1.0],
            "gamma": ["scale", "auto"],
        }

        result = self.converter.fit_transform(param_grid)

        # Verify result is a list of dictionaries
        assert isinstance(result, list)
        assert all(isinstance(combo, dict) for combo in result)

        # Verify all parameter names are present in each combination
        for combo in result:
            assert set(combo.keys()) == set(param_grid.keys())

        # Verify reduced grid is smaller than full grid
        full_grid_size = len(list(ParameterGrid(param_grid)))
        assert len(result) < full_grid_size

    def test_parameter_grid_object_input(self):
        """Test fit_transform with ParameterGrid object input."""
        param_dict = {"kernel": ["linear", "rbf", "poly"], "C": [0.1, 1, 10]}
        param_grid_obj = ParameterGrid(param_dict)

        result = self.converter.fit_transform(param_grid_obj)

        assert isinstance(result, list)
        assert len(result) > 0
        # Should handle ParameterGrid object same as dict
        result_from_dict = self.converter.fit_transform(param_dict)
        assert result == result_from_dict

    def test_fixed_parameters_single_values(self):
        """Test handling of parameters with single values (fixed parameters)."""
        param_grid = {
            "kernel": ["linear", "rbf"],
            "C": [1.0],  # Single value in list - should be treated as fixed
            "verbose": True,  # Single value not in list - should be converted to fixed
            "gamma": ["scale", "auto"],
        }

        result = self.converter.fit_transform(param_grid)

        # All combinations should have the fixed parameter values
        for combo in result:
            assert combo["C"] == 1.0
            assert combo["verbose"] is True
            assert combo["kernel"] in ["linear", "rbf"]
            assert combo["gamma"] in ["scale", "auto"]

    def test_example_from_main(self):
        """Test the example usage from the __main__ section."""
        sample_grid = {
            "kernel": ["linear", "rbf", "poly"],
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
            "verbose": [True],  # Fixed parameter as single-item list
        }

        reduced = self.converter.fit_transform(sample_grid)

        # Test with ParameterGrid object as well
        grid = ParameterGrid(sample_grid)
        reduced2 = self.converter.fit_transform(grid)

        assert reduced2 == reduced
        assert len(reduced) > 0

        # Verify all combinations have verbose=True (fixed parameter)
        for combo in reduced:
            assert combo["verbose"] is True

    def test_empty_parameter_grid_error(self):
        """Test that empty parameter grid raises ValueError."""
        with pytest.raises(
            ValueError, match="must contain at least one parameter with multiple values"
        ):
            self.converter.fit_transform({})

    def test_all_fixed_parameters_error(self):
        """Test that grid with only fixed parameters raises ValueError."""
        param_grid = {
            "kernel": ["linear"],  # Single value
            "C": [1.0],  # Single value
            "verbose": True,  # Single value
        }

        with pytest.raises(
            ValueError, match="must contain at least one parameter with multiple values"
        ):
            self.converter.fit_transform(param_grid)

    def test_empty_parameter_values_error(self):
        """Test that empty parameter value lists raise ValueError."""
        param_grid = {"kernel": [], "C": [0.1, 1.0]}  # Empty list

        with pytest.raises(ValueError, match="has an empty list of values"):
            self.converter.fit_transform(param_grid)

    def test_invalid_input_type_error(self):
        """Test that invalid input types raise TypeError."""
        with pytest.raises(
            TypeError, match="must be either a dictionary or ParameterGrid"
        ):
            self.converter.fit_transform("invalid_input")

        with pytest.raises(
            TypeError, match="must be either a dictionary or ParameterGrid"
        ):
            self.converter.fit_transform(123)

    def test_non_string_parameter_names_error(self):
        """Test that non-string parameter names raise TypeError."""
        param_grid = {123: ["linear", "rbf"], "C": [0.1, 1.0]}  # Non-string key

        with pytest.raises(TypeError, match="must be a string"):
            self.converter.fit_transform(param_grid)

    def test_large_parameter_grid(self):
        """Test with larger parameter grid to verify scaling."""
        param_grid = {
            "param1": list(range(10)),
            "param2": list(range(5)),
            "param3": ["a", "b", "c"],
            "param4": [True, False],
        }

        result = self.converter.fit_transform(param_grid)
        full_grid_size = len(list(ParameterGrid(param_grid)))

        assert len(result) < full_grid_size
        assert len(result) > 0

        # Verify each combination is valid
        for combo in result:
            assert combo["param1"] in param_grid["param1"]
            assert combo["param2"] in param_grid["param2"]
            assert combo["param3"] in param_grid["param3"]
            assert combo["param4"] in param_grid["param4"]

    def test_single_parameter_multiple_values(self):
        """Test with only one parameter having multiple values."""
        param_grid = {
            "kernel": ["linear", "rbf", "poly", "sigmoid"],
            "fixed_param": [42],  # Fixed parameter
        }

        result = self.converter.fit_transform(param_grid)

        assert len(result) > 0
        assert len(result) < len(param_grid["kernel"])  # Should be reduced

        # All combinations should have fixed_param=42
        for combo in result:
            assert combo["fixed_param"] == 42
            assert combo["kernel"] in param_grid["kernel"]

    def test_verbose_logging(self, caplog):
        """Test that verbose mode produces debug logs."""
        param_grid = {"kernel": ["linear", "rbf"], "C": [0.1, 1.0]}

        with caplog.at_level(logging.DEBUG):
            result = self.converter_verbose.fit_transform(param_grid)

        # Check that debug messages were logged
        debug_messages = [
            record.message for record in caplog.records if record.levelname == "DEBUG"
        ]
        assert len(debug_messages) > 0
        assert any("Creating reduced grid" in msg for msg in debug_messages)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = TaguchiGridSearchConverter()

    def test_parameter_grid_with_none_values(self):
        """Test parameter grid containing None values."""
        param_grid = {"param1": [None, "value1", "value2"], "param2": [1, 2, None]}

        result = self.converter.fit_transform(param_grid)

        assert len(result) > 0
        # Verify None values are preserved in combinations
        param1_values = [combo["param1"] for combo in result]
        param2_values = [combo["param2"] for combo in result]

        assert None in param1_values or None in param2_values

    def test_duplicate_combinations_handling(self):
        """Test that duplicate combinations are properly handled."""
        # Create a grid that might generate duplicates due to modulo operation
        param_grid = {
            "param1": ["a", "b"],  # 2 values
            "param2": ["x", "y", "z", "w"],  # 4 values
        }

        result = self.converter.fit_transform(param_grid)

        # Convert to tuples for comparison
        result_tuples = [tuple(sorted(combo.items())) for combo in result]

        # Verify no duplicates
        assert len(result_tuples) == len(set(result_tuples))

    def test_mixed_parameter_types(self):
        """Test parameter grid with mixed value types."""
        param_grid = {
            "string_param": ["a", "b", "c"],
            "int_param": [1, 2, 3, 4],
            "float_param": [0.1, 0.5],
            "bool_param": [True, False],
            "none_param": [None],  # Fixed parameter
        }

        result = self.converter.fit_transform(param_grid)

        assert len(result) > 0

        # Verify type preservation
        for combo in result:
            assert isinstance(combo["string_param"], str)
            assert isinstance(combo["int_param"], int)
            assert isinstance(combo["float_param"], float)
            assert isinstance(combo["bool_param"], bool)
            assert combo["none_param"] is None


if __name__ == "__main__":
    pytest.main([__file__])
