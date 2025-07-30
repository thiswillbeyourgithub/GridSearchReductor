"""
Configuration and fixtures for pytest.
This file provides shared test configuration and fixtures for the test suite.
"""

import pytest
import logging


@pytest.fixture(autouse=True)
def reset_logging():
    """
    Reset logging configuration before each test.
    This ensures tests don't interfere with each other's logging settings.
    """
    # Store original level
    original_level = logging.getLogger().level

    yield

    # Reset to original level after test
    logging.getLogger().setLevel(original_level)

    # Clear any handlers that might have been added
    logger = logging.getLogger()
    logger.handlers.clear()


@pytest.fixture
def sample_param_grid():
    """
    Provides a standard parameter grid for testing.
    This reduces code duplication across tests.
    """
    return {
        "kernel": ["linear", "rbf", "poly"],
        "C": [0.1, 1, 10],
        "gamma": ["scale", "auto"],
    }


@pytest.fixture
def complex_param_grid():
    """
    Provides a more complex parameter grid for advanced testing.
    Includes mixed types and edge cases.
    """
    return {
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
        "p": [1, 2],
        "leaf_size": [10, 30, 50],
    }
