"""Shared fixtures for rollups tests.
"""
import pytest
from rollups import DataSet


@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging for tests"""
    import logging
    logging.basicConfig(level=logging.DEBUG)

# --- Common Sample Dataset Fixtures ---


@pytest.fixture
def sample_dataset():
    """Basic dataset with mixed types for common test scenarios.

    Contains 3 rows with id (int), name (str), and value (float) columns.
    """
    ds = DataSet([
        {'id': 1, 'name': 'A', 'value': 100.0},
        {'id': 2, 'name': 'B', 'value': 200.0},
        {'id': 3, 'name': 'C', 'value': 300.0},
    ])
    ds.columns = (('id', int), ('name', str), ('value', float))
    return ds


# --- Copy Method Fixtures ---

@pytest.fixture(params=['copy', 'shallowcopy', 'deepcopy'])
def copy_method_name(request):
    """Parameterized fixture providing copy method names.

    Use this fixture to test all three copy methods with the same test logic.
    """
    return request.param


@pytest.fixture
def copy_method(request, sample_dataset, copy_method_name):
    """Parameterized fixture for testing all copy methods.

    Returns a tuple of (method_name, copied_dataset).
    """
    method = getattr(sample_dataset, copy_method_name)
    return copy_method_name, method()
