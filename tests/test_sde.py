import pytest

from sde import SDE


def test_if_construction_fails_without_function():
    with pytest.raises(TypeError):
        def test_drift(x):
            return x

        SDE(test_drift, 2)
