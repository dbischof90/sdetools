import pytest

from sde import SDE
from simulation.scheme import Scheme


def test_missing_parameter():
    with pytest.raises(KeyError):
        sample_sde = SDE(lambda x: x, lambda c: c)
        Scheme(sample_sde, parameter={}, steps=10)


def test_path_size_consistency():
    with pytest.raises(ValueError):
        sample_sde = SDE(lambda x: x, lambda x: x)
        dW = list(range(5))
        Scheme(sample_sde, parameter={}, steps=10, path=dW)


def test_scheme_derivation():
    with pytest.raises(TypeError):
        class SomeSpecialScheme(Scheme):
            def __init__(self, sde, parameter, steps, **kwargs):
                super().__init__(sde, parameter, steps, **kwargs)

            def propagation(self, x, t):
                self.x = x * t

        sample_sde = SDE(lambda x: x, lambda x: x)
        SomeSpecialScheme(sample_sde, parameter={}, steps=10)
