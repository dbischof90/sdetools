from collections import OrderedDict

import pytest

from sde import SDE
from simulation.strong.explicit.taylor import Order_05 as NumericalScheme


def test_missing_parameter():
    with pytest.raises(KeyError):
        sample_sde = SDE(lambda x: x, lambda c: c)
        NumericalScheme(sample_sde, parameter={}, steps=10)


def test_path_size_consistency():
    with pytest.raises(ValueError):
        sample_sde = SDE(lambda x: x, lambda x: x)
        dW = list(range(5))
        NumericalScheme(sample_sde, parameter=OrderedDict(), steps=10, path=dW)


def test_scheme_derivation():
    with pytest.raises(TypeError):
        class SomeSpecialScheme(NumericalScheme):
            def __init__(self, sde, parameter, steps, **kwargs):
                super().__init__(sde, parameter, steps, **kwargs)

            def propagation(self, x, t):
                self.x = x * t

        sample_sde = SDE(lambda x: x, lambda x: x)
        SomeSpecialScheme(sample_sde, parameter={}, steps=10)
