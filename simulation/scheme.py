
import abc
from collections import deque, OrderedDict
from math import sqrt

from numpy.random import standard_normal, randint

from sde import build_information


class Scheme(metaclass=abc.ABCMeta):
    """Base class for numerical schemes.

    This abstract base class in inherited by every numerical scheme in order to provide
    similar features and compatibility. It takes care of functional mappings and implements every scheme
    as an iterator.

    A specific numerical scheme only has to implement the method 'propagation', which is supposed
    to modify the discretized process at time t given t-1 inplace

    """
    def __init__(self, sde, parameter, steps, save_path=True, small_memory=False, **kwargs):
        """Initializes the numerical scheme."""

        self.drift = self.map_to_parameter_set(sde.drift, parameter, sde.information['drift'])
        self.diffusion = self.map_to_parameter_set(sde.diffusion, parameter, sde.information['diffusion'])
        self.steps = steps
        self.currentstep = 0
        self.h = (sde.timerange[1] - sde.timerange[0]) / steps
        self.sqrt_h = sqrt(self.h)
        self.x = sde.startvalue
        self.t = sde.timerange[0]
        self.save_path = save_path
        self.small_memory = small_memory

        if type(self.x) not in (int, float, complex):
            raise NotImplementedError('Currently just scalar SDEs supported!')
        if type(parameter) is not OrderedDict:
            raise TypeError('Parameter set needs to be specified as collections.OrderedDict!')

        if 'derivatives' in kwargs:
            if 'diffusion_x' in kwargs['derivatives']:
                self.diffusion_x = self.map_to_parameter_set(kwargs['derivatives']['diffusion_x'], parameter,
                                                             build_information(kwargs['derivatives']['diffusion_x']))
            if 'diffusion_xx' in kwargs['derivatives']:
                self.diffusion_xx = self.map_to_parameter_set(kwargs['derivatives']['diffusion_xx'], parameter,
                                                              build_information(kwargs['derivatives']['diffusion_xx']))
            if 'drift_x' in kwargs['derivatives']:
                self.diffusion_x = self.map_to_parameter_set(kwargs['derivatives']['drift_x'], parameter,
                                                             build_information(kwargs['derivatives']['drift_x']))
            if 'drift_xx' in kwargs['derivatives']:
                self.diffusion_x = self.map_to_parameter_set(kwargs['derivatives']['drift_xx'], parameter,
                                                             build_information(kwargs['derivatives']['drift_xx']))

        if 'path' in kwargs:
            self.driving_stochastic_differential = deque(kwargs['path'])
        else:
            if 'strong' in self.__module__:
                self.driving_stochastic_differential = deque(standard_normal(steps) * self.sqrt_h)
                if 'Order_15' in self.__module__:
                    self.dZ = deque([0.5 * self.h * self.driving_stochastic_differential[idx] + self.h * self.sqrt_h / sqrt(12) * rnd for idx, rnd in standard_normal(steps)])
            elif 'weak' in self.__module__:
                if 'Order_10' in self.__module__:
                    self.driving_stochastic_differential = deque((2 * randint(0, 2, steps) - 1) * self.sqrt_h)
                elif 'Order_20' in self.__module__:
                    s = sqrt(3 * self.h)
                    self.driving_stochastic_differential = deque([0 if l in (1, 2, 3, 4) else s if l == 5 else -s for l in randint(0, 6, steps)])
            else:
                raise TypeError('The proposed scheme is neither weak nor strong; no convergence order can be set.')

        if len(self.driving_stochastic_differential) != steps:
            raise ValueError('The resolution of the driving stochastic differential does not match the discretization.')

        self.dW = []

    def map_to_parameter_set(self, func, parameter, information):
        func_parameter = tuple(parameter[key] for key in information['parameter'])
        if information['spatial']:
            if information['time']:
                return lambda x, t: func(x, t, *func_parameter)
            else:
                return lambda x, t: func(x, *func_parameter)
        elif information['time']:
            return lambda x, t: func(t, *func_parameter)
        else:
            return lambda x, t: func(*func_parameter)

    @abc.abstractmethod
    def propagation(self, x, t):
        pass

    def __next__(self):
        if self.currentstep <= self.steps:
            if self.currentstep > 0:
                self.propagation(self.x, self.t)
                self.t += self.h

            if not self.currentstep == self.steps:
                dW_step = self.driving_stochastic_differential.popleft()
                self.dW.append(dW_step)

            self.currentstep += 1
            return self.x
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def return_path(self):
        return self.dW
