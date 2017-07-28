
"""
A generic base class to implement a numerical scheme.
Once inherited, the specific scheme has to implement the method 'propagation' to represent
a functional implementation.
"""

from collections import deque

import numpy as np

from sde import build_information


class Scheme:
    def __init__(self, sde, parameter, steps, **kwargs):
        self.drift = self.map_to_parameter_set(sde.drift, parameter, sde.information['drift'])
        self.diffusion = self.map_to_parameter_set(sde.diffusion, parameter, sde.information['diffusion'])
        self.steps = steps
        self.currentstep = 0
        self.h = (sde.timerange[1] - sde.timerange[0]) / steps
        self.x = sde.startvalue
        self.t = sde.timerange[0]

        if 'derivatives' in kwargs:
            if 'diffusion_x' in kwargs['derivatives']:
                self.diffusion_x = self.map_to_parameter_set(kwargs['derivatives']['diffusion_x'], parameter,
                                                             build_information(kwargs['derivatives']['diffusion_x']))
                kwargs['derivatives'].pop('diffusion_x')
            if 'diffusion_xx' in kwargs['derivatives']:
                self.diffusion_xx = self.map_to_parameter_set(kwargs['derivatives']['diffusion_xx'], parameter,
                                                              build_information(kwargs['derivatives']['diffusion_xx']))
                kwargs['derivatives'].pop('diffusion_xx')
            if 'drift_x' in kwargs['derivatives']:
                self.diffusion_x = self.map_to_parameter_set(kwargs['derivatives']['drift_x'], parameter,
                                                             build_information(kwargs['derivatives']['drift_x']))
                kwargs['derivatives'].pop('drift_x')
            if 'drift_xx' in kwargs['derivatives']:
                self.diffusion_x = self.map_to_parameter_set(kwargs['derivatives']['drift_xx'], parameter,
                                                             build_information(kwargs['derivatives']['drift_xx']))
                kwargs['derivatives'].pop('drift_xx')

        if 'path' in kwargs:
            self.driving_stochastic_differential = deque(kwargs['path'])
        else:
            if 'strong' in self.__module__:
                self.driving_stochastic_differential = deque(np.random.standard_normal(steps) * np.sqrt(self.h))
            elif 'weak' in self.__module__:
                if 'Order_10' in self.__module__:
                    self.driving_stochastic_differential = deque(
                        (2 * np.random.randint(0, 2, steps) - 1) * np.sqrt(self.h))
                elif 'Order_20' in self.__module__:
                    s = np.sqrt(3 * self.h)
                    self.driving_stochastic_differential = (
                    deque([0 if l in (1, 2, 3, 4) else s if l == 5 else -s for l in np.random.randint(0, 6, steps)]))
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

    def propagation(self, x, t):
        pass

    def __next__(self):
        if self.currentstep <= self.steps:
            if self.currentstep > 0:
                self.propagation(self.x, self.t)
                self.t += self.h

            if not self.currentstep == self.steps:
                self.dW.append(self.driving_stochastic_differential.popleft())

            self.currentstep += 1
            return self.x
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def return_path(self):
        return self.dW
