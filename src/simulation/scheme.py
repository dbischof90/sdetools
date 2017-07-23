
"""
This file contains several simulation methods for SDEs.
"""

import numpy as np

class Scheme(object):
    def __init__(self, sde, parameter, steps, **kwargs):
        self.drift = self.map_to_parameter_set(sde.drift, parameter, sde.information['drift'])
        self.diffusion = self.map_to_parameter_set(sde.diffusion, parameter, sde.information['diffusion'])
        if 'derivatives' in kwargs:
            if 'diffusion_x' in kwargs['derivatives']:
                self.diffusion_x = self.map_to_parameter_set(kwargs['derivatives']['diffusion_x'], parameter, sde.information['diffusion'])
        if 'path' in kwargs:
            self.given_path = kwargs['path']
            self.has_path = True
        else:
            self.has_path = False
        self.dW = []

        self.steps = steps
        self.currentstep = 0
        self.h = (sde.timerange[1] - sde.timerange[0])/steps
        self.x = sde.startvalue
        self.t = sde.timerange[0]

    def map_to_parameter_set(self, func, parameter, information):
        func_parameter = {key: parameter[key] for key in information['parameter']}
        if information['spatial']:
            if information['time']:
                return lambda x, t: func(x=x, t=t, **func_parameter)
            else:
                return lambda x, t: func(x=x, **func_parameter)
        elif information['time']:
            return lambda x, t: func(t=t, **func_parameter)
        else:
            return lambda x, t: func(**func_parameter)

    def propagation(self, x, t):
        pass

    def __next__(self):
        if self.currentstep <= self.steps:
            if self.currentstep > 0:
                self.propagation(self.x, self.t)
                self.t += self.h

            if not self.currentstep == self.steps:
                if self.has_path:
                    self.dW.append(self.given_path[self.currentstep])
                else:
                    self.dW.append(np.random.standard_normal() * np.sqrt(self.h))

            self.currentstep += 1
            return self.x
        else:
            raise StopIteration

    def __iter__(self):
        return self

    def return_path(self):
        return self.dW
