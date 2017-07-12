
"""
This file contains the main object definition for a general Ito diffusion.

Version 0.1: 1D-Problem driven by a Brownian Motion.
"""
from inspect import signature

class SDE(object):
    """
    Main object representing an SDE. Contains main methods for functional representation.
    """

    def __init__(self, drift, diffusion, timerange = [0, 1], startvalue = 1):
        self._drift = drift
        self._diffusion = diffusion
        self._timerange = timerange
        self._startvalue = startvalue
        self.build_information()

    def build_information(self):
        drift_parameter = list(signature(self.drift).parameters)
        diffusion_parameter = list(signature(self.diffusion).parameters)

        self._information = dict()
        self._information['drift'] = dict()
        self._information['diffusion'] = dict()
        if "x" in drift_parameter:
            self._information['drift']['spatial'] = True
            drift_parameter.remove("x")
        else:
            self._information['drift']['spatial'] = False

        if "t" in drift_parameter:
            self._information['drift']['time'] = True
            drift_parameter.remove("t")
        else:
            self._information['drift']['time']  = False

        if "x" in diffusion_parameter:
            self._information['diffusion']['spatial'] = True
            diffusion_parameter.remove("x")
        else:
            self._information['diffusion']['spatial'] = False

        if "t" in diffusion_parameter:
            self._information['diffusion']['time'] = True
            diffusion_parameter.remove("t")
        else:
            self._information['diffusion']['time'] = False

        self._information['drift']['parameter'] = drift_parameter
        self._information['diffusion']['parameter'] = diffusion_parameter
        self._information['parameter'] = [drift_parameter, diffusion_parameter]

    @property
    def drift(self):
        return self._drift

    @property
    def diffusion(self):
        return self._diffusion

    @property
    def timerange(self):
        return self._timerange

    @property
    def startvalue(self):
        return self._startvalue

    @property
    def information (self):
        return self._information

