
"""
This file contains the main object definition for a general Ito diffusion.

Version 0.1: 1D-Problem driven by a Brownian Motion.
"""
from inspect import signature

class SDE(object):
    """
    Main object representing an SDE. Contains main methods for functional representation.
    """

    def __init__(self, drift, diffusion, timerange=None, startvalue=1):
        if timerange is None:
            timerange = [0, 1]
        self._drift = drift
        self._diffusion = diffusion
        self._timerange = timerange
        self._startvalue = startvalue
        self._information = dict()
        self._information['drift'] = build_information(drift)
        self._information['diffusion'] = build_information(diffusion)
        self._driving_process = self

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


def build_information(func):
    try:
        parameter = set(signature(func).parameters)
    except TypeError:
        print("ERROR: No proper function was given, the information set can't be built.")
        raise
    
    information = {
        'spatial': 'x' in parameter,
        'time': 't' in parameter,
        'parameter': list(parameter - {'x', 't'})
    }
    return information