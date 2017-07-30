
import numpy as np

from simulation.scheme import Scheme


class Order_10(Scheme):
    def __init__(self, sde, parameter, steps, **kwargs):
        super().__init__(sde, parameter, steps, **kwargs)

    def propagation(self, x, t):
        drift = self.drift(x, t)
        diffusion = self.diffusion(x, t)
        dW = self.dW[-1]
        predictor = x + drift * self.h + diffusion * np.sqrt(self.h)
        self.x += drift * self.h + diffusion * dW + (self.diffusion(predictor, t) - diffusion) / np.sqrt(4 * self.h) * (dW ** 2 - self.h)
