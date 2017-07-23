
import numpy as np

from src.simulation.scheme import Scheme


class Order_10(Scheme):
    def __init__(self, sde, parameter, steps, **kwargs):
        super().__init__(sde, parameter, steps, **kwargs)

    def predictor(self):
        return self.x + self.drift(self.x, self.t) * self.h + \
               self.diffusion(self.x, self.t) * np.sqrt(self.h)

    def propagation(self, x, t):
        drift = self.drift(x, t)
        diffusion = self.diffusion(x, t)
        dW = self.dW[-1]
        self.x += drift * self.h + diffusion * dW + 0.5 / np.sqrt(self.h) * (
            self.diffusion(self.predictor(), t) - diffusion) * (dW ** 2 - self.h)
