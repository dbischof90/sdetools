
import numpy as np
from src.simulation.scheme import Scheme


class Order_10(Scheme):
    def __init__(self, sde, parameter, steps):
        super().__init__(sde, parameter, steps)

    def predictor(self):
        return self.x + self.drift(self.x, self.t) * self.h + \
               self.diffusion(self.x, self.t) * np.sqrt(self.h)

    def propagation(self, x, t):
        drift = self.drift(x, t)
        diffusion = self.diffusion(x, t)
        dW = np.random.standard_normal() * np.sqrt(self.h)
        self.x += drift * self.h + diffusion * dW + \
                  1/np.sqrt(4 * self.h) * (self.diffusion(self.predictor(), t) - diffusion) * (dW ** 2 - self.h)
