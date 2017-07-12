
from src.scheme import Scheme
import numpy as np

class Platen(Scheme):
    def __init__(self, sde, parameter, steps):
        super().__init__(sde, parameter, steps)

    def predictor(self):
        return self.x + self.drift(self.x, self.t) * self.h + \
               self.diffusion(self.x, self.t) * np.sqrt(self.h)

    def propagation(self, x, t):
        self.x += self.drift(x, t) * self.h + self.diffusion(x, t) * np.random.normal(0, np.sqrt(self.h)) + \
                  1/np.sqrt(4 * self.h) * (self.diffusion(self.predictor(), t) - self.diffusion(x, t)) * \
                  (np.random.normal(0, np.sqrt(self.h)) ** 2 - self.h)