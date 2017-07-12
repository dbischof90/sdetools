
from src.scheme import Scheme
import numpy as np

class Order_05(Scheme):
    def __init__(self, sde, parameter, steps):
        super().__init__(sde, parameter, steps)

    def propagation(self, x, t):
        self.x += self.drift(x, t) * self.h + \
                  self.diffusion(x, t) * np.random.normal(0.0, np.sqrt(self.h))

class Order_10(Scheme):
    def __init__(self, sde, parameter, steps, derivatives):
        super().__init__(sde, parameter, steps)
        self.diffusion_x = derivatives['diffusion']['x']

    def propagation(self, x, t):
        self.x += self.drift(x, t) * self.h + \
                  self.diffusion(x, t) * np.random.normal(0, np.sqrt(self.h)) + \
                  self.diffusion_x(x, t) * self.diffusion(x, t) * 0.5 * (np.random.normal(0, np.sqrt(self.h)) ** 2 - self.h)
