
from src.scheme import Scheme
import numpy as np

class Euler(Scheme):
    def __init__(self, sde, parameter, steps, derivatives, alpha=0.5, beta=0.5):
        super().__init__(sde, parameter, steps)
        self.diffusion_x = derivatives['diffusion']['x']
        self.alpha = alpha
        self.beta = beta

    def predictor(self):
        return self.x + self.drift(self.x, self.t) * self.h + \
               self.diffusion(self.x, self.t) * np.sqrt(self.h)

    def corrector(self, x, t):
        return self.drift(x, t) - self.beta * self.diffusion(x, t) * self.diffusion_x(x, t)

    def propagation(self, x, t):
        self.x += (self.alpha * self.corrector(self.predictor(), self.t + self.h) + (1 - self.alpha) * self.corrector(self.x, self.t)) * self.h + \
                  (self.beta * self.diffusion(self.predictor(), self.t + self.h) + (1 - self.beta) * self.diffusion(self.x, self.t)) * np.random.normal(0.0, np.sqrt(self.h))

