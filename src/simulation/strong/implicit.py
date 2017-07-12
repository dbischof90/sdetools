
from src.scheme import Scheme
from scipy.optimize import newton
import numpy as np

class Trapez(Scheme):
    def __init__(self, sde, parameter, steps, alpha=1):
        super().__init__(sde, parameter, steps)
        self.alpha = alpha

    def state_equation(self, x, dW):
        return x - (self.x + (self.alpha * self.drift(x, self.t + self.h) + (1 - self.alpha) * self.drift(self.x, self.t)) *
                    self.h + self.diffusion(self.x, self.t) * dW)

    def propagation(self, x, t):
        dW = np.random.normal(0.0, np.sqrt(self.h))
        self.x = newton(self.state_equation, self.x, args=(dW,))

