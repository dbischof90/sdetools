
from scipy.optimize import newton

from src.simulation.scheme import Scheme


class Order_10(Scheme):
    def __init__(self, sde, parameter, steps, alpha=1):
        super().__init__(sde, parameter, steps)
        self.alpha = alpha

    def state_equation(self, x, dW):
        return x - (self.x + (self.alpha * self.drift(x, self.t + self.h) + (1 - self.alpha) * self.drift(self.x, self.t)) *
                    self.h + self.diffusion(self.x, self.t) * dW)

    def propagation(self, x, t):
        dW = self.dW[-1]
        self.x = newton(self.state_equation, self.x, args=(dW,))

