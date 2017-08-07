
from scipy.optimize import newton

from simulation.scheme import Scheme


class Order_05_Trapez(Scheme):
    def __init__(self, sde, parameter, steps, alpha=1, **kwargs):
        super().__init__(sde, parameter, steps, **kwargs)
        self.alpha = alpha

    def state_equation(self, x, dW):
        return x - (
        self.x + (self.alpha * self.drift(x, self.t + self.h) + (1 - self.alpha) * self.drift(self.x, self.t)) *
        self.h + self.diffusion(self.x, self.t) * dW)

    def propagation(self, x, t):
        dW = self.dW[-1]
        self.x = newton(self.state_equation, self.x, args=(dW,))


class Order_10_Trapez(Scheme):
    def __init__(self, sde, parameter, steps, alpha=1, **kwargs):
        super().__init__(sde, parameter, steps, **kwargs)
        self.alpha = alpha

    def state_equation(self, x, dW):
        explixit_diffusion = self.diffusion(self.x, self.t)
        return x - (
        self.x + (self.alpha * self.drift(x, self.t + self.h) + (1 - self.alpha) * self.drift(self.x, self.t)) * \
        self.h + explixit_diffusion * dW + 0.5 * self.diffusion_x(self.x, self.t) * explixit_diffusion * (
        dW ** 2 - self.h))

    def propagation(self, x, t):
        dW = self.dW[-1]
        self.x = newton(self.state_equation, self.x, args=(dW,))
