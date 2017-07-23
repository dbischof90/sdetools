
import numpy as np

from simulation.scheme import Scheme


class Order_10(Scheme):
    def __init__(self, sde, parameter, steps, derivatives, alpha=0.5, beta=0.5):
        super().__init__(sde, parameter, steps)
        self.diffusion_x = derivatives['diffusion']['x']
        self.alpha = alpha
        self.beta = beta

    def corrector(self, x, t):
        return self.drift(x, t) - self.beta * self.diffusion(x, t) * self.diffusion_x(x, t)

    def propagation(self, x, t):
        drift = self.drift(x, t)
        diffusion = self.diffusion(x, t)
        predictor = x + drift * self.h + diffusion * np.sqrt(self.h)
        dW = self.dW[-1]
        self.x += (self.alpha * self.corrector(predictor, self.t + self.h) + (1 - self.alpha) * self.corrector(self.x,
                                                                                                               self.t)) * self.h + \
                  (self.beta * self.diffusion(predictor, self.t + self.h) + (1 - self.beta) * diffusion) * dW
