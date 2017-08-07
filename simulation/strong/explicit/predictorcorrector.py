
from simulation.scheme import Scheme

class Order_05(Scheme):
    def __init__(self, sde, parameter, steps, derivatives, alpha=0.5, beta=0.5, **kwargs):
        super().__init__(sde, parameter, steps, derivatives=derivatives, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def corrector(self, x, t):
        return self.drift(x, t) - self.beta * self.diffusion(x, t) * self.diffusion_x(x, t)

    def propagation(self, x, t):
        drift = self.drift(x, t)
        diffusion = self.diffusion(x, t)
        dW = self.dW[-1]
        predictor = x + drift * self.h + diffusion * dW
        self.x += (self.alpha * self.corrector(predictor, t + self.h) + (1 - self.alpha) * self.corrector(x, t)) * self.h + \
                  (self.beta * self.diffusion(predictor, t + self.h) + (1 - self.beta) * diffusion) * dW
