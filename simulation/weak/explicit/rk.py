
from simulation.scheme import Scheme


class Order_20(Scheme):
    def __init__(self, sde, parameter, steps, **kwargs):
        super().__init__(sde, parameter, steps, **kwargs)

    def predictor_p(self):
        return self.x + self.drift(self.x, self.t) * self.h + \
               self.diffusion(self.x, self.t) * self.sqrt_h

    def predictor_m(self):
        return self.x + self.drift(self.x, self.t) * self.h + \
               self.diffusion(self.x, self.t) * self.sqrt_h

    def predictor_euler(self, dW):
        return self.x + self.drift(self.x, self.t) * self.h + \
               self.diffusion(self.x, self.t) * dW

    def propagation(self, x, t):
        drift = self.drift(x, t)
        diffusion = self.diffusion(x, t)
        diffusion_pm = self.diffusion(self.predictor_m(), t)
        diffusion_pp = self.diffusion(self.predictor_p(), t)
        dW = self.dW[-1]
        drift_pe = self.drift(self.predictor_euler(dW), t)
        self.x += 0.5 * ((drift_pe + drift) * self.h + \
                         0.5 * (diffusion_pp + diffusion_pm + 2 * diffusion) * dW + \
                         0.5 * (diffusion_pp - diffusion_pm) * (dW ** 2 - self.h) / self.sqrt_h)
