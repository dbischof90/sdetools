
from simulation.scheme import Scheme


class Order_10(Scheme):
    def __init__(self, sde, parameter, steps, **kwargs):
        super().__init__(sde, parameter, steps, **kwargs)

    def propagation(self, x, t):
        drift = self.drift(x, t)
        diffusion = self.diffusion(x, t)
        dW = self.dW[-1]
        predictor = x + drift * self.h + diffusion * self.sqrt_h
        self.x += drift * self.h + diffusion * dW + 0.5 * (self.diffusion(predictor, t) - diffusion) / self.sqrt_h * (dW ** 2 - self.h)


class Order_15(Scheme):
    def __init__(self, sde, parameter, steps, **kwargs):
        super().__init__(sde, parameter, steps, **kwargs)

    def propagation(self, x, t):
        drift = self.drift(x, t)
        diffusion = self.diffusion(x, t)
        dW = self.dW[-1]
        dZ = self.dZ.popleft()

        predictor_p = x + drift * self.h + diffusion * self.sqrt_h
        predictor_m = predictor_p - 2 * diffusion * self.sqrt_h
        diffusion_p = self.diffusion(predictor_p, t)
        diffusion_m = self.diffusion(predictor_m, t)
