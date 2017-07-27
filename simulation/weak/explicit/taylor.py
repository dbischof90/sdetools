from simulation.scheme import Scheme


class Order_10(Scheme):
    def __init__(self, sde, parameter, steps, **kwargs):
        super().__init__(sde, parameter, steps, **kwargs)

    def propagation(self, x, t):
        drift = self.drift(x, t)
        diffusion = self.diffusion(x, t)
        dW = self.dW[-1]
        self.x += drift * self.h + diffusion * dW


class Order_20(Scheme):
    def __init__(self, sde, parameter, steps, derivatives, **kwargs):
        super().__init__(sde, parameter, steps, derivatives=derivatives, **kwargs)

    def propagation(self, x, t):
        drift = self.drift(x, t)
        drift_x = self.drift_x(x, t)
        drift_xx = self.drift_xx(x, t)
        diffusion = self.diffusion(x, t)
        diffusion_x = self.diffusion_x(x, t)
        diffusion_xx = self.diffusion_xx(x, t)
        dW = self.dW[-1]
        self.x += drift * self.h + diffusion * dW + \
                  0.5 * ((diffusion_x * diffusion) * (dW ** 2 - self.h) + \
                         (
                         drift_x * diffusion + drift * diffusion_x + 0.5 * diffusion_xx * diffusion ** 2) * dW * self.h + \
                         (drift * drift_x + 0.5 * drift_xx * diffusion ** 2) * self.h ** 2)
