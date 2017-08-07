
from simulation.scheme import Scheme


class Order_05(Scheme):
    def __init__(self, sde, parameter, steps, **kwargs):
        super().__init__(sde, parameter, steps, **kwargs)

    def propagation(self, x, t):
        self.x += self.drift(x, t) * self.h + self.diffusion(x, t) * self.dW[-1]


class Order_10(Scheme):
    def __init__(self, sde, parameter, steps, derivatives, **kwargs):
        super().__init__(sde, parameter, steps, derivatives=derivatives, **kwargs)

    def propagation(self, x, t):
        drift = self.drift(x, t)
        diffusion = self.diffusion(x, t)
        dW = self.dW[-1]
        self.x += drift * self.h + diffusion * dW + 0.5 * self.diffusion_x(x, t) * diffusion * (dW ** 2 - self.h)

class Order_15(Scheme):
    def __init__(self, sde, parameter, steps, derivatives, **kwargs):
        super().__init__(sde, parameter, steps, derivatives=derivatives, **kwargs)

    def propagation(self, x, t):
        drift = self.drift(x, t)
        drift_x = self.drift_x(x, t)
        drift_xx = self.drift_xx(x, t)
        diffusion = self.diffusion(x, t)
        diffusion_x = self.diffusion_x(x, t)
        diffusion_xx = self.diffusion_xx(x ,t)
        dW = self.dW[-1]
        dZ = self.dZ.popleft()
        self.x += drift * self.h + diffusion * dW + 0.5 * diffusion_x * diffusion * (dW ** 2 - self.h) \
                  + drift_x * diffusion * dZ + 0.5 * (drift * drift_x + 0.5 * diffusion ** 2 * drift_xx) * self.h ** 2 \
                  + (drift * diffusion_x + 0.5 * diffusion ** 2 * diffusion_xx) * (dW * self.h - dZ) \
                  + 0.5 * diffusion * (diffusion * diffusion_xx + diffusion_x ** 2) * (dW ** 2 / 3 - self.h) * dW
