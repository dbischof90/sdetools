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
