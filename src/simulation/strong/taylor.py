import numpy as np

from src.simulation.scheme import Scheme


class Order_05(Scheme):
    def __init__(self, sde, parameter, steps):
        super().__init__(sde, parameter, steps)

    def propagation(self, x, t):
        drift = self.drift(x, t)
        diffusion = self.diffusion(x, t)
        dW = np.random.standard_normal() * np.sqrt(self.h)
        self.x += drift * self.h + diffusion * dW


class Order_10(Scheme):
    def __init__(self, sde, parameter, steps, derivatives):
        super().__init__(sde, parameter, steps, derivatives=derivatives)

    def propagation(self, x, t):
        drift = self.drift(x, t)
        diffusion = self.diffusion(x, t)
        diffusion_x = self.diffusion_x(x, t)
        dW = np.random.standard_normal() * np.sqrt(self.h)
        self.x += drift * self.h + diffusion * dW + diffusion_x * diffusion * (dW ** 2 - self.h)
