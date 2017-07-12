
from time import time

import numpy as np

from src.sde import SDE
from src.simulation.strong.taylor import Euler
from src.simulation.strong.explicit import Platen


def cir_drift(x, a, b):
    return a * (b - x)

def cir_diffusion(x, c):
    return np.sqrt(x) * c


cir_process = SDE(cir_drift, cir_diffusion, timerange=[0,2])

euler_path = np.zeros([10, 2001])
platen_path = np.zeros([10, 2001])

print("Run time estimation between Euler and Platen discretization of an CIR process.")

parameter = {'a': 2, 'b': 2.5, 'c' : 0.2}
t = time()
for i in range(10):
    tmp = []
    for path in Euler(cir_process, parameter, steps = 2000):
        tmp.append(path)
    euler_path[i] = tmp
print("Euler: " + str(time() - t))

t = time()
for i in range(10):
    tmp = []
    for path in Platen(cir_process, parameter, steps = 2000):
        tmp.append(path)
    platen_path[i] = tmp
print("Platen-Runge-Kutta: " + str(time() - t))

print('Simulation complete.')