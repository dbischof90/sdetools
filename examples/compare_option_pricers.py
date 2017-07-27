from time import time

import numpy as np
from scipy.stats import norm

from sde import SDE
from simulation.strong.explicit.taylor import Order_05
from simulation.strong.explicit.taylor import Order_10

"""
In this example we will see the benefits of higher-order schemes in the application of option pricing through Monte Carlo methods.
While less difficult to compute, we need more precision with the lower order schemes in comparison to the higher order
schemes.

This script calculates the steps needed to get the pricing precision under a needed precision threshold.
To isolate the error given through the discretization, we use a high sample size for the Monte Carlo estimator.

Note that this script might take some time to run.
Another note is that the method presented here is not step-optimal. Nonlinear integer programming is a difficult problem
and delving into this field oversteps the objective here.

Do also note that for smaller number of steps the absolute error still has a considerable variance, so the
error will not necessarily be continuously decreasing.
"""

T = 1
r = 0.01
sigma = 0.2
K = 70
S0 = 80
parameter = {'mu': r, 'sigma': sigma}


"""
First, we define the standard Black-Scholes European Call price function
"""

def bs_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    Nd1 = norm.cdf(d1, 0.0, 1.0)
    Nd2 = norm.cdf(d2, 0.0, 1.0)
    return S0 * Nd1 - np.exp(-r * T) * K * Nd2


"""
The next step is to define a geometric Brownian Motion.
"""

def gbm_drift(x, mu):
    return mu * x


def gbm_diffusion(x, sigma):
    return sigma * x


def gbm_diffusion_x(x, sigma):
    return sigma


gbm_process = SDE(gbm_drift, gbm_diffusion, timerange=[0, T], startvalue=S0)


"""
A next step is the generation of 500.000 possible option pay-offs. The option price will be
the discounted mean of the possible pay-offs.
"""

def euler_payoffs(steps):
    euler_values = []
    for i in range(1000000):
        for path in Order_05(gbm_process, parameter, steps=steps): pass
        euler_values.append(max(path - K, 0))

    return euler_values


def milstein_payoffs(steps):
    milstein_values = []
    for i in range(1000000):
        for path in Order_10(gbm_process, parameter, steps=steps, derivatives={'diffusion_x': gbm_diffusion_x}): pass
        milstein_values.append(max(path - K, 0))

    return milstein_values


def euler_discretization_error(steps):
    error = abs(bs_call(S0, K, T, r, sigma) - np.exp(-r * T) * np.mean(euler_payoffs(int(np.ceil(steps)))))
    print("Euler discretization with {} steps: Absolute error {}".format(int(np.ceil(steps)), error))
    return error


def milstein_discretization_error(steps):
    error = abs(bs_call(S0, K, T, r, sigma) - np.exp(-r * T) * np.mean(milstein_payoffs(int(np.ceil(steps)))))
    print("Milstein discretization with {} steps: Absolute error {}".format(int(np.ceil(steps)), error))
    return error


"""
We do now calculate the absolute error of the price approximation. If the error is too large, we increase the step count
and repeat the option price calculation.
This procedure will be done for both the Euler and the Milstein scheme. At the end, we time the option pricing
with both step sizes to demonstrate the differences.
"""

euler_error = 1
milstein_error = 1
steps_euler = 50
steps_milstein = 50

while euler_error > 10e-5:
    euler_error = euler_discretization_error(steps_euler)
    steps_euler += 50

while milstein_error > 10e-5:
    milstein_error = euler_discretization_error(steps_milstein)
    steps_milstein += 50

t = time()
np.mean(euler_discretization_error(steps_euler))
time_euler = time() - t
t = time()
np.mean(milstein_discretization_error(steps_milstein))
time_milstein = time() - t

print("Calculation time of the Euler scheme with error smaller than 10e-5: {} seconds. Resolution needed: {}".format(
    time_euler, T / steps_euler))
print("Calculation time of the Milstein scheme with error smaller than 10e-5: {} seconds. Resolution needed: {}".format(
    time_milstein, T / steps_milstein))
