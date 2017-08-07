
import math
from collections import OrderedDict

import numba
import numpy as np

from sde import SDE
from simulation.strong.explicit.rk import Order_10 as Platen
from simulation.strong.explicit.taylor import Order_05 as Euler
from simulation.strong.explicit.taylor import Order_10 as Milstein

"""
We compare the order of convergence between two strong Taylor schemes and one derivative-free strong
Schemes.

As a benchmark process we choose a geometric Brownian motion. Since the derivatives of the diffusion function
are known and explicitly computable, we expect to gain higher order of convergence through the 1.0 - Taylor scheme
which is known as the Milstein scheme.
A similar order of convergence should be expected with the explicit scheme but we do also expect some
linear penalty through the approximation error of the derivative.

In this script we will create a process and solve it with the mentioned three schemes.
For this we create n paths of a Brownian Motion and compute the schemes with various resolutions.
The coefficients can then be estimated by linear regression against the log-errors.
"""

end_point = 1
num_samples = 500
parameter = OrderedDict(mu=0.6, sigma=0.3)
resolutions = [2 ** -4, 2 ** -5, 2 ** -6, 2 ** -7, 2 ** -8, 2 ** -9]

@numba.jit
def gbm_endval_given_bm_endval(t, x0, mu, sigma, bm_t):
    return x0 * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * bm_t)

@numba.jit
def gbm_drift(x, mu):
    return mu * x

@numba.jit
def gbm_diffusion(x, sigma):
    return sigma * x

@numba.jit
def gbm_difusion_x(sigma):
    return sigma


gbm_process = SDE(gbm_drift, gbm_diffusion, timerange=[0, end_point])

stepsizes = [int(np.ceil(end_point / i)) for i in resolutions]
analytical_values = np.full([num_samples, len(stepsizes)], np.nan)
euler_values = np.full([num_samples, len(stepsizes)], np.nan)
platen_values = np.full([num_samples, len(stepsizes)], np.nan)
milstein_values = np.full([num_samples, len(stepsizes)], np.nan)

for i in range(num_samples):
    dW_full = np.random.standard_normal(max(stepsizes)) * math.sqrt(end_point / max(stepsizes))
    for r_count, res in enumerate(stepsizes, start=0):

        dW = [sum(dW_full[int(i * len(dW_full) / res): int((i + 1) * len(dW_full) / res)]) for i in range(res)]

        for path_value in Euler(gbm_process, parameter, steps=res, path=dW): pass
        euler_values[i, r_count] = path_value

        for path_value in Milstein(gbm_process, parameter, steps=res, derivatives={'diffusion_x': gbm_difusion_x},
                                   path=dW, alpha=0.5, beta=0.5): pass
        milstein_values[i, r_count] = path_value

        for path_value in Platen(gbm_process, parameter, steps=res, path=dW): pass
        platen_values[i, r_count] = path_value

        analytical_values[i, r_count] = gbm_endval_given_bm_endval(end_point, 1, parameter['mu'], parameter['sigma'],
                                                                   np.cumsum(dW)[-1])

euler_errors = np.mean(abs((euler_values - analytical_values)), axis=0)
milstein_errors = np.mean(abs((milstein_values - analytical_values)), axis=0)
platen_errors = np.mean(abs((platen_values - analytical_values)), axis=0)

log_errors = np.log2(resolutions)
error_regression_matrix = np.array([np.ones(log_errors.shape), log_errors]).transpose()
euler_coefficients = np.linalg.solve(error_regression_matrix.T.dot(error_regression_matrix),
                                     error_regression_matrix.T.dot(np.log2(euler_errors)))
milstein_coefficients = np.linalg.solve(error_regression_matrix.T.dot(error_regression_matrix),
                                        error_regression_matrix.T.dot(np.log2(milstein_errors)))
platen_coefficients = np.linalg.solve(error_regression_matrix.T.dot(error_regression_matrix),
                                      error_regression_matrix.T.dot(np.log2(platen_errors)))

print("Upper bound Euler errors: {} * h^{}".format(np.exp(euler_coefficients[0]), euler_coefficients[1]))
print("Upper bound Milstein errors: {} * h^{}".format(np.exp(milstein_coefficients[0]), milstein_coefficients[1]))
print("Upper bound Platen errors: {} * h^{}".format(np.exp(platen_coefficients[0]), platen_coefficients[1]))
