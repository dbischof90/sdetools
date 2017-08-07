import functools as ft
import math
from collections import OrderedDict

import numba
import numpy as np

from sde import SDE
# from simulation.strong.explicit.predictorcorrector import Order_10 as pc_e
from simulation.strong.explicit.rk import Order_10 as rk_e
from simulation.strong.explicit.taylor import Order_05 as Euler_e
from simulation.strong.explicit.taylor import Order_10 as Milstein_e
from simulation.strong.implicit.taylor import Order_05_Trapez as Euler_i
from simulation.strong.implicit.taylor import Order_10_Trapez as Milstein_i


def map_scheme_to_arguments(cls, *args, **kwargs):
    class MappedScheme(cls):
        __init__ = ft.partialmethod(cls.__init__, *args, **kwargs)

    return MappedScheme


def build_instance_list_of_mapped_schemes(mapped_scheme, step_list, differentials):
    if len(step_list) != len(differentials):
        raise ValueError('Wrong number of resolutions or differentials!')

    scheme_list = []
    for steps, diff in zip(step_list, differentials):
        scheme_list.append(mapped_scheme(steps=steps, path=diff))
    return scheme_list


def list_has_equal_strong_convergence_order(list_of_schemes, resolutions, order):

    stepsizes = [int(np.ceil(end_point / i)) for i in resolutions]
    differentials = [np.random.standard_normal(max(stepsizes)) * math.sqrt(end_point / max(stepsizes)) for i in
                               range(num_samples)]
    analytical_values = np.full([num_samples, len(stepsizes)], np.nan)
    scheme_values = [np.full([num_samples, len(stepsizes)], np.nan) for s in list_of_schemes]
    list_scheme_instances = [list_of_schemes for i in range(num_samples * len(resolutions))]

    for i in range(num_samples):
        dW_full = differentials.pop()
        for r_count, res in enumerate(stepsizes, start=0):
            dW = [sum(dW_full[int(i * len(dW_full) / res): int((i + 1) * len(dW_full) / res)]) for i in range(res)]
            scheme_instance = list_scheme_instances.pop()
            for idx, scheme in enumerate(scheme_instance):
                for path_value in scheme(steps=res, path=dW): pass
                scheme_values[idx][i, r_count] = path_value
            analytical_values[i, r_count] = gbm_endval_given_bm_endval(end_point, 1, 0.8, 0.6, np.cumsum(dW)[-1])
    scheme_errors = [np.mean(abs((scheme_values[idx] - analytical_values)), axis=0) for idx in
                     range(len(list_of_schemes))]
    log_errors = np.log2(resolutions)
    error_regression_matrix = np.array([np.ones(log_errors.shape), log_errors]).transpose()
    scheme_coefficients = [np.linalg.solve(error_regression_matrix.T.dot(error_regression_matrix),
                                           error_regression_matrix.T.dot(np.log2(scheme_errors[idx]))) for idx in
                           range(len(list_of_schemes))]
    scheme_orders = [coeff[1] for coeff in scheme_coefficients]
    print(' Tested {} schemes of order {}.'.format(len(list_of_schemes), order))
    return all(np.isclose(scheme_orders, order, 10e-2))


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


end_point = 1
num_samples = 500
gbm_process = SDE(gbm_drift, gbm_diffusion, timerange=[0, end_point])
resolutions = [2 ** -4, 2 ** -5, 2 ** -6, 2 ** -7, 2 ** -8, 2 ** -9]
gbm_para_sample = OrderedDict(mu=0.8, sigma=0.6)
gbm_derivatives = {'diffusion_x': gbm_difusion_x}
stepsizes = [int(np.ceil(end_point / i)) for i in resolutions]


def test_convergence_order_05():
    mapped_euler_e = map_scheme_to_arguments(Euler_e, sde=gbm_process, parameter=gbm_para_sample)
    mapped_euler_i = map_scheme_to_arguments(Euler_i, sde=gbm_process, parameter=gbm_para_sample)
#    mapped_pc_e = map_scheme_to_arguments(pc_e, sde=gbm_process, parameter=gbm_para_sample, derivatives=gbm_derivatives, beta=1)
    list_schemes = [mapped_euler_e, mapped_euler_i] #mapped_pc_e]
    assert list_has_equal_strong_convergence_order(list_schemes, resolutions, 0.5)


def test_convergence_order_10():
    mapped_milstein_e = map_scheme_to_arguments(Milstein_e, sde=gbm_process, parameter=gbm_para_sample, derivatives=gbm_derivatives)
    mapped_milstein_i = map_scheme_to_arguments(Milstein_i, sde=gbm_process, parameter=gbm_para_sample, derivatives=gbm_derivatives)
    mapped_rk_e = map_scheme_to_arguments(rk_e, sde=gbm_process, parameter=gbm_para_sample)
    list_schemes = [mapped_milstein_e, mapped_milstein_i, mapped_rk_e]
    assert list_has_equal_strong_convergence_order(list_schemes, resolutions, 1.0)


def test_if_path_is_handed_through_correctly():
    steps_used = 50
    sample_differential_brownian_motion = np.array([0.03329902, 0.20850244, 0.12094308, -0.14159548, 0.02973983,
                                                    0.06103259, -0.00915205, 0.01928274, 0.09207789, -0.13199381,
                                                    0.17663064, 0.1333172, -0.01288733, -0.31281056, -0.05924482,
                                                    -0.01702982, 0.18025385, -0.17514341, 0.03477228, 0.31712905,
                                                    -0.25351569, -0.19384718, -0.29929325, 0.20444405, 0.08353272,
                                                    0.09427778, 0.05516237, -0.18329133, -0.18365494, -0.13901742,
                                                    -0.15492822, 0.0384501, -0.0544241, -0.15041881, -0.07649629,
                                                    0.07692755, -0.12122493, 0.18393892, 0.12113368, 0.10871338,
                                                    -0.1328373, -0.05468304, 0.08074539, 0.52846189, -0.00426639,
                                                    0.04982364, 0.16280621, -0.03664431, 0.22651330, -0.08565257])

    sample_sde = SDE(lambda x: x, lambda x: x)
    euler_instance = Euler_e(sample_sde, parameter=OrderedDict(), steps=steps_used, path=sample_differential_brownian_motion)
    for _ in euler_instance: pass
    assert all(euler_instance.return_path() == sample_differential_brownian_motion)
