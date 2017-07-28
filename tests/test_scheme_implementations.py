import numpy as np

from sde import SDE
from simulation.strong.explicit.taylor import Order_05

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


def test_if_path_is_handed_through_correctly():
    sample_sde = SDE(lambda x: x, lambda x: x)
    euler_instance = Order_05(sample_sde, parameter={}, steps=steps_used, path=sample_differential_brownian_motion)
    for _ in euler_instance: pass
    assert all(euler_instance.return_path() == sample_differential_brownian_motion)
