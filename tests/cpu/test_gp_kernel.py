import numpy as np

from physiworld_uav.trajectories.gp_kernel import PeriodicKernel, make_composite


def test_kernel_symmetry():
    k = PeriodicKernel(sigma=1.0, ell=1.3, period=10.0)
    tau = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    assert np.allclose(k.k(tau), k.k(-tau), atol=1e-12)
    assert k.k(np.array([0.0]))[0] > 0


def test_derivative_shapes_and_limits():
    k = PeriodicKernel(sigma=1.0, ell=1.3, period=10.0)
    tau = np.linspace(0.0, 5.0, 50)
    assert k.k(tau).shape == tau.shape
    assert k.k_t(tau).shape == tau.shape
    assert k.k_tt(tau).shape == tau.shape
    # finite-difference consistency
    h = 1e-5
    fd = (k.k(tau + h) - k.k(tau - h)) / (2 * h)
    assert np.allclose(k.k_t(tau), fd, atol=1e-6)


def test_composite_is_sum_with_total_variance_sigma2():
    """Plan T7: n components each carry sigma²/n, so k(0) = sigma² for the sum."""
    kc = make_composite(sigma=1.0, length_scales=[1.3, 3.0], periods=[10.0, 20.0])
    s = 1.0 / np.sqrt(2.0)
    k1 = PeriodicKernel(s, 1.3, 10.0)
    k2 = PeriodicKernel(s, 3.0, 20.0)
    tau = np.linspace(0, 5, 20)
    assert np.allclose(kc.k(tau), k1.k(tau) + k2.k(tau), atol=1e-12)
    assert np.isclose(kc.k(np.zeros(1))[0], 1.0)