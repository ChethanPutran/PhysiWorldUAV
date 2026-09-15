import torch


def rk4_step(dynamics, state, dt):
    k1 = dynamics(state)
    k2 = dynamics(state + 0.5 * dt * k1)
    k3 = dynamics(state + 0.5 * dt * k2)
    k4 = dynamics(state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)