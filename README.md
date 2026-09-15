# UAV Physics-Prior Latent World Model

Research prototype implementing:

    ds/dt = f_phys(s,u) + d_theta(h,s,u)

The code currently uses a synthetic differentiable quadrotor simulator
so that the complete pipeline can be tested before connecting Isaac Sim.

## State

    [position(3), velocity(3), rotation_vector(3), angular_velocity(3)]

## Action

    [rotor_1, rotor_2, rotor_3, rotor_4]

## Main baseline

Physics-only:
    ds/dt = f_phys(s,u)

Pure neural:
    s_{t+1} = s_t + NN(s_t,u_t)

Physics-prior:
    ds/dt = f_phys(s,u) + d_theta(h,s,u)

## Run

    pip install -r requirements.txt
    python main.py --device cuda

For CPU:

    python main.py --device cpu

This is an MVP research implementation. The synthetic dynamics should
be replaced by Isaac Sim trajectories for the thesis experiments.