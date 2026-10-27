Body drag is a good starting point, but it is not a complete quadrotor aerodynamic model.









A. Rotor thrust changes due to airflow

Wind can alter rotor inflow and thrust. A simple body-drag model does not capture this. A more advanced model accounts for the air velocity relative to each rotor and modifies thrust and rotor moments accordingly.











B. Aerodynamic moments

Airflow acting away from the center of mass can produce torque. A simple approximation is to model body-frame angular damping, but geometry-based aerodynamic moments require additional parameters.











C. Rotor and blade effects

Blade flapping, rotor drag, and induced airflow can matter at higher airspeeds. These are more involved and are best added only if your simulator or vehicle model supports them reliably.



For your thesis, I would implement wind velocity plus body-frame aerodynamic drag first. That gives you a meaningful upgrade over direct force injection without requiring a full CFD model.


Experiment	Wind model	What it tests
A	Uniform steady wind	Basic dynamics prediction
B	Uniform, time-varying gusts	Generalization to temporal disturbances
C	Spatially varying wind	Generalization to wind gradients
D	Stronger and unseen disturbances	Robustness under OOD conditions




Check wind:

Profiles:
    none
    constant
    step
    sinusoidal
    gust
    dryden
    random

Steady wind 
    direction
    maginitude

Steady wind + gusts
    direction
    magnitude
    frequency


Steady wind + gradients
    direction
    magnitude
    gradient: magnitude and direction

Steady wind + gusts + spatial gradients
    direction
    magnitude
    frequency
    gradient