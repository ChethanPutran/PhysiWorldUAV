# Scenarios and experiment types

## Data types (`requirement.md` #2)

| `data_type` | How it is generated | Purpose |
|---|---|---|
| `closed_loop` | OmniDrones controller tracks a reference (`position`: LeePositionController; `attitude`: AttitudeController with a slow yaw-aware position hold) | Realistic flight; feedback-dependent actions |
| `open_loop` | Predefined actuator programs from hover trim (see below), no tracking controller | System identification; free response |
| `physics_validation` | Thesis Case A (no wind) and Case C (mass +5 % counterfactual) on the nominal vehicle | Evaluating the nominal f_phys against the simulator |

Counterfactual groups are orthogonal to the data type. They are identified by `counterfactual.group_id`, with `mode` = `closed_loop` (Mode A) or `fixed_action` (Mode B: siblings replay the baseline's applied rotor commands).

### The three open-loop abstractions (prompt §6B; not physically equivalent)

| `open_loop_kind` | Input | Feedback |
|---|---|---|
| `rotor_command` | Mixer-space Δ collective [N] and Δ torques [N m] around hover trim → per-rotor thrust via the exact allocation → OmniDrones rotor command | none |
| `collective_rate` | Collective thrust [N] + body-rate targets [rad/s] → OmniDrones RateController | inner rate loop closed |
| `body_wrench` | External body wrench with rotors at trim | none; identification-only, excluded from model-ready data; needs the Run 4 injector |

## Scenario templates (`config/scenario/`)

| Group | Templates |
|---|---|
| Basic | `hover`, `vertical`, `translation` (forward/backward/lateral relative to heading), `line` (constant velocity, any direction), `accel_decel` |
| Complex | `circle`, `helix`, `figure_eight`, `sinusoidal`, `waypoints`, `combined` (circle + yaw oscillation) |
| Attitude | `yaw_step`, `yaw_sine`, `roll_doublet`, `pitch_doublet`, `attitude_combined` |
| Open loop | `open_loop_rotor`, `open_loop_collective_rate` |

Guarantees of the reference library (`trajectories/library.py`):

* Every position reference starts at the initial state with zero velocity and acceleration (minimum-jerk ramps).
* Position is C³.
* Parameters are fixed for the whole episode.
* Directions can be world azimuths or relative to the initial heading.

Every reference passes the plan-time feasibility check (`trajectories/feasibility.py`): thrust within 15–85 % of T_max, tilt, tilt rate and yaw rate, speed, acceleration, jerk, and workspace. Infeasible draws are resampled up to `max_resample` times; after that the plan fails.

## Excitation (`config/excitation/`)

| Template | Applied to | Design-doc group |
|---|---|---|
| `ref_smooth` | position reference: band-limited smooth random | §11.3 C |
| `multisine` | position reference: Schroeder multisine, low crest factor | D |
| `chirp` | position reference: linear chirp | D |
| `prbs` | position reference: PRBS with min-jerk edges | — |
| `yaw_excitation` | yaw reference | §11.2-9 |
| `rotor_doublets`, `rotor_chirp` | `rotor_command` programs | E |
| `rate_doublets` | `collective_rate` programs | E |

Perturbations must start at rest (`ramp_s` / `edge_s` / `start_s`), so references stay anchored. Signals are bounded by their amplitude. All parameters, including PRBS bits and multisine phases, are sampled once and stored.

## Identifiability mix

The pilot's `id_cl` mix follows `docs/datasets/problems/identifiability.md`: hover is 10 %, and the rest is translation, acceleration, yaw, attitude and combined motion. Open-loop parts are separate (`id_ol`). The `helix` and `figure_eight` families are held out of every ID mix and appear only in `test_ood_trajectory` (Q8).
