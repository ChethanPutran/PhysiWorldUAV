# Dataset schema (v2.0)

> Generated from `src/physiworld_uav/data/schema.py` — edit the code, then regenerate this page.

## Files of a dataset

`<output_root>/<dataset_id>_v<version>/`

| File | Content |
|---|---|
| `resolved_config.yaml` | Fully resolved Hydra config |
| `provenance.json` | Config hash, git state (project + OmniDrones), Isaac/extension/library versions, GPUs, timestamp |
| `manifest.jsonl` | One fully specified plan row per episode (every sampled value and seed) |
| `shard_XXX.h5` | Episodes: `/episodes/<episode_id>/<group>/<field>`, metadata JSON in group attribute `metadata` |
| `index.json` | Episode rows (split role, group, shard, status), model layout, validation summary, `complete` flag |
| `validation.json` | Full validation report (hard and soft findings) |
| `regenerate_<episode>.json` | Optional: max abs difference after re-simulating one episode |

A dataset is `complete` only if every planned episode was written and every hard validation check passed.

## Sampling semantics

Physics rate: k = 0…N, t_k = k·dt (dt = 0.005 s).

* **inst** — sampled at the instant t_k (after step k−1 → k): N+1 rows.
* **intv** — applied/held over [t_k, t_(k+1)): N rows.

Actions are held over control intervals of D = `control_decimation` physics steps, so they are exactly piecewise constant at the model rate.

## Physics-rate fields

| Field | Shape | dtype | Unit | Frame | Sem. | Req. | Description |
|---|---|---|---|---|---|---|---|
| `time/t` | (N+1) | float64 | s | - | inst | yes | simulation time t_k = k·dt |
| `ground_truth/position_W` | (N+1,3) | float32 | m | world ENU | inst | yes | base_link origin position |
| `ground_truth/orientation_wxyz` | (N+1,4) | float32 | - | body→world | inst | yes | Hamilton unit quaternion, sign-continuous |
| `ground_truth/velocity_W` | (N+1,3) | float32 | m/s | world | inst | yes | linear velocity |
| `ground_truth/angular_velocity_B` | (N+1,3) | float32 | rad/s | body FLU | inst | yes | body angular velocity ω_B = R_WBᵀ ω_W (thesis state) |
| `ground_truth/angular_velocity_W` | (N+1,3) | float32 | rad/s | world | inst | yes | world angular velocity (diagnostic; simulator native) |
| `ground_truth/linear_acceleration_W` | (N+1,3) | float32 | m/s² | world | inst | no | DERIVED: central difference of PhysX velocity |
| `ground_truth/angular_acceleration_B` | (N+1,3) | float32 | rad/s² | body | inst | no | DERIVED: central difference of ω_B |
| `ground_truth/rotor_speed` | (N+1,4) | float32 | rad/s | rotor | inst | yes | Ω_i = throttle_i·Ω_max,i after the motor lag (D3 model action) |
| `ground_truth/rotor_thrust` | (N+1,4) | float32 | N | +Z_B | inst | yes | k_F·Ω_i² |
| `ground_truth/wrench/external_force_W` | (N,3) | float32 | N | world | intv | yes | total force injected by project force models (exactly what was applied) |
| `ground_truth/wrench/external_torque_B` | (N,3) | float32 | N·m | body | intv | yes | total torque injected by project force models |
| `observations/imu_accel_B` | (N+1,3) | float32 | m/s² | body (sensor=body) | inst | yes | specific force from Isaac IMUSensor (clean; noise applied at load time, D7) |
| `observations/imu_gyro_B` | (N+1,3) | float32 | rad/s | body | inst | yes | IMU angular rate |
| `observations/imu_valid` | (N+1) | bool | - | - | inst | yes | fresh sensor frame at t_k |
| `actions/rotor_cmd` | (N,4) | float32 | [-1,1] | - | intv | yes | applied (clamped) OmniDrones rotor command, held over the control interval |
| `actions/rotor_cmd_raw` | (N,4) | float32 | - | - | intv | yes | controller output before clamping |
| `actions/saturated` | (N,4) | bool | - | - | intv | yes | rotor_cmd_raw outside [-1,1] |
| `actions/rotor_speed_cmd` | (N,4) | float32 | rad/s | rotor | intv | yes | steady-state Ω target of the command: sqrt((cmd+1)/2)·Ω_max (Q6) |
| `actions/collective_thrust_cmd` | (N) | float32 | N | +Z_B | intv | yes | commanded collective thrust (NaN when the mode has none) |
| `actions/body_rate_cmd` | (N,3) | float32 | rad/s | body | intv | yes | commanded body rates (NaN when the mode has none) |
| `control/reference/position_W` | (N,3) | float32 | m | world | intv | no | reference position |
| `control/reference/velocity_W` | (N,3) | float32 | m/s | world | intv | no | reference velocity |
| `control/reference/acceleration_W` | (N,3) | float32 | m/s² | world | intv | no | reference accel |
| `control/reference/yaw` | (N) | float32 | rad | world | intv | no | reference yaw |
| `control/reference/roll` | (N) | float32 | rad | - | intv | no | commanded roll (attitude mode) |
| `control/reference/pitch` | (N) | float32 | rad | - | intv | no | commanded pitch (attitude mode) |
| `control/tracking_error/position_W` | (N,3) | float32 | m | world | intv | no | p − p_ref |
| `control/tracking_error/velocity_W` | (N,3) | float32 | m/s | world | intv | no | v − v_ref |
| `environment/wind_W` | (N+1,3) | float32 | m/s | world | inst | yes | wind at the CoM at t_k |
| `environment/wind_applied_W` | (N,3) | float32 | m/s | world | intv | yes | wind used to compute the wrench for interval k |

## Model-ready view (`data/dataset_builder.py`, `data/dataset.py`)

Model step Δt = D·dt (default 0.02 s). States are **decimated**, never filtered.

* state (13): p_W[0:3], v_W[3:6], q_wxyz[6:10], omega_B[10:13]
* observations (6): IMU specific force (body) | IMU gyro (body)
* action keys: `rotor_speed_mean` (ground_truth/rotor_speed__mean, mean), `rotor_speed` (ground_truth/rotor_speed, start), `rotor_cmd` (actions/rotor_cmd, held), `rotor_speed_cmd` (actions/rotor_speed_cmd, held); default `rotor_speed_mean` (decision D3)

Sample at current model step t (H history, P horizon):

| Key | Shape | Content |
|---|---|---|
| `state_history` | (H, 13) | s_(t−H+1 … t) |
| `obs_history` | (H, 6) | o_(t−H+1 … t) |
| `action_history` | (H, A) | u_(t−H … t−1), u_k produced s_(k+1) |
| `future_actions` | (P, A) | u_(t … t+P−1) |
| `future_states` | (P, 13) | s_(t+1 … t+P) |
| `dt` | () | model step |

Auxiliary data (wind, injected wrench, parameters) is never part of the primary sample; use `AuxiliaryView`.

## Labels

* `data_type` ∈ closed_loop, open_loop, physics_validation
* `split_role` ∈ train, val_id, test_id, val_ood_magnitude, test_ood_magnitude, val_ood_direction, test_ood_direction, val_ood_temporal, test_ood_temporal, val_ood_gust, test_ood_gust, val_ood_physics, test_ood_physics, val_ood_trajectory, test_ood_trajectory, val_ood_joint, test_ood_joint, physics_validation
* `ood_category`: set only for `*_ood_*` roles; counterfactual groups carry `variable`, `mode` (closed_loop = Mode A, fixed_action = Mode B), `sibling_index`, `is_baseline` and the exact `diff` against the baseline.
