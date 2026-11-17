# Architecture overview

Status: Run 4 (physics models implemented and validated in Isaac). See also
`docs/models/` for force accounting, aerodynamics, drone, IMU and wind. See `docs/audit/run2_implementation_plan.md` for the plan
and `docs/audit/run3_report.md` for what is implemented.

## Data flow

```
config/ (Hydra groups) ──compose──► resolved config
                                      │
                                      ▼
experiments/plan.py ──► manifest.jsonl     every random choice made here, from the seed tree
   domains (randomization/)               (core/seeds.py): domains, initial conditions,
   references (trajectories/)             references + feasibility, excitation, wind
   excitation (excitation/)               realisations (+ realised stats), counterfactual
   wind regimes (wind/regimes.py)         siblings, split role per group
   counterfactual siblings
                                      │
                                      ▼
experiments/runner.py  (one manifest row → one episode, Isaac via core/simulation_interface.py)
   control every D physics steps (control/controllers.py over OmniDrones Lee/Attitude/Rate)
   rotor command held for D steps; data/recorder.py fills device buffers at 200 Hz
                                      │
                                      ▼
data/writer.py (HDF5 shards, schema-validated) ──► data/dataset_validator.py ──► index.json
                                      │
                                      ▼
data/dataset_builder.py + data/dataset.py  (model-rate view: decimate, windows, future actions)
```

## Packages (`src/physiworld_uav/`)

| Package | Responsibility |
|---|---|
| `config/` | Hydra composition (`compose`) and named templates (`load_template`) |
| `core/` | Simulator wrapper, state layout and frame conversions, seeds, provenance, YAML loader |
| `drones/` | `DroneParams` (generic multirotor parameters, derived quantities, OmniDrones cross-check); `sim_binding` (domain → PhysX, read-back, effective inertia) |
| `control/` | Control laws over OmniDrones controllers; frame handling per controller |
| `trajectories/` | Pure-time reference library, feasibility checks, GP sampler |
| `excitation/` | Bounded signals, reference perturbations, open-loop programs (three abstractions) |
| `wind/` | Wind components, field, manager, ID/OOD regimes with realised-statistics guarantees |
| `randomization/` | Coupled physical-domain sampling (Sobol/LHS), ID/OOD regions, held-out corners |
| `forces/` | Aerodynamic models (rotor drag, body drag, ground effect, rotational damping), applicability registry, wrench injector into OmniDrones' base-link call |
| `experiments/` | Planner, counterfactual siblings, episode runner, CLI |
| `data/` | Schema v2, recorder, HDF5 I/O, decimation, model-ready dataset, validation, splits |
| `models/ training/ losses/ baselines/ evaluation/ visualization/` | World-model code — out of scope until after Run 6 (D10) |

## Design rules

* **Plan, then simulate.** The simulator run is a pure function of a manifest row, so episodes are individually reproducible and can be distributed across workers in any order.
* **Reuse, don't reimplement.** Position and attitude control, rotor mixing and the motor model are OmniDrones'. The project adds only the control-law glue and frame handling.
* **Fail loudly.** Anything that could silently mislabel data raises instead: missing simulator APIs, schema violations, unvalidated force models, wind without force models, non-nominal domains before PhysX binding, and perturbations that don't start at rest.
* **Truth, observations and actions are separate** (`requirement.md` #4), and auxiliary data is never a model input.

## Known issues (tracked)

| Issue | Status |
|---|---|
| Isaac `IMUSensor` returned no fresh frames | **Fixed in Run 4**: sensor prim created before `sim.reset()` + `gpu_dynamics_cpu_readback` pipeline (decision (a)); verified against the state |
| Force injection, frames, PhysX damping 0.2, domain binding | **Fixed and validated in Run 4** (docs/models/force_accounting.md) |
| Drag coefficients (Q3) | Proposed with sources; `truth_default` is `validated: false` until approved — the runner refuses it (dev flag `--allow-unvalidated-forces` produces datasets that never validate) |
| Small articulation roll–pitch coupling with unequal rotor thrusts (~0.3 % of authority) | Documented; part of the learned residual (Q4) |
| `training/train.py` imports the deleted `physiworld_uav.physics` | Out of scope (D10) |

## Replicator

Not used. Replicator is for visual synthetic data; batched physics uses vectorised environments (Run 6).
