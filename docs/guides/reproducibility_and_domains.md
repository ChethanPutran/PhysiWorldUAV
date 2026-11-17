# Reproducibility, splits and physical domains

## Seed tree (`core/seeds.py`)

```
experiment seed
 └ part:<name>
     ├ domains └ region:<region>                 → domain design (Sobol/LHS)
     ├ assignment                                 → group order / role interleaving
     └ group:<nnnnn>
         ├ duration
         ├ attempt:<k>                            (feasibility resampling)
         │    ├ initial_conditions
         │    ├ trajectory
         │    └ excitation
         ├ wind                                    → regime sampling; per component:
         │                                           child_seed(wind_seed, "wind_component:<name>")
         └ cf_select
 episode seed = root.child(<episode_id>)
```

* Children are derived from stable string keys (SHA-256 → `SeedSequence.spawn_key`), not list positions. Adding a part, a scenario type or a template field does not change any other seed (tested).
* Every seed (integer and tree path) is stored in the manifest and in episode metadata.
* Counterfactual siblings share every stream except the bookkeeping `episode` seed.

## Provenance

`provenance.json` holds:
* the resolved-config SHA-256;
* git commit and dirty flag (project and OmniDrones);
* Isaac Sim and extension versions;
* library versions, CUDA and GPUs;
* a UTC timestamp.

Each episode stores the `config_hash` and its complete manifest row (`metadata.plan`).

## Regenerating one episode

```bash
python -m physiworld_uav.experiments.generate regenerate --dataset <dir> --episode <id>
```

This re-runs the manifest row and writes `regenerate_<id>.json`, the max |Δ| for every field. Run 3 result for the smoke episode (fresh Isaac launch, 1 env): **0.0** for positions and commands.

Determinism across environment counts and GPUs is evaluated in Runs 4 and 6.

## Splits by design (plan §6.5-DS8)

* **Roles are assigned to whole groups at plan time.** A group is a counterfactual group or a single episode.
* **Randomised domains are partitioned between the roles of a part,** so no domain is shared across roles.
* **OOD conditions come from separate regimes and regions** (`wind/regimes/*`, `domain region`) and are generated before training data on their own seed branches.
* **The validator re-checks everything after generation:** group, scenario, counterfactual-group, trajectory-seed and domain disjointness; wind category and domain region allowed per role; and that the realised logged wind statistics equal the planned ones and satisfy the regime limits.

## Physical domains (`randomization/`, decision D9)

Relative factors around nominal, driven by a scrambled Sobol (or LHS) design:

| Parameter | ID | OOD shell |
|---|---|---|
| mass | ±10 % | 10–20 % |
| inertia | J = J₀ · (m/m₀) · shape, shape ±5 % per axis | — |
| thrust constant k_F | ±10 % | 10–20 % |
| k_M/k_F ratio | ±5 % | — |
| motor time constant | ±20 % | 20–40 % |
| rotor / body drag scales | ±10 % | 10–20 % |

* **Feasibility:** T/W ≥ 2 and a valid inertia (positive, triangle inequality); otherwise the design point is rejected.
* **Held-out combinations:** corners such as `{mass: +0.5, force_scale: −0.5}` in design coordinates are excluded from ID sampling and are the only region of `ood_combination`.
* **Status:** domains are sampled and recorded now. Binding them to PhysX (masses, inertias, k_F, k_M, motor α) is Run 4, and until then the runner refuses non-nominal domains.
