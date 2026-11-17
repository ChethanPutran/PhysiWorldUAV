# Configuration guide

Hydra 1.3 + OmegaConf. Root: `config/defaults.yaml`.

```
config/
  defaults.yaml            defaults list: drone, physics, limits, controller, dataset, experiment
  drone/                   physical parameters (single source of truth; checked against OmniDrones)
  physics/                 dt, control decimation, device, requested PhysX settings
  limits/                  plan-time feasibility limits + run-time termination envelope
  controller/              controller-parameter policy ("true" = domain parameters, Q2), gains
  dataset/                 storage, model-ready builder (H, P, action key)
  experiment/              experiments = list of parts (split role(s), counts, templates)
  scenario/                reusable scenario templates (reference family, yaw, duration, data type)
  initial_conditions/      initial-state templates
  excitation/              perturbation / open-loop signal templates
  wind/regimes/            ID and OOD wind regimes (D1, D2)
  domain/                  physical-domain templates (nominal, randomized)
  counterfactual/          counterfactual group templates
  forces/                  force-model sets (`none`; `truth_default` pending Run 4)
```

## Two levels of composition

1. **Hydra groups** (`drone`, `physics`, `limits`, `controller`, `dataset`, `experiment`) are selected in the defaults list or on the command line:
   `physiworld-generate run experiment=pilot physics.control_decimation=2 dataset.builder.horizon=40`
2. **Named templates** inside an experiment (`scenario`, `wind_regime`, `domain`, `excitation`, `counterfactual`, `forces`, `initial_conditions`) are loaded by name, and per-use overrides are merged on top:

```yaml
parts:
  - name: id_cl
    roles: {train: 0.7, val_id: 0.15, test_id: 0.15}   # groups (and domains) split by role
    n_groups: 40
    domain: randomized          # template
    domain_overrides: {region: id}
    n_domains: 12
    wind_regime: id
    forces: truth_default
    counterfactual: {template: wind_magnitude, fraction: 0.3}
    mix:
      - {scenario: translation, weight: 0.10, excitation: ref_smooth, experiment_group: B,
         overrides: {params: {speed_mps: [0.5, 1.0]}}}       # scenario-specific override
```

## Templates may contain ranges

Templates hold *distributions* (`[lo, hi]` uniform, `{choice: [...]}`); the planner draws concrete values with the seed tree and stores them in the manifest. Configs never contain sampled values.

## Validation

* `DroneParams.from_dict` validates mass, inertia (positive, triangle inequality) and T/W, and the planner cross-checks the drone yaml against OmniDrones'.
* `DomainSamplerConfig`, `WindRegime`, `FeasibilityLimits`, `ParameterSpec`, `OpenLoopProgram` and `BuilderConfig` validate their sections.
* The planner raises `PlanError` for impossible plans: a role with zero groups, too few domains to partition between roles, or no feasible reference.
