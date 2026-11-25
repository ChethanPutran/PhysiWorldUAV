# Glossary

Definitions of the terms used in the configs, the manifest, the validators and the reports. Numbers are from the Run 5 pilot (`physiworld_pilot_v0.5.0`: 147 episodes, 99 groups, 40 distinct domains) unless stated. Where a term is a config key, the key is in `code font`.

## The unit hierarchy

| Term | Definition | Where it lives |
|---|---|---|
| **Episode** | One simulated flight: one initial condition, one reference / open-loop programme, one wind realisation, one vehicle (domain), run until the planned duration or a termination (ground contact, tilt). Stored as one HDF5 group with ~36 arrays at the physics rate (200 Hz) plus metadata. `episode_id` = `<group_id>_e<i>`, e.g. `id_cl_g00002_e1`. | `data/writer.py`, manifest row |
| **Manifest row** | The complete plan of one episode (scenario, seeds, wind spec, domain factors, controller, forces, split role, counterfactual info). A row fully determines the episode, so it can be regenerated from the row alone. | `manifest.jsonl` |
| **Group** | One *experiment unit*: a scenario instance (a trajectory or excitation, a wind realisation and a domain) together with its **counterfactual siblings**. A group with no counterfactual has one episode; a counterfactual group has 2–3 (e.g. `id_cl_g00002`: a no-wind baseline `e0` and siblings `e1`, `e2` with the same vehicle and scenario but wind scaled by 0.5 and 1.0). `group_id` = `<part>_g<NNNNN>`. **A group is the unit of independence and of splitting**: all episodes of a group go to the same split role, and statistics over windows or episodes of one group are not independent samples. 147 episodes form 99 groups. | `group_id` in the manifest |
| **Domain** | One **vehicle**: a set of physical parameters drawn around the nominal Hummingbird. A domain is defined by relative *factors*: `mass`, `inertia_shape_x/y/z`, `force_scale`, `km_kf_ratio`, `motor_time_constant`, `rotor_drag_scale`, `body_drag_scale` (config `config/domain/randomized.yaml`). In-distribution (ID) factors lie within ±`id_halfwidth` of nominal (mass ±10 %, inertia shape ±5 %, force scale ±10 %, km/kf ±5 %, motor time constant ±20 %, drag scales ±10 %); out-of-distribution (OOD) domains lie in a shell outside it (e.g. mass 10–20 %), or are a held-out corner (heavy vehicle with weak rotors). `domain_id` = e.g. `id_cl_dom_00002`. **Several groups share one domain** (the pilot has 40 distinct domains for 99 groups; `id_cl_dom_00000` is used by 4 groups); a domain is never shared across split roles (the validator fails the dataset otherwise). | `domain_id`, `factors` in the manifest |
| **Part** | A block of the experiment plan: a number of groups (`n_groups`) with one data type, one wind regime, one domain policy and `n_domains` domains, split into roles by fractions (`roles: {train: 0.7, val_id: 0.15, test_id: 0.15}`). Pilot parts: `id_cl`, `id_ol`, `ood_mag`, `ood_dir`, `ood_tmp`, `ood_gst`, `ood_phy`, `ood_cmb`, `ood_trj`, `phys_val`. | `config/experiment/*.yaml` |
| **Window** | One *model sample*: a context of `history` steps (default 10 = 0.2 s) and `horizon` future steps at the model rate (50 Hz). Windows of one episode overlap heavily and are **not** independent; evaluation therefore resamples whole episodes. | `data/dataset.py` |

**Group vs domain in one line:** a *domain* is *which vehicle* is flying (its parameters); a *group* is *which experiment* is flown (scenario, wind, vehicle, and the sibling episodes that differ in one controlled variable). The number of independent *scenarios/wind conditions* in a role is its number of groups; the number of distinct *vehicles* is its number of domains. The correlation check between vehicle parameters runs over distinct domains, because the factors are constant within a domain.

## Counterfactual groups

| Term | Definition |
|---|---|
| **Counterfactual group** | A group whose siblings differ in exactly one variable (`wind_magnitude`: wind scaled by a factor, the baseline has no wind; or `mass`). Used to separate the effect of the variable from everything else. |
| **Mode A** (`closed_loop`) | Siblings are flown by the same controller reacting to their own wind, so the commands `u` differ between siblings. |
| **Mode B** (`fixed_action`) | Siblings replay the **same rotor commands** as the baseline (`controller.mode = replay`), so `u` is identical across the group and cannot explain a difference in outcome. In the pilot, Mode B exists only in the `train` and `val_id` roles. |
| **Sibling index / baseline** | `sibling_index` 0 is the baseline (`is_baseline`). Mode B siblings need their baseline generated first (and on the same worker). |

## Splits

| Term | Definition |
|---|---|
| **Split role** | The role of a group: `train`, `val_id`, `test_id` (in-distribution); `val_ood_*` / `test_ood_*` with `magnitude`, `direction`, `temporal`, `gust`, `physics` (OOD families) and `test_ood_trajectory`; `physics_validation` (mass-sibling pairs for the model sanity check G10). Roles are assigned per group; the validator fails the dataset if a group, a domain or a scenario instance appears in two roles. |
| **ID / OOD** | In-distribution / out-of-distribution. OOD is defined by a condition never present in training: wind magnitude above the ID range, wind direction (vertical component), temporal structure (faster / different), gust structure, vehicle parameters in the OOD shell or corner, or the trajectory family. |
| **Independent groups per role** | The number of groups in a role; the validator warns below 5 (too few to report per-role results). |
| **Model-ready episode** | A completed episode that passed the filters of `data/dataset_builder.py`: status `completed`, inside the force-model applicability envelope, not excluded by the open-loop `body_wrench` rule. Ground-contact terminations are excluded. |

## Wind

| Term | Definition |
|---|---|
| **Wind regime** | A distribution of wind realisations (`config/wind/regimes/<name>.yaml`): `id`, `none`, `ood_magnitude`, `ood_direction`, `ood_temporal`, `ood_gust`. It specifies the sampled mean speed, direction band, profile types (steady, cosine gust, step, sinusoid, Ornstein–Uhlenbeck, random gusts) and the **realised-statistics limits** below. |
| **Realised-statistics limits** | `limits` of a regime: `peak_speed_max`, `vertical_max`, `mean_horizontal`. They are checked on the *realised* wind series of each episode, so an episode cannot silently leave its regime. |
| **Peak / mean wind speed** | Peak = maximum of ‖w(t)‖ over the episode; mean horizontal = ‖mean(w_xy)‖. The "ID range 0–4 m/s" is a limit on the **peak**, while the sampled mean is lower (0–3 m/s). |
| **Wind magnitude / airspeed** | Wind is the air velocity in the world frame. The **airspeed** the vehicle experiences is the vehicle's velocity relative to the air (v − w), which is what the aerodynamic models and the envelope use; flying into the wind adds to it. |

## Aerodynamics and the validity envelope

| Term | Definition |
|---|---|
| **Aero truth** (`config/forces/truth_default.yaml`) | The approved aerodynamic model injected in the simulator: rotor drag (linear in rotor speed, d = 0.30 s⁻¹), body drag with per-axis C_D = (1.1, 1.1, 1.3) and projected areas (0.0070, 0.0076, 0.0180) m². Gravity, thrust and rotor reaction torque come from PhysX / OmniDrones, not from here. |
| **Total injected aero wrench** | The recorded `external_force_W` / `external_torque_B`: the sum of all injected components. It is **not wind only**: drag also acts with zero wind because it depends on the vehicle's own velocity. |
| **Wind-attributable part** | Total wrench minus the same wrench evaluated with zero wind (reconstructed in `data/disturbance.py`). |
| **Advance ratio μ** | In-plane relative air speed at the rotor divided by the rotor tip speed: μ = ‖v_rel,⊥‖ / (Ω R), with R = 0.10 m. |
| **Applicability envelope** | The conditions under which an aero model is validated: rotor drag μ ≤ 0.20 (fidelity reduced between 0.15 and 0.20) and rotor speed ≥ 20 % of maximum; body drag airspeed ≤ 12 m/s. |
| **Violation fraction / `applicability_ok`** | Fraction of physics steps of an episode outside a model's envelope. An episode is `applicability_ok` when every component's fraction is ≤ 0.05 (`max_violation_fraction`); other episodes are excluded from model-ready data. |

## Control, data and sensing

| Term | Definition |
|---|---|
| **Closed loop / open loop** | Closed loop: a controller tracks a reference (data type `closed_loop`, parts `id_cl`…). Open loop: scripted rotor or rate commands with no feedback (`id_ol`, excitation such as rotor doublets). |
| **Model step / physics step** | Physics 5 ms (200 Hz); controller and model step 20 ms (`control_decimation` = 4, 50 Hz). The action `u` is the interval-mean rotor speed. |
| **State** | Model state `p, v, q, ω_B` (13): world position and velocity, unit quaternion (wxyz, body→world), body angular velocity. |
| **Specific force (IMU)** | The accelerometer measures `R(q)ᵀ(a − g)` in the body frame (≈ +g on the body z axis in hover). Datasets from v0.6.0 on remove Isaac's `−ω×v` transport term at generation (`physics.imu_transport_correction`); the Run 5 pilot v0.5.0 holds the raw sensor. |
| **Dataset version** | `dataset_version` identifies the generation semantics; a dataset is never overwritten (v0.5.0 raw IMU, v0.6.0 corrected IMU). |
| **Shard / index / merge** | Episodes are stored in HDF5 shards; `index.json` lists them. Parallel workers write their own shards and `index.workerII.json`; `generate merge` validates the whole dataset and writes `index.json`. |
| **Checksum** | `sha256` over the arrays of an episode, stored with it and verified on resume, merge and validate. |
