# Model Specification — physics-prior latent world model and baselines

Status: DRAFT for approval of the open decisions (§9). **Implemented and unit-tested:** `f_phys`, B0, B1, B2, B3, B4 (`src/physiworld_uav/models/`), the loss, evaluator, trainer with checkpoint/resume/curriculum, IMU noise. **Not implemented:** B5, B6, the §3.4 variants (B3a–c), the wind-estimator probe. This version follows the thesis text (`thesis.tex`, "Progress Achieved: Proposed Method" and "Experimental Baselines") equation by equation. Where the thesis leaves a choice open, the choice is listed in §9 with a recommendation. Where the thesis cannot be implemented as written, §3.1 explains why and offers options.

Data definitions: `data_specification.md`. Training and evaluation: `training_pipeline.md`.

Decision already made by you: the attitude loss uses the **unit quaternion** with a geodesic distance, not rotation matrices (§5).

## 1. Notation and mapping to the dataset

| Thesis | Meaning | Dataset (model-ready sample) |
|---|---|---|
| `s_t = [p, v, R, ω]` | Physical state | `state` (13): p_W · v_W · q_wxyz · ω_B. The attitude is stored as a Hamilton quaternion body→world, `R = R(q)` |
| `u_t` | Control input | `rotor_speed_mean` (4, rad/s), the mean rotor speed over [t, t+1) (decision D3) |
| `a_t^IMU, ω_t^IMU` | Proprioceptive observation | `obs` (6): specific force (body) · gyro (body); clean in the data, noise added at load time |
| `h_t` | Latent state | Computed by the model |
| `Δt` | Model step | 0.02 s |
| Wind, `f_ext`, `τ_ext` | Disturbances | Auxiliary data, not model inputs (only B6 may use them) |

Sample keys: `state_history (H,13)`, `obs_history (H,6)`, `action_history (H,4)` = u_{t−H…t−1}, `future_actions (P,4)` = u_{t…t+P−1}, `future_states (P,13)`.

## 2. Current code state

| Component | File | State |
|---|---|---|
| Model-ready episodes and dataset | `data/dataset_builder.py`, `data/dataset.py` | Implemented and tested (`tests/cpu/test_data_layer.py`) |
| World model, encoder, latent, residual | `models/*.py` | **Stale and structurally different from the thesis**: 12-dim state, encoder takes state and action (no IMU), no decoder, no state→latent coupling, Euler-style step |
| Losses | `losses/losses.py` | **Stale**: rotation term is an MSE on old indices 6:9 |
| Baselines | `baselines/*.py` | **Stale** (12-dim, no history, no latent) |
| Metrics | `evaluation/metrics.py`, `reliable_horizon.py` | Position error and first-failure horizon usable; attitude error missing |
| Trainer | `training/train.py` | Broken (imports a deleted module); no checkpoint, resume or tracking |
| Nominal physics `f_phys` | – | Does not exist for the 13-dim state |

All model code is to be rewritten against the new layout rather than patched.

## 3. The proposed model (B3), as in the thesis

### 3.1 Components

| # | Thesis equation | Specification |
|---|---|---|
| 1 | `e_t = E_θ(a_t^IMU, ω_t^IMU, u_t)` | Observation encoder: MLP on the 6 IMU channels and the 4 actions → e_t ∈ R^64. **No physical state is an input.** Inputs z-scored |
| 2 | `h_t = F_θ(h_{t−1}, e_t, u_{t−1})` | Recurrent latent update, a GRU cell (hidden 128) with input `[e_t, u_{t−1}]` |
| 3 | `ŝ_t = D_θ(h_t)` | Decoder MLP (128→256→13): p, v, q, ω. The quaternion output is normalised. Supervised against simulator states |
| 4 | `ṡ = f_phys(ŝ,u) + d_θ(h,ŝ,u)` | `f_phys`: nominal rigid body (§3.2). `d_θ`: residual MLP on `[h, ŝ, u]` (§3.3) |
| 5 | `s̃_{t+1} = RK4(ŝ_t, f_phys + d_θ, Δt)` | The **sum** is integrated jointly with RK4 (4 stages, h and u held over the step); the quaternion is renormalised after the step |
| 6 | `ĥ_{t+1} = F_θ(h_t, φ(s̃_{t+1}), u_t)` | The same GRU cell as in 2, with the state embedding `φ(s̃)` (MLP 13→64) in the place of `e`, so the latent is updated by the predicted state |
| 7 | Open-loop imagination | Observation phase over the history window (equations 1–3), then recursion `h_t → s̃_{t+1} → h_{t+1} → …` using **only** `future_actions`; no observations, wind or true states are used |

Alignment of indices: in the observation phase the encoder uses (o_j, u_j) at each history step j; for the last history step t, u_t is `future_actions[0]`, and `u_{j−1}` for the recurrence comes from `action_history`. The command u_t is the command the controller issues at time t, held over [t, t+1), which the model is allowed to know (it is a model input in imagination as well).

**Open observability problem (needs your decision, §9-D1).** The thesis restricts the observation to IMU and control. An IMU measures specific force and angular rate; from IMU and rotor commands alone the **absolute position and the yaw are not observable**, and velocity is observable only through the (weak) effect of drag. A decoder `ŝ_t = D(h_t)` therefore cannot recover absolute position, and the rollout's start state will be uncertain. Options:

| Option | Rollout start state | Comment |
|---|---|---|
| **(a) thesis-exact** | `ŝ_t = D(h_t)`; position and yaw expressed **relative to the start of the history window** (p − p_{t−H+1}, yaw − yaw_{t−H+1}); the target is changed to match | Honest to the thesis; initial error large for v and attitude |
| **(b) given initial state** (recommended as the primary evaluation) | `ŝ_t = s_t` (true state at the window end); the latent `h_t` is still built from IMU only; the decoder is still trained and reported as a state-estimation diagnostic | Isolates dynamics prediction from state estimation, which is the thesis's stated focus ("dynamics learning rather than visual representation learning") |
| (c) both | Run (a) and (b) as two evaluation settings with the same trained weights | Recommended: costs nothing at inference |

All of (a)–(c) use the same model code; a config switch `initial_state: decoder | given` selects the start.

### 3.2 f_phys (nominal quadrotor, differentiable, 13-dim state)

State derivative: `ṗ = v`; `v̇ = g + R(q)·(ΣT_i/m)·ẑ_B`; `q̇ = ½ q ⊗ [0, ω_B]`; `J ω̇ = τ − ω × Jω`, with `T_i = k_F·Ω_i²` and the torque from thrust differences (arm geometry) and the yaw reaction `k_M·Ω_i²` with alternating signs. Nominal parameters from `config/drone/`. It contains **no wind, no drag, no ground effect and no motor lag** (the action is already the interval-mean rotor speed), so these are left to `d_θ` as in the thesis. Unit tests: hover trim (zero acceleration at the trim rotor speed), free fall, pure yaw rotation, quaternion norm after RK4, and agreement with the logged no-wind `physics_validation` episodes within a stated tolerance.

### 3.3 Residual `d_θ`

* Input `[h_t, ŝ_t, u_t]`; output in the **tangent space** of the state, 12 values: (ṗ-correction 3, v̇ 3, attitude-rate 3, ω̇ 3). The quaternion derivative is `q̇ = ½ q ⊗ [0, ω_B + δ]` with the 3-vector attitude-rate correction δ, so the quaternion stays on the manifold.
* 2 hidden layers of 256, SiLU; last layer initialised near zero (training starts from pure physics).
* Config `residual_channels`: `all` (default, thesis-exact: d_θ acts on the whole ṡ) or `acc_only` (only v̇ and ω̇; ṗ and attitude-rate corrections are zero, since kinematics are exact). `acc_only` is an ablation, §9-D2.
* No loss ties `d_θ` to the wind. The decomposition `d_wind + d_aero + d_param + d_other` is an interpretation used in analysis: the evaluation reports `d_θ` by wind regime and by domain factor (§training_pipeline) but never as a training target.

### 3.4 Architectural-bias options against the action shortcut (from `docs/datasets/problems/identifiability.md`)

The identifiability note argues that under correlated action/wind data the residual can fit `d_θ = φ(u)` and ignore the IMU, and that the data fix (free flight, steps, counterfactual pairs) should be amplified by three architectural measures. They are **switchable options on the proposed model (B3)**; the thesis-exact B3 of §3.1–3.3 stays the reference, and each option is run as an ablation (§4, B3a–B3c). None of them uses wind as a target.

| Option | Specification | Config flag |
|---|---|---|
| **(a) IMU-only residual path** ("forbids the action shortcut") | The latent that feeds `d_θ` has **no action input anywhere**: `e_t = E_IMU(a_t, ω_t)` (the IMU encoder of equation 1 without `u`), the recurrence is `h_t = F(h_{t−1}, e_t)`, the coupling is `ĥ_{t+1} = F(h_t, φ(s̃_{t+1}))`, and the residual is `d_θ(h_t, ŝ_t)` with **no `u` input**. The action still enters the model through `f_phys` and through the predicted state, which is the physically correct route. In imagination there is no IMU, so the latent evolves from the predicted state only (equation 7 is unchanged); the bias acts on how the latent is built during observation and on the residual's inputs | `residual_path: imu_only` |
| **(b) Soft zero-residual penalty** `λ₀‖d_θ(0)‖²` | A soft term in the loss (not a hard constraint, so the model is not brittle to latent mean shifts): `L_0 = mean_k ‖d_θ(h = 0, ŝ_k, u_k)‖²` evaluated on the rollout's states, i.e. the residual must vanish when the latent carries no information. PROPOSED `λ₀ = 1e-3`, in the total loss `L + λ₀ L_0` | `lambda_zero: 1e-3` |
| **(c) Shared decoder path** | The residual head and the state decoder `D` share a trunk: `z = T(h)`, `ŝ = D_head(z)`, `d_θ = R_head(z, ŝ, u)`, so the residual cannot ignore the representation that the decoder is trained to keep state-informative. The note itself rates this the weakest option alone | `shared_trunk: true` |

Notes from the note's own caveats: (a) is the only option that removes the shortcut, but physically a residual that cannot see the rotor speeds cannot model effects that depend on `Ω` directly (for example rotor drag); those must then reach it through the state and the IMU, so (a) may cost accuracy and must be judged by the OOD results, not by training loss. (b) and (c) do not remove the shortcut by themselves and are meaningful only together with (a) or with the data fix.

**Diagnostic wind estimator (evaluation only, never a loss).** After training, fit a small probe `F̂_w = h_φ(h_t)` on the **frozen** latent (train split) and report its error on `val_id` and the OOD roles. If the probe predicts the logged wind well, the latent carries wind information; if the model is accurate but the probe is poor, wind was handled through another pathway and the OOD claim is weakened. This uses wind ground truth only for evaluation, as allowed by `data_specification.md` §1.4 (Part B).

## 4. Baselines B1–B6 (thesis numbering)

All share the data, the loss (§5), the interface and the evaluation. Parameter counts are matched (below).

| ID | Thesis description | Definition here | Latent? | Initial state options |
|---|---|---|---|---|
| **B1** | Physics-only | `ṡ = f_phys(ŝ,u)`, RK4, nominal parameters, no wind. Nothing is learned | no | `given` (no encoder exists); (a) is not defined |
| **B2** | Pure neural latent dynamics | Same encoder E, latent F, decoder D and coupling φ as B3, but the transition is a learned map `ŝ_{t+1} = F_pure(ŝ_t, u_t, h_t)` (MLP 128+13+4→256→256→12, output = state increment: Δp, Δv, rotation-vector Δθ composed with q, Δω). **No `f_phys`**. The hidden width of F_pure is set so that the total parameter count is within ±10 % of B3 (thesis: "approximately comparable capacity") | yes | (a), (b) |
| **B3** | Physics + residual + latent memory | The model of §3 | yes | (a), (b) |
| **B4** | Physics + residual without latent memory | `ṡ = f_phys + d_θ(s_t, u_t)`; the residual MLP is sized to match B3's parameter count (and is a bit wider to compensate for no encoder/decoder). Without a latent, no state is decoded | no | `given` only |
| **B5** | Residual-only | `ṡ = d_θ(h_t, ŝ_t, u_t)` with `f_phys` removed; the same encoder, latent, decoder and coupling as B3, integrated with RK4 | yes | (a), (b) |
| **B6** | Physics + oracle disturbance | `ṡ = f_phys(ŝ,u) + d_oracle(t)`, where `d_oracle` is the exact disturbance from the logged auxiliary data: `[0, F_ext/m_nom, 0, J_nom⁻¹ τ_ext]` from `ground_truth/wrench/external_force_W` and `external_torque_B` at the model step. Wind and wrench are used **only** here | no | `given` |

Notes:
* B6 supplies the exact aerodynamic force. It is an upper bound for the wind/aerodynamic part only: it still uses the **nominal** mass and inertia, so the parameter mismatch (±10 % ID, 10–20 % OOD physics) remains uncorrected. The thesis's `d_param` is therefore not removed by B6. This is the correct reading of "oracle disturbance" and should be stated in the results.
* Initial-state option (a) needs an encoder; B1, B4 and B6 have none, so their start state is `given`. Fair comparisons between models with and without an encoder are made under option (b).
* B2's pure-neural step also receives the latent through the same coupling, so memory is not what separates B2 from B3: the physical prior is.

**B3 variants (ablations of §3.4, outside the thesis table):** B3a = B3 with the IMU-only residual path, B3b = B3 + zero-residual penalty, B3c = B3 with the shared trunk, B3a+b = (a) with (b). Parameter counts stay matched to B3 (the IMU-only path removes the action inputs, so widths are adjusted).

Added by me, outside the thesis table: **B0 (extra)** = hold the last state (zero-order). It costs nothing and gives the lowest bar (§7 gate G3). Not part of the thesis comparison.

## 5. Loss (thesis form, quaternion attitude)

```
L_state = λ_p L_p + λ_v L_v + λ_R L_R + λ_ω L_ω
L_roll  = Σ_{k=1..H} γ^(k−1) · L_state(ŝ_{t+k}, s_{t+k})
L       = L_state + λ_roll · L_roll          (+ optional L_smooth, ablation only)
```

| Term | Definition |
|---|---|
| `L_p` | `‖p̂ − p‖²` |
| `L_v` | `‖v̂ − v‖²` |
| `L_R` | `d_q(q̂, q)²`, with `d_q(q̂, q) = 2·atan2(‖Im(q̂ ⊗ q*)‖, |Re(q̂ ⊗ q*)|)` (radians; sign-invariant, so q and −q are the same rotation; numerically stable at 0, unlike `arccos|⟨q̂,q⟩|`). Equals the geodesic angle on SO(3) of the corresponding rotation matrices |
| `L_ω` | `‖ω̂ − ω‖²` (body frame) |
| `L_state` supervision | In the observation phase, `ŝ_j = D(h_j)` against `s_j` for every history step j (this trains the decoder; for B1, B4, B6 there is no decoder and the term is absent). In the rollout, `L_state(ŝ_{t+k}, s_{t+k})` is inside `L_roll` |
| `L_smooth` | `Σ ‖d_{t+1} − d_t‖²`, **ablation only**. The thesis cautions that rapid disturbances legitimately produce rapidly changing residuals, so it is off by default |

The thesis gives no `λ` values. **PROPOSED:** set `λ_p, λ_v, λ_R, λ_ω` to the inverse of each term's mean value of a hold-last-state baseline on the training set, so that no term dominates (a way to choose λ, not a change to the loss), γ = 0.99, `λ_roll` = 1; revisit with evidence. The thesis sums `L_roll` without normalising by H, so its scale grows with the horizon: when the curriculum raises H, `λ_roll` is rescaled by `1/Σγ^(k−1)` to keep the gradient scale comparable (§9-D5). No residual-magnitude penalty is used (it is not in the thesis). The zero-residual term `λ₀ L_0` of §3.4(b) is added **only** in the variants that enable it.

## 6. Training stages (thesis §"Training Stages")

| Stage | Thesis | Action |
|---|---|---|
| 1 | Physics-only | Evaluate B1 (nothing trained) |
| 2 | Pure neural baseline | Train B2 |
| 3 | Physics + residual | Train B3 |
| 4 | Physics + residual without latent memory | Train B4 |
| 5 | Long-horizon training | Curriculum on the rollout horizon `H_train = 1 → 5 → 10 → 20 → 50`, applied to every trained model (B2–B5). The final horizon is chosen from cost and stability; stage advances when `val_id` loss has plateaued for N epochs (PROPOSED N = 3) or after a fixed epoch budget |

B5 and B6: B5 is trained like B3; B6 has nothing to train. Dataset prerequisite for stage 5: the builder must be run with `horizon: 50` (currently 20; the pilot episodes allow it, since every completed episode has at least 101 model steps ≥ 10 + 50, but the window count drops). Evaluation rollouts up to 100 steps need `horizon: 100` with episodes of at least 110 steps; windows that do not fit are skipped, never padded.

## 7. Data loading

Implemented: `load_model_episodes` → `ModelReadyDataset` (lazy windows) → `AuxiliaryView` (evaluation and B6 only). To add:

| Item | Specification |
|---|---|
| Split selection | By `split_role`. Training uses `train` only |
| Normalisation | Mean/std from `train` only, saved in the checkpoint. IMU, v, ω and rotor speed are z-scored; the quaternion is left unscaled; positions are used relative to the window's reference point (translation invariance), in line with option (a)/(b) of §3.1 |
| IMU noise | Load-time transform on `obs_history` (not implemented; proposed Gaussian white noise plus bias random walk, seeded) |
| Batching | Stride-1 windows, shuffled, batch 256, `drop_last`, multiple workers |
| Tests | Shapes and dtypes of all keys; no NaN; no auxiliary key in a sample (enforced); consecutive actions at the history/future boundary; `future_states` equals the raw episode at the same index; normalise→denormalise round trip; loader throughput |

## 8. Gate experiments on the pilot

Numeric thresholds are PROPOSED starting points to be tuned after the first results. The pilot has 63 training trajectories, so the gates test learnability and ordering, not final accuracy.

| # | Experiment | Pass criterion |
|---|---|---|
| G1 | **Tiny overfit**: B2 (then B3) on 8 training episodes, no regularisation | Training loss falls ≥ 100× and 20-step position RMSE < 2 cm. If not, debug data, targets or loss first |
| G2 | **Curves**: train/val loss on the full pilot | Both decrease; the `val_id` gap is reported; no divergence |
| G3 | **Beats trivial** | B2 and B3 < B0 on `val_id` at 1, 5, 10, 20 steps |
| G4 | **Physics value** | B3 < B2 on `val_id` and on OOD roles (thesis claim); B3 < B1 at 20 steps (residual earns its keep) |
| G5 | **Memory value** | B3 vs B4 (does the latent help) on `val_id` and OOD roles |
| G6 | **Residual-only** | B3 vs B5 (does the physics prior help beyond learning the whole dynamics) |
| G7 | **Wind learnability** | B6 < B3 on `val_id` and `test_ood_magnitude`. If the oracle does not help, the data does not make the disturbance learnable; revisit the excitation (`docs/datasets/problems/identifiability.md`) |
| G8 | **Per-role evaluation** | Report all six models on `test_id`, five OOD families, trajectory OOD; the pilot is too small to claim a result, the gate is only that it runs and is sensible |
| G9 | **Counterfactual consistency** | On Mode B pairs (same actions, different wind) the predicted difference has the right sign and order of magnitude |
| G10 | **Label sanity** | B1 on `physics_validation` Case A (no wind, nominal): small 20-step error; Case C (mass +5 %): larger error in the expected direction |
| G11 | **State-estimation diagnostic** | Decoder error for p, v, q, ω at the window end under option (a) (how much the IMU-only latent can recover; expect poor position and yaw, §3.1) |
| G12 | **Action-shortcut test (identifiability)** | Compare B3 with B3a/b/c on `test_ood_magnitude` and on a **decorrelation stress set**: re-evaluate with the actions of a trajectory but the wind of another group (Mode B pairs give this naturally). A model that relies on `u → wind` degrades on the stress set; a model that uses the IMU does not. Also report the wind-estimator probe (§3.4) |

## 9. Decisions needed from you

| # | Decision | Recommendation |
|---|---|---|
| D1 | Initial state for the rollout (observability): (a) decoder, (b) given, (c) both | **(c) both**, with (b) as the primary result |
| D2 | `d_θ` channels: `all` (thesis-exact) vs `acc_only` | `all` as the main model; `acc_only` as an ablation |
| D3 | The thesis's B2 transition `ŝ_{t+1} = F_pure(ŝ,u,h)`: predict an increment with a rotation-vector update (keeps `|q| = 1`) vs the absolute next state | Increment (implementation detail; does not change the model's inputs) |
| D4 | `f_phys` scope: rigid body and thrust from rotor speed only (leaves drag, wind and motor effects to `d_θ`) vs adding nominal linear drag | Rigid body only |
| D5 | Loss weights: set from the hold-last-state baseline (proposed) and `λ_roll` rescaled with the curriculum; confirm γ = 0.99 | As proposed |
| D6 | Checkpoint selection on `val_id` only vs also `val_ood_*` | `val_id` only, to keep the OOD claim clean |
| D7 | Latent and embedding sizes (latent 128, e_t 64, φ 64, hidden 256) | As proposed; the parameter counts of B2–B5 are matched to B3 |
| D8 | Include B0 (hold-last-state) as an extra reference outside the thesis table | Yes |
| D10 | Which B3 is the **main** model: thesis-exact (residual sees `u`) with the §3.4 options as ablations (recommended, keeps the thesis form and measures the shortcut), or the IMU-only residual path (§3.4a) as the main model | Thesis-exact main; B3a, B3b, B3c as ablations; decide the main model from G12 |
| D11 | `λ₀` for the zero-residual penalty and the meaning of "0" (zero latent with real state and action, as proposed) | `λ₀ = 1e-3`, zero latent |
| D9 | First implementation target | B1 and B2 with the gates G1–G3 and G10, then B3–B6 (B3–B6 are not meaningful if B2 cannot learn) |

## 10. Differences from the thesis text (all of them)

1. B0 is an addition, not in the thesis.
2. The thesis does not define the initial state of the rollout from IMU-only observations; §3.1 documents the observability issue and the config switch.
3. The thesis does not specify `f_phys` in detail; §3.2 defines it.
4. The thesis gives no network sizes or `λ` values; §3.1, §5 propose them.
5. Quaternion rather than matrix for the attitude loss, at your request; the distance is the same geodesic angle.
6. `d_θ`'s attitude component is a tangent-space (angular-rate) correction because the state stores a quaternion.
7. `L_roll` weights: thesis unnormalised sum; the curriculum rescales `λ_roll`.

8. §3.4 adds three optional architectural-bias variants and an evaluation-only wind probe from `docs/datasets/problems/identifiability.md`; none is in the thesis text. The IMU-only residual path (§3.4a) departs from thesis equations 1–2 (no `u` in the encoder and latent) and is therefore an ablation, not the main model, unless you decide otherwise (D10).

Removed from my earlier draft because they are not in the thesis: state history as an encoder input, a residual magnitude penalty, increment prediction for the physics models (the thesis integrates with RK4), and the 6-dim acceleration-only residual as the default.
