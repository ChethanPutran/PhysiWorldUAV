# PhysiWorldUAV — Master README (project story to 2026-11-26)

M.Tech thesis, IISc. Title: *Physics-Prior Latent World Models for Reliable Long-Horizon UAV Prediction under Out-of-Distribution Wind Disturbances.*

This file tells the whole story: what the idea is, how it started, how it changed, what was built, what went wrong, what the numbers say, and what is left. The setup and run commands are in `README.md` and `CLAUDE.md`. The current position is in `HANDOFF.md`. Every claim here points to a file that holds the evidence. Where a number is a pilot-scale, single-seed result, it says so.

Contents: 1 Idea · 2 Timeline · 3 How the idea evolved · 4 What was built · 5 Results · 6 Problems and fixes · 7 Decisions · 8 Open questions · 9 Remaining work · 10 Where to find things · 11 Midterm deliverables

---

## 1. The idea in one page

**Problem.** A learned world model predicts the future by feeding its own predictions back in. Small one-step errors compound, so long-horizon prediction is unreliable, and it gets worse when conditions differ from training. A quadrotor is a good test case: its rigid-body dynamics are known, but wind, drag and actuator effects are not.

**Model.**

```
ds/dt = f_phys(s, u) + d_θ(h, s, u)
```

* `s` = physical state (position, velocity, attitude quaternion, body angular rate).
* `f_phys` = known rigid-body dynamics (thrust, gravity, kinematics), integrated with RK4.
* `d_θ` = learned residual on forces and torques. `h` = recurrent latent memory built from IMU history. The latent is context for what physics cannot explain, not a second copy of the state.

**Research question (RQ1).** Does adding the known physics as a prior make long-horizon prediction under OOD wind more reliable?
**Hypothesis H1:** the physics-prior model's error grows more slowly than a pure learned latent model's.
**H2 (conditional):** the benefit shrinks as the nominal physics model gets worse. A negative result is acceptable if the conditions where physics fails are identified.
**Primary OOD axis:** wind magnitude. Secondary: direction, temporal structure, gusts, physics (vehicle parameters), unseen trajectories.

**Gap.** Physics-informed learning, residual dynamics and latent world models are each studied, but there is little controlled comparison of *pure latent* against *physics-prior latent* under recursive prediction with controlled OOD wind. The contribution is that comparison and the benchmark behind it, not a new component.

**Reliable imagination horizon:** the longest horizon for which mean position error stays below a tolerance (`evaluation/reliable_horizon.py`).

---

## 2. Timeline (from git history, file dates and run reports)

| Date | What happened |
|---|---|
| 2026-09-15 / 09-21 | First commit; first-generation scaffold (`main.py`, generators, force models). |
| Oct 2026 | Setup in an Isaac Sim 4.1 + OmniDrones + Isaac Lab v1.0.0 docker container. Studied the **AirDreamer** paper (`.idea/airdreamer.md`): Hummingbird, Lee controller, 200 Hz physics / 20 Hz policy, domain randomisation. It served as the blueprint for the data pipeline. |
| 2026-11-03 | `sim/` files and force configs added; `.idea/existing_dataset.md` (survey of existing data sources) written. |
| 11-12 | **Run 1: audit.** Found the first-generation pipeline had never produced a usable dataset (18 episodes of length 1) and that wind never acted on the drone. |
| ~11-13 | **Run 2: implementation plan**, decisions D1–D10 (`docs/audit/run2_implementation_plan.md`). |
| 11-16 | **Run 3** (Hydra config and experiment architecture) and **Run 4** (physical models validated in Isaac: 17 Isaac tests, 131 CPU tests). |
| 11-17 | **Run 5:** pilot dataset v0.5.0, 147/147 episodes, validated. Approved aero truth. Baseline gate report. |
| 11-17 to 11-20 | World-model code B0–B4. Design discussions in `.idea/` (rollout training, residual channels, identifiability, data leakage, IMU, gaps, baselines). |
| 11-19 | "Models were added" commit. |
| 11-24 | B5/B6 implemented. IMU transport artefact found and fixed, giving pilot v0.6.0. 16 baseline runs trained and evaluated. |
| 11-25 | Audit found `best.pt` was not the best checkpoint (C9); trainer fixed. Sweep `fix_v1` (29 runs). Wind probe. 20 Hz tracking check. Run 6 wind plan decided. |
| 11-26 | **Pilot v7** (4,530 episodes, 20 Hz) generating on 9 workers. Auto-resume after reboot. Live report page. Legacy code removed. |
| 11-30 | Midterm report due. 12-03/04: presentations. |

---

## 3. How the idea evolved

**Stage 0: the proposal (`docs/thesis.tex`).** A physics-prior latent world model, baselines B1–B6, an OOD wind protocol, and the reliable-horizon metric. Vision, uncertainty and MPC were optional extensions only.

**Stage 1: where to get the data.** Using an existing dataset was weighed first (`.idea/existing_dataset.md`). The need for ground-truth state, wind, controlled OOD wind and counterfactuals led to **simulation with Isaac Sim + OmniDrones**, following AirDreamer.

**Stage 2: the first pipeline was not trustworthy.** The audit showed the data could not have been used even if it ran: wind and drag were silent no-ops, ω was mislabelled as body-frame, mass randomisation never reached PhysX, the controller ignored yaw, data was windowed at 200 Hz, and training wind was 2–16 m/s instead of [0, 4]. **The project changed from "train a model" to "build a physically validated data generator first".** That is why Runs 1–6 exist.

**Stage 3: the data became the main work.** Every approximation had to be justified, validated and used only inside its applicability envelope:
* Aero truth: rotor drag (linear in Ω), body drag with per-axis C_D, ground effect, rotational damping, a registry of envelopes, and episodes outside an envelope excluded.
* Wind: OU, Dryden (MIL-F-8785C), gusts, regimes with realised-statistics guarantees.
* Counterfactual siblings: Mode A (closed loop) and Mode B (fixed-action replay), so wind effects can be separated from controller reactions.

**Stage 4: identifiability and leakage.** The notes `.idea/identifiability.md` and `.idea/dataleakage` argued that under a good controller the state trajectory barely changes with wind, so the residual is undetermined; and that in closed loop `u` correlates with wind. The fixes were bounded excitation signals, open-loop segments, counterfactual pairs, several controllers, and a permutation test for action–wind correlation.

**Stage 5: model design decisions** (`.idea/choices.md`, `.idea/residual_channels`, `.idea/rollout_training`, `.idea/gaps`, `.idea/baselines`):
* Training protocol: context window with teacher forcing, then free-running rollout with a horizon curriculum.
* Residual on `v̇` and `ω̇` only (`acc_only`); the kinematics are exact.
* The physics state carries forward in imagination. The decoder only initialises it.
* IMU-only start state does not work (attitude error about 120°), so evaluation starts from the given true state.
* B5 and B6 each got several variants because the thesis left them undefined.
* Compared with Dreamer, the latent is memory attached to a state, not the state itself.

**Stage 6: first results complicated the story.** The physics prior helped in distribution but not on strong unseen wind. Careful audit then removed some false conclusions (the checkpoint bug, the missing constant-velocity baseline, the weak weight-decay test). The thesis question has become sharper: *when and why does a residual learned on bounded wind fail to extrapolate, and what in the data or model fixes it?*

**Stage 7: fix the data against what the diagnostics found.** The training wind never exceeded 3.7 m/s while OOD peaks were 4.3–8.2 m/s, and a probe found no readable wind information in the latent. This led to the **wide-wind, stratified Run 6 plan**, 20 Hz model steps, 10 s episodes, and about 470 vehicles.

---

## 4. What was built

**Data pipeline** (`src/physiworld_uav/`): plan → simulate → write → validate. A manifest row fully determines an episode.
* Hydra config groups (drone, physics, limits, controller, dataset, scenario ×18, excitation ×8, wind regimes, domain, counterfactual, forces, experiment).
* Seed tree with stable keys; provenance (config hash, git, versions, GPUs).
* Domain randomisation: Sobol/LHS, ID and OOD shells, held-out corners.
* Trajectory library with feasibility checks; controllers built from the true vehicle parameters (decision Q2).
* Excitation signals: step, doublet, multisine, chirp, PRBS, smooth random, open loop.
* Schema v2 HDF5, resampler (decimation, never filtering), dataset builder and validator, splits assigned per group.
* **Parallel generation** (`--shard I/N`, merge), per-episode checksums, crash-safe resume, auto-resume after reboot, live report page (`visualization/`), phone notifications.

**Physics validated inside Isaac** (`docs/models/validation_run4.json`): hover 9.810 m/s², free fall 0, 1 N force gives 1.3957 vs 1.3966 m/s² expected, motor lag error 6e-8, torque responses within about 0.5 %, wind 3 m/s gives 0.9614 vs 0.9618 m/s² expected, bitwise repeatability within a process.

**World-model code** (`models/`, `training/`, `losses/`, `evaluation/`): baselines B0–B6, curriculum training, geodesic attitude loss, resumable trainer, per-role evaluation, wind/no-wind gap, re-observation, residual diagnostics, shuffled-`u` test, latent probe.

| ID | Model | Purpose |
|---|---|---|
| B0 | hold last state | floor |
| B1 | physics only | what the nominal model alone does |
| B2 | pure neural latent | the thesis's "pure learned" side |
| B3 | physics + residual + latent memory | **proposed model** |
| B4 | physics + residual, no memory | isolates memory |
| B5a/b | learned field / exact kinematics + learned dynamics | isolates the physics split |
| B6a/b | oracle wrench (no learning / with learning), future or current interval | upper-bound reference |

**Tests:** 250 CPU tests and 17 Isaac tests.

---

## 5. Results so far

All numbers are position RMSE after 20 model steps (0.4 s at 20 ms steps), from `docs/results/sweep_fix_v1.md`, dataset pilot v0.6.0. **Limits:** 63 training episodes, 4–7 episodes per OOD role, 3 seeds for B3/B4/B5b, 1 seed for the rest.

| Model | test ID | OOD magnitude | OOD direction | OOD temporal | OOD gust |
|---|---|---|---|---|---|
| B0 hold | 0.376 | 0.420 | 0.424 | 0.359 | 0.169 |
| B1 physics only | 0.053 | 0.185 | 0.058 | 0.089 | 0.054 |
| B2 pure neural | 0.105 | 0.122 | 0.056 | 0.032 | 0.043 |
| **B3 proposed** | **0.036** | 0.166 | 0.039 | **0.026** | **0.022** |
| B4 no memory | 0.037 | 0.183 | 0.051 | 0.047 | 0.036 |
| Constant velocity | 0.072 | **0.050** | **0.027** | 0.032 | 0.036 |
| B6a oracle wrench | 0.033 | 0.018 | 0.034 | 0.074 | 0.047 |

**Pilot dataset quality** (v0.5.0): 147/147 episodes, validator 0 hard and 0 soft findings; Newton's-law check on ground truth p99 0.030 m/s²; realised wind statistics within regime limits; 0 NaN.

**What the numbers say (plainly):**
1. The physics prior **helps in distribution**: B3 is 0.036 m against B2's 0.105 m and B1's 0.053 m. It also helps on OOD temporal and gust wind.
2. **It does not help on OOD wind magnitude**, the primary axis. B3 (0.166 m) is about the same as physics-only B1 (0.185 m). B2 is better there (0.122 m), and the oracle shows the information exists (0.018 m).
3. **A constant-velocity extrapolation beats B3 on OOD magnitude and direction** (0.050 and 0.027 m). B5b's good OOD numbers are that baseline: it picks its almost untrained epoch-1 checkpoint.
4. B4 and B5b get worse as they train. Lower learning rate did nothing; the weight-decay test was too weak to count (C12).
5. B3 variants (no action in residual, zero-residual penalty, shared trunk) are within seed noise of B3.
6. The latent of B3, B2 and B5b carries **no linearly readable wind-force information** (probe R² ≤ 0; the torque control gives 0.8).
7. The IMU correction (v0.5 vs v0.6) shows no effect beyond seed noise.
8. 20 Hz controller check: stable on 12 scenarios with and without wind; position scenarios identical to 50 Hz; one attitude-mode case worse (n = 1).

**Not yet shown:** the B2-vs-B3 comparison on long horizons under OOD wind at proper scale, the reliable-horizon curves, multiple seeds for B2, and the effect of the wide-wind data.

---

## 6. Problems faced and how they were tackled

Full log with evidence: `docs/problems_log.md` (IDs A–E). Summary:

**Simulation and physics**
* *First pipeline invalid* (audit): rewrote the force path as one `WrenchInjector` call; fixed frames (ω_B), mass/inertia binding with read-back, yaw-aware controller, 20–50 Hz timing.
* *IMU never updated / gravity-only frames:* sensor created before `sim.reset()` and pipeline `gpu_dynamics_cpu_readback`.
* *IMU accelerometer was not the specific force* (p99 1.65 m/s², max 4.7). A sensor lever-arm explanation was tested and **refuted**; the residual fits −ω×v (R² 0.83). Corrected at generation (p99 5.04 → 0.173 on worst episodes) → v0.6.0.
* *Rotor-drag envelope* too narrow for OOD wind: widened μ ≤ 0.15 → 0.20 with the fidelity cost stated (`.idea/problems/2.md`).
* *Episode depended on earlier episodes in the same process (≈ 6e-6 m).* After several wrong hypotheses (B20), the cause was found: realised inertia computed from float32 world poses before reset. Fixed by resetting to a canonical pose first. IMU accelerometer remains reproducible only to about 2e-5 m/s² (accepted, documented).
* *Dryden profile silently produced no turbulence:* a unit test caught it before generation.
* *Run aborted at episode 144 and the traceback was swallowed* (`SimulationApp.close()` kills the process): failing episodes are now logged and the run continues; the index is checkpointed after every episode; `--resume` added.
* *Simulator segfault with `physics.device=cuda:N`:* choose GPUs only with `CUDA_VISIBLE_DEVICES`.

**Data validity**
* Recorded wrench is the *total* aero wrench, not wind only: decided to use the total for B6 and reconstruct the wind-attributable part.
* Open-loop action–wind correlation 0.39 (permutation p = 0.085): reported with the p-value.
* Too few independent groups per OOD role, calm-weighted wind, few Mode B episodes in test roles, correlated domain factors: addressed in the v7 plan (about 17 groups per OOD part, at least 100 domains, stratified wind).

**Model and training**
* **`best.pt` bug (C9):** the best-metric reset at each curriculum stage, so 8 of 16 runs used a worse checkpoint. I had also written that it was "not a checkpoint bug", which was false. Fixed with a regression test; all affected claims were corrected and the runs repeated in sweep `fix_v1`.
* Missing constant-velocity baseline (C11) added; it changed several conclusions.
* Weight-decay test too weak (C12): AdamW decay = lr × wd, i.e. 2e-6. A proper test still needs wd 0.1–1.
* Review of an external gap list (`.idea/v6/problems_identifid.md`, G1–G13): 3 became model flags, 4 were code fixes, others were outdated or deliberate.

**Process**
* Tests timing out on the shared 96-core host: `OMP_NUM_THREADS=4`.
* `pkill -f` killed its own shell (twice): kill by PID only.
* Docs and specs deleted from the working tree: specs moved to `.idea/specs`.
* A wrong claim attributing an episode difference to the GPU was written into the docs, then corrected (B13).
* Generation memory: about 6–16 GB per worker; worker count must stay the same on resume (duplicate episode ids otherwise).

---

## 7. Decisions taken (by the user)

* Aero truth coefficients: rotor drag 0.30 s⁻¹, C_D = (1.1, 1.1, 1.3), areas (0.0070, 0.0076, 0.0180) m², R = 0.10 m, ρ = 1.225; envelope μ ≤ 0.20.
* Controllers use the true vehicle mass and inertia (Q2). The physics step is 5 ms.
* Terminated (ground-contact) episodes are kept and flagged, not discarded.
* Headline runs use the `nominal` IMU-noise preset; bias is not modelled (limitation).
* B6 uses the total recorded wrench.
* Wind plan for Run 6: wide ranges, stratified, upper-band-weighted (ID peak ≤ 6.2 m/s; OOD mean > 6.2, peak ≤ 9 m/s).
* 20 Hz model step for v7: a 20-step horizon then covers 1 s instead of 0.4 s.

---

## 8. Open questions

* Why does B3's residual not extrapolate in wind: training wind range, 0.2 s history with latent reset per window (audit F3), a linear probe, or something else? Next tests: longer history, non-linear probe, wind-range augmentation, OOD-aware checkpoint selection.
* B4/B5b degradation: cause not established.
* Whether to change the open-loop action–wind rule (fixed 0.30 limit ignores sample size).
* Which of the future-work items in `docs/checklist.md` §5 enter the final data.
* Wind range, sizing and model experiments for the next stage need approval.

---

## 9. Remaining work (to the thesis)

1. Finish pilot v7 generation (4,530 episodes, 906k model steps, 473 vehicles; **29.6 % done at 12:59 on 2026-11-26**), then validate and run the quality report.
2. Re-run baselines on v7 with several seeds (B2, B3, B4, B5b, B1, B6) and the horizon curriculum to 50 steps.
3. Test the hypotheses for B3's OOD failure: wind-range augmentation, OOD-aware checkpoint selection, longer history, a stronger weight-decay test, a non-linear probe, a disturbance-supervision head.
4. Primary result: B2 vs B3 reliability curves and reliable imagination horizon under OOD wind, with confidence intervals over groups.
5. Physics-mismatch and training-data-size ablations (RQ2, RQ3).
6. Optional: uncertainty head (RQ4), ground effect and rotor faults, a second airframe.
7. Thesis writing; update `docs/thesis.tex` for the changes listed in `.idea/specs/model_specification.md` §10/§11 (synthetic-IMU φ, IMU-only encoder).

---

## 10. Where to find things

| Need | File |
|---|---|
| Current status and pending decisions | `HANDOFF.md` |
| Architecture, commands, gotchas | `CLAUDE.md`, `docs/guides/architecture.md` |
| Everything that went wrong | `docs/problems_log.md` |
| Run 1–5 reports | `docs/audit/run{1..5}_*.md`, `AUDIT_REPORT.md` |
| Plan and settled decisions D1–D10 | `docs/audit/run2_implementation_plan.md` |
| Baseline results | `docs/results/baseline_comparison_pilot_v05_v06.md`, `docs/results/sweep_fix_v1.md`, `docs/results/tracking_20hz_vs_50hz.md` |
| Physics and force accounting | `docs/models/*.md` |
| Assumptions and limitations | `docs/guides/assumptions_limitations.md` |
| Checklist with evidence | `docs/checklist.md`, `.idea/specs/e2e_done.md` |
| Design reasoning (why) | `.idea/` (`choices.md`, `identifiability.md`, `dataleakage`, `residual_channels`, `rollout_training`, `gaps`, `baselines`, `v6/`, `problems/`) |
| Thesis text | `docs/thesis.tex`, `docs/thesis.pdf` |
| Midterm | `report/midterm_report.tex` (+ PDF), `report/MIDTERM_GUIDE.md` |

---

## 11. Midterm deliverables

See `report/MIDTERM_GUIDE.md`: what the instruction asks for, a 25-minute slide plan, speaker notes, and likely questions. The written report is `report/midterm_report.tex`, built from the IEEEtran template already in `report/`.
