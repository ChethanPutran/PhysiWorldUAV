## **Pre-generation checklist**

Status as of 2026-11-24. An item is marked done only when there is evidence (a file, a test, a measured result). "Partly" items stay open. Items I added beyond the original 18 are marked **(added)**. Per-item evidence for the e2e requirements is in `.idea/specs/e2e_done.md`; problems met along the way are in `docs/problems_log.md`.

**Original 18 items: 18 / 18 done. Including my added items: 52 / 75 done.**

**1. Problem specification**

- [x] Input and output dimensions defined — `specs/data_specification.md` A.2
- [x] Units and coordinate frames defined — A.2, `guides/dataset_schema.md`
- [x] Sampling frequency and temporal alignment defined — A.1, A.3
- [x] Training distributions and OOD conditions defined — A.6, A.7 (`config/wind/regimes`, `config/domain`)
- [ ] **(added)** Approve the PROPOSED choices (model, loss weights, normalisation, model selection, D1–D11) — `specs/model_specification.md` §9
- [x] **(added)** Missing-value handling and exclusion rules defined — A.4
- [x] **(added)** IMU noise model defined and implemented — `data/noise.py` (presets none/low/nominal/high), tests in `tests/cpu/test_quality.py`; the preset values are proposals

**2. Model pipeline**

- [x] Baseline architecture implemented — B0–B4 (`models/world_model.py`) plus B5 (`b5a`, `b5b`) and B6 (`b6a_future/current`, `b6b_future/current`); 50 model tests pass
- [x] Dataloader implemented and tested — `ModelReadyDataset` + normalisation + noise transform; shapes, dtypes, finite values, no auxiliary leakage tested
- [x] Loss functions and evaluation metrics defined — thesis loss with quaternion geodesic (`losses/losses.py`); position, attitude, velocity, ω errors, reliable horizon, episode-level bootstrap (`evaluation/`); tested
- [x] Checkpointing and experiment tracking implemented — atomic checkpoints, exact resume (tested), `metrics.jsonl`, `runs.jsonl`, provenance, optional TensorBoard
- [x] **(added)** Nominal physics `f_phys` for the 13-dim state implemented and unit-tested — hover trim, free fall, RK4 quaternion norm; yaw-torque sign checked against the pilot data
- [x] **(added)** Trainer unit tests: resume equals continuous training; tiny-overfit; curriculum; checkpoint pruning and config-mismatch refusal
- [x] **(added)** YAML configuration for training, model choice and evaluation (`config/training`, `config/model`, `config/evaluation`) with unknown-key errors and a default-drift test
- [x] **(added)** Frozen-latent disturbance probe implemented (both targets: total wrench and reconstructed wind-attributable part), plus the evaluation analyses (wind/no-wind gap, re-observation, residual diagnostics, shuffled-u): `evaluation/probe.py`, `evaluation/analysis.py`, 10 tests
- [x] **(added)** B5 (residual-only) and B6 (physics + oracle disturbance) implemented and tested — `tests/cpu/test_models.py`; B6 uses the total recorded wrench; B6a on the pilot: `val_id` 20-step RMSE 0.0276 m vs B1 0.0447 m, `test_ood_magnitude` 0.0181 vs 0.1845 m. Not yet trained/evaluated as gates G6/G7
- [x] **(added)** Architectural-bias variants B3a–c (`.idea/specs/model_specification.md` §3.4) implemented as flags, defaults = thesis B3 (parameter counts unchanged): `resid_uses_u: false` (a, also B4 and B6b), `lambda_zero` (b, soft penalty on d_θ(h=0), proposed 1e-3), `shared_trunk: true` (c, residual reads the decoder trunk). Configs `config/model/b3a.yaml`, `b3b.yaml`, `b3c.yaml`; tests in `tests/cpu/test_models.py`. **Not yet trained or evaluated** (needs approval)
- [x] **(added)** Loader throughput benchmark — `data/loader_benchmark.py`: pilot `train` 57 episodes/s load, 13,378 windows/s (no worker) to 36,585 (8 workers) vs ≈ 878 windows/s consumed by training: ≥ 15× headroom

**3. Pilot validation**

- [x] Representative pilot dataset generated — `physiworld_pilot_v0.5.0`, 147 / 147 episodes, every part and role
- [x] Data quality and statistical distributions inspected — `data/statistical_validator.py`: range, distribution, duplicates, correlation, physical consistency, coverage; report `outputs/reports/physiworld_pilot_v0.5.0_quality/` (0 fail, 4 warn)
- [x] Labels validated against ground truth — wrench sum, wind vs plan, free fall/hover, **Newton balance on the ground truth p99 0.03 m/s²**, kinematics, quaternion; (the model-based check G10 is a separate item below)
- [x] Baseline successfully trained and evaluated — B2, B3, B4 trained on the pilot and evaluated per role against B0 and B1 (`audit/baseline_gate_report.md`, `outputs/results/`)
- [x] Overfitting test on a tiny dataset passed (G1) — B2 1.8 mm, B3 0.5 mm, B4 0.7 mm 20-step position error on 8 pilot episodes
- [x] **(added)** Pilot validator passes with no hard findings — `validation.json`: ok, 141 + 6 episodes
- [x] **(added)** Ground-contact and saturation outcomes inspected and decided — `audit/run5_report.md`
- [x] **(added)** Doc sanity checks run on the pilot (state coverage, residual vs wind, decorrelation, counterfactual share) — closed loop passes; findings recorded
- [x] **(added)** Statistical validator implemented and tested (13 tests)
- [x] **(added)** Outlier episodes: robust per-episode p99 check (`outliers.episodes`) and `outliers.png`
- [x] **(added)** IMU lever arm checked: refuted; transport-term cause identified (see the IMU item above)
- [x] **(added)** Coverage report of vehicle-factor space, wind regimes, scenario × role cells, initial conditions and novelty vs the training range
- [ ] **(added)** Per-role counts large enough to evaluate — 2–4 independent groups in several OOD roles (limit 5), see the quality report
- [ ] **(added)** Open-loop decorrelation: ρ(rotor speed, wind) = 0.39 > 0.3 on 24 episodes. Permutation test (1000 permutations): chance gives a 95th percentile of 0.42, p = 0.085 (borderline, not significant at 5 %); all 6 sibling groups share identical `u`. Needs more open-loop episodes and a decision on the rule
- [ ] **(added)** Domain-factor decorrelation: |ρ| = 0.59 between two ID factors over 32 domains (Bonferroni p 0.009); more domains or a joint sequence
- [x] **(added)** IMU vs ground-truth acceleration: cause found and **fixed at generation** (`physics.imu_transport_correction`). The lever arm was refuted (CoM offset 0.56 mm; fitted lever arm R² 0.007); the residual was ≈ −ω_B × v_B (R² 0.83). Verified on the real simulator on the 6 worst pilot episodes: p99 5.04 → 0.173 m/s², median 0.024 → 0.003; flagged in the manifest and episode statistics; raw-IMU datasets trigger a warning in training/evaluation
- [x] **(added)** Pilot regenerated as v0.6.0 (IMU fixed, determinism fixed) — 147 / 147 episodes in 26.5 min on 3 workers, validator ok (0 hard, 0 soft), checksums ok, identical plan to v0.5.0 (same episodes, statuses, step counts, wind); trajectories within 9e-5 m of v0.5.0 (bitwise equal in 61 / 147 episodes); IMU-vs-truth residual p99 1.645 → 0.051 m/s², episodes with p99 > 1: 47 / 141 → 0 / 141; quality report 0 fail, 3 warn (was 4)
- [x] **(added)** `physical.acc_imu` limit tightened for datasets with the transport correction (p99 ≤ 0.25; v0.6.0: 0.051); the raw-IMU limit (1.0) stays for v0.5.0
- [x] **(added)** Train all baselines on pilot v0.5.0 and v0.6.0 and compare: 16 runs finished (6–34 min each, ≈ 450–565 MiB GPU, ≈ 1 GB CPU RAM), evaluated on own and the other dataset, analyses run. Report `outputs/reports/training_compare_v05_v06/report.html`; written results `docs/results/baseline_comparison_pilot_v05_v06.md`. Findings: B3 best non-oracle model in distribution (test ID 20-step RMSE 0.035 / 0.037 m v0.5 / v0.6) but **not on OOD wind magnitude** (0.163 m vs B6a 0.018 m); B4, B4 (B3 width) and B5b get worse with training (C6); `best.pt` was not the best epoch for 8 of 16 runs (C9, fixed in code, runs not repeated: those numbers are provisional); IMU correction has no effect distinguishable from one-seed noise. One seed, 63 training episodes, 2–7 episodes per OOD role
- [x] **(added)** Resource tracking: host-side sampler `scripts/resource_sampler.py` (per-process CPU, peak RSS, GPU memory, epoch; per-GPU utilisation) for the running jobs; the trainer now logs per-epoch peak GPU/CPU memory, CPU time and wall time and run totals in `metrics.jsonl` / `summary.json` (`training/resources.py`)
- [x] **(added)** Report generator `evaluation/report.py` (figures, CSV tables, `REPORT.md`, self-contained `report.html`; validated palette; 4 tests)
- [x] **(added)** Baseline ordering shown (G3–G7) on the pilot, one seed: all learned models < hold-last; B3 best non-oracle in distribution; B5a/B5b worse than B3; B6b (oracle) best; B3 not best on OOD magnitude. Multi-seed confirmation open
- [ ] **(added)** Action-shortcut test G12 and wind-estimator probe on trained models: shuffled-u done for B1–B4 on both pilots (jump 1.00 / 1.01 / 1.02 / 1.24 at k=5; low power, siblings' commands differ little); wind-estimator probe run on 8 trained v0.6.0 models (2026-11-25): the latent carries no linearly readable wind force (problems log C14)
- [x] **(added)** Label sanity by model G10: B1 on no-wind nominal episodes 0.0092 m at 20 steps vs 0.0438 m on mass +5 % siblings (expected direction)
- [x] **(added)** State-estimation diagnostic G11: decoder start from the IMU only gives 0.5 m position and ≈ 120° attitude error; use the given initial state

**4. Generation infrastructure**

- [x] Random seeds and configurations recorded — seed tree, config hash, `provenance.json`, manifest (validator enforces)
- [x] Checkpointing and recovery implemented — atomic `index.json` per episode, `--resume`, failure isolation, restart wrapper (tests in `tests/cpu/test_planner_runner.py`)
- [x] Storage and generation time estimated — ≈ 24 s and ≈ 0.49 MB per episode; projections in `data_generation_pipeline.md` §6; one generator process is CPU-bound (≈ 3 % GPU, ≈ 4.5 busy cores, ≈ 55 steps/s)
- [x] Sharding and integrity checks implemented — shards of 200 episodes, validator, resume scan, **per-episode sha256 checksum verified on resume, merge and validate** (pilot episodes written before it have none: `generate checksums`)
- [x] Train/validation/test split strategy finalized — per-group roles, leakage hard-fail, five OOD families + trajectory + corner (A.7); the checkpoint-selection choice is part of the approval item above
- [x] **(added)** Live monitoring and alerts — status page, notifier (`info.md`); the real ntfy/Telegram services are untested
- [x] **(added)** Cross-process reproducibility checked (a resumed episode vs a one-shot episode) — a fresh-process episode is bitwise identical across processes, after `kill -9` + `--resume` and across GPUs; an episode generated after others in the same process differs by ≤ 6e-6 m (order dependence, see the next item)
- [ ] **(added)** Parallel generation measured: implemented (`--shard I/N`, `merge`, `scripts/generate_parallel.sh`, tested); a 2-worker real run showed no per-worker slowdown (51.4 s vs 50.9 s) and a 75/72 split; speed-up with more workers (and several per GPU) not measured
- [ ] **(added)** Final dataset size and mix decided (time budget)
- [ ] **(added)** Gate report written in `docs/audit/`, configuration frozen
- [ ] **(added)** Remove or explain the order dependence between episodes in one process (≈ 6e-6 m): the leaking simulator state is not identified (`docs/problems_log.md` B15)
- [x] **(added)** Parallel generation implemented and tested (group-level LPT split, per-worker shards and index, merge with whole-dataset validation, launcher); 9 CPU tests plus a real 2-worker run
- [ ] **(added)** Add checksums to the pilot shards (`generate checksums --dataset data/datasets/physiworld_pilot_v0.5.0`; modifies the shards by adding an attribute: needs your approval)
- [ ] **(added)** Mode B (fixed-action) episodes in the test roles: the pilot has them only in `train` (18) and `val_id` (6)

**5. Diversity and generalisation (added 2026-11-24)**

*Priority 1: config-level additions, implemented and verified in the real simulator*

- [x] **(added)** Perturbed initial states — `initial_conditions: randomized` via scenario `overrides` (config only; the pilot used hover trim for all 147 episodes)
- [x] **(added)** Longer episodes (20–30 s) — scenario `overrides: {duration_s: [20, 30]}` (config only)
- [x] **(added)** Dryden turbulence as a wind profile (`type: dryden`, regime `id_diverse`) — small code addition in `wind/regimes.py`. A test caught that the turbulence spec needs `enabled: true`, otherwise the wind manager silently ignores it
- [x] **(added)** Wind "switching" — `type: composite` profile (a wind level that steps up and back down, with an OU or Dryden overlay). Level switches only: switching the profile *type* inside an episode (e.g. calm → turbulence → calm) is not supported (needs time windows in the WindManager)
- [x] **(added)** Air-density randomisation — domain config `randomized_env` with the extra factor `air_density_scale` (±10 %, OOD shell 10–20 %); scales ρ in the body drag and the rotor-drag coefficient; drawn from its own generator after the design, so every other domain factor is unchanged (code: `parameter_sampler.py`, `forces/system.py`)
- [x] **(added)** Controller gain randomisation — part key `controller_randomisation` (position and velocity gains and the attitude-hold gains × 0.7–1.3); the sampled gains are stored in the manifest row (code: `experiments/plan.py`)
- [x] **(added)** Tests: `tests/cpu/test_diversity.py` (10 tests); experiment `config/experiment/diversity_check.yaml` (29 episodes plan, all six additions)
- [x] **(added)** `diversity_check` run in the real simulator: 29 / 29 episodes completed, validator ok, all inside the force-model envelope; perturbed starts non-zero (|v₀| 0.22–0.42 m/s, recovered with tracking error ≤ 0.23 m); long episodes 25.8–29.9 s; Dryden episodes show wind fluctuations on all three axes (std 0.15–0.31 m/s); composite episodes show the level switch (std up to 1.07 m/s); randomised gains ran without saturation (one attitude episode with hold scale 0.83 drifted 1.2 m, to be looked at)
- [ ] **(added)** Decide whether the diversity additions enter the Run 6 plan (they are not in `pilot.yaml`)

*Wind range (Option C: ID 0–6 m/s, OOD (6, 9] m/s), verified on the real simulator (40 episodes, randomised vehicles)*

- [x] **(added)** Envelope check of Option C — ID (peak ≤ 6, observed up to 5.0 m/s): 14 / 14 inside the force-model envelope, worst μ 0.123. OOD mean 6.3–8.6 m/s: 24 / 26 inside (92 %); 14 / 14 up to 7.5 m/s, 18 / 19 up to 8.0 m/s; the two exceptions have mean wind ≥ 7.9 m/s with strong rotors (force scale 1.07–1.09 lowers the hover rotor speed, so μ reaches 0.201–0.207 against the 0.20 limit)
- [ ] **(added)** Decide the OOD upper bound: keep (6, 9] and accept that ≈ 1 in 6 episodes above 7.5 m/s are excluded from model-ready data, or narrow to (6, 8]. ID range 0–6 m/s: inside the envelope in all 14 episodes

*Priority 2 (future work: code changes, medium effort)*

- [ ] **(future)** Ground effect: enable it in the aero truth (the model exists, `truth_default` disables it) and add low-altitude / landing scenarios
- [ ] **(future)** Vehicle faults and asymmetry: per-rotor thrust asymmetry (a degraded rotor) and a centre-of-mass offset as domain factors; `force_scale` currently scales all rotors together
- [ ] **(future)** Battery sag: thrust scale decaying over an episode (needs longer episodes, see above)
- [ ] **(future)** Sensor realism: IMU bias random walk, latency and jitter, rotor-speed-dependent vibration, dropouts (today: white noise plus a per-window bias at load time)
- [ ] **(future)** Held-out families for the new factors (turbulence, ground proximity, rotor fault, sensor degradation): each needs its own split roles and at least 5 independent groups

*Priority 3 (future work: larger)*

- [ ] **(future)** A second airframe as a held-out vehicle family (OmniDrones ships several quadrotor models): needs its own drone config and an approved aero truth (drag coefficients, areas) like the Q3 step for the Hummingbird
- [ ] **(future)** Time-windowed wind regimes (the profile type changes inside an episode): needs time windows in the WindManager

Reset checklist: edit the boxes above.
- [x] **(added)** All model sizes (widths, depths, encoder/decoder layers, activation) are YAML-driven through `TrainConfig` → `ModelConfig` (default parameter counts unchanged, tests in `tests/cpu/test_models.py`)
- [x] **(added)** Sweep fix_v1 (29 runs: repeats with the fixed trainer, seeds, B3a–c, lr/wd): finished and evaluated; `docs/results/sweep_fix_v1.md`, `outputs/reports/sweep_fix_v1/`. B3a–c do not help OOD wind; B4 overfits (lr did not help; wd test too weak, C12); constant-velocity baseline added (C11)
- [ ] **(added)** Next experiments (need approval): longer history; non-linear probe; wind-range augmentation; stronger weight decay; OOD-aware checkpoint selection
- [x] **(added)** Wind plan for Run 6: wide ID (0–5 m/s mean) and OOD (6.2–8.5 m/s) ranges with stratified, upper-band-weighted mean speed implemented in the planner (`speed_strata`, `speed_strata_weights`), tested, plan preview `config/experiment/pilot_wide_wind.yaml`; nothing generated
- [ ] **(added)** Run 6 not started: sizing (episodes, domains, Mode B in test roles), envelope re-check of the wide wind, learning-curve experiment for the model-size / data question
- [x] **(added)** 20 Hz model step supported end to end (decimation and model step read from the manifest; evaluation refuses a rate mismatch); tracking check at 20 Hz vs 50 Hz: `docs/results/tracking_20hz_vs_50hz.md`
- [ ] **(added)** Pilot v7 generation (4,530 episodes, 20 Hz, stratified wide wind): planned and previewed, waits for approval
- [x] **(added)** Automatic resume of an unfinished generation after a server restart (`scripts/resume_on_boot.sh`, container restart policy, `@reboot` cron; stop switch `logs/AUTO_RESUME`); dataset report page shows running / queued / pending / done per episode while a parallel generation runs
