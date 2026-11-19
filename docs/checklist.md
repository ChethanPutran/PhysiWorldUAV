## **Pre-generation checklist**

Status as of 2026-11-17. An item is marked done only when there is evidence (a file, a test, a measured result). "Partly" items stay open. Items I added beyond the original 18 are marked **(added)**. Per-item evidence for the e2e requirements is in `specs/e2e_done.md`.

**Original 18 items: 18 / 18 done. Including my added items: 30 / 44 done.**

**1. Problem specification**

- [x] Input and output dimensions defined — `specs/data_specification.md` A.2
- [x] Units and coordinate frames defined — A.2, `guides/dataset_schema.md`
- [x] Sampling frequency and temporal alignment defined — A.1, A.3
- [x] Training distributions and OOD conditions defined — A.6, A.7 (`config/wind/regimes`, `config/domain`)
- [ ] **(added)** Approve the PROPOSED choices (model, loss weights, normalisation, model selection, D1–D11) — `specs/model_specification.md` §9
- [x] **(added)** Missing-value handling and exclusion rules defined — A.4
- [x] **(added)** IMU noise model defined and implemented — `data/noise.py` (presets none/low/nominal/high), tests in `tests/cpu/test_quality.py`; the preset values are proposals

**2. Model pipeline**

- [x] Baseline architecture implemented — B0, B1, B2, B3, B4 (`models/world_model.py`), 27 unit tests pass; B5 and B6 are not written (listed separately below)
- [x] Dataloader implemented and tested — `ModelReadyDataset` + normalisation + noise transform; shapes, dtypes, finite values, no auxiliary leakage tested
- [x] Loss functions and evaluation metrics defined — thesis loss with quaternion geodesic (`losses/losses.py`); position, attitude, velocity, ω errors, reliable horizon, episode-level bootstrap (`evaluation/`); tested
- [x] Checkpointing and experiment tracking implemented — atomic checkpoints, exact resume (tested), `metrics.jsonl`, `runs.jsonl`, provenance, optional TensorBoard
- [x] **(added)** Nominal physics `f_phys` for the 13-dim state implemented and unit-tested — hover trim, free fall, RK4 quaternion norm; yaw-torque sign checked against the pilot data
- [x] **(added)** Trainer unit tests: resume equals continuous training; tiny-overfit; curriculum; checkpoint pruning and config-mismatch refusal
- [ ] **(added)** B5 (residual-only) and B6 (physics + oracle disturbance) implemented
- [ ] **(added)** Architectural-bias variants B3a–c and the frozen-latent wind-estimator probe implemented (`specs/model_specification.md` §3.4)
- [ ] **(added)** Loader throughput benchmark (windows/s, GPU never waiting)

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
- [x] **(added)** Coverage report of vehicle-factor space, wind regimes, scenario × role cells, initial conditions and novelty vs the training range
- [ ] **(added)** Per-role counts large enough to evaluate — 2–4 independent groups in several OOD roles (limit 5), see the quality report
- [ ] **(added)** Open-loop decorrelation: ρ(rotor speed, wind) = 0.39 > 0.3 on 24 episodes; more open-loop episodes or a fix
- [ ] **(added)** Domain-factor decorrelation: |ρ| = 0.59 between two ID factors over 32 domains (Bonferroni p 0.009); more domains or a joint sequence
- [ ] **(added)** IMU vs ground-truth acceleration: p99 1.65 m/s² in high-angular-rate phases (cause unverified, lever-arm lead); verify the IMU position and fix or compensate
- [ ] **(added)** Baseline ordering shown (G3–G7): done for B0–B4 (learned < hold-last at 20 steps; B3 best in distribution, not on OOD magnitude; B3 vs B4 mixed); B5 and B6 (G6, G7) not implemented
- [ ] **(added)** Action-shortcut test G12 and wind-estimator probe run
- [x] **(added)** Label sanity by model G10: B1 on no-wind nominal episodes 0.0092 m at 20 steps vs 0.0438 m on mass +5 % siblings (expected direction)
- [x] **(added)** State-estimation diagnostic G11: decoder start from the IMU only gives 0.5 m position and ≈ 120° attitude error; use the given initial state

**4. Generation infrastructure**

- [x] Random seeds and configurations recorded — seed tree, config hash, `provenance.json`, manifest (validator enforces)
- [x] Checkpointing and recovery implemented — atomic `index.json` per episode, `--resume`, failure isolation, restart wrapper (tests in `tests/cpu/test_planner_runner.py`)
- [x] Storage and generation time estimated — ≈ 24 s and ≈ 0.49 MB per episode; projections in `specs/data_generation_pipeline.md` §6 (parallel speed-up not measured)
- [x] Sharding and integrity checks implemented — shards of 200 episodes, validator, resume scan (no per-episode checksum)
- [x] Train/validation/test split strategy finalized — per-group roles, leakage hard-fail, five OOD families + trajectory + corner (A.7); the checkpoint-selection choice is part of the approval item above
- [x] **(added)** Live monitoring and alerts — status page, notifier (`info.md`); the real ntfy/Telegram services are untested
- [ ] **(added)** Cross-process reproducibility checked (a resumed episode vs a one-shot episode)
- [ ] **(added)** Parallel generation (Run 6) measured: speed-up over one process
- [ ] **(added)** Final dataset size and mix decided (time budget)
- [ ] **(added)** Gate report written in `docs/audit/`, configuration frozen

Reset checklist: edit the boxes above.
