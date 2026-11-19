# E2E requirements — what is done

Tracks `docs/specs/e2e_requirements.md` item by item. `[x]` = done with evidence, `[~]` = partly done, `[ ]` = not done. Updated 2026-11-17. Evidence is a file, a test or a measured number; nothing is marked done on the strength of intent.

Legend for evidence: **spec** = `docs/specs/*.md`; **tests** = `tests/cpu/`; **report** = `docs/audit/run5_report.md` and `outputs/reports/physiworld_pilot_v0.5.0_quality/` (git-ignored).

## 1. Define the learning problem

- [x] Input and output definitions (features, dimensions, types, units, frames, sampling, temporal alignment) — `data_specification.md` A.1–A.3
- [x] Missing-value handling — A.4 (ground-contact episodes excluded; 0 NaN)
- [x] Data distribution: normal, extreme, rare, environmental variation, parameter variation, OOD — A.6, A.7 (`config/wind/regimes`, `config/domain`)
- [x] Noise distributions defined **and implemented** — `data/noise.py` (presets none/low/nominal/high), `tests/cpu/test_quality.py` (4 tests); the preset values are proposals
- [x] Reserved validation / test distributions — A.7 (per-group roles, 5 OOD families + trajectory + corner)

## 2. Baseline model before full-scale generation

### A. Data loading pipeline
- [x] Dataset returns correctly shaped batches — `ModelReadyDataset`, `tests/cpu/test_data_layer.py`
- [x] Data types, tensor dimensions, missing values tested — `test_data_layer.py`, `test_models.py` (shapes, finite, no auxiliary keys)
- [x] Normalisation — fitted on the training split only, stored in the checkpoint (`fit_stats`)
- [ ] Data loading speed measured (the trainer logs windows/s, but no standalone loader benchmark)

### B. Baseline architecture
- [x] Simple temporal baseline (GRU) — B2; plus B0, B1, B3 (proposed), B4 — `models/world_model.py`, `tests/cpu/test_models.py` (27 tests pass)
- [ ] B5 (residual-only) and B6 (physics + oracle disturbance) — gates G6 and G7 depend on them

### C. Training and evaluation pipeline
- [x] Load the dataset; train; training loss; validation loss — `training/trainer.py`
- [x] Save checkpoints (atomic, best / last / rolling) and resume exactly — tests: resume equals continuous training
- [x] Evaluate on unseen data per split role — `evaluation/evaluator.py`, `evaluation/run.py`
- [x] Experiment tracking — `metrics.jsonl`, `runs.jsonl`, `provenance.json`, `training.tracking` CLI (optional TensorBoard)
- [x] Gradient/curriculum features of the thesis (horizon curriculum, quaternion geodesic loss) — tests

## 3. Pilot data and its tests

- [x] Pilot generated and validated — 147 / 147 episodes, `physiworld_pilot_v0.5.0`, validator ok
- [x] Representative: all data types, roles, OOD categories, counterfactual groups — `data_specification.md` A.7

### A. Data quality checks (`statistical_validator`, report)
| Check | Status | Evidence |
|---|---|---|
| Completeness | [x] | schema validator, hard |
| NaN / Inf | [x] | 0 non-finite values |
| Range | [x] | per-channel stats and limits; 0 violations |
| Distribution | [x] | wind regimes, domain factors (range + KS), scenario mix (TVD ≤ 0.08) |
| Correlation | [x] done, [~] findings | action–wind closed loop 0.10 (pass); **open loop 0.39 (warn)**; domain factors 0.59 (warn) |
| Temporal consistency | [x] | uniform strictly increasing time, decimation multiple |
| Label correctness | [x] | wrench sum, wind vs plan, force balance p99 0.03 m/s² |
| Duplicates | [x] | no identical or near-identical episodes, unique seeds |
- [x] Histograms, correlation matrices, time-series and input–output (residual vs wind) plots — PNGs in the report folder; per-episode time series via `visualization/episode_plots.py`
- [~] Outlier plots (only range limits and percentiles)

### B. Physical / logical consistency
- [x] Position–velocity consistency (p99 0.009 m/s), quaternion normalisation and sign, gyro vs truth
- [x] Force and torque consistency — external wrench equals the sum of components; **Newton balance** Δv/dt vs rotor thrust + gravity + injected force (p99 0.03 m/s²)
- [x] Free fall / hover trim (Run 4 Isaac tests)
- [ ] **IMU specific force vs ground-truth acceleration: open finding** (p99 1.65 m/s², max 4.7 in high-angular-rate phases; cause unverified, lever-arm lead) — `run5_report.md` addendum

### C. Train the baseline (gate experiments on the pilot)
- [x] Training loss decreases on the full pilot — B2/B3/B4 trained to early stopping (29/35/33 epochs), loss down 75× / 104× (curriculum horizons change the loss scale, see `training_summary.md`)
- [x] Validation loss decreases, train–validation gap reported — best `val_id` 20-step position error B2 0.065, B3 0.027, B4 0.050 m; gap vs train 5–11× (`docs/audit/baseline_gate_report.md`)
- [x] Tiny-overfit test passed (G1) — B2 1.8 mm, B3 0.5 mm, B4 0.7 mm at 20 steps on 8 episodes (target 2 cm)
- [x] Does it generalise to unseen trajectories? Evaluated on `val_id`, `test_id`, five OOD roles and trajectory OOD: all models beat hold-last-state at 20 steps; B3 best in distribution; **OOD magnitude not improved by the physics prior** (B2 best); see the report
- [x] Can the model learn a simple relationship? Yes (G1) and B3/B4 learn a constant wind acceleration on synthetic data (tests)
- [x] Code verified on synthetic data: loss falls and tiny-overfit works for B2, B3, B4 — `tests/cpu/test_models.py`

## 4. Test the data-generation pipeline

### A. Reproducibility
- [x] Seeds, config hash, versions, units, timestamps, success/failure status stored — `provenance.json`, manifest, per-episode metadata
- [x] Reproduce an individual episode — `generate regenerate`
- [ ] Cross-process bit-exactness (a resumed episode vs a one-shot episode)

### B. Failure handling
- [x] Periodic checkpoints, small independent jobs, retry, logging, atomic writes, progress tracking — `generate.py`, `generate_until_complete.sh`, `start_generation.sh`, tests
- [x] Corrupted-group detection on resume
- [ ] Per-episode checksum; restart after a host reboot

### C. Performance and storage
- [x] Time and storage per episode and projections — ≈ 24 s and 0.49 MB per episode (`data_generation_pipeline.md` §6)
- [~] Training throughput (logged per epoch; ≈ 1,100 windows/s for B2 on one RTX A6000 in the tiny test)
- [ ] Parallel generation speed-up, GPU/CPU/memory profile under load

## 5. Generalisation, not just size

- [x] Coverage across factors — `check_coverage`: domain factors (≥ 90 % bins, ≥ 76 % pairs), wind speed 82 %, azimuth 100 %, elevation 75 %, scenario × role table, initial conditions
- [~] Coverage gaps identified, not closed: calm-weighted wind (50 of 147 episodes < 0.5 m/s), only 2–4 independent groups in several OOD roles, 32 ID domains
- [x] Data-leakage control — validator hard-fails on group / domain / scenario shared across roles
- [x] Genuinely unseen OOD conditions reserved — wind regimes, vehicle shell and corner, trajectory families; novelty check shows `test_ood_magnitude` states lie inside the training range (OOD is in the wind)

## 6. Workflow phases

- [x] Phase 1 problem specification (drafted; open decisions D1–D11 await approval)
- [x] Phase 2 baseline development — B0–B4 implemented, trained and evaluated on the pilot (B5, B6 pending)
- [x] Phase 3 pilot generation
- [~] Phase 4 validation gate: quality done with open findings (IMU deviation, open-loop decorrelation, few OOD groups); learnability shown (`baseline_gate_report.md`); reproducibility (cross-process) and performance (parallel generation) pending
- [ ] Phases 5–7

## 7. Advanced considerations
- [~] Iterate data generation with model results — first results are in (`docs/audit/baseline_gate_report.md`: large train→val gap, OOD-magnitude behaviour, calm-weighted wind); the generator changes they suggest are not yet decided
- [x] Architecture kept open — ablations and variants in `model_specification.md` §3.4
- [x] Dataset versions never overwritten

## 8. Pre-generation checklist
See `docs/checklist.md` (same evidence, same marks).

## 9. Recommended project structure
- [x] Data generation module independent of the model (`experiments/`, `data/`)
- [x] Data validation modules: schema, physical, integrity, **statistical** (`data/dataset_validator.py`, `data/statistical_validator.py`)
- [x] Models, training, evaluation modules
- [~] Separate `generation_config / model_config / evaluation_config` YAMLs (generation is Hydra; model and training use dataclasses and `--set`)
- [x] Metadata and experiment folders (`outputs/experiments`, manifests, provenance)
