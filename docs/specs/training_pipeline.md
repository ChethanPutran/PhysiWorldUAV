# Training and Evaluation Pipeline

Status: DRAFT for approval; none of this is implemented yet (`training/train.py` is broken and is to be rewritten). It is built for the baselines in `model_specification.md` and the data in `data_specification.md`. Items marked **PROPOSED** need your approval.

## 1. Requirements (from the e2e requirements)

The pipeline must: load the dataset, train, compute training loss, compute validation loss, save checkpoints, resume training, and evaluate on unseen data, so that we can verify the generated data contains learnable information before the final generation.

## 2. Layout (PROPOSED)

```
src/physiworld_uav/
├── models/        physics.py (f_phys), encoder/latent/decoder/residual, world_model.py (B1–B6, B0 extra)   # rewritten for the 13-dim state
├── losses/        losses.py (L_state with quaternion geodesic, L_roll)
├── training/      trainer.py, checkpoint.py, run.py (CLI), tracking.py
├── evaluation/    evaluator.py, metrics.py, reliable_horizon.py, report.py
config/
├── model/         b1.yaml … b6.yaml (+ b0 extra)
├── training/      default.yaml (optimiser, batch size, epochs, seed)
└── evaluation/    default.yaml (horizons, thresholds, roles)
experiments/<name>_<time>/   config.yaml  metrics.jsonl  checkpoints/  eval/  provenance.json
```

Run via the container as for all code. CLI sketch: `python -m physiworld_uav.training.run model=b2 dataset=<dir>`; `--resume` continues the latest checkpoint.

## 3. Training loop

| Step | Specification |
|---|---|
| Data | `ModelReadyDataset(train)`; DataLoader with `shuffle=True`, `drop_last=True`; validation on `val_id` |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4, cosine decay with 200-step warm-up, grad-clip 1.0. PROPOSED, tune after G1 |
| Precision | float32 (the data and the physics integration are float32; mixed precision only if needed) |
| Batch / epochs | Batch 256; epoch = one pass over all windows; max 100 epochs with early stopping (patience 10 on `val_id` 20-step position error) |
| Loss | `model_specification.md` §5 (L = L_state + λ_roll·L_roll, quaternion geodesic attitude); log each term separately |
| Validation | After every epoch on the full `val_id` windows: loss terms and position error at k = 1, 5, 10, 20 |
| Seeds | One seed per run set in `torch`, `numpy`, `random`, the loader generator; stored in the run config. A multi-seed (≥ 3) repeat for every reported number |
| Determinism | `torch.use_deterministic_algorithms(True)` where supported; record that it was on |

## 4. Checkpointing and resume

* Save at the end of every epoch to `checkpoints/last.pt` **atomically** (write to a temporary file, then `os.replace`), and `best.pt` when `val_id` improves. Keep the last 3 epoch checkpoints.
* A checkpoint holds: model and optimiser state, scheduler, epoch and step, RNG states (torch, numpy, python, CUDA), normalisation statistics, the resolved config, the dataset id/version/config hash, git commit, and the best metric.
* `--resume` loads `last.pt`, restores everything, and continues at the next step; a test must show that "train 2 epochs, stop, resume 2 more" equals "train 4 epochs" within float tolerance on CPU.
* On load, refuse a checkpoint whose dataset hash or model config differs unless `--force` is given.
* Same philosophy as the generator: a crash never loses more than one epoch.

## 5. Experiment tracking

**Implemented** (`training/tracking.py`): local and dependency-free. One `metrics.jsonl` line per epoch (train/val loss terms, error by horizon, lr, throughput, GPU memory), `config.yaml`, `provenance.json` (git, library versions, dataset hash). A TensorBoard writer is optional. The existing HTML dataset-report style can be reused for a run report.

## 6. Evaluation

Run on a checkpoint, per split role, with the autoregressive rollout from the start state (`initial_state: decoder | given`, `model_specification.md` §3.1; no teacher forcing), using only `future_actions` (never future observations or wind, except B4).

| Metric | Definition |
|---|---|
| Position error vs horizon | mean and p90 of ‖p̂ − p‖ at each step k = 1…P (training horizon) and up to 100 steps (long evaluation, windows that do not fit are skipped, never padded) |
| Reliable imagination horizon | first step at which position error exceeds 0.5 m (first-failure definition, `evaluation/reliable_horizon.py`) |
| Attitude error | quaternion geodesic angle d_q (deg) vs horizon |
| Velocity, ω error | RMSE vs horizon |
| Energy/physical sanity | quaternion norm drift, speed within the data's range, no NaN |
| Residual magnitude (B3) | mean ‖d_θ‖ by role, to see where the residual works hardest |
| Aggregation | per role, per scenario family, per wind regime; mean ± std over seeds; bootstrap confidence interval over **episodes** (not windows, since windows overlap) |

Reports: a table per role (rows = models, columns = horizons), degradation ID→OOD, and per-episode plots (predicted vs true) for the worst cases, written to `experiments/<run>/eval/`.

## 7. Validation gates for the pilot (feed the Phase 4 gate)

Defined in `model_specification.md` §8 (G1 tiny-overfit … G11 state-estimation diagnostic, covering baselines B1–B6). The gate result is written to `docs/audit/` as a short report and decides: *freeze the generation configuration* or *debug, adjust the generator, regenerate the pilot*.

## 8. Experiments beyond the baseline gate (Phase 7)

* Sample-efficiency: train on nested group subsets 10 / 25 / 50 / 100 % (`data/splits.py`).
* Ablations: no residual, no latent, no IMU, no history, with/without IMU noise; B3a–B3c architectural-bias variants (`model_specification.md` §3.4: IMU-only residual path, zero-residual penalty λ₀‖d_θ(0)‖², shared decoder trunk) and the frozen-latent wind-estimator probe (evaluation only).
* Curriculum (thesis Stage 5): H_train 1 → 5 → 10 → 20 → 50; the dataset builder needs `horizon: 50`.
* Open-loop vs closed-loop data mix (does system-identification data help the residual?).
* Counterfactual pairs as an evaluation (and later, possibly, training signal).

## 9. Performance checks (before long runs)

Measure and record: loader throughput (windows/s), step time, GPU utilisation and memory, time per epoch, time to the stopping criterion. If the GPU idles on the loader, raise `num_workers` or preload tensors to the GPU (the pilot fits in memory many times over).

## 10. Compute

Training runs outside the generation container's GPU load: the host has 4 × RTX A6000. The baselines are small (≤ 1 M parameters); one GPU is enough, and CPU runs are feasible for G1. Do not train on a GPU that is running generation.

## 11. Open decisions

1. Optimiser settings and the early-stopping rule (proposed values above).
2. Number of seeds (proposed 3).
3. Where to run training (container vs a separate environment; torch is already in the container's venv).
4. Whether the unit-test suite for the trainer is required before first use (recommended: yes, the resume-equals-continuous test and the tiny-overfit test).
