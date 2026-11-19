# Data Generation Pipeline Specification

Status: describes what is **implemented and measured** (Runs 3–5) and lists the gaps. Numbers are from the pilot `physiworld_pilot_v0.5.0` (147 episodes) unless stated. Data definitions: `data_specification.md`. Commands: `info.md`, `README.md`.

## 1. Pipeline

```
config/ (Hydra) ──► plan ──► manifest.jsonl ──► runner (Isaac Sim 4.1 + OmniDrones) ──► shard_XXX.h5
                                                                                           │
index.json (atomic, after every episode) ◄── validator ◄──────────────────────────────────┘
        ▲
        └── live status page, phone notifier, resume
```

| Stage | Code | Notes |
|---|---|---|
| Plan | `experiments/plan.py` | Every random choice is made at plan time from a seed tree; an episode is a pure function of one manifest row. Writes `manifest.jsonl`, `resolved_config.yaml`, `provenance.json` |
| Simulate | `experiments/runner.py` | One episode: domain → PhysX, controller or open-loop program, wind, aerodynamic wrench, IMU |
| Write | `data/writer.py` | One HDF5 shard, closed after every episode |
| Validate | `data/dataset_validator.py` | Per episode and across the dataset; writes `validation.json` |
| Index | `experiments/generate.py` | `index.json` rewritten atomically after every episode |
| Wrapper | `scripts/generate_until_complete.sh`, `scripts/start_generation.sh` | Restart with `--resume`; detached launcher with status page and notifier |

## 2. A. Data quality checks

Mapping of the e2e requirement table to the implementation. "Hard" findings make the dataset incomplete; "soft" are reported.

| Check | Implemented | Where / result on the pilot |
|---|---|---|
| Completeness (all expected fields) | Yes, hard | `validate_arrays` against the schema registry (required fields, shapes, dtypes) |
| NaN / Inf | Yes, hard | Non-finite values in any non-optional float field; pilot: 0 |
| Range | **Yes** (`statistical_validator.check_range`) | Per-channel min/p1/p99/max/std, frozen-channel test, limits on altitude, position, speed, rate, tilt, IMU force, rotor speed; pilot: 0 violations. Quaternion norm and wind regime limits are in the schema validator |
| Distribution (matches intended) | **Yes** (`check_distribution`) | Realised wind statistics vs regime limits (hard); domain factors vs the intended ID range / OOD shell and KS test against uniform; scenario mix vs the planned weights (total-variation distance). Pilot: all pass |
| Correlation | **Yes** (`check_correlation`) | Action–wind correlation per data type, domain-factor correlation (Fisher/Bonferroni), feature redundancy, and the identifiability check Δv(nominal physics) vs wind. Pilot: open-loop action–wind ρ = 0.39 and a domain-factor pair at 0.59 are flagged |
| Temporal consistency | Yes, hard | Strictly increasing uniform timestamps, steps a multiple of the decimation, metadata step count equals data |
| Label correctness | Yes, hard | Logged external wrench equals the sum of the logged components; IMU sanity (hover = 9.81 m/s², free fall ≈ 0) verified in Run 4; rotor speed/thrust consistency; drone parameters read back from PhysX |
| Duplicates | **Yes** | Duplicate ids and split leakage (schema validator, hard); identical or near-identical episode content across groups and repeated seeds (`check_duplicates`). Pilot: none |

Figures (`statistical_validator`, PNG): channel histograms by split family, correlation matrices, nominal-physics residual vs wind, coverage (wind speed, domain factors, scenario × role), physical-consistency histograms.

Visualisation: the dataset report (`visualization/dataset_report.py`) shows splits, scenarios, parts, wind, vehicle domains, counterfactuals and an episode table, live while a run is in progress. Per-episode time series: `visualization/episode_plots.py`. **Gaps:** feature histograms, input–output relationships and outlier plots across the dataset.

## 3. B. Physical and logical consistency

| Consistency | Status |
|---|---|
| Position / velocity / acceleration | **Checked** (`check_physical`): Δp/dt vs mean velocity p99 0.009 m/s; Newton balance on the ground truth, Δv/dt vs R·(ΣT/m)ẑ + g + F_ext/m, p99 0.030 m/s² |
| Quaternion normalisation and continuity | Checked (hard) |
| Rotation-matrix orthogonality | Implied by unit quaternions; checked in `tests/cpu/test_frames.py` |
| Force and torque consistency | Checked (hard): sum of components equals the injected wrench; Run 4 verified a 1 N injected force against the logged acceleration (1.3957 vs 1.3966 m/s²) |
| Conservation / free fall / hover trim | Isaac tests (Run 4): free fall −9.8100 m/s², hover trim residual 1e-6 m/s² |
| Sensor vs ground truth | Gyro vs ω_B exact (p99 8e-6 rad/s). **IMU specific force vs Δv/dt deviates** in high-angular-rate phases (p99 1.65 m/s², max 4.7); open, see `docs/audit/run5_report.md` addendum |
| Force-model applicability | Per-episode envelope check; violating episodes are excluded from model data (pilot: 0) |
| Repeatability | Bit-exact within one process (Run 4). **Cross-process bit-exactness is unchecked** (four pilot episodes were produced in a second process) |
| Ground contact | Ends an episode (`terminated_ground`), kept and flagged (6 pilot episodes) |

A dataset can be numerically valid but physically wrong, so the baseline gates (`model_specification.md` G10) add a second, independent check: nominal physics on the no-wind episodes must reproduce the data.

## 4. C. Reproducibility

Stored per dataset: `resolved_config.yaml`, `provenance.json` (config hash, git state of this repo and OmniDrones, Isaac/library versions, GPUs, host, timestamp), `manifest.jsonl` (all sampled values and seeds), and per episode metadata (seeds, config hash, domain, wind plan, controller, simulation settings read back from PhysX, status, termination reason, statistics).

| Requirement | Status |
|---|---|
| Dataset / generator version | `dataset_version` in config and in the directory name; **gap:** generator (code) version is the git commit only, no separate version string |
| Seeds for independent random processes | Yes: `SeedSequence` tree with stable keys (domain, scenario, wind, excitation, noise…) |
| Config hash | Yes (validator rejects a missing hash) |
| Software and simulator versions | Yes (`provenance.json`) |
| Units and conventions | In the schema registry and `docs/guides/dataset_schema.md` |
| Timestamps, success/failure status | Yes (`generated_utc`; per-episode `status`; run-level `failures`) |
| Reproduce a single episode | `generate regenerate --dataset … --episode <id>` and reports max abs difference |
| Never overwrite a dataset | Yes: an existing dataset directory is refused without `--resume`; versions in the directory name |

## 5. D. Failure handling

| Requirement | Implemented |
|---|---|
| Periodic checkpoints | `index.json` rewritten atomically (temp file + `os.replace`) after every episode; shard closed after every episode |
| Small independent jobs | One shard per 200 episodes (`episodes_per_shard`); an episode is a pure function of its manifest row, so any subset can run independently (Run 6 parallelism) |
| Automatic retry | `generate_until_complete.sh` restarts with `--resume` (up to 5 attempts, stops if no progress) |
| Logging | Timestamped logs in `logs/`; every failing episode is logged with traceback |
| Atomic writes | Index: yes. Shard: an interrupted episode group (no metadata) is removed on resume |
| Corrupted-file detection | On resume, shards are scanned; a group without metadata is removed. **Gap:** no per-episode checksum |
| Progress tracking | `index.json`, live status page, `start_generation.sh --status`, phone notifier |
| Failure isolation | A failing episode is recorded under `failures` and the run continues |
| Gap | A reboot does not restart the generation (container restart policy `no`); a host-level supervisor is not set up |

## 6. E. Performance and storage

Measured on one RTX A6000, one process, pilot settings (200 Hz physics, ≈ 11 s median episode):

| Quantity | Value |
|---|---|
| Time per episode | ≈ 24 s (≈ 2,000 physics steps) |
| Storage per episode | ≈ 0.49 MB (72.4 MB / 147 episodes, gzip 4, all physics-rate fields) |
| Model-ready windows per episode | ≈ 460 (65,112 windows / 141 eligible episodes) |
| Pilot total | ≈ 1 h generation, 72 MB |

Projection (linear; assumes one process, no speed-up from parallelism, which is not yet measured):

| Episodes | Time, 1 process | Storage | Windows (stride 1) |
|---|---|---|---|
| 1,000 | ≈ 6.7 h | ≈ 0.5 GB | ≈ 0.46 M |
| 5,000 | ≈ 33 h | ≈ 2.5 GB | ≈ 2.3 M |
| 20,000 | ≈ 133 h (5.6 days) | ≈ 9.8 GB | ≈ 9.2 M |

Run 6 targets parallel generation on the 4 GPUs; the speed-up (and whether several processes per GPU help) must be measured, not assumed. Storage is not the constraint; time is.

Still to measure: dataloader throughput (windows/s), GPU utilisation and memory of generation (already sampled once: see the container usage note in Run 5) and training memory. The data volume in `data_specification.md` Part B §3 (200–500 train, 50–100 val, 50–100 per OOD regime) corresponds to roughly 1,000–1,500 episodes, i.e. 7–10 h on one process. The **final size is your decision**.

## 7. F. Generalisation and coverage

Coverage to examine (factor → what the pilot has → gap):

| Factor | Controlled by | Pilot coverage | Gap |
|---|---|---|---|
| Environment / wind | Regimes (5 families) | 0–7.5 m/s means, several temporal/gust profiles | Per-profile counts are small |
| Dynamics | Domain factors (Sobol) | 37 randomised domains (12 id_cl, 6 id_ol, 19 across the OOD parts) plus the nominal one | Few domains per part; coverage of the 9-D factor space is sparse |
| Noise | IMU noise at load time | Not applied | Implement and evaluate at several levels |
| Initial conditions | `config/initial_conditions` | Randomised per episode | Not summarised in the report |
| Disturbance profiles | Wind profile types | steady, step, gust, sinusoid, OU, random gusts | – |
| Rare / difficult events | Open-loop programs, ground contact | 6 ground contacts | Rare events are few |
| Trajectory diversity | 14 closed-loop + 2 held-out families | Present | Diversity is by family, not yet by feature-space coverage |

Leakage control: splits are per group; the validator hard-fails when a group, domain or scenario appears in more than one role. Held-out OOD conditions are genuinely unseen (wind regime, vehicle shell and corner, trajectory family).

**Open item:** a coverage report (per-factor histograms, pairwise coverage of the factor space, effective number of distinct trajectories per role) and an identifiability analysis (parameter sensitivity / FIM) before freezing the configuration (see `data_specification.md` Part B notes).

## 8. Iteration and versioning

Generation and modelling are iterative (e2e §7): the baseline results can send us back to change the generator (more excitation, more domains, targeted scenarios), producing a new dataset version. Version scheme: v0.x pilots; v0.5.0 is the Run 5 pilot; a change in labels or schema bumps the version and the old one is kept. Freeze the generation configuration only after the validation gate passes (`worlkflow.md`).

## 9. Gaps to close before the final generation (summary)

1. ~~IMU noise transform~~ and ~~distribution / correlation / duplicate / kinematic checks~~: done (`data/noise.py`, `data/statistical_validator.py`).
2. Resolve the IMU-vs-ground-truth acceleration deviation (verify the IMU prim offset; fix or compensate).
3. Cross-process bit-exactness check on a resumed episode.
4. Parallel generation (Run 6) and measured speed-up.
5. A supervisor for restarts after a reboot, and a checksum per episode (optional).
6. Final dataset size, and the decision on sample weights for the scenario mix.
