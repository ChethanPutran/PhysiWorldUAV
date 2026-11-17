# Environment

Everything runs inside the `isaac-sim` docker container.

```bash
docker exec -it isaac-sim bash
cd /workspace/PhysiWorldUAV
source omnidrones_env/bin/activate          # puts Isaac's bundled scipy etc. on PYTHONPATH
export PYTHONPATH=$PWD/src:$PYTHONPATH      # append — do not overwrite (activate sets it)
```

| Component | Version |
|---|---|
| Isaac Sim | 4.1.0 (Kit 106.0.1, omni.isaac.core 3.18.1, omni.isaac.sensor 12.7.1) |
| OmniDrones | 0.1.1 (editable install, `OmniDrones/`) |
| PyTorch | 2.2.2+cu121 |
| Hydra / OmegaConf / h5py / SciPy | 1.3.7 / 2.3.1 / 3.16 / 1.10.1 |
| pytest | 9.2 (installed in Run 3, decision Q1) |

Exact versions of every run are recorded in each dataset's `provenance.json`.

## Commands

```bash
python -m pytest                        # CPU tests (~75 s)
python -m pytest --isaac tests/isaac    # Isaac integration tests (~2 min, needs a GPU)
python -m physiworld_uav.experiments.generate plan experiment=pilot
python -m physiworld_uav.experiments.generate run  experiment=smoke
python -m physiworld_uav.experiments.generate validate   --dataset data/datasets/smoke_v0.3.0
python -m physiworld_uav.experiments.generate regenerate --dataset data/datasets/smoke_v0.3.0 --episode smk_g00000_e0
```

## Notes

* `SimulationApp.close()` terminates the Python process. The CLI therefore validates and writes `index.json` *before* closing Isaac, and the Isaac tests drive the CLI in a subprocess.
* Files written from inside the container are owned by root. Run `chown -R 1003:1003 <path>` from the container when the host user needs to edit them.
