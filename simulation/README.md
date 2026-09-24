# Simulation

Training and playback for the deployed MARBLE policies, in MuJoCo through
[mjlab](https://github.com/mujocolab/mjlab) (GPU, MuJoCo Warp). The robot model is
`asset/ball_linear_complex/`, the full-assembly export of the robot. Code names use `ballbot`.

## Quick start

From the MARBLE root, on a desktop with an NVIDIA GPU and a display:

```bash
micromamba create -n marble-sim python=3.12 -y && micromamba activate marble-sim
pip install -r simulation/requirements.txt
cd simulation
python mj_envs/run.py play --task BallbotVelComplexFlatDRLatency    # land
python mj_envs/run.py play --task BallbotVelRingCageComplexDR       # water
python mj_envs/run.py play --task BallbotVelRingCageComplexSmooth   # water, smoother
```

No training needed: `play` loads the shipped checkpoint from `../policies/<task>/`. In the viewer:

| Input | Action |
| --- | --- |
| Numpad 8 / 5 | Command forward / back |
| Numpad 4 / 6 | Command left / right |
| Numpad 0 | Zero the command |
| K / L, [ / ] | Push the robot forward / back, left / right |
| Space | Pause |
| Enter | Reset |
| Mouse | Left drag rotates, right drag pans, scroll zooms |

## Install

Python 3.12 and an NVIDIA GPU. This is a separate environment from the robot's.

```bash
micromamba create -n marble-sim python=3.12 -y && micromamba activate marble-sim
pip install -r simulation/requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

## Tasks

`--task` names an experiment class in
[`mj_envs/tasks/ballbot_velocity/experiments.py`](mj_envs/tasks/ballbot_velocity/experiments.py).

| Task | Medium | Deployed as |
| --- | --- | --- |
| `BallbotVelComplexFlatDRLatency` | land | `policies/BallbotVelComplexFlatDRLatency` (land default) |
| `BallbotVelRingCageComplexDR` | water | `policies/BallbotVelRingCageComplexDR` (water default) |
| `BallbotVelRingCageComplexSmooth` | water | `policies/BallbotVelRingCageComplexSmooth` |
| `BallbotVelComplexHeuristic` | land | Geometric controller, no training |

The other classes in that file are base classes of these and are not meant to be trained.

## Play

Run from `simulation/`:

```bash
cd simulation
python mj_envs/run.py play --task BallbotVelComplexFlatDRLatency
python mj_envs/run.py play --task BallbotVelComplexHeuristic --agent experiment
```

`play` picks the latest checkpoint in `runs/<task>/` with at least 500 iterations, and otherwise
the shipped export in `../policies/<task>/policy_deployed.pt`, so it works on a fresh clone. Pass
`--checkpoint <path>` for a specific one. It opens a native MuJoCo viewer and needs a display.

## Train

```bash
python mj_envs/run.py train --task BallbotVelComplexFlatDRLatency
```

FlashSAC, 4096 environments. Logs and checkpoints go to `runs/<task>/<timestamp>_flash_sac/`.
`--num_envs`, `--max_iterations` and `--seed` override the class defaults.

## Export to the robot

```bash
python mj_envs/run.py play --task BallbotVelComplexFlatDRLatency \
    --checkpoint runs/BallbotVelComplexFlatDRLatency/<run>/model_<iter>.pt --export_policy
```

Writes `policy_deployed.pt`, `env_config.yaml` and `robot.xml` to
`mj_envs/deploy/runs/<task>/`. Copy that folder to `../policies/<name>/` and check it as in
[`docs/DEVELOPING.md`](../docs/DEVELOPING.md#exporting-a-learned-policy).

## Source

This directory is generated from the lab's internal training repository by
`scripts/make_marble_sim.py` there; files keep their internal paths so imports work unchanged.
Helpers the ballbot code shares with other lab robots (reward terms, the ring-cage water model,
randomisation events) are collected in `mj_envs/asset_zoo/ballbot/shared.py` and
`mj_envs/tasks/ballbot_velocity/shared.py`.
