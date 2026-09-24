# Developing

Run the examples from the repository root.

## Code map

| Path | Contents |
|---|---|
| `src/ballbot_runtime.py` | Constants, calibration, joint-to-motor mapper, IMU frame, both controllers, policy contract check. No hardware I/O; imports only `numpy` at load. |
| `src/ballbot_terminal.py` | `BallbotController`: 100 Hz loop, keyboard, arming, homing, safety, logging. Also the CLI. |
| `xiao_can/xiao_gl_motor.py` | `GLMotorController`: GL40 motors through the XIAO serial-to-CAN bridge. |
| `xiao_can/xiao_can_bridge/` | Bridge firmware. Host sends `S:<id>:<length>:<data>` (hex); bridge returns `R:` lines; `V` returns the version string. |
| `hardware_bindings/imu/py_imu.py` | `IMU` driver ([README](../hardware_bindings/README.md)). |
| `scripts/` | Calibration, bench tools, [offline checks](#offline-checks). |

## Units and frames

**Joints.** Every three-element array (observation, action, target) is in
`JOINT_NAMES` order: `base_link_Slider-5`, `base_link_Slider-6`,
`base_link_Slider-7`. A joint position is metres from the 114 mm idle position
(`IDLE_POSITION_MM`) on 220 mm of travel (`MAX_TRAVEL_MM`). Positive motion
moves the mass toward base −Z, +Y and −X respectively (`JOINT_POSITIVE_AXIS`).

**Motors.** Motor positions are radians, in `MOTOR_IDS` order `0x08`, `0x07`,
`0x06`. One turn is 100 mm (`MM_PER_TURN`). After homing, 0 rad is the 0 mm
hardstop. `MotorJointMapper` converts using `config/calibration.json`.

**Calibration file.** `schema_version` 2. Per joint: `motor_id` (8, 7 or 6,
each once), `encoder_sign` and `joint_to_travel_sign` (each ±1). The loader
rejects any other schema version and any mapping that does not cover every
joint and motor exactly once. The procedure is in
[`SETUP.md`](SETUP.md#slider-calibration).

**Orientation.** `ImuState` holds `rotation_world_base` (R_wb) and
`angular_velocity_world` (rad/s). `ImuState.from_device(imu)` computes
R_wb = R_ws R_bs^T and ω_world = R_ws ω_sensor, with R_bs the IMU mounting
rotation (`R_BASE_SENSOR`). It raises `ValueError` on non-finite or
non-orthonormal data.

**Command frame.** Commands are in world XY. `with_yaw_reference(yaw)` rotates
the world frame about +Z so command +X points at `yaw`, leaving roll and pitch
unchanged. The terminal sets it from `--yaw-offset-deg` and Q/E.

**Travel limits.**

| Stage | Limit |
|---|---|
| `HeuristicPolicy.act` | −0.104 to +0.096 m (`HEURISTIC_LOWER_M`, `HEURISTIC_UPPER_M`) |
| `TrainedPolicy.act_from_frame` | −0.114 to +0.106 m (`JOINT_LOWER_M`, `JOINT_UPPER_M`) |
| `MotorJointMapper`, every target | Joint clipped to −0.114/+0.106 m, then travel to 10–195 mm (`SAFE_MIN_TRAVEL_MM`, `SAFE_MAX_TRAVEL_MM`) |

Reach from the 114 mm centre is asymmetric (104 mm down, 81 mm up);
`joint_to_travel_sign` sets which side is short.

## Runtime API

### Geometric controller

```python
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from ballbot_runtime import HeuristicPolicy, ImuState, MotorJointMapper, SliderCalibration

# The robot uses config/calibration.json; the backup runs on a fresh clone.
cal = SliderCalibration.load(Path("config/calibration.robot-backup.json"))
mapper = MotorJointMapper(cal)

policy = HeuristicPolicy()

# Upright and at rest. On the robot this comes from ImuState.from_device(imu).
imu_state = ImuState(rotation_world_base=np.eye(3), angular_velocity_world=np.zeros(3))

direction_xy = np.array([1.0, 0.0])   # toward +X of the command frame
magnitude = 0.5                       # 0 to 1

joint_targets_m = policy.act(direction_xy, magnitude, imu_state, np.zeros(3), np.zeros(3))
motor_targets_rad = mapper.motor_targets_from_joint_position(joint_targets_m)
```

`HeuristicPolicy.act()` places a point 1.5 × magnitude m out along the
direction and sends each slider to the closest point on its rail; the example
returns `[0, 0, -0.104]`.

### Learned policy

```python
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
from ballbot_runtime import ImuState, TrainedPolicy

policy = TrainedPolicy(Path("policies/BallbotVelComplexFlatDRLatency/policy_deployed.pt"))

imu_state = ImuState(np.eye(3), np.zeros(3))
joint_pos = np.zeros(3, dtype=np.float32)    # m, from MotorJointMapper.joint_state_from_motors
joint_vel = np.zeros(3, dtype=np.float32)    # m/s
last_action = np.zeros(3, dtype=np.float32)

policy.reset()
for _ in range(5):                           # one call per 50 Hz policy step
    frame = TrainedPolicy.build_frame(
        np.array([0.3, 0.0]),                # commanded velocity in the command frame, m/s
        imu_state, joint_pos, joint_vel, last_action,
    )
    joint_targets_m, last_action = policy.act_from_frame(frame)
```

`TrainedPolicy(policy_path, action_scale=1.0, policy_hz=POLICY_HZ, strict=True)`
loads the model on the CPU, checks the [`env_config.yaml`](#env_configyaml-contract)
beside it, and confirms a finite `(1, 3)` output for a zero input. It raises on
any failure.

- `build_frame(command_world_xy, imu_state, joint_pos, joint_vel, last_action)`
  (static) returns the 23-element [frame](#observation-frame).
- `act_from_frame(frame)` appends to the three-frame history and returns
  `(joint_targets_m, action)`. Pass `action` back as `last_action`.
- `reset(frame=None)` fills the history with copies of `frame`, or zeros.
- `set_action_scale(scale)` sets the runtime scale, clipped to [0, 1].

Call `act_from_frame` once every 20 ms (`POLICY_HZ` = 50). Always pass
`policy_path`: `DEFAULT_POLICY_PATH` names a retired checkpoint
([`OPERATIONS.md`](OPERATIONS.md#learned-policy)).

### Motors

The terminal opens the bridge like this (run with `PYTHONPATH=src:xiao_can`):

```python
from ballbot_runtime import MOTOR_IDS
from xiao_gl_motor import GL40, GL_MODE_POS_VEL, GLMotorController

motor = GLMotorController(
    [[motor_id, "xiao", GL40] for motor_id in MOTOR_IDS],
    control_mode=GL_MODE_POS_VEL,
    default_vel_limit=30.0,   # rad/s; the terminal passes --vel-limit
    port_name=None,           # autodetect; the terminal passes --motor-port
)
```

Arrays are in `MOTOR_IDS` order: write `mech_pos_ref`/`mech_vel_ref`, read
`mech_pos`, `mech_vel`, temperatures, `error_code` and `last_feedback_time`.
`motion_control_once()` sends one command to every motor.

The constructor raises `RuntimeError` unless the bridge reports
`BALLBOT_XIAO_BRIDGE_V3` ([`FLASH_XIAO.md`](FLASH_XIAO.md)) and leaves the
motors disabled. Feedback streams only after `enable()`; follow
`BallbotController.arm()` for a safe start.

`ballbot_runtime.py` has no safety logic. A program that drives the motors
itself gets none of the terminal's
[automatic stops](OPERATIONS.md#automatic-stops) or learned-policy limits.

## Adding a controller

A controller needs one method, called positionally:
`act(direction_world_xy, magnitude, imu_state, joint_pos, joint_vel)`,
returning (3,) float32 slider targets in metres from centre. Inputs are in the
command frame, and `direction_world_xy` is a unit vector.

There is no plug-in hook: replace `self.heuristic = HeuristicPolicy()` in
`BallbotController.__init__` and launch with `--policy heuristic`. The terminal
calls `act()` every 100 Hz tick while a movement command is active and centres
the sliders otherwise. `magnitude` starts at `--heuristic-magnitude`; N/M move
it in 0.1 steps within [0.01, 10.0], so clip it yourself. With smoothing off
(the default), targets go from the mapper clip straight to the drives.

## Exporting a learned policy

The runtime accepts any TorchScript module that meets this contract:

| | |
|---|---|
| Input | float32 `(1, 3, 23)`: three [frames](#observation-frame), oldest first |
| Output | `(1, 3)`, in `JOINT_NAMES` order |
| Output meaning | Clipped to [−1, 1], × 0.106 m (`ACTION_SCALE_M`) × runtime action scale (`--action-scale`, default 1.0), then clipped to −0.114/+0.106 m. An absolute target from centre, not an increment. |
| Rate | 50 Hz (`POLICY_HZ`) |
| Device | CPU: `torch.jit.load(path, map_location="cpu")` |

Save the traced or scripted actor as `policy_deployed.pt`:

```python
import torch

model.eval()
example = torch.zeros((1, 3, 23), dtype=torch.float32)   # (batch, history, frame)
with torch.no_grad():
    traced = torch.jit.trace(model, example)
traced.save("policies/MyPolicy/policy_deployed.pt")
```

Put the `env_config.yaml` exported by training beside it
([contract](#env_configyaml-contract)); a hand-written one checks nothing.
Without `env_config.yaml` the runtime prints a `NOTE` and runs only the shape
check, which cannot catch a wrong action scale or a reordered frame.

Check it, then run it:

```bash
python src/ballbot_terminal.py --preflight --policy-path policies/MyPolicy/policy_deployed.pt
python src/ballbot_terminal.py --policy trained --policy-path policies/MyPolicy/policy_deployed.pt
```

`./run.sh` lists the new folder in options 5 and 6 automatically.

## Observation frame

`TrainedPolicy.build_frame` packs each frame in this order. Terms are named as
in `env_config.yaml`.

| Index | Term | Source | Units |
|---|---|---|---|
| 0-2 | `angular_velocity` | `imu_state.angular_velocity_world` | rad/s, command frame |
| 3-4 | `commands_xy` | commanded velocity; in the terminal, unit direction × `--speed` | m/s, command frame |
| 5-7 | `dofPosition` | `joint_pos` | m from centre |
| 8-10 | `dofVelocity` | `joint_vel` | m/s |
| 11-13 | `actions` | previous `action` from `act_from_frame`, in [−1, 1] | unitless |
| 14-22 | `base_rotation_matrix` | `imu_state.rotation_world_base`, row-major | unitless |

An empty history is filled from the first frame. The terminal feeds back the
raw `action`, so `--action-scale` below 1 and `--target-rate-limit` above 0
change the applied target but not the fed-back action.

## `env_config.yaml` contract

`PolicyDeployConfig` checks these fields against the runtime constants.

| Field | Required value |
|---|---|
| `policy_obs_group` | Observation group to read; `actor` if absent |
| `observation.<group>` term names, in order | `angular_velocity`, `commands_xy`, `dofPosition`, `dofVelocity`, `actions`, `base_rotation_matrix` |
| Sum of `term_dim` | 23 |
| `history_length` of every term | Equal, and 3 |
| `action.dim` | 3 |
| `action_scale` of every `actuators` entry | Equal, and 0.106 |
| `simulation.control_freq` | 50 |
| `simulation.timestep` × `simulation.decimation` | 1 / `control_freq` |

A mismatch raises `RuntimeError` listing every problem. Unequal histories or
action scales raise `ValueError` at load. Other fields are not read.
`strict=False` turns mismatches into warnings; only `policy_sim_check.py` uses
it. The terminal is always strict.

## Offline checks

None of these opens the motors or the IMU.

| Check | Command | Needs |
|---|---|---|
| Readiness | `python scripts/preflight_heuristic.py [--require-calibration]` | `numpy`, `pyserial`; on Linux, Python 3.12 and the built IMU binding |
| Policy contract | `python src/ballbot_terminal.py --preflight --policy-path <pt>` | `torch`, `pyyaml`, `tyro`, `msgpack`, `pyserial` |
| Policy in simulation | `python scripts/policy_sim_check.py --robot-xml <xml> --policy-path <pt>` | `torch`, `pyyaml`, `mujoco`, a `robot.xml` that compiles |
| Run-log analysis | `python scripts/analyze_policy_log.py <log>.msgpack [--radius 0.19]`, or `--self-check` | `numpy`, `msgpack` |

`preflight_heuristic.py` (`./run.sh` option 1, also run by
`prepare_orangepi.sh`) checks the Python version, `pyserial`, controller and
mapper output, the firmware source (text match only), the IMU binding, the
calibration file and script permissions. A missing calibration is a warning
unless `--require-calibration` is passed.

`policy_sim_check.py` runs `TrainedPolicy` in CPU MuJoCo on a ground plane
(water checkpoints too) and reports speed, tracking error and slider speed;
`--help` lists its sweeps. The shipped `robot.xml` files need meshes that are
not included ([known issues](#known-issues)).

## Run logs

Each run log ([`OPERATIONS.md`](OPERATIONS.md#status-line-and-logs)) is
concatenated msgpack objects: one header, then one step per policy step while
a movement command is active.

```python
import msgpack

with open(path, "rb") as f:
    header, *steps = msgpack.Unpacker(f, raw=False)
```

The header records the policy, rate, joint and motor IDs and observation
terms; each step records the frame (`obs`), `action`, `target_m`, motor
feedback and the operator settings. `PolicyRunLogger` in
`src/ballbot_terminal.py` defines every field.

Heuristic runs log `action` as zeros and `target_m` as the controller output.
`obs` is in the command frame, so steps compare only at equal `yaw_ref_deg`.

## Known issues

- `analyze_policy_log.py` stops with `KeyError: 'stiffness'` on current logs,
  which record `kp` and `kd`. Only `--self-check` works; compute the achieved
  rate from `t_s`.
- The achieved policy rate is not checked.
- The `policies/*/robot.xml` files do not compile as shipped
  ([`PROVENANCE.md`](../PROVENANCE.md#the-shipped-checkpoints)).
