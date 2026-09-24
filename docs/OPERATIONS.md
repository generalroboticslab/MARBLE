# Operations

This assumes [`SETUP.md`](SETUP.md) is done and that a new or rebuilt robot
has passed [`validation.md`](validation.md). Run commands from the repository
root in the Python 3.12 environment.

## Safety

- Keep a physical power cutoff within reach. The latching E-stop (X), quitting
  and `scripts/disable_motors.py` all send the GL40 exit-control command. None
  of them cuts battery or bus power.
- Restrain or lift the robot for the first run of a new robot, checkpoint or
  flag. Secure the sphere before applying motor power for calibration.
- Keep the first water run of a checkpoint tethered, with manual override
  ready.

## Before each session

1. Power on the Orange Pi, the motors and the IMU.
2. Log in with `ssh -t`. The controller reads single key presses; without a
   TTY it exits before touching any hardware.

   ```bash
   ssh -t <user>@<robot-host> "cd <path-to-this-repository> && ./run.sh"
   ```

3. Check that `config/calibration.json` exists; to restore or create it, see
   [`SETUP.md`](SETUP.md#slider-calibration). The committed backup's motor IDs
   were remapped, not measured, so calibrate a new robot before its first run.

## Homing

Every slider target assumes the encoder zero is at the 0 mm hardstop; a wrong
zero shifts every target.

> [!WARNING]
> Without homing, the controller arms at the pose the drivers report,
> unverified, and immediately steps the sliders to the 114 mm centre. This is
> safe only after a trusted rehome.

Rehome after any mechanical disturbance, or whenever you do not trust where
the sliders are.

Use `rehome_auto.py`. The `--home` centring move did not always converge on
hardware.

```bash
python scripts/rehome_auto.py
```

It seeks the 0 mm hardstop at 8 rad/s for 4 s, disables the motors and waits
at `Press ENTER to zero encoders and hold at 0 mm...`. Check that every slider
is on its hardstop, then press Enter. It zeros the encoders, holds 0 mm at
1.5 rad/s until every motor is within 0.08 rad or 15 s pass, disables and
exits.

| Alternative | Behaviour |
|---|---|
| `python src/ballbot_terminal.py --home`, or `y` at the `run.sh` rehome prompt | Seeks at `--home-vel` (20 rad/s) for 5 s, zeros without confirmation, centres at 1.5 rad/s and starts control, armed. Fails after 15 s with `Sliders did not reach the 114 mm center after homing`. X is not read until control starts; only Ctrl+C stops homing. |
| `python src/ballbot_terminal.py --home-only` | Same homing, then disables and exits. Needs no TTY. |
| `python scripts/home_to_endstop.py` | Seeks at 20 rad/s for 5 s, zeros, and holds 0 mm enabled until Ctrl+C. |

- Every method takes the seek direction from `encoder_sign` in
  `calibration.json`, so home only after a measured calibration.
- `rehome_auto.py` and `home_to_endstop.py` use a fixed port
  ([`SETUP.md`](SETUP.md#serial-ports)).
- With `--no-telem`, homing proceeds with one live motor and the centring
  timeout becomes a warning.

## The run.sh menu

```bash
./run.sh
```

It runs the interpreter in `BALLBOT_PYTHON`
([`SETUP.md`](SETUP.md#python-environment)).

| Option | Runs | Hardware |
|---|---|---|
| 1 | `scripts/preflight_heuristic.py` (offline readiness check) | none |
| 2 | `scripts/calibrate_imu.py` ([`SETUP.md`](SETUP.md#imu-mounting)) | IMU |
| 3 | `scripts/calibrate_sliders.py` ([`SETUP.md`](SETUP.md#slider-calibration)) | motors, robot restrained |
| 4 | `ballbot_terminal.py --policy heuristic` | robot |
| 5 | checkpoint picker, then `--policy trained --policy-path <picked>` | robot |
| 6 | checkpoint picker, then `--preflight` | none |
| 7 | exit | |

Options 4 and 5 ask `Force a full rehome first ...? [y/N]`. Enter means no;
`y` adds `--home`.

The picker lists every `policies/*/policy_deployed.pt`; Enter picks
`BallbotVelComplexFlatDRLatency`. Folders with a `requires_smoothing` file are
tagged `legacy` and launched with [smoothing on](#trajectory-smoothing).

- `BALLBOT_POLICY` sets the checkpoint Enter picks. Give it as
  `policies/<dir>`, with or without `/policy_deployed.pt`; any other form
  falls back to entry 1.

  ```bash
  BALLBOT_POLICY=policies/BallbotVelRingCageComplexDR ./run.sh
  ```

- `BALLBOT_ARGS` is split on spaces and appended to option 4 and 5 launches,
  not option 6.

  ```bash
  BALLBOT_ARGS="--speed 0.15" ./run.sh
  ```

## Geometric controller

```bash
python src/ballbot_terminal.py --policy heuristic
```

Or `run.sh` option 4. `--heuristic-magnitude` sets how far the masses lean;
above about 0.07 the aligned slider already hits its travel limit. On a new
robot, start at 0.02:

```bash
python src/ballbot_terminal.py --policy heuristic --heuristic-magnitude 0.02
```

## Learned policy

```bash
# Land
python src/ballbot_terminal.py --policy trained \
    --policy-path policies/BallbotVelComplexFlatDRLatency/policy_deployed.pt

# Water
python src/ballbot_terminal.py --policy trained \
    --policy-path policies/BallbotVelRingCageComplexDR/policy_deployed.pt
```

`run.sh` option 5 preselects land; pick entry 3 for water. The other
checkpoints are in [`policies/README.md`](../policies/README.md).

Always pass `--policy-path`. The code default, `DEFAULT_POLICY_PATH`, still
names the retired `BallbotVelRingCageShellDR`, which then runs without the
smoother it needs.

At startup the checkpoint is shape-checked and its `env_config.yaml` is
compared with the runtime. On a mismatch the controller refuses to start
([`DEVELOPING.md`](DEVELOPING.md#env_configyaml-contract)). To run the same
checks without hardware, use `run.sh` option 6 or add `--preflight`; it prints
`PASS: ...` on success.

### Trajectory smoothing

Smoothing is off by default: the software PD smoother (`--kp 125`, `--kd 25`)
is a low-pass filter at about 1.78 Hz that the simulator does not model.
Checkpoints with a `requires_smoothing` file need it on. `run.sh` adds
`--no-no-trajectory-smoothing` for them; a hand-typed command must include
it.

With smoothing off, SPACE snaps the sliders to centre instead of ramping, and
`--kp`, `--kd` and their keys have no effect.

### Limits for the learned policy

- `--vel-limit` must be at least 10 rad/s (default 30); below that the sliders
  cannot build the lean. It is not a hard speed cap: logged motor speeds have
  exceeded it.
- `--action-scale` must be at least 0.8; leave it at 1.0. It is not a derate:
  below 0.8 the robot rolls but stops tracking the command.
- The floors apply at startup and to the keys. A P switch raises a lower
  velocity limit to 10 and resets a lower action scale to 1.0.
- To go slower, lower `--speed` or press N. For a first move on an unfamiliar
  robot, use `--speed 0.15`.
- Leave `--target-rate-limit` at 0; the simulator does not model it.

### Switching controllers (P)

P centres the sliders for 1 s, then swaps between the geometric controller
and the `--policy-path` checkpoint loaded at startup. Geometric launches load
it too, so `run.sh` option 4 switches to the retired default without its
smoother. To switch to the land checkpoint, launch with:

```bash
BALLBOT_ARGS="--policy-path policies/BallbotVelComplexFlatDRLatency/policy_deployed.pt" ./run.sh
```

`BALLBOT_ARGS` also applies to option 5, so set it for that launch only.

If the checkpoint failed to load, P is refused. To change checkpoints, quit
and relaunch.

## Flags

`python src/ballbot_terminal.py --help` prints the full list.

| Flag | Default | Note |
|---|---|---|
| `--policy` | `heuristic` | `heuristic` (geometric) or `trained` (learned) |
| `--policy-path` | retired checkpoint | Always pass it |
| `--speed` | 0.3 m/s | 0.05 to 0.7 |
| `--heuristic-magnitude` | 0.98 | 0 to 1 |
| `--action-scale` | 1.0 | 0 to 1; at least 0.8 for `trained` |
| `--vel-limit` | 30 rad/s | 0.2 to 100; at least 10 for `trained` |
| `--yaw-offset-deg` | 0 | [Aiming the drive axes](#aiming-the-drive-axes) |
| `--home`, `--home-only` | off | [Homing](#homing) |
| `--home-vel` | 20 rad/s | 0.2 to 100 |
| `--no-telem` | off | Allow silent motors; one must answer |
| `--no-no-trajectory-smoothing` | smoothing off | For `requires_smoothing` checkpoints |
| `--kp`, `--kd` | 125, 25 | Smoother gains |
| `--target-rate-limit` | 0 (off) | Leave at 0 |
| `--imu-port` | `$BALLBOT_IMU_PORT`, else autodetect | |
| `--motor-port` | `$BALLBOT_MOTOR_PORT`, else autodetect | Bridge port |
| `--motor-temp-limit`, `--drive-temp-limit` | 85 °C, 85 °C | Thermal stop |

## Keys

| Key | Action | Step, range |
|---|---|---|
| `W` / `S` | Move +X / −X | |
| `A` / `D` | Move +Y / −Y | |
| `SPACE` | Stop and centre | |
| `P` | Switch geometric / learned | |
| `N` / `M` | Speed (learned) or magnitude (geometric) down / up | speed 0.05 m/s, 0.05 to 0.7; magnitude 0.1, no effect above 1.0 |
| `[` / `]` | Action scale down / up | 0.1, 0.8 to 1.0 (learned) |
| `-` / `+` (or `_` / `=`, `C` / `V`) | Velocity limit down / up | 5 rad/s, 10 to 100 (learned), 0.2 to 100 (geometric) |
| `Q` / `E` | Rotate drive axes +15° / −15° | |
| `,` / `.` (or `<` / `>`) | kp down / up | 10, 5 to 1000 |
| `O` / `I` | kd down / up | 2, 1 to 200 |
| `X` | E-stop, latching | |
| `Ctrl+C` | Quit and disable the motors | |

W/A/S/D are fixed command-frame directions, not relative to the robot.
Diagonals are normalised; pressing a direction key cancels its opposite. Q does not quit.

### Aiming the drive axes

The command frame's yaw origin is the heading at which the IMU mounting was
captured, not the way the robot was set down. Tap W, then press Q or E (safe
while armed) until the robot drives the way you mean. The controller prints
the matching `--yaw-offset-deg` (again on exit) and shows it as `yawref`.

## Automatic stops

| Condition | Result |
|---|---|
| No movement key for 3 s | Target returns to 114 mm centre; stays armed |
| No movement key for 120 s | Motors disabled; next movement key re-arms |
| Motor feedback older than 5 s while armed | Disabled (`motor telemetry lost`); arming refused until feedback returns |
| IMU data older than 5 s | Disabled; program exits |
| Motor or drive at its limit (85 °C) | `THERMAL STOP`: disabled; clears 5 °C below the limits, then a new movement key re-arms |
| CAN send fails | Disabled |

Hold a movement key (terminal key repeat) to keep moving. With `--no-telem`,
one fresh motor is enough.

## Status line and logs

```text
policy=trained   speed=0.30m/s pos=[114, 114, 114]mm Tdrv=[31, 30, 31]C err=[1, 1, 1] dir=(+1.00,+0.00) yawref=+0 armed=True
```

- `pos`: mm from the encoder zero, motor order 0x08, 0x07, 0x06. 114 (or −114
  when that motor's `encoder_sign` is −1) is centre.
- `Tdrv`: drive temperature.
- `err`: GL40 status. 0 disabled, 1 enabled, 8+ a fault (names in
  `GL40_STATUS_NAMES`, `xiao_can/xiao_gl_motor.py`).

Each run logs to `logs/<policy>_policy_YYYYMMDD_HHMMSS.msgpack`, named after
the controller at launch: one record per policy step (at most 50 Hz) while a
movement command is active. `--no-log` turns it off; `--log-path` moves it.

Beside it, `<name>_can_errors.txt` gets a line whenever a motor's status code,
stale-feedback state or CAN send error changes, flagged `FAULT`,
`UNEXPECTED_DISABLE` or `STALE_RX`; the console prints it as `CAN ERR: ...`.
Look here first when a motor sticks. The record format is in
[`DEVELOPING.md`](DEVELOPING.md#run-logs).

## Disabling the motors

Stop any running session first, then:

```bash
python scripts/disable_motors.py
```

It sends exit-control to all three motors twice and prints
`Motors successfully disabled.` It autodetects the bridge only and exits at
once if none is found.

## Bench scripts

| Command | What it does | Motion |
|---|---|---|
| `python scripts/verify_xiao_firmware.py` | Reads the bridge firmware version; no CAN traffic | none |
| `python scripts/gl_bench.py read` | Prints position, velocity, torque, temperatures and status, drives disabled | none |
| `python scripts/gl_bench.py hold --confirm HOLD` | Enables and holds the current position for 1 s | holds |
| `python scripts/scan_can_motors.py` | Briefly enables each ID from 0x01 to 0x24 (1 rad/s limit) and lists which answer, and missing or extra IDs. IDs above 0x0F are not reported under their own ID. | brief |
| `python scripts/monitor_temps.py` | Prints temperatures every 2 s; warns at 50 °C, over limit at 60 °C | none |
| `python scripts/motor_test.py` | Drives `MOTOR_ID` (default 0x08) at 10 rad/s for 5 s toward 0 mm, if the script's `ENCODER_SIGN` (default +1) matches that motor's calibrated `encoder_sign` | yes |

Each opens the bridge port, so run them only with no control session running.
[`SETUP.md`](SETUP.md#serial-ports) lists the port each uses.
