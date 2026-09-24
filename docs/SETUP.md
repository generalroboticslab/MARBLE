# Setup

Run these steps on the robot computer, from the repository root. Then see
[`OPERATIONS.md`](OPERATIONS.md) to run the robot, and
[`validation.md`](validation.md) for a new or rebuilt robot.

## Python environment

The stack needs Python 3.12 with its headers and `libpython3.12`; the IMU
binding is built against it. With micromamba:

```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)   # no root; open a new shell afterwards
micromamba create -n py312 python=3.12 -y && micromamba activate py312
export BALLBOT_PYTHON="$(command -v python)"
```

Or a venv on a system Python 3.12
(`sudo apt install python3.12-dev python3.12-venv`):

```bash
python3.12 -m venv "$HOME/marble-py312"
export BALLBOT_PYTHON="$HOME/marble-py312/bin/python"
```

Install PyTorch separately; only `--policy trained`, `--preflight` and
`scripts/policy_sim_check.py` need it. `prepare_orangepi.sh` installs the rest.

```bash
"$BALLBOT_PYTHON" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The `torch.jit.load` `FutureWarning` is harmless.

## Build the IMU binding

Prerequisites:

- CMake 3.15 or later, Ninja, and `g++` with C++17.
- gcc-12. `CMakeLists.txt` pins the C compiler to
  `/usr/bin/aarch64-linux-gnu-gcc-12` (aarch64) or `/usr/bin/gcc-12` (x86_64)
  when present. Override with `CC` or `-DCMAKE_C_COMPILER=<path>`.
- vcpkg, which needs `git`, `curl`, `zip`, `unzip`, `tar` and `pkg-config`. It
  installs `eigen3`, `cserialport` and `fmt` from `vcpkg.json` at configure
  time.

  ```bash
  git clone https://github.com/microsoft/vcpkg "$HOME/vcpkg" && "$HOME/vcpkg/bootstrap-vcpkg.sh" -disableMetrics
  export VCPKG_ROOT="$HOME/vcpkg"
  ```

- `nanobind`, from the requirements file.

Then build and check:

```bash
./scripts/prepare_orangepi.sh
```

It installs the requirements, builds `imu_nanobind` with CMake and the vcpkg
toolchain, and runs the preflight. The preflight must end with
`PASS: heuristic policy files and safety bounds are ready`.
`WARN: calibration.json is absent` is expected until
[slider calibration](#slider-calibration). If a copy lost the file modes, run
`chmod +x run.sh scripts/prepare_orangepi.sh`.

`CMakePresets.json` is for editors (`$HOME/repo/` paths, Debug). The script
does not use it.

## Serial ports

The XIAO bridge and the IMU must be different devices; `ballbot_terminal.py`
refuses equal ports. Autodetection:

- XIAO bridge: the single port whose USB description, manufacturer, product or
  hardware ID contains `xiao`, `seeed`, `samd` or `arduino`. Failing that, the
  single non-STMicroelectronics `/dev/ttyACM*` or `/dev/ttyUSB*`. Zero or
  several candidates is an error that asks for `--motor-port`.
- IMU: the port described as `STMicroelectronics Virtual COM Port`.

If detection is ambiguous, set the ports:

```bash
export BALLBOT_IMU_PORT=/dev/ttyACM0
export BALLBOT_MOTOR_PORT=/dev/ttyACM1
```

| Script | Port source, in order |
|---|---|
| `src/ballbot_terminal.py` | `--motor-port` and `--imu-port`; `BALLBOT_MOTOR_PORT` and `BALLBOT_IMU_PORT`; autodetect |
| `scripts/calibrate_sliders.py`, `scripts/gl_bench.py` | `--motor-port`; `BALLBOT_MOTOR_PORT`; autodetect |
| `scripts/calibrate_imu.py` | `--imu-port`; `BALLBOT_IMU_PORT`; autodetect |
| `scripts/verify_xiao_firmware.py` | `--motor-port`; autodetect |
| `scripts/disable_motors.py`, `scripts/scan_can_motors.py` | autodetect only |
| `scripts/rehome_auto.py`, `scripts/home_to_endstop.py`, `scripts/monitor_temps.py`, `scripts/motor_test.py` | fixed `/dev/ttyACM1` (edit `port_name` in the script) |

### Environment variables

| Variable | Read by | Default |
|---|---|---|
| `BALLBOT_PYTHON` | `run.sh`, `scripts/prepare_orangepi.sh` | `/home/orangepi/repo/micromamba/envs/py312/bin/python` (the reference robot's) |
| `VCPKG_ROOT` | `scripts/prepare_orangepi.sh` | `/home/orangepi/repo/vcpkg` |

`BALLBOT_POLICY` and `BALLBOT_ARGS` configure `run.sh`
([`OPERATIONS.md`](OPERATIONS.md#the-runsh-menu)).

## Flash the bridge firmware

With motor power off, flash and verify the XIAO as in
[`FLASH_XIAO.md`](FLASH_XIAO.md).

## Calibration

- Slider mapping: `config/calibration.json` (gitignored), written by
  `scripts/calibrate_sliders.py`. Both controllers and homing need it.
- IMU mounting: the constant `IMU_QUAT_BASE_SENSOR_WXYZ` in
  `src/ballbot_runtime.py`. `scripts/calibrate_imu.py` checks it; no script
  writes it.

### Slider calibration

`config/calibration.robot-backup.json` is a reference only: its motor IDs were
remapped for this release, not measured
([`PROVENANCE.md`](../PROVENANCE.md#motor-ids)). Run this procedure once on the
restrained robot before its first control run. Recalibrate after replacing motors, sliders or other mechanical parts, or when
[section 3 of `validation.md`](validation.md#3-slider-zero-and-mapping) fails.

> [!CAUTION]
> Secure the sphere before you apply motor power. Keep an emergency power
> switch within reach.

1. Start `./run.sh` over `ssh -t`
   ([`OPERATIONS.md`](OPERATIONS.md#before-each-session)).
2. Option 1: offline readiness check.
3. Option 2: IMU mounting check ([below](#imu-mounting)).
4. Put the secured robot in the reference pose: +Y forward, +Z up.
5. Option 3, with motor power on:
   1. The drives are disabled. Push all three sliders by hand to the 0 mm
      fully extended hardstop, press Enter, and type `ZERO`. Anything else
      cancels without writing.
   2. For each motor (0x08, 0x07, 0x06), the script moves the slider 30 mm at
      5 rad/s to find the encoder sign, trying the other sign if it hits the
      hardstop. Answer which marked axis (X, Y or Z) the slider follows and
      whether the mass moved toward + or −. The slider returns to 0 mm.
   3. It writes `config/calibration.json` (`--output` to change).

   It aborts at 70 °C motor or 80 °C drive temperature, or on lost feedback.

   Do not use `scripts/rehome_auto.py` instead of pushing by hand on a new
   build, after a motor was replaced or remounted, or after a failed
   [slider check](validation.md#3-slider-zero-and-mapping). It takes the
   direction from the existing `config/calibration.json`; with a wrong sign it
   drives the slider away from the hardstop at 8 rad/s for 4 s.
6. Save the result as the backup:

   ```bash
   cp config/calibration.json config/calibration.robot-backup.json
   ```

To restore it later (for example on a reflashed Orange Pi):

```bash
cp -n config/calibration.robot-backup.json config/calibration.json
```

### IMU mounting

`IMU_QUAT_BASE_SENSOR_WXYZ` is the `imu_site` of the MuJoCo model `Sim_Model`
(in every `policies/*/robot.xml`), rotated +90° about the sensor Z axis.

| Command | What it does |
|---|---|
| `python scripts/calibrate_imu.py` | Prints the base axes and angular velocity in world coordinates, live |
| `python scripts/calibrate_imu.py --viz` | Live 3D view with tilt from vertical, on port 8080 (`--viz-port`) |
| `python scripts/calibrate_imu.py --capture-upright` | Solves the mounting rotation from a held reference pose |

Check it after any remount: at the reference pose the tilt reads about 0°. To
re-solve it:

1. Run `--capture-upright`. Hold the robot with the shell on the ground, the
   chassis vertical and +Y toward forward. Keep it still and press Enter.
2. The script averages 200 samples (`--samples`), warning on motion (median
   gyro above 0.05 rad/s, or more than 1° spread). It prints a quaternion for
   `IMU_QUAT_BASE_SENSOR_WXYZ` and compares it with the CAD mounting in 90°
   steps. The capture heading becomes the world yaw origin.
3. Nothing is written. Edit the constant, re-run `--viz`, and confirm the tilt
   reads about 0°.
