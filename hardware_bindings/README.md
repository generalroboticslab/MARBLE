# hardware_bindings

The C++ driver for the SYD Dynamics TM171 IMU, exposed to Python through nanobind.

```python
from hardware_bindings.imu.py_imu import IMU
```

The motors do not use this package; they use
[`xiao_can/xiao_gl_motor.py`](../xiao_can/xiao_gl_motor.py).

## Contents

| Path | Contents |
|---|---|
| `imu/imu.hpp`, `imu/imu_nanobind.cpp` | The driver and its nanobind module, `imu_nanobind`. |
| `imu/py_imu.py` | The Python wrapper, `IMU`. Loads `imu_nanobind.abi3.so` from its own directory. |
| `imu/EasyProfile/` | The vendor's protocol SDK, BSD-2-Clause. Ship [`NOTICE.md`](imu/EasyProfile/NOTICE.md) with any binary build. |
| `common/publisher.py` | Socket publisher used by the live viewer. |
| `CMakeLists.txt`, `imu/CMakeLists.txt` | Build rules. |
| `pyproject.toml` | Packaging metadata for use from another project. |

## Building

Build from the repository root with
[`scripts/prepare_orangepi.sh`](../scripts/prepare_orangepi.sh) (see
[`docs/SETUP.md`](../docs/SETUP.md)). It writes `imu_nanobind.abi3.so` and a
copy of `libnanobind-abi3.so` into `imu/`. Before the build, importing
`py_imu` raises `FileNotFoundError` naming the missing `.so`.

The control stack needs no install step. From another project, build first,
then `pip install -e hardware_bindings` (editable, so `libnanobind-abi3.so` is
found).

## The `IMU` class

`IMU(port_name=None)` opens the port at 4,000,000 baud (autodetected when
`None`, see [`docs/SETUP.md`](../docs/SETUP.md#serial-ports)) and starts reading.

| Attribute | Meaning |
|---|---|
| `counter` | Increments with each parsed sample. If it stalls for 5 s, the terminal disables the motors and exits. |
| `rotation_matrix` | 3x3 sensor orientation in the IMU's world frame. |
| `ang_vel` | Body rate in the sensor frame, rad/s. |

The other properties are bound in [`imu/imu_nanobind.cpp`](imu/imu_nanobind.cpp).
`shutdown()` closes the port. `ImuState.from_device` converts the IMU readings
into the controllers' frame; see
[`docs/DEVELOPING.md`](../docs/DEVELOPING.md#units-and-frames).

## Live viewer

```bash
python hardware_bindings/imu/py_imu.py
```

Serves a 3D view with live plots on port 8080 (viser) and publishes each sample
as msgpack over UDP to `localhost:9870`. Needs the built extension plus `viser`,
`scipy`, `orjson` and `msgpack` (all in `config/requirements-robot.txt`).

