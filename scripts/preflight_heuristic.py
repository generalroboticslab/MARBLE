#!/usr/bin/env python3
"""Hardware-free readiness checks for the physical heuristic controller."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ballbot_runtime import (
    CALIBRATION_FILE,
    HEURISTIC_LOWER_M,
    HEURISTIC_UPPER_M,
    MOTOR_IDS,
    R_BASE_SENSOR,
    SAFE_MAX_TRAVEL_MM,
    SAFE_MIN_TRAVEL_MM,
    HeuristicPolicy,
    ImuState,
    JointMapping,
    JOINT_NAMES,
    MotorJointMapper,
    SliderCalibration,
    rad_to_mm,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline heuristic deployment checks")
    parser.add_argument(
        "--require-calibration",
        action="store_true",
        help="fail instead of warn when calibration.json is absent",
    )
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []

    if sys.platform.startswith("linux") and sys.version_info[:2] != (3, 12):
        failures.append(
            f"Python {sys.version.split()[0]} is active; the Orange Pi binding is built for 3.12"
        )

    try:
        import serial  # noqa: F401
    except ImportError:
        failures.append("pyserial is missing; install requirements-robot.txt")

    if not np.allclose(R_BASE_SENSOR.T @ R_BASE_SENSOR, np.eye(3), atol=1e-6):
        failures.append("fixed IMU quaternion does not produce an orthonormal rotation")

    imu_state = ImuState(np.eye(3), np.zeros(3))
    policy = HeuristicPolicy()
    outputs = np.stack(
        [
            policy.act(direction, 1.0, imu_state, np.zeros(3), np.zeros(3))
            for direction in (
                np.array([1.0, 0.0]),
                np.array([-1.0, 0.0]),
                np.array([0.0, 1.0]),
                np.array([0.0, -1.0]),
            )
        ]
    )
    if not np.isfinite(outputs).all():
        failures.append("heuristic cardinal-direction test produced non-finite targets")
    if np.any(outputs < HEURISTIC_LOWER_M) or np.any(outputs > HEURISTIC_UPPER_M):
        failures.append("heuristic cardinal-direction test exceeded policy limits")
    if any(np.allclose(outputs[i], outputs[j]) for i in range(4) for j in range(i)):
        failures.append("two cardinal directions produced the same slider target")

    synthetic = SliderCalibration(
        {
            name: JointMapping(
                motor_id=MOTOR_IDS[idx],
                encoder_sign=1 if idx != 1 else -1,
                joint_to_travel_sign=1 if idx != 2 else -1,
            )
            for idx, name in enumerate(JOINT_NAMES)
        }
    )
    mapper = MotorJointMapper(synthetic)
    for target in (np.full(3, -1.0), np.zeros(3), np.full(3, 1.0)):
        travel_mm = np.abs(rad_to_mm(mapper.motor_targets_from_joint_position(target)))
        if np.any(travel_mm < SAFE_MIN_TRAVEL_MM - 1e-5) or np.any(
            travel_mm > SAFE_MAX_TRAVEL_MM + 1e-5
        ):
            failures.append("motor mapper escaped safe hardstop/feedback limits")
            break

    firmware = PROJECT_ROOT / "xiao_can" / "xiao_can_bridge" / "xiao_can_bridge.ino"
    if not firmware.exists():
        failures.append(f"XIAO bridge firmware is missing: {firmware}")
    else:
        source = firmware.read_text(encoding="utf-8")
        setup = source.partition("void setup()")[2].partition(
            "void processSerialCommand"
        )[0]
        if re.search(r"byte\s+ping_buf\s*\[[^\]]+\]\s*=\s*\{[^}]*0xFE", setup):
            failures.append("XIAO startup still contains an encoder-zero command")
        if "BALLBOT_XIAO_BRIDGE_V3" not in source:
            failures.append("XIAO firmware does not expose the required version handshake")

    native_files = (
        PROJECT_ROOT / "hardware_bindings" / "imu" / "imu_nanobind.abi3.so",
        PROJECT_ROOT / "hardware_bindings" / "imu" / "libnanobind-abi3.so",
    )
    missing_native = [str(path) for path in native_files if not path.exists()]
    if missing_native:
        message = "IMU native binding is not built: " + ", ".join(missing_native)
        if sys.platform.startswith("linux"):
            failures.append(message)
        else:
            warnings.append(message)
    elif sys.platform.startswith("linux"):
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from hardware_bindings.imu.py_imu import IMU  # noqa: F401
        except Exception as exc:
            failures.append(f"IMU native binding failed to import: {exc}")

    if CALIBRATION_FILE.exists():
        try:
            SliderCalibration.load(CALIBRATION_FILE)
        except Exception as exc:
            failures.append(f"calibration.json is invalid: {exc}")
    else:
        message = "calibration.json is absent; create it on the assembled robot"
        (failures if args.require_calibration else warnings).append(message)

    for script in ("run.sh", "scripts/prepare_orangepi.sh"):
        path = PROJECT_ROOT / script
        if not path.exists():
            failures.append(f"missing launcher: {path}")
        elif not os.access(path, os.X_OK):
            failures.append(f"launcher is not executable: {path}")

    for message in warnings:
        print(f"WARN: {message}")
    for message in failures:
        print(f"FAIL: {message}")
    if failures:
        raise SystemExit(1)
    print("PASS: heuristic policy files and safety bounds are ready")


if __name__ == "__main__":
    main()
