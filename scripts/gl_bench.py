#!/usr/bin/env python3
"""Non-moving GL40/XIAO connectivity and hold test."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "xiao_can"))

from xiao_gl_motor import GL40, GL_MODE_POS_VEL, GLMotorController  # noqa: E402


MOTOR_IDS = (0x08, 0x07, 0x06)
TELEMETRY_TIMEOUT_S = 2.0


def wait_for_feedback(motor: GLMotorController) -> None:
    deadline = time.monotonic() + TELEMETRY_TIMEOUT_S
    while time.monotonic() < deadline:
        motor.disable()
        if np.all(time.monotonic() - motor.last_feedback_time < 1.0):
            return
        time.sleep(0.05)
    raise RuntimeError("Telemetry is missing from one or more GL40 motors")


def show(motor: GLMotorController) -> None:
    for idx, motor_id in enumerate(MOTOR_IDS):
        print(
            f"0x{motor_id:02X}: pos={motor.mech_pos[idx]:+.3f} rad "
            f"vel={motor.mech_vel[idx]:+.2f} rad/s "
            f"torque={motor.mech_torque[idx]:+.2f} Nm "
            f"motor={motor.motor_temp[idx]:.0f} C "
            f"drive={motor.drive_temp[idx]:.0f} C "
            f"status={motor.mode_status[idx]} error={motor.error_code[idx]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("read", "hold"), nargs="?", default="read")
    parser.add_argument("--motor-port", default=os.environ.get("BALLBOT_MOTOR_PORT"))
    parser.add_argument(
        "--confirm",
        default="",
        help="must equal HOLD before enabling the non-moving hold test",
    )
    args = parser.parse_args()

    if args.stage == "hold" and args.confirm != "HOLD":
        raise SystemExit("Refusing to enable motors; rerun with --confirm HOLD")

    setup = [[motor_id, "xiao", GL40] for motor_id in MOTOR_IDS]
    motor = GLMotorController(
        setup,
        control_mode=GL_MODE_POS_VEL,
        default_vel_limit=2.0,
        port_name=args.motor_port,
    )
    motor.should_print_send = False
    motor.should_print_recv = False
    try:
        wait_for_feedback(motor)
        show(motor)
        if args.stage == "hold":
            motor.enable()
            motor.mech_pos_ref[:] = motor.mech_pos
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                motor.motion_control_once()
                if np.any(time.monotonic() - motor.last_feedback_time >= 1.0):
                    raise RuntimeError("Motor telemetry became stale during hold")
                time.sleep(0.01)
            print("One-second current-position hold passed.")
    finally:
        motor.disable()
        motor.shutdown()


if __name__ == "__main__":
    main()
