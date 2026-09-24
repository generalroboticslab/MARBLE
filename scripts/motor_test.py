#!/usr/bin/env python3

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "xiao_can"))

from ballbot_runtime import MAX_TRAVEL_MM, mm_to_rad
from xiao_gl_motor import GL40, GL_MODE_POS_VEL, GLMotorController

MOTOR_ID = 0x08       # <-- change this
VEL = 10.0            # rad/s
DRIVE_TIME = 5.0      # seconds
ENCODER_SIGN = 1      # change to -1 if direction is reversed


def main():
    motor = GLMotorController(
        [[MOTOR_ID, "xiao", GL40]],
        control_mode=GL_MODE_POS_VEL,
        default_vel_limit=VEL,
        port_name="/dev/ttyACM1",
    )

    motor.should_print_send = False
    motor.should_print_recv = False

    # Get current motor state
    for _ in range(30):
        motor.motion_control_once()
        time.sleep(0.02)

    motor.enable()
    time.sleep(0.05)

    # Command far toward the 0 mm hardstop (if ENCODER_SIGN is right for this motor)
    delta = float(mm_to_rad(-MAX_TRAVEL_MM))

    motor.mech_pos_ref[0] = (
        float(motor.mech_pos[0]) + ENCODER_SIGN * delta
    )
    motor.mech_vel_ref[0] = VEL

    print(f"Driving motor 0x{MOTOR_ID:02X} for {DRIVE_TIME} seconds...")

    deadline = time.monotonic() + DRIVE_TIME

    while time.monotonic() < deadline:
        motor.motion_control_once()
        time.sleep(0.01)

    motor.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()