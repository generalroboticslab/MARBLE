#!/usr/bin/env python3
"""Drive all sliders to the 0 mm hardstop and leave them there (no center)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "xiao_can"))

from ballbot_runtime import (  # noqa: E402
    CALIBRATION_FILE,
    MAX_TRAVEL_MM,
    MOTOR_IDS,
    MotorJointMapper,
    SliderCalibration,
    mm_to_rad,
    rad_to_mm,
)
from xiao_gl_motor import GL40, GL_MODE_POS_VEL, GLMotorController  # noqa: E402

HOME_VEL = 20.0
HOME_SEEK_S = 5.0


def main() -> None:
    cal = SliderCalibration.load(CALIBRATION_FILE)
    mapper = MotorJointMapper(cal)
    motor = GLMotorController(
        [[motor_id, "xiao", GL40] for motor_id in MOTOR_IDS],
        control_mode=GL_MODE_POS_VEL,
        default_vel_limit=HOME_VEL,
        port_name="/dev/ttyACM1",
    )
    motor.should_print_send = False
    motor.should_print_recv = False

    print(
        f"Homing all sliders toward the 0 mm hardstop at {HOME_VEL:.0f} rad/s "
        f"for {HOME_SEEK_S:.0f} s..."
    )
    for _ in range(30):
        motor.motion_control_once()
        time.sleep(0.02)

    motor.enable()
    time.sleep(0.05)
    inward_delta = float(mm_to_rad(-MAX_TRAVEL_MM))
    for joint_name in mapper.calibration.joints:
        mapping = mapper.calibration.joints[joint_name]
        idx = mapper.motor_index[mapping.motor_id]
        motor.mech_pos_ref[idx] = (
            float(motor.mech_pos[idx]) + mapping.encoder_sign * inward_delta
        )
    motor.mech_vel_ref[:] = HOME_VEL

    deadline = time.monotonic() + HOME_SEEK_S
    while time.monotonic() < deadline:
        motor.motion_control_once()
        time.sleep(0.01)

    print("Seek complete; zeroing encoders at 0 mm hardstop.")
    motor.set_pos_zero()
    time.sleep(0.3)
    for _ in range(20):
        motor.motion_control_once()
        time.sleep(0.02)

    motor.mech_pos_ref[:] = 0.0
    motor.mech_vel_ref[:] = HOME_VEL
    for _ in range(25):
        motor.motion_control_once()
        time.sleep(0.02)

    for idx, motor_id in enumerate(MOTOR_IDS):
        mm = float(rad_to_mm(float(motor.mech_pos[idx])))
        print(f"  Motor 0x{motor_id:02X}: {mm:+.1f} mm ({motor.mech_pos[idx]:+.3f} rad)")

    print("Holding at 0 mm hardstop (enabled). Ctrl+C to disable and exit.")
    try:
        while True:
            motor.motion_control_once()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nDisabling...")
    finally:
        motor.disable()
        time.sleep(0.1)
        motor.shutdown()


if __name__ == "__main__":
    main()
