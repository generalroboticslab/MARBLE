#!/usr/bin/env python3
"""First-zero and manually identify each motor in the MuJoCo joint convention."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
for d in (ROOT / "src", ROOT / "xiao_can", PROJECT_ROOT / "src", PROJECT_ROOT / "xiao_can"):
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))

from ballbot_runtime import (  # noqa: E402
    CALIBRATION_FILE,
    JOINT_NAMES,
    JOINT_POSITIVE_AXIS,
    JointMapping,
    MOTOR_IDS,
    SliderCalibration,
    mm_to_rad,
)
from xiao_gl_motor import GL40, GL_MODE_POS_VEL, GLMotorController  # noqa: E402


TEST_TRAVEL_MM = 30.0
TEST_VELOCITY_RAD_S = 5.0
CONTACT_TORQUE_NM = 0.5
TELEMETRY_TIMEOUT_S = 5.0


def ask_choice(prompt: str, choices: set[str]) -> str:
    while True:
        answer = input(prompt).strip().upper()
        if answer in choices:
            return answer
        print(f"Enter one of: {', '.join(sorted(choices))}")


def feedback_fresh(motor: GLMotorController, idx: int) -> bool:
    return time.monotonic() - float(motor.last_feedback_time[idx]) < TELEMETRY_TIMEOUT_S


def wait_for_all_feedback(
    motor: GLMotorController,
    timeout_s: float = TELEMETRY_TIMEOUT_S,
    *,
    hold_disabled: bool = False,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if hold_disabled:
            motor.disable()
        else:
            motor.motion_control_once()
        if all(feedback_fresh(motor, idx) for idx in range(len(MOTOR_IDS))):
            return
        time.sleep(0.02)
    raise RuntimeError("Telemetry is missing from one or more motors")


def move_one(
    motor: GLMotorController,
    idx: int,
    target_rad: float,
    timeout_s: float = 3.0,
    allow_contact: bool = False,
) -> bool:
    start = time.monotonic()
    contact_since = 0.0
    motor.mech_pos_ref[idx] = target_rad
    while time.monotonic() - start < timeout_s:
        motor.motion_control_once()
        now = time.monotonic()
        if now - start > TELEMETRY_TIMEOUT_S and not feedback_fresh(motor, idx):
            raise RuntimeError(f"No telemetry from motor 0x{int(MOTOR_IDS[idx]):02X}")
        if feedback_fresh(motor, idx):
            if float(motor.motor_temp[idx]) >= 70.0 or float(motor.drive_temp[idx]) >= 80.0:
                raise RuntimeError("Temperature limit reached during calibration")
            contact = abs(float(motor.mech_torque[idx])) >= CONTACT_TORQUE_NM
            if contact:
                if contact_since == 0.0:
                    contact_since = now
                elif now - contact_since >= 0.2:
                    motor.mech_pos_ref[idx] = float(motor.mech_pos[idx])
                    motor.motion_control_once()
                    return allow_contact
            else:
                contact_since = 0.0
            if abs(float(motor.mech_pos[idx]) - target_rad) < 0.08:
                return True
        time.sleep(0.01)
    return False


def detect_encoder_sign(motor: GLMotorController, idx: int) -> int:
    test_rad = float(mm_to_rad(TEST_TRAVEL_MM))
    for sign in (1, -1):
        print(
            f"  Testing motor encoder direction {sign:+d} "
            f"({TEST_TRAVEL_MM:.0f} mm at {TEST_VELOCITY_RAD_S:.0f} rad/s)..."
        )
        motor.mech_pos_ref[:] = motor.mech_pos[:]
        succeeded = move_one(motor, idx, sign * test_rad)
        moved = abs(float(motor.mech_pos[idx])) > 0.5 * test_rad
        if succeeded and moved:
            return sign
        print("  Direction reached the 0 mm hardstop; trying the opposite direction.")
        motor.mech_pos_ref[idx] = 0.0
        motor.motion_control_once()
        time.sleep(0.2)
    raise RuntimeError(
        f"Motor 0x{int(MOTOR_IDS[idx]):02X} did not move away from its zero hardstop"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Map physical sliders to BallbotVel joints")
    parser.add_argument("--motor-port", default=os.environ.get("BALLBOT_MOTOR_PORT"))
    parser.add_argument("--output", default=str(CALIBRATION_FILE))
    args = parser.parse_args()

    print(
        "\nSLIDER CALIBRATION\n"
        "The sphere must be in the marked MuJoCo reference pose: +Y forward, +Z up.\n"
        "This procedure writes the motor-to-Slider-5/6/7 mapping required by both policies.\n"
        "Motors will be software-disabled so you can hand-move them; bus power can stay on."
    )

    setup = [[motor_id, "xiao", GL40] for motor_id in MOTOR_IDS]
    motor = GLMotorController(
        setup,
        control_mode=GL_MODE_POS_VEL,
        default_vel_limit=TEST_VELOCITY_RAD_S,
        port_name=args.motor_port,
    )
    motor.should_print_send = False
    motor.should_print_recv = False

    mappings: dict[str, JointMapping] = {}
    unused_axes = {"X", "Y", "Z"}
    try:
        print("\nWaiting for fresh telemetry, then disabling drives...")
        wait_for_all_feedback(motor, hold_disabled=True)
        motor.disable()
        print(
            "Drives disabled. Manually push ALL sliders to the 0 mm fully extended "
            "hardstop (robot stays restrained, power can stay on)."
        )
        input("Press ENTER when the sliders are at the hardstop... ")
        confirmation = input(
            "Type ZERO to confirm all three sliders are at that hardstop: "
        ).strip()
        if confirmation != "ZERO":
            raise SystemExit("Calibration cancelled; no file was written.")

        # Re-poll while still disabled so zero is taken at the hand-set pose.
        wait_for_all_feedback(motor, hold_disabled=True)
        print("\nPersisting the manually confirmed hardstop as encoder position 0...")
        motor.set_pos_zero()
        time.sleep(0.3)
        motor.enable()
        motor.mech_pos_ref[:] = 0.0
        motor.mech_vel_ref[:] = TEST_VELOCITY_RAD_S
        for _ in range(25):
            motor.motion_control_once()
            time.sleep(0.02)

        for idx, motor_id in enumerate(MOTOR_IDS):
            print(f"\nMotor 0x{motor_id:02X}")
            encoder_sign = detect_encoder_sign(motor, idx)

            axis = ask_choice(
                f"  Which marked model axis does this slider follow ({'/'.join(sorted(unused_axes))})? ",
                unused_axes,
            )
            unused_axes.remove(axis)
            movement_answer = ask_choice(
                f"  While moving away from the 0 mm hardstop (0 to {TEST_TRAVEL_MM:.0f} mm), "
                f"did the internal mass move toward +{axis} or -{axis}? (+/-): ",
                {"+", "-"},
            )
            movement_sign = 1 if movement_answer == "+" else -1

            joint_name = next(
                name for name, (joint_axis, _) in JOINT_POSITIVE_AXIS.items()
                if joint_axis == axis
            )
            _, simulation_positive_sign = JOINT_POSITIVE_AXIS[joint_name]
            joint_to_travel_sign = (
                1 if movement_sign == simulation_positive_sign else -1
            )
            mappings[joint_name] = JointMapping(
                motor_id=motor_id,
                encoder_sign=encoder_sign,
                joint_to_travel_sign=joint_to_travel_sign,
            )
            print(
                f"  Saved: {joint_name} -> motor 0x{motor_id:02X}, "
                f"encoder_sign={encoder_sign:+d}, "
                f"joint_to_travel_sign={joint_to_travel_sign:+d}"
            )

            print("  Returning this slider to the 0 mm hardstop...")
            motor.mech_pos_ref[:] = motor.mech_pos[:]
            if not move_one(motor, idx, 0.0, timeout_s=4.0, allow_contact=True):
                raise RuntimeError(f"Motor 0x{motor_id:02X} failed to return to zero")

        calibration = SliderCalibration(mappings)
        output = Path(args.output)
        calibration.save(output)
        print(f"\nCalibration saved: {output}")
        print("Back up this file with: cp config/calibration.json config/calibration.robot-backup.json")
        print("\nJoint mapping:")
        for joint_name in JOINT_NAMES:
            mapping = mappings[joint_name]
            print(
                f"  {joint_name}: motor=0x{mapping.motor_id:02X}, "
                f"encoder={mapping.encoder_sign:+d}, "
                f"travel={mapping.joint_to_travel_sign:+d}"
            )
    finally:
        motor.disable()
        motor.shutdown()


if __name__ == "__main__":
    main()
