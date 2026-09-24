#!/usr/bin/env python3
"""Live motor/drive temperature monitor. Read-only: never enables the motors.

Prints a status line periodically and warns loudly when a reading crosses
MOTOR_TEMP_LIMIT_C / DRIVE_TEMP_LIMIT_C (60 °C), below the 85 °C thermal lock in
ballbot_terminal.py. Ctrl+C to exit.
"""

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
    MOTOR_IDS,
    SliderCalibration,
)
from xiao_gl_motor import GL40, GL_MODE_POS_VEL, GLMotorController  # noqa: E402

POLL_HZ = 20.0
PRINT_INTERVAL_S = 2.0
WARN_MARGIN_C = 10.0
# Below ballbot_terminal.py's 85 °C thermal lock, so this warns first.
MOTOR_TEMP_LIMIT_C = 60.0
DRIVE_TEMP_LIMIT_C = 60.0


def main() -> None:
    # Not needed to read temps; fails early on a bad calibration.json.
    SliderCalibration.load(CALIBRATION_FILE)

    print("Connecting to GL40 motor controller on /dev/ttyACM1 (read-only, motors stay disabled)...")
    motor = GLMotorController(
        [[motor_id, "xiao", GL40] for motor_id in MOTOR_IDS],
        control_mode=GL_MODE_POS_VEL,
        default_vel_limit=0.0,
        port_name="/dev/ttyACM1",
    )
    motor.should_print_send = False
    motor.should_print_recv = False

    print(
        f"Monitoring {len(MOTOR_IDS)} motors. Limits: motor {MOTOR_TEMP_LIMIT_C:.0f}C, "
        f"drive {DRIVE_TEMP_LIMIT_C:.0f}C. Ctrl+C to stop.\n"
    )
    last_print = 0.0
    tripped = False
    try:
        while True:
            motor.motion_control_once()
            now = time.monotonic()
            if now - last_print >= PRINT_INTERVAL_S:
                last_print = now
                mtemp = [float(t) for t in motor.motor_temp]
                dtemp = [float(t) for t in motor.drive_temp]
                hot = any(t >= MOTOR_TEMP_LIMIT_C for t in mtemp) or any(
                    t >= DRIVE_TEMP_LIMIT_C for t in dtemp
                )
                warm = any(t >= MOTOR_TEMP_LIMIT_C - WARN_MARGIN_C for t in mtemp) or any(
                    t >= DRIVE_TEMP_LIMIT_C - WARN_MARGIN_C for t in dtemp
                )
                stamp = time.strftime("%H:%M:%S")
                mstr = " ".join(f"{mid:#04x}={t:+.0f}C" for mid, t in zip(MOTOR_IDS, mtemp))
                dstr = " ".join(f"{mid:#04x}={t:+.0f}C" for mid, t in zip(MOTOR_IDS, dtemp))
                if hot and not tripped:
                    tripped = True
                    print(f"[{stamp}] !!! OVER LIMIT !!! motor={{{mstr}}} drive={{{dstr}}}")
                elif hot:
                    print(f"[{stamp}] still over limit: motor={{{mstr}}} drive={{{dstr}}}")
                elif warm:
                    tripped = False
                    print(f"[{stamp}] WARM  motor={{{mstr}}} drive={{{dstr}}}")
                else:
                    tripped = False
                    print(f"[{stamp}] ok    motor={{{mstr}}} drive={{{dstr}}}")
            time.sleep(1.0 / POLL_HZ)
    except KeyboardInterrupt:
        print("\nStopping monitor (motors were never enabled).")
    finally:
        motor.shutdown()


if __name__ == "__main__":
    main()
