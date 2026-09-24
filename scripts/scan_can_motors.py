#!/usr/bin/env python3
"""Probe GL40 motor IDs on the XIAO CAN bridge and print who answers."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / "src", ROOT / "xiao_can"):
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))

from xiao_gl_motor import (  # noqa: E402
    CMD_CLEAR_ERRORS,
    CMD_ENTER_CONTROL,
    CMD_EXIT_CONTROL,
    GL40,
    GL_MODE_POS_VEL,
    GLMotorController,
    _get_can_id,
)

EXPECTED = (0x08, 0x07, 0x06)
CANDIDATES = list(range(1, 0x25))


def main() -> None:
    print("Connecting to XIAO CAN bridge...")
    mc = GLMotorController(
        [[mid, "xiao", GL40] for mid in CANDIDATES],
        control_mode=GL_MODE_POS_VEL,
        default_vel_limit=1.0,
        connect_timeout_s=20.0,
    )
    for mid in CANDIDATES:
        mc._send_can(_get_can_id(mid, GL_MODE_POS_VEL), CMD_CLEAR_ERRORS)
        time.sleep(0.005)

    print("Enabling each ID briefly and listening for feedback...")
    seen: dict[int, dict] = {}
    try:
        for idx, mid in enumerate(CANDIDATES):
            mc._send_can(_get_can_id(mid, GL_MODE_POS_VEL), CMD_ENTER_CONTROL)
            time.sleep(0.03)
            mc.mech_pos_ref[idx] = float(mc.mech_pos[idx])
            mc.mech_vel_ref[idx] = 1.0
            t0 = time.monotonic()
            while time.monotonic() - t0 < 0.35:
                mc.motion_control_once()
                time.sleep(0.001)
            stamp = float(mc.last_feedback_time[idx])
            if stamp > 0 and time.monotonic() - stamp < 0.5:
                seen[mid] = {
                    "err": int(mc.error_code[idx]),
                    "pos": float(mc.mech_pos[idx]),
                    "vel": float(mc.mech_vel[idx]),
                    "temp_m": float(mc.motor_temp[idx]),
                    "temp_d": float(mc.drive_temp[idx]),
                }
            mc._send_can(_get_can_id(mid, GL_MODE_POS_VEL), CMD_EXIT_CONTROL)
            time.sleep(0.001)
    finally:
        for mid in CANDIDATES:
            mc._send_can(_get_can_id(mid, GL_MODE_POS_VEL), CMD_EXIT_CONTROL)
        mc.disable()
        mc.shutdown()

    print()
    print("=== CAN scan results ===")
    if not seen:
        print("No motors responded.")
    else:
        for mid in sorted(seen):
            s = seen[mid]
            print(
                f"  0x{mid:02X}  err={s['err']}  pos={s['pos']:+.3f} rad  "
                f"vel={s['vel']:+.2f}  Tmot={s['temp_m']:.0f}C  Tdrv={s['temp_d']:.0f}C"
            )
    print()
    print("Expected ballbot IDs:", ", ".join(f"0x{m:02X}" for m in EXPECTED))
    missing = [m for m in EXPECTED if m not in seen]
    extra = [m for m in seen if m not in EXPECTED]
    if missing:
        print("Missing expected:", ", ".join(f"0x{m:02X}" for m in missing))
    else:
        print("All expected IDs present.")
    if extra:
        print("Extra IDs on bus:", ", ".join(f"0x{m:02X}" for m in sorted(extra)))


if __name__ == "__main__":
    main()
