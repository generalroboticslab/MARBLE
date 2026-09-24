#!/usr/bin/env python3
"""Confirm the XIAO bridge is running the firmware this checkout expects.

Read-only: opens the serial port, asks the bridge for its version string, and
prints a pass/fail. Sends no CAN traffic, so it is safe to run with motor power
off (and that is the recommended way to run it).

    python scripts/verify_xiao_firmware.py [--motor-port /dev/ttyACM0]
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "xiao_can"))

import serial  # noqa: E402
from xiao_gl_motor import BRIDGE_VERSION, _autodetect_port  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motor-port", default=None,
                        help="Serial port of the XIAO bridge (default: autodetect)")
    args = parser.parse_args()

    try:
        port = args.motor_port or _autodetect_port()
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1

    print(f"Checking XIAO bridge on {port}...")
    try:
        ser = serial.Serial(port, 115200, timeout=0.01, write_timeout=0.1)
    except Exception as exc:
        print(f"FAIL: could not open {port}: {exc}")
        return 1

    with ser:
        time.sleep(2.0)  # nRF52840 re-enumerates its USB serial after reset
        ser.reset_input_buffer()
        ser.write(b"V\n")
        message = ""
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if ser.in_waiting:
                message += ser.read(ser.in_waiting).decode("ascii", errors="ignore")
            if BRIDGE_VERSION in message:
                break
            time.sleep(0.02)

    if BRIDGE_VERSION in message:
        print(f"PASS: bridge reports {BRIDGE_VERSION}. Firmware is up to date.")
        return 0

    reported = " | ".join(line.strip() for line in message.splitlines() if line.strip())
    print(f"FAIL: expected {BRIDGE_VERSION}, bridge said: {reported or '(no response)'}")
    print("The XIAO still has old firmware. Re-flash it: see docs/FLASH_XIAO.md")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
