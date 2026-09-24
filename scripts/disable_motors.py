#!/usr/bin/env python3
"""Connect to the XIAO CAN bridge and disable all motors on startup."""

import sys
import time
from pathlib import Path

# Resolve paths so it works whether run from scripts/ or repo root
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
for d in (ROOT / "src", ROOT / "xiao_can", PROJECT_ROOT / "src", PROJECT_ROOT / "xiao_can"):
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))

from ballbot_runtime import MOTOR_IDS
from xiao_gl_motor import GL40, GL_MODE_POS_VEL, GLMotorController

def main():
    print("Disabling all motors on startup...")
    motor_setup = [[motor_id, "xiao", GL40] for motor_id in MOTOR_IDS]
    
    try:
        mc = GLMotorController(
            motor_setup,
            control_mode=GL_MODE_POS_VEL,
            # Up to 30 s of retries if a found port fails to open or the MCP2515 fails to start;
            # exits at once if no bridge is found.
            connect_timeout_s=30.0
        )
    except Exception as e:
        print(f"Failed to connect to motor bridge: {e}")
        sys.exit(1)
        
    mc.disable()
    time.sleep(0.2)
    mc.disable()
    print("Motors successfully disabled.")
    mc.shutdown()

if __name__ == "__main__":
    main()
