"""
CubeMars GL40 II motor controller through the Seeed XIAO USB-serial CAN bridge.

The bridge firmware is xiao_can/xiao_can_bridge/xiao_can_bridge.ino. When the port
opens, the bridge must report BRIDGE_VERSION, at boot or in reply to a "V" query
(see docs/FLASH_XIAO.md).
"""

import time
import threading
import numpy as np
import serial
from serial.tools import list_ports
import struct
import glob
import sys

GL40 = 40
GL_MODE_MIT = 0
GL_MODE_POS_VEL = 1
GL_MODE_VEL = 2
BRIDGE_VERSION = "BALLBOT_XIAO_BRIDGE_V3"

# Scaling Limits (must match the GL40 II datasheet)
P_MIN, P_MAX = -12.5, 12.5
V_MIN, V_MAX = -120.0, 120.0
T_MIN, T_MAX = -10.0, 10.0

# Universal CAN command payloads (byte 7 determines the command)
CMD_CLEAR_ERRORS    = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFB"
CMD_ENTER_CONTROL   = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFC"
CMD_EXIT_CONTROL    = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFD"
CMD_SET_ZERO_POS    = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFE"

# CubeMars GL II feedback nibble (data[0] >> 4). 0/1 are status; >=8 are faults.
GL40_STATUS_NAMES = {
    0: "disable",
    1: "enable",
    8: "over_voltage",
    9: "under_voltage",
    10: "over_current",
    11: "mos_over_temp",
    12: "motor_over_temp",
    13: "comm_loss",
    14: "overload",
}
GL40_FAULT_CODES = frozenset({8, 9, 10, 11, 12, 13, 14})


def gl40_status_name(code: int) -> str:
    return GL40_STATUS_NAMES.get(int(code), f"unknown_{int(code)}")


def _get_can_id(motor_id, mode):
    """Encode mode into the CAN arbitration ID: (mode << 8) | motor_id."""
    return (mode << 8) | (motor_id & 0xFF)


def uint_to_float(x_int, x_min, x_max, bits):
    span = x_max - x_min
    return float(x_int) * span / float((1 << bits) - 1) + x_min


def _autodetect_port():
    infos = list(list_ports.comports())
    keywords = ("xiao", "seeed", "samd", "arduino")
    preferred = [
        info.device
        for info in infos
        if any(
            keyword in " ".join(
                (
                    info.description or "",
                    info.manufacturer or "",
                    info.product or "",
                    info.hwid or "",
                )
            ).lower()
            for keyword in keywords
        )
    ]
    if len(preferred) == 1:
        return preferred[0]
    if len(preferred) > 1:
        raise RuntimeError(
            f"Multiple possible XIAO serial ports found: {preferred}. "
            "Pass --motor-port explicitly."
        )

    patterns = (
        ('/dev/cu.usbmodem*', '/dev/tty.usbmodem*')
        if sys.platform.startswith('darwin')
        else ('/dev/ttyACM*', '/dev/ttyUSB*')
    )
    candidates = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    imu_ports = {
        info.device
        for info in infos
        if "stmicroelectronics" in (info.manufacturer or "").lower()
        or "stmicroelectronics" in (info.description or "").lower()
    }
    candidates = [path for path in candidates if path not in imu_ports]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(
            "No XIAO serial-to-CAN bridge found. Pass --motor-port explicitly."
        )
    raise RuntimeError(
        f"Ambiguous motor serial ports: {candidates}. Pass --motor-port explicitly."
    )


class GLMotorController:
    def __init__(self, motor_setup, control_mode=GL_MODE_POS_VEL,
                 default_kp=0.0, default_kd=0.1, default_vel_limit=2.0,
                 port_name=None, connect_timeout_s=10.0):
        self.control_mode = control_mode
        self.can_ids = np.array([e[0] for e in motor_setup], dtype=np.uint32)
        self.motor_types = np.array([e[2] for e in motor_setup], dtype=np.uint8)
        self.num_motors = len(self.motor_types)

        n = self.num_motors
        self.mech_pos_ref = np.zeros(n, dtype=np.float32)
        self.mech_vel_ref = np.zeros(n, dtype=np.float32)
        self.kp = np.full(n, default_kp, dtype=np.float32)
        self.kd = np.full(n, default_kd, dtype=np.float32)

        self.mech_pos = np.zeros(n, dtype=np.float32)
        self.mech_vel = np.zeros(n, dtype=np.float32)
        self.mech_torque = np.zeros(n, dtype=np.float32)
        self.drive_temp = np.zeros(n, dtype=np.float32)
        self.motor_temp = np.zeros(n, dtype=np.float32)
        self.mode_status = np.zeros(n, dtype=np.uint8)
        self.error_code = np.zeros(n, dtype=np.uint8)
        self.last_feedback_time = np.zeros(n, dtype=np.float64)

        self.should_print_send = False
        self.should_print_recv = False
        self.last_send_error = None

        # ── Serial init with boot handshake ──
        self.ser = None
        self.port_name = None
        self._buffer = ""
        connect_deadline = time.monotonic() + connect_timeout_s
        while True:
            port = port_name or _autodetect_port()
            print(f"GLMotorController: Attempting serial on {port}...")
            try:
                self.ser = serial.Serial(port, 115200, timeout=0.01, write_timeout=0.1)
                self.ser.reset_input_buffer()
                self.ser.reset_output_buffer()
            except Exception as e:
                print(f"GLMotorController: Failed to open {port}: {e}")
                self.ser = None
                if time.monotonic() >= connect_deadline:
                    raise RuntimeError(
                        f"Could not open XIAO motor bridge on {port}"
                    ) from e
                print("Retrying in 1 second... (Ctrl+C to abort)")
                time.sleep(1.0)
                continue

            # Wait for XIAO to boot and send its status string
            print("GLMotorController: Waiting for XIAO CAN bridge to boot...")
            time.sleep(1.5)

            startup_msg = ""
            try:
                if self.ser.in_waiting:
                    startup_msg = self.ser.read(self.ser.in_waiting).decode("ascii", errors="ignore")
            except Exception as e:
                print(f"GLMotorController: Serial read error during boot check: {e}")

            if "Error Initializing MCP2515" in startup_msg:
                print("ERROR: XIAO reported MCP2515 CAN chip init failure!")
                print("Check SPI wiring between XIAO and MCP2515 / CAN breakout power.")
                self.ser.close()
                self.ser = None
                if time.monotonic() >= connect_deadline:
                    raise RuntimeError(
                        "XIAO repeatedly reported MCP2515 initialization failure"
                    )
                print("Retrying in 1 second... (Ctrl+C to abort)")
                time.sleep(1.0)
                continue

            try:
                self.ser.write(b"V\n")
                version_deadline = time.monotonic() + 1.0
                version_msg = startup_msg
                while time.monotonic() < version_deadline:
                    if self.ser.in_waiting:
                        version_msg += self.ser.read(self.ser.in_waiting).decode(
                            "ascii", errors="ignore"
                        )
                    if BRIDGE_VERSION in version_msg:
                        break
                    time.sleep(0.02)
            except Exception as exc:
                self.ser.close()
                self.ser = None
                raise RuntimeError("Could not query XIAO bridge firmware") from exc
            
            if BRIDGE_VERSION not in version_msg:
                self.ser.close()
                self.ser = None
                raise RuntimeError(
                    f"XIAO bridge firmware is outdated; flash {BRIDGE_VERSION} "
                    "(see docs/FLASH_XIAO.md) before running the robot"
                )

            self.port_name = port
            print(f"GLMotorController: Connected to Seeed XIAO on {port}")
            break

        # ── Start receive thread ──
        self.running = True
        self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.recv_thread.start()

        # ── Initial state: disable + clear faults ──
        self.disable()
        time.sleep(0.01)
        self.clear_errors()
        time.sleep(0.01)

        self.mech_vel_ref[:] = default_vel_limit if control_mode == GL_MODE_POS_VEL else 0.0

    # ──────────────────── Serial I/O ────────────────────

    def _send_can(self, arb_id, data):
        """Send a CAN frame via the XIAO serial bridge."""
        if not self.ser:
            self.last_send_error = "serial bridge is closed"
            return False
        try:
            hex_data = data.hex().upper()
            msg = f"S:{arb_id:X}:{len(data)}:{hex_data}\n"
            self.ser.write(msg.encode('ascii'))
            self.last_send_error = None
            if self.should_print_send:
                print(f"  TX> id=0x{arb_id:03X} data={data.hex()}")
            return True
        except Exception as exc:
            self.last_send_error = str(exc)
            return False

    def _recv_loop(self):
        """Background thread: read serial lines and parse CAN feedback."""
        while self.running and self.ser:
            try:
                if self.ser.in_waiting:
                    incoming = self.ser.read(self.ser.in_waiting).decode("ascii", errors="ignore")
                    self._buffer += incoming

                while "\n" in self._buffer:
                    line, self._buffer = self._buffer.split("\n", 1)
                    line = line.strip()
                    if line.startswith("R:"):
                        parts = line.split(":")
                        if len(parts) >= 4:
                            try:
                                dlc = int(parts[2])
                                data = bytes.fromhex(parts[3][:dlc * 2])
                                self._handle_feedback(data)
                            except Exception:
                                pass
            except Exception:
                pass
            time.sleep(0.001)

    def _handle_feedback(self, data):
        """Parse an 8-byte GL40 feedback frame."""
        if len(data) != 8:
            return
        motor_id = data[0] & 0x0F

        idx = -1
        for i in range(self.num_motors):
            if self.can_ids[i] == motor_id:
                idx = i
                break
        if idx == -1:
            return

        pos_int = (data[1] << 8) | data[2]
        self.mech_pos[idx] = uint_to_float(pos_int, P_MIN, P_MAX, 16)

        spd_int = (data[3] << 4) | ((data[4] >> 4) & 0x0F)
        self.mech_vel[idx] = uint_to_float(spd_int, V_MIN, V_MAX, 12)

        t_int = ((data[4] & 0x0F) << 8) | data[5]
        self.mech_torque[idx] = uint_to_float(t_int, T_MIN, T_MAX, 12)

        self.drive_temp[idx] = struct.unpack("b", bytes([data[6]]))[0]
        self.motor_temp[idx] = struct.unpack("b", bytes([data[7]]))[0]
        self.last_feedback_time[idx] = time.monotonic()

        err_raw = (data[0] >> 4) & 0x0F
        self.error_code[idx] = err_raw
        self.mode_status[idx] = 1 if err_raw == 1 else 0

        if self.should_print_recv:
            print(f"  RX< M[{motor_id:02x}] pos={self.mech_pos[idx]:+.3f} vel={self.mech_vel[idx]:+.2f} err={err_raw}")

    # ──────────────────── Motor commands ────────────────────

    def enable(self):
        """Enable motor control. CAN ID encodes the current control mode."""
        sent_all = True
        send_error = None
        for i in range(self.num_motors):
            can_id = _get_can_id(int(self.can_ids[i]), self.control_mode)
            if not self._send_can(can_id, CMD_ENTER_CONTROL):
                sent_all = False
                send_error = self.last_send_error
            time.sleep(0.01)
        if not sent_all:
            raise RuntimeError(
                f"Failed to enable one or more GL40 motors: {send_error}"
            )
        time.sleep(0.05)
        self.mech_pos_ref[:] = self.mech_pos[:]

    def disable(self, clear_fault=False):
        """Disable motor control. Sends on the current control mode's CAN ID."""
        cmd = CMD_CLEAR_ERRORS if clear_fault else CMD_EXIT_CONTROL
        for i in range(self.num_motors):
            can_id = _get_can_id(int(self.can_ids[i]), self.control_mode)
            self._send_can(can_id, cmd)
            time.sleep(0.01)

    def clear_errors(self):
        """Send clear-errors command on the current mode."""
        for i in range(self.num_motors):
            can_id = _get_can_id(int(self.can_ids[i]), self.control_mode)
            self._send_can(can_id, CMD_CLEAR_ERRORS)
            time.sleep(0.01)

    def set_pos_zero(self):
        """Set current shaft position as the new encoder zero."""
        sent_all = True
        send_error = None
        for i in range(self.num_motors):
            can_id = _get_can_id(int(self.can_ids[i]), self.control_mode)
            if not self._send_can(can_id, CMD_SET_ZERO_POS):
                sent_all = False
                send_error = self.last_send_error
            time.sleep(0.01)
        if not sent_all:
            raise RuntimeError(
                f"Failed to zero one or more GL40 motors: {send_error}"
            )

    def motion_control_once(self):
        """Send one round of motion commands to all motors."""
        if self.control_mode == GL_MODE_POS_VEL and (
            not np.isfinite(self.mech_pos_ref).all()
            or not np.isfinite(self.mech_vel_ref).all()
        ):
            raise ValueError("GL40 position/velocity references must be finite")
        if self.control_mode == GL_MODE_VEL and not np.isfinite(self.mech_vel_ref).all():
            raise ValueError("GL40 velocity references must be finite")
        sent_all = True
        send_error = None
        for i in range(self.num_motors):
            mid = int(self.can_ids[i])
            if self.control_mode == GL_MODE_POS_VEL:
                can_id = _get_can_id(mid, GL_MODE_POS_VEL)
                data = struct.pack("<ff", float(self.mech_pos_ref[i]), float(self.mech_vel_ref[i]))
                sent = self._send_can(can_id, data)
            elif self.control_mode == GL_MODE_VEL:
                can_id = _get_can_id(mid, GL_MODE_VEL)
                data = struct.pack("<f", float(self.mech_vel_ref[i]))
                sent = self._send_can(can_id, data)
            else:
                # MIT mode (mode 0)
                can_id = _get_can_id(mid, GL_MODE_MIT)
                sent = self._send_can(can_id, b"\x00" * 8)
            if not sent:
                sent_all = False
                send_error = self.last_send_error
        if not sent_all:
            raise RuntimeError(
                f"Failed to send one or more GL40 commands: {send_error}"
            )

    def shutdown(self):
        self.running = False
        self.disable()
        time.sleep(0.01)
        self.disable(clear_fault=True)
        if self.ser:
            self.ser.close()
