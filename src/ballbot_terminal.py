#!/usr/bin/env python3
"""SSH-friendly terminal controller with heuristic and trained TorchScript policies."""

from __future__ import annotations

import dataclasses
import math
import os
from datetime import datetime
from pathlib import Path
import select
import sys
import termios
import threading
import time
import tty
from typing import Annotated, Literal

import msgpack
import numpy as np
import tyro

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
for d in (
    ROOT,
    ROOT / "src",
    ROOT / "xiao_can",
    ROOT / "hardware_bindings",
    PROJECT_ROOT,
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "xiao_can",
    PROJECT_ROOT / "hardware_bindings",
):
    if d.exists() and str(d) not in sys.path:
        sys.path.insert(0, str(d))

from ballbot_runtime import (
    CALIBRATION_FILE,
    DEFAULT_POLICY_PATH,
    IDLE_POSITION_MM,
    ImuState,
    HeuristicPolicy,
    wrap_to_pi,
    JOINT_NAMES,
    MAX_TRAVEL_MM,
    MotorJointMapper,
    MOTOR_IDS,
    POLICY_FRAME_DIM,
    POLICY_HZ,
    SliderCalibration,
    TrainedPolicy,
    mm_to_rad,
)
from xiao_gl_motor import GL40_FAULT_CODES, gl40_status_name

LOG_DIR = PROJECT_ROOT / "logs"

CONTROL_HZ = 100.0
COMMAND_TTL_S = 3.0
IDLE_DISARM_S = 120.0
SWITCH_CENTER_S = 1.0
TELEMETRY_TIMEOUT_S = 5.0
HOME_VELOCITY_RAD_S = 20.0
HOME_SEEK_S = 5.0
# Gentle approach to the 114 mm centre; at seek speed it overshoots the tolerance.
CENTER_VELOCITY_RAD_S = 1.5
MOTOR_TEMP_LIMIT_C = 85.0
DRIVE_TEMP_LIMIT_C = 85.0
TEMP_HYSTERESIS_C = 5.0
MAX_COMMAND_SPEED_MPS = 0.7
# Degrees per Q/E press when re-aiming the command frame.
YAW_STEP_DEG = 15.0
DEFAULT_COMMAND_SPEED_MPS = 0.3
DEFAULT_HEURISTIC_MAGNITUDE = 0.98
DEFAULT_CONTROL_VEL_LIMIT_RAD_S = 30.0
MAX_VEL_LIMIT_RAD_S = 100.0
MIN_VEL_LIMIT_RAD_S = 0.2
VEL_LIMIT_STEP_RAD_S = 5.0
# Below this the trained policy's targets clip every step and the robot barely moves
# (docs/OPERATIONS.md).
TRAINED_MIN_VEL_LIMIT_RAD_S = 10.0
# Not a derate (use --speed/--vel-limit): below this the policy stops tracking the command.
TRAINED_MIN_ACTION_SCALE = 0.8

def _as_f64_list(values: np.ndarray | list[float]) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=np.float64).reshape(-1)]


class PolicyRunLogger:
    """Append-only msgpack stream of heuristic/trained policy observation/action traces.

    File layout: concatenated msgpack objects. First object is a header dict
    (`type="header"`); each subsequent object is a step (`type="step"`).
    Read with `msgpack.Unpacker(f, raw=False)` or
    `hardware_bindings.common.publisher.unpack_data_from_file`.
    """

    def __init__(self, path: Path, *, policy: str, yaw_ref_deg: float = 0.0) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("wb")
        self.step = 0
        header = {
            "type": "header",
            "schema_version": 2,
            "policy": str(policy),
            "policy_hz": float(POLICY_HZ),
            "frame_dim": int(POLICY_FRAME_DIM),
            "joint_names": list(JOINT_NAMES),
            "motor_ids": [int(mid) for mid in MOTOR_IDS],
            "obs_terms": [
                "ang_vel_world[3]",
                "cmd_world_xy[2]",
                "joint_pos[3]",
                "joint_vel[3]",
                "last_action[3]",
                "rotation_world_base[9]",
            ],
            "created_at": datetime.now().isoformat(timespec="seconds"),
            # World yaw the command frame's +X pointed at when the log opened. Per-step
            # `yaw_ref_deg` is authoritative: the operator can re-aim it mid-run.
            "yaw_ref_deg": float(yaw_ref_deg),
        }
        self._pack(header)
        self._can_error_path = path.with_name(path.stem + "_can_errors.txt")
        self._can_error_file = self._can_error_path.open("a", encoding="utf-8")
        self._can_error_file.write(
            f"# CAN/GL40 status log for {path.name}\n"
            f"# created_at={header['created_at']} policy={policy}\n"
        )
        self._can_error_file.flush()

    def log_can_event(self, t_s: float, message: str) -> None:
        line = f"{t_s:9.3f}  {message}\n"
        if self._can_error_file is not None:
            self._can_error_file.write(line)
            self._can_error_file.flush()
        print(f"\nCAN ERR: {message}", flush=True)

    def _pack(self, obj: dict) -> None:
        self._file.write(msgpack.packb(obj, use_bin_type=True))

    def log(
        self,
        *,
        t_s: float,
        policy: str,
        frame: np.ndarray,
        action: np.ndarray,
        target: np.ndarray,
        virtual_pos: np.ndarray,
        virtual_vel: np.ndarray,
        motor_pos: np.ndarray,
        motor_vel: np.ndarray,
        motor_torque: np.ndarray,
        motor_temp: np.ndarray,
        drive_temp: np.ndarray,
        error_code: np.ndarray,
        mode_status: np.ndarray,
        feedback_age_s: np.ndarray,
        direction: np.ndarray,
        command_speed: float,
        action_scale: float,
        vel_limit: float,
        target_rate_limit: float = 0.0,
        kp: float,
        kd: float,
        armed: bool,
        send_error: str | None,
        yaw_ref_deg: float = 0.0,
    ) -> None:
        frame = np.asarray(frame, dtype=np.float64).reshape(-1)
        if frame.shape[0] != POLICY_FRAME_DIM:
            raise ValueError(f"expected {POLICY_FRAME_DIM}D frame, got {frame.shape}")
        err = [int(x) for x in np.asarray(error_code).reshape(-1)]
        record = {
            "type": "step",
            "t_s": float(t_s),
            "step": int(self.step),
            "policy": str(policy),
            "obs": _as_f64_list(frame),
            "action": _as_f64_list(action),
            "target_m": _as_f64_list(target),
            "virtual_pos": _as_f64_list(virtual_pos),
            "virtual_vel": _as_f64_list(virtual_vel),
            "motor_pos_rad": _as_f64_list(motor_pos),
            "motor_vel_rad_s": _as_f64_list(motor_vel),
            "motor_torque_nm": _as_f64_list(motor_torque),
            "motor_temp_c": _as_f64_list(motor_temp),
            "drive_temp_c": _as_f64_list(drive_temp),
            "error_code": err,
            "error_name": [gl40_status_name(c) for c in err],
            "mode_status": [int(x) for x in np.asarray(mode_status).reshape(-1)],
            "feedback_age_s": _as_f64_list(feedback_age_s),
            "direction": _as_f64_list(direction[:2]),
            "command_speed_mps": float(command_speed),
            "action_scale": float(action_scale),
            "vel_limit_rad_s": float(vel_limit),
            "target_rate_limit_mps": float(target_rate_limit),
            "kp": float(kp),
            "kd": float(kd),
            "armed": bool(armed),
            "send_error": None if send_error is None else str(send_error),
            # `obs` (cmd_world_xy, ang_vel_world, rotation_world_base) is expressed in the
            # command frame, so it is only comparable across steps at equal yaw_ref_deg.
            "yaw_ref_deg": float(yaw_ref_deg),
        }
        self._pack(record)
        self.step += 1
        if self.step % 10 == 0:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
        if getattr(self, "_can_error_file", None) is not None:
            self._can_error_file.flush()
            self._can_error_file.close()
            self._can_error_file = None


class TerminalInput:
    def __init__(self, command_ttl_s: float = COMMAND_TTL_S):
        self.command_ttl_s = command_ttl_s
        self.key_expiry = {"w": 0.0, "a": 0.0, "s": 0.0, "d": 0.0}
        self.stop_requested = False
        self.quit_requested = False
        self.estop_requested = False
        self.switch_requested = False
        self.speed_delta = 0
        self.scale_delta = 0
        self.vel_limit_delta = 0
        self.stiffness_delta = 0
        self.damping_delta = 0
        self.yaw_delta = 0
        self.last_movement_at = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        if not sys.stdin.isatty():
            # Without `ssh -t` stdin is a pipe and `tty.setcbreak` throws. Flush so the
            # message shows before the run loop sees `quit_requested`.
            print(
                "\nInteractive TTY required for keyboard control; "
                "use `ssh -t` when launching remotely.",
                flush=True,
            )
            with self._lock:
                self.quit_requested = True
            return
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while self._running:
                if not select.select([sys.stdin], [], [], 0.05)[0]:
                    continue
                key = sys.stdin.read(1)
                now = time.monotonic()
                with self._lock:
                    lower = key.lower()
                    if lower in self.key_expiry:
                        opposite = {"w": "s", "s": "w", "a": "d", "d": "a"}[lower]
                        self.key_expiry[opposite] = 0.0
                        self.key_expiry[lower] = now + self.command_ttl_s
                        self.last_movement_at = now
                        self.stop_requested = False
                    elif key == " ":
                        self._clear_directions()
                        self.stop_requested = True
                    elif lower == "x":
                        self._clear_directions()
                        self.estop_requested = True
                    elif lower == "p":
                        self.switch_requested = True
                    elif lower == "m":
                        self.speed_delta += 1
                    elif lower == "n":
                        self.speed_delta -= 1
                    elif key == "]":
                        self.scale_delta += 1
                    elif key == "[":
                        self.scale_delta -= 1
                    elif key in ("=", "+") or lower == "v":
                        self.vel_limit_delta += 1
                    elif key in ("-", "_") or lower == "c":
                        self.vel_limit_delta -= 1
                    elif key in (",", "<"):
                        self.stiffness_delta -= 1
                    elif key in (".", ">"):
                        self.stiffness_delta += 1
                    elif lower == "o":
                        self.damping_delta -= 1
                    elif lower == "i":
                        self.damping_delta += 1
                    elif lower == "q":
                        self.yaw_delta += 1
                    elif lower == "e":
                        self.yaw_delta -= 1
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _clear_directions(self) -> None:
        for key in self.key_expiry:
            self.key_expiry[key] = 0.0

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self._lock:
            x = float(self.key_expiry["w"] > now) - float(self.key_expiry["s"] > now)
            y = float(self.key_expiry["a"] > now) - float(self.key_expiry["d"] > now)
            direction = np.array([x, y], dtype=np.float32)
            norm = float(np.linalg.norm(direction))
            if norm > 0.0:
                direction /= norm
            result = {
                "direction": direction,
                "last_movement_at": self.last_movement_at,
                "stop": self.stop_requested,
                "quit": self.quit_requested,
                "estop": self.estop_requested,
                "switch": self.switch_requested,
                "speed_delta": self.speed_delta,
                "scale_delta": self.scale_delta,
                "vel_limit_delta": self.vel_limit_delta,
                "stiffness_delta": self.stiffness_delta,
                "damping_delta": self.damping_delta,
                "yaw_delta": self.yaw_delta,
            }
            self.stop_requested = False
            self.estop_requested = False
            self.switch_requested = False
            self.speed_delta = 0
            self.scale_delta = 0
            self.vel_limit_delta = 0
            self.stiffness_delta = 0
            self.damping_delta = 0
            self.yaw_delta = 0
            return result

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=0.2)


class BallbotController:
    def __init__(self, args: "Args"):
        self.args = args
        if args.imu_port and args.motor_port and args.imu_port == args.motor_port:
            raise ValueError("IMU and XIAO motor bridge must use different serial ports")
        self.calibration = SliderCalibration.load(Path(args.calibration))
        self.mapper = MotorJointMapper(self.calibration)
        self.heuristic = HeuristicPolicy()
        self.trained: TrainedPolicy | None = None
        self.active_policy_name = args.policy
        self.command_speed = args.speed
        self.heuristic_magnitude = args.heuristic_magnitude
        # World yaw that the command frame calls +X (the W key). 0 = the frame the IMU
        # mount calibration left behind, whose yaw origin is unrelated to which way the
        # robot is set down or driven from. Q/E turn it; see rotate_yaw_reference.
        self.yaw_reference_rad = wrap_to_pi(math.radians(args.yaw_offset_deg))
        self.action_scale = args.action_scale
        self.last_action = np.zeros(3, dtype=np.float32)
        self.current_joint_target = np.zeros(3, dtype=np.float64)
        self.virtual_joint_pos = np.zeros(3, dtype=np.float64)
        self.virtual_joint_vel = np.zeros(3, dtype=np.float64)
        self.last_policy_step = 0.0
        self.switch_until = 0.0
        self.pending_policy_name: str | None = None
        self.armed = False
        self.estopped = False
        self.thermal_lock = False
        self.last_thermal_poll = 0.0
        self.last_imu_counter = -1
        self.last_imu_update = time.monotonic()
        self._shutdown = False
        self.policy_logger: PolicyRunLogger | None = None
        self._run_started_at = 0.0
        self._last_can_error_codes: tuple[int, ...] | None = None
        self._last_can_send_error: str | None = None
        self._last_can_stale_mask: tuple[bool, ...] | None = None

        policy_exists = Path(args.policy_path).exists()
        if self.active_policy_name == "trained" or policy_exists:
            try:
                self._ensure_trained_policy()
            except Exception as exc:
                if self.active_policy_name == "trained":
                    raise
                print(
                    f"\nTrained policy unavailable ({exc}); "
                    "heuristic control remains usable."
                )

        from hardware_bindings.imu.py_imu import IMU
        from xiao_gl_motor import GL40, GL_MODE_POS_VEL, GLMotorController

        print("Initializing fixed-pose TM171 IMU...")
        self.imu = IMU(port_name=args.imu_port)
        time.sleep(1.0)

        self.vel_limit = float(args.vel_limit)
        self.home_vel = float(args.home_vel)
        self.kp = float(args.kp)
        self.kd = float(args.kd)
        self.no_trajectory_smoothing = bool(args.no_trajectory_smoothing)
        self.target_rate_limit = float(getattr(args, "target_rate_limit", 0.0))
        motor_setup = [[motor_id, "xiao", GL40] for motor_id in MOTOR_IDS]
        print("Initializing XIAO serial-to-CAN motor bridge...")
        try:
            self.motor = GLMotorController(
                motor_setup,
                control_mode=GL_MODE_POS_VEL,
                default_vel_limit=self.vel_limit,
                port_name=args.motor_port,
            )
        except Exception:
            self.imu.shutdown()
            raise
        self.motor.should_print_send = False
        self.motor.should_print_recv = False

    def _ensure_trained_policy(self) -> TrainedPolicy:
        if self.trained is None:
            print(f"\nLoading trained TorchScript policy: {self.args.policy_path}")
            self.trained = TrainedPolicy(Path(self.args.policy_path), self.action_scale)
            print("Policy preflight passed.")
        return self.trained

    def _motor_feedback_fresh(self) -> bool:
        """True when motor RX is acceptable for the current --no-telem setting.

        Default: every motor must have fresh feedback.
        `--no-telem`: at least one motor fresh (silent axes are allowed).
        """
        timestamps = np.asarray(self.motor.last_feedback_time, dtype=np.float64)
        if timestamps.shape != (len(MOTOR_IDS),):
            return False
        ages = time.monotonic() - timestamps
        fresh = (timestamps > 0.0) & (ages < TELEMETRY_TIMEOUT_S)
        if getattr(self.args, "no_telem", False):
            return bool(np.any(fresh))
        return bool(np.all(fresh))

    def _stale_motor_ids(self) -> list[int]:
        now = time.monotonic()
        stamps = np.asarray(self.motor.last_feedback_time, dtype=np.float64)
        stale: list[int] = []
        for mid, stamp in zip(MOTOR_IDS, stamps):
            if stamp <= 0.0 or now - float(stamp) >= TELEMETRY_TIMEOUT_S:
                stale.append(int(mid))
        return stale

    def _warn_missing_telem(self, context: str) -> None:
        stale = self._stale_motor_ids()
        if not stale:
            return
        ids = ", ".join(f"0x{mid:02x}" for mid in stale)
        print(
            f"\nWARNING ({context}): no/stale motor telemetry from [{ids}] "
            "(continuing because --no-telem).",
            flush=True,
        )

    def _wait_for_feedback(self, timeout_s: float = TELEMETRY_TIMEOUT_S) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.motor.motion_control_once()
            if self._motor_feedback_fresh():
                return True
            time.sleep(0.02)
        return False

    def _temperature_trip(self) -> bool:
        motor_hot = bool(np.any(np.asarray(self.motor.motor_temp) >= self.args.motor_temp_limit))
        drive_hot = bool(np.any(np.asarray(self.motor.drive_temp) >= self.args.drive_temp_limit))
        return motor_hot or drive_hot

    def _temperature_cooled(self) -> bool:
        return bool(
            np.all(
                np.asarray(self.motor.motor_temp)
                < self.args.motor_temp_limit - TEMP_HYSTERESIS_C
            )
            and np.all(
                np.asarray(self.motor.drive_temp)
                < self.args.drive_temp_limit - TEMP_HYSTERESIS_C
            )
        )

    def _trip_thermal_lock(self) -> None:
        if not self.thermal_lock:
            print(
                "\nTHERMAL STOP: motors disabled "
                f"(motor={self.motor.motor_temp}, drive={self.motor.drive_temp})."
            )
        self.motor.disable()
        self.armed = False
        self.thermal_lock = True

    def _poll_thermal_lock(self) -> None:
        now = time.monotonic()
        if now - self.last_thermal_poll < 1.0:
            return
        self.last_thermal_poll = now
        self.motor.disable()
        time.sleep(0.03)
        if self._motor_feedback_fresh() and self._temperature_cooled():
            self.thermal_lock = False
            print("\nTemperatures recovered. Send a new movement command to re-arm.")

    def read_imu(self) -> ImuState:
        counter = int(self.imu.counter)
        now = time.monotonic()
        if counter != self.last_imu_counter:
            self.last_imu_counter = counter
            self.last_imu_update = now
        if now - self.last_imu_update > TELEMETRY_TIMEOUT_S:
            raise RuntimeError("IMU telemetry is stale")
        return ImuState.from_device(self.imu, self.yaw_reference_rad)

    @property
    def yaw_reference_deg(self) -> float:
        return math.degrees(self.yaw_reference_rad)

    def rotate_yaw_reference(self, delta_deg: float) -> None:
        """Turn the command frame about world +Z, so W/A/S/D drive somewhere else.

        Only the command/observation frame moves: slider calibration, joint targets and
        the travel clamps are untouched, so this is safe to do while armed.
        """
        self.yaw_reference_rad = wrap_to_pi(
            self.yaw_reference_rad + math.radians(delta_deg)
        )
        print(
            f"\nYaw reference {delta_deg:+.0f} deg -> W now drives "
            f"{self.yaw_reference_deg:+.1f} deg in the calibrated IMU frame "
            f"(--yaw-offset-deg {self.yaw_reference_deg:.1f})"
        )

    def home_and_center(self) -> None:
        print(
            f"Homing all sliders toward the 0 mm hardstop at {self.home_vel:.1f} rad/s for "
            f"{HOME_SEEK_S:.0f} s, then assuming hardstop..."
        )
        # Match arm(): enable first. GL40s only stream feedback in control mode;
        # waiting before enable always times out on a cold start.
        self.motor.clear_errors()
        time.sleep(0.02)
        self.motor.enable()
        time.sleep(0.05)
        if not self._wait_for_feedback():
            stale = ", ".join(f"0x{mid:02x}" for mid in self._stale_motor_ids())
            codes = [int(c) for c in self.motor.error_code]
            raise RuntimeError(
                "Motor telemetry missing before homing "
                f"(no/stale RX from [{stale}]; error_code={codes})"
                + ("" if self.args.no_telem else "; pass --no-telem to allow silent motors")
            )
        self._warn_missing_telem("homing")
        if self._temperature_trip():
            raise RuntimeError("Temperature limit reached before homing")
        # Drive all axes hard toward the 0 mm hardstop for a fixed window; no torque latch.
        inward_delta = float(mm_to_rad(-MAX_TRAVEL_MM))
        for joint_name in self.mapper.calibration.joints:
            mapping = self.mapper.calibration.joints[joint_name]
            idx = self.mapper.motor_index[mapping.motor_id]
            self.motor.mech_pos_ref[idx] = (
                float(self.motor.mech_pos[idx]) + mapping.encoder_sign * inward_delta
            )
        self.motor.mech_vel_ref[:] = self.home_vel
        deadline = time.monotonic() + HOME_SEEK_S
        while time.monotonic() < deadline:
            self.motor.motion_control_once()
            if self._temperature_trip():
                raise RuntimeError("Temperature limit reached during homing")
            time.sleep(0.01)

        print("  Seek complete; zeroing encoders at assumed hardstop.")
        self.motor.set_pos_zero()
        # Hold at the current position before centring; otherwise the settle loop keeps
        # driving the seek's stale, out-of-range target into the hardstop.
        self.motor.mech_vel_ref[:] = 0.0
        self.motor.mech_pos_ref[:] = self.motor.mech_pos[:]
        time.sleep(0.3)
        for _ in range(20):
            self.motor.motion_control_once()
            if self._temperature_trip():
                raise RuntimeError("Temperature limit reached after zeroing")
            time.sleep(0.02)

        center_targets = self.mapper.center_motor_targets()
        self.motor.mech_pos_ref[:] = center_targets
        self.motor.mech_vel_ref[:] = CENTER_VELOCITY_RAD_S
        center_deadline = time.monotonic() + 15.0
        while time.monotonic() < center_deadline:
            self.motor.motion_control_once()
            if not self._motor_feedback_fresh():
                raise RuntimeError("Motor telemetry lost while moving to center")
            if self._temperature_trip():
                raise RuntimeError("Temperature limit reached while moving to center")
            pos = np.asarray(self.motor.mech_pos, dtype=np.float64)
            if self.args.no_telem:
                live = [
                    i
                    for i, mid in enumerate(MOTOR_IDS)
                    if mid not in self._stale_motor_ids()
                ]
                if live and bool(
                    np.all(np.abs(pos[live] - center_targets[live]) < 0.08)
                ):
                    break
            elif bool(np.all(np.abs(pos - center_targets) < 0.08)):
                break
            time.sleep(0.01)
        else:
            if self.args.no_telem:
                self._warn_missing_telem("homing center timeout")
                print(
                    "\nWARNING: center move timed out; continuing with --no-telem.",
                    flush=True,
                )
            else:
                raise RuntimeError("Sliders did not reach the 114 mm center after homing")

        self.motor.mech_vel_ref[:] = self.vel_limit
        actual_joint_pos, _ = self.mapper.joint_state_from_motors(
            np.asarray(self.motor.mech_pos), np.asarray(self.motor.mech_vel)
        )
        self.current_joint_target = actual_joint_pos.copy()
        self.virtual_joint_pos = actual_joint_pos.copy()
        self.virtual_joint_vel[:] = 0.0
        self.motor.mech_pos_ref[:] = self.mapper.motor_targets_from_joint_position(
            self.virtual_joint_pos
        )
        self.motor.motion_control_once()
        self.armed = True
        print(
            f"Homing complete; sliders holding {IDLE_POSITION_MM:.0f} mm "
            "(motors stay enabled)."
        )

    def arm(self) -> bool:
        if self.estopped or self.thermal_lock:
            return False
        self.motor.clear_errors()
        time.sleep(0.02)
        self.motor.enable()
        if not self._wait_for_feedback():
            stale = ", ".join(f"0x{mid:02x}" for mid in self._stale_motor_ids())
            codes = [int(c) for c in self.motor.error_code]
            self.motor.disable()
            print(
                "\nARM REFUSED: motor telemetry is missing or stale "
                f"(no/stale RX from [{stale}]; error_code={codes})"
                + ("" if self.args.no_telem else "; pass --no-telem to allow silent motors")
                + "."
            )
            return False
        self._warn_missing_telem("arm")
        if self._temperature_trip():
            self._trip_thermal_lock()
            return False
        actual_joint_pos, _ = self.mapper.joint_state_from_motors(
            np.asarray(self.motor.mech_pos), np.asarray(self.motor.mech_vel)
        )
        self.current_joint_target = actual_joint_pos.copy()
        self.virtual_joint_pos = actual_joint_pos.copy()
        self.virtual_joint_vel[:] = 0.0
        self.motor.mech_pos_ref[:] = self.mapper.motor_targets_from_joint_position(
            self.virtual_joint_pos
        )
        self.motor.motion_control_once()
        self.armed = True
        print("\nMotors armed.")
        return True

    def disable(self, reason: str) -> None:
        if self.armed:
            print(f"\nMotors disabled: {reason}.")
        self.motor.disable()
        self.armed = False

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self.disable("controller shutdown")
        self.motor.shutdown()
        self.imu.shutdown()

    def request_policy_switch(self, now: float) -> None:
        target = "trained" if self.active_policy_name == "heuristic" else "heuristic"
        if target == "trained" and self.trained is None:
            print("\nPolicy switch rejected: trained policy did not pass startup preflight.")
            return
        self.pending_policy_name = target
        self.switch_until = now + SWITCH_CENTER_S
        print(f"\nCentering before switch to {target}...")

    def finish_policy_switch(self, imu_state: ImuState, joint_pos: np.ndarray, joint_vel: np.ndarray) -> None:
        if self.pending_policy_name is None:
            return
        self.active_policy_name = self.pending_policy_name
        self.pending_policy_name = None
        self.last_action[:] = 0.0
        if self.active_policy_name == "trained":
            # Heuristic control is usable at low slider speeds; the trained policy is not.
            # Switching without raising these would hand over a policy that cannot move.
            if self.vel_limit < TRAINED_MIN_VEL_LIMIT_RAD_S:
                self.vel_limit = TRAINED_MIN_VEL_LIMIT_RAD_S
                self.motor.mech_vel_ref[:] = self.vel_limit
                print(
                    f"\nRaised velocity limit to {self.vel_limit:.1f} rad/s "
                    "for the trained policy."
                )
            if self.action_scale < TRAINED_MIN_ACTION_SCALE:
                self.action_scale = 1.0
                print("\nReset action scale to 1.0 for the trained policy.")
            frame = TrainedPolicy.build_frame(
                np.zeros(2, dtype=np.float32),
                imu_state,
                joint_pos,
                joint_vel,
                self.last_action,
            )
            policy = self._ensure_trained_policy()
            # The policy may already exist from startup, in which case the constructor did
            # not see the value corrected above.
            policy.set_action_scale(self.action_scale)
            policy.reset(frame)
        print(f"\nControl policy: {self.active_policy_name}.")

    def update_target(
        self,
        now: float,
        direction: np.ndarray,
        imu_state: ImuState,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
    ) -> None:
        command_world_xy = direction * self.command_speed
        if self.active_policy_name == "heuristic":
            raw_target = self.heuristic.act(
                direction,
                self.heuristic_magnitude,
                imu_state,
                joint_pos,
                joint_vel,
            )
            self.current_joint_target[:] = raw_target
            if (
                self.policy_logger is not None
                and now - self.last_policy_step >= 1.0 / POLICY_HZ
            ):
                frame = TrainedPolicy.build_frame(
                    command_world_xy,
                    imu_state,
                    joint_pos,
                    joint_vel,
                    self.last_action,
                )
                self._log_policy_step(
                    now=now,
                    direction=direction,
                    frame=frame,
                    action=np.zeros(3, dtype=np.float64),
                    target=raw_target,
                )
                self.last_policy_step = now
            return

        if now - self.last_policy_step >= 1.0 / POLICY_HZ:
            policy = self._ensure_trained_policy()
            frame = policy.build_frame(
                command_world_xy, imu_state, joint_pos, joint_vel, self.last_action
            )
            target, action = policy.act_from_frame(frame)
            target = np.asarray(target, dtype=np.float64).copy()
            if self.target_rate_limit > 0.0:
                # Measured dt, not 1/POLICY_HZ: the loop steps the policy below its nominal rate.
                dt_step = max(now - self.last_policy_step, 1e-3)
                max_step = self.target_rate_limit * dt_step
                delta = np.clip(target - self.current_joint_target, -max_step, max_step)
                target = self.current_joint_target + delta
            self.current_joint_target = target
            self.last_action = action
            self.last_policy_step = now
            self._log_policy_step(
                now=now,
                direction=direction,
                frame=frame,
                action=action,
                target=target,
            )

    def _feedback_age_s(self) -> np.ndarray:
        now = time.monotonic()
        stamps = np.asarray(self.motor.last_feedback_time, dtype=np.float64)
        ages = np.full(len(MOTOR_IDS), float("inf"), dtype=np.float64)
        for i, stamp in enumerate(stamps):
            if stamp > 0.0:
                ages[i] = now - float(stamp)
        return ages

    def _report_can_status(self, now: float) -> None:
        """Print + file-log GL40 status changes, faults, stale RX, and send failures."""
        codes = tuple(int(c) for c in np.asarray(self.motor.error_code).reshape(-1))
        ages = self._feedback_age_s()
        stale = tuple(bool(a >= TELEMETRY_TIMEOUT_S) for a in ages)
        send_error = self.motor.last_send_error

        changed = (
            codes != self._last_can_error_codes
            or stale != self._last_can_stale_mask
            or send_error != self._last_can_send_error
        )
        if not changed:
            return

        parts: list[str] = []
        for i, (mid, code, age, is_stale) in enumerate(
            zip(MOTOR_IDS, codes, ages, stale)
        ):
            name = gl40_status_name(code)
            age_txt = "inf" if not np.isfinite(age) else f"{age:.3f}s"
            flag = ""
            if code in GL40_FAULT_CODES:
                flag = " FAULT"
            elif (
                self.armed
                and code == 0
                and self._last_can_error_codes is not None
                and self._last_can_error_codes[i] == 1
            ):
                flag = " UNEXPECTED_DISABLE"
            if is_stale:
                flag += " STALE_RX"
            parts.append(f"M[{mid:#04x}]={code}:{name} age={age_txt}{flag}")
        if send_error:
            parts.append(f"send_error={send_error}")

        message = " | ".join(parts)
        t_s = now - self._run_started_at if self._run_started_at else 0.0
        if self.policy_logger is not None:
            self.policy_logger.log_can_event(t_s, message)
        else:
            print(f"\nCAN ERR: {message}", flush=True)

        self._last_can_error_codes = codes
        self._last_can_stale_mask = stale
        self._last_can_send_error = send_error

    def _log_policy_step(
        self,
        *,
        now: float,
        direction: np.ndarray,
        frame: np.ndarray,
        action: np.ndarray,
        target: np.ndarray,
    ) -> None:
        if self.policy_logger is None:
            return
        self.policy_logger.log(
            t_s=now - self._run_started_at,
            policy=self.active_policy_name,
            frame=frame,
            action=action,
            target=target,
            virtual_pos=self.virtual_joint_pos,
            virtual_vel=self.virtual_joint_vel,
            motor_pos=np.asarray(self.motor.mech_pos, dtype=np.float64),
            motor_vel=np.asarray(self.motor.mech_vel, dtype=np.float64),
            motor_torque=np.asarray(self.motor.mech_torque, dtype=np.float64),
            motor_temp=np.asarray(self.motor.motor_temp, dtype=np.float64),
            drive_temp=np.asarray(self.motor.drive_temp, dtype=np.float64),
            error_code=np.asarray(self.motor.error_code, dtype=np.int64),
            mode_status=np.asarray(self.motor.mode_status, dtype=np.int64),
            feedback_age_s=self._feedback_age_s(),
            direction=direction,
            command_speed=self.command_speed,
            action_scale=self.action_scale,
            vel_limit=self.vel_limit,
            target_rate_limit=self.target_rate_limit,
            kp=self.kp,
            kd=self.kd,
            armed=self.armed,
            send_error=self.motor.last_send_error,
            yaw_ref_deg=self.yaw_reference_deg,
        )

    def center_target(self) -> None:
        self.current_joint_target[:] = 0.0
        self.last_action[:] = 0.0

    def _step_virtual_trajectory(self) -> None:
        dt = 1.0 / CONTROL_HZ
        pos_error = self.current_joint_target - self.virtual_joint_pos
        vel_error = 0.0 - self.virtual_joint_vel
        accel = self.kp * pos_error + self.kd * vel_error
        self.virtual_joint_vel += accel * dt
        self.virtual_joint_vel[:] = np.clip(self.virtual_joint_vel, -self.vel_limit, self.vel_limit)
        self.virtual_joint_pos += self.virtual_joint_vel * dt

    def run(self) -> None:
        if self.args.home:
            self.home_and_center()
        else:
            # Loud on purpose: the reported pose is not checked against the hardware.
            print(
                "\n" + "!" * 60 + "\n"
                "SKIPPING HOMING (default) - arming at the pose the motor drivers\n"
                "currently report, then centring from it, unverified against\n"
                "physical reality. Only safe right after a trusted rehome. Pass\n"
                "--home to force a full contact-seek rehome instead.\n" + "!" * 60 + "\n"
            )
            if not self.arm():
                raise RuntimeError(
                    "Could not arm motors at the current pose. If the sliders are not in a "
                    "known-good position, quit and rerun with --home."
                )
        keyboard = TerminalInput()
        keyboard.last_movement_at = time.monotonic()
        started = time.monotonic()
        self._run_started_at = started
        last_status = 0.0
        if getattr(self.args, "log", False) or not getattr(self.args, "no_log", False):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = Path(self.args.log_path) if self.args.log_path else (
                LOG_DIR / f"{self.active_policy_name}_policy_{stamp}.msgpack"
            )
            self.policy_logger = PolicyRunLogger(
                log_path,
                policy=self.active_policy_name,
                yaw_ref_deg=self.yaw_reference_deg,
            )
            print(f"Logging policy obs/actions to {log_path}")
            print(
                f"Logging CAN/GL40 status changes to {self.policy_logger._can_error_path}"
            )
        print(
            "\nCONTROL ACTIVE: W/A/S/D move in fixed world axes "
            "(W=+X, A=+Y; combine keys for diagonals).\n"
            "SPACE stop/center | P switch policy | N/M speed | [/] scale | "
            "-/+/C/V vel limit | ,/./O/I kp/kd | X E-stop | Ctrl+C quit\n"
            f"Q/E rotate the drive axes +/-{YAW_STEP_DEG:.0f} deg (yawref on the status "
            f"line, now {self.yaw_reference_deg:+.1f} deg): tap W, then Q/E until it "
            "drives the way you mean."
        )
        if self.no_trajectory_smoothing:
            print(
                "TRAJECTORY SMOOTHING OFF (default; --no-no-trajectory-smoothing turns it on): "
                "policy targets go straight to motor driver. SPACE/E-stop will snap targets "
                "instead of ramping."
            )
        try:
            next_tick = time.monotonic()
            while True:
                now = time.monotonic()
                command = keyboard.snapshot()
                if command["quit"]:
                    break
                if command["estop"]:
                    self.estopped = True
                    self.disable("latching E-stop")
                    print("\nE-STOP LATCHED. Restart the program to clear it.")

                if self.active_policy_name == "heuristic":
                    self.heuristic_magnitude = float(
                        np.clip(
                            self.heuristic_magnitude + 0.1 * command["speed_delta"],
                            0.01,
                            10.0,
                        )
                    )
                else:
                    self.command_speed = float(
                        np.clip(
                            self.command_speed + 0.05 * command["speed_delta"],
                            0.05,
                            MAX_COMMAND_SPEED_MPS,
                        )
                    )
                # Enforce the trained-policy floors mid-run too, so '['/']' and '-'/'C'
                # cannot walk below them.
                trained_active = self.active_policy_name == "trained"
                scale_floor = TRAINED_MIN_ACTION_SCALE if trained_active else 0.0
                vel_floor = (
                    TRAINED_MIN_VEL_LIMIT_RAD_S if trained_active else MIN_VEL_LIMIT_RAD_S
                )
                if command["scale_delta"]:
                    self.action_scale = float(
                        np.clip(
                            self.action_scale + 0.1 * command["scale_delta"], scale_floor, 1.0
                        )
                    )
                    if self.trained is not None:
                        self.trained.set_action_scale(self.action_scale)
                if command["vel_limit_delta"]:
                    self.vel_limit = float(
                        np.clip(
                            self.vel_limit + VEL_LIMIT_STEP_RAD_S * command["vel_limit_delta"],
                            vel_floor,
                            MAX_VEL_LIMIT_RAD_S,
                        )
                    )
                    self.motor.mech_vel_ref[:] = self.vel_limit
                if command["stiffness_delta"]:
                    self.kp = float(
                        np.clip(self.kp + 10.0 * command["stiffness_delta"], 5.0, 1000.0)
                    )
                if command["damping_delta"]:
                    self.kd = float(
                        np.clip(self.kd + 2.0 * command["damping_delta"], 1.0, 200.0)
                    )
                if command["yaw_delta"]:
                    self.rotate_yaw_reference(YAW_STEP_DEG * command["yaw_delta"])
                if command["switch"]:
                    self.request_policy_switch(now)

                if self.thermal_lock:
                    self._poll_thermal_lock()
                    time.sleep(0.02)
                    continue

                try:
                    imu_state = self.read_imu()
                except Exception:
                    self.disable("IMU telemetry invalid or stale")
                    raise
                joint_pos, joint_vel = self.mapper.joint_state_from_motors(
                    np.asarray(self.motor.mech_pos), np.asarray(self.motor.mech_vel)
                )

                direction = command["direction"]
                moving = bool(np.linalg.norm(direction) > 0.0) and not self.estopped
                if command["stop"]:
                    moving = False
                    direction[:] = 0.0

                if self.pending_policy_name is not None:
                    moving = False
                    self.center_target()
                    if now >= self.switch_until:
                        self.finish_policy_switch(imu_state, joint_pos, joint_vel)
                elif moving:
                    if not self.armed and not self.arm():
                        moving = False
                    if self.armed:
                        self.update_target(now, direction, imu_state, joint_pos, joint_vel)
                else:
                    self.center_target()

                if self.armed:
                    if not self._motor_feedback_fresh():
                        self._report_can_status(now)
                        self.disable("motor telemetry lost")
                    elif self._temperature_trip():
                        self._trip_thermal_lock()
                    else:
                        if self.no_trajectory_smoothing:
                            # Smoother off: the raw policy target goes straight to the mapper.
                            self.virtual_joint_pos[:] = self.current_joint_target
                            # Not advanced on this path; zeroed so re-enabling cannot jump.
                            self.virtual_joint_vel[:] = 0.0
                        else:
                            self._step_virtual_trajectory()
                        self.motor.mech_pos_ref[:] = self.mapper.motor_targets_from_joint_position(
                            self.virtual_joint_pos
                        )
                        try:
                            self.motor.motion_control_once()
                        except RuntimeError as exc:
                            self._report_can_status(now)
                            self.disable(f"CAN send failed: {exc}")
                    self._report_can_status(now)

                if (
                    self.armed
                    and not moving
                    and now - command["last_movement_at"] >= IDLE_DISARM_S
                ):
                    self.disable(f"{IDLE_DISARM_S:.0f}-second movement idle timeout")

                if now - last_status >= 0.25:
                    direction_text = f"({direction[0]:+.2f},{direction[1]:+.2f})"
                    command_text = (
                        f"magnitude={self.heuristic_magnitude:.2f}"
                        if self.active_policy_name == "heuristic"
                        else f"speed={self.command_speed:.2f}m/s"
                    )
                    # Shaft angle (rad, zero at the 0 mm hardstop) to slider travel (mm).
                    rad_to_mm = 100.0 / (2.0 * math.pi)
                    pos_mm = [
                        int(round(float(p) * rad_to_mm))
                        for p in self.motor.mech_pos
                    ]
                    tdrv_list = [int(t) for t in self.motor.drive_temp]
                    err_list = [int(c) for c in self.motor.error_code]
                    print(
                        f"\rpolicy={self.active_policy_name:<9} "
                        f"{command_text} "
                        f"pos={pos_mm}mm "
                        f"Tdrv={tdrv_list}C "
                        f"err={err_list} "
                        f"dir={direction_text} "
                        f"yawref={self.yaw_reference_deg:+.0f} "
                        f"armed={self.armed}",
                        end="",
                        flush=True,
                    )
                    last_status = now

                next_tick += 1.0 / CONTROL_HZ
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.monotonic()
        finally:
            print("\nStopping...")
            keyboard.stop()
            if self.policy_logger is not None:
                rows = self.policy_logger.step
                path = self.policy_logger.path
                self.policy_logger.close()
                self.policy_logger = None
                print(f"Wrote {rows} policy steps to {path}")
            self.shutdown()
            if self.yaw_reference_rad != 0.0:
                print(
                    "Final yaw reference: "
                    f"--yaw-offset-deg {self.yaw_reference_deg:.1f}"
                )
            print(f"Session duration: {time.monotonic() - started:.1f}s")


@dataclasses.dataclass
class Args:
    """Physical ballbot terminal controller."""

    policy: Literal["heuristic", "trained"] = "heuristic"
    policy_path: str = str(DEFAULT_POLICY_PATH)
    calibration: str = str(CALIBRATION_FILE)
    speed: float = DEFAULT_COMMAND_SPEED_MPS
    action_scale: float = 1.0
    heuristic_magnitude: float = DEFAULT_HEURISTIC_MAGNITUDE
    yaw_offset_deg: Annotated[
        float,
        tyro.conf.arg(
            help=(
                "Start with the drive axes rotated this many degrees CCW about world +Z, "
                "i.e. where the W key points in the calibrated IMU frame. Q/E change it "
                "live and print the value to pass back in here next run."
            )
        ),
    ] = 0.0
    vel_limit: Annotated[
        float, tyro.conf.arg(help="Position-mode velocity limit during control (rad/s)")
    ] = DEFAULT_CONTROL_VEL_LIMIT_RAD_S
    kp: Annotated[
        float, tyro.conf.arg(help="Software PD position gain for trajectory generator")
    ] = 125.0
    kd: Annotated[
        float, tyro.conf.arg(help="Software PD velocity gain for trajectory generator")
    ] = 25.0
    no_trajectory_smoothing: Annotated[
        bool,
        tyro.conf.arg(
            help=(
                "Pass --no-no-trajectory-smoothing to turn ON the software PD trajectory "
                "smoother. It is off by default; '(default: True)' at the end of this text "
                "means smoothing is off. Checkpoints with a requires_smoothing marker need "
                "it, and run.sh adds it for them. The smoother is a 1.78 Hz second-order "
                "low-pass (kp=125, kd=25, slow pole 0.145 s = 7 control steps at 50 Hz) "
                "between the policy and the motor. It passes the heuristic's step-and-hold "
                "untouched but strips ~76 percent of a 3 Hz RL slider gait, and the sim "
                "models no such filter. With it off, the policy's raw joint target goes "
                "directly to the motor driver's mech_pos_ref (the motor's internal velocity "
                "limit and current loop still apply), and SPACE/E-stop snap targets instead "
                "of ramping. Travel clipping is NOT part of the smoother: policy targets are "
                "clipped to JOINT_LOWER/UPPER_M (trained) or HEURISTIC_LOWER/UPPER_M, and "
                "motor_targets_from_joint_position clips to SAFE_MIN/MAX_TRAVEL_MM, both "
                "regardless of this flag."
            )
        ),
    ] = True
    target_rate_limit: Annotated[
        float,
        tyro.conf.arg(
            help=(
                "Max slider speed the trained policy may command, m/s. 0 = off (default). "
                "This bounds |target - previous target| per policy step using the MEASURED "
                "step time, not the nominal one. Why it exists: the GL40's own mech_vel_ref "
                "(set from --vel-limit) does NOT bound slider speed in position mode - a "
                "hardware log taken with --vel-limit 30 rad/s recorded motor velocities up "
                "to 120 rad/s (1.91 m/s). With trajectory smoothing off nothing else limits "
                "the jump between policy steps, and the trained policy was measured "
                "commanding 159 mm in a single step against 220 mm of total travel. "
                "0.477 reproduces the nominal 30 rad/s. "
                "WARNING: any value > 0 is a filter the SIM DOES NOT MODEL, i.e. the same "
                "class of sim/real mismatch that the trajectory smoother was. Prefer "
                "training smoothness in (higher action_rate_l2) and leaving this at 0; use "
                "it as a hardware safety clamp set well above the gait's real bandwidth, "
                "not as the primary smoother."
            )
        ),
    ] = 0.0
    home_vel: Annotated[
        float,
        tyro.conf.arg(help="Position-mode velocity limit during homing/centering (rad/s)"),
    ] = HOME_VELOCITY_RAD_S
    home: Annotated[
        bool,
        tyro.conf.arg(
            help=(
                "Force a full contact-seek rehome before arming (default: skip homing, "
                "arm at the current pose, then centre from it)"
            )
        ),
    ] = False
    home_only: Annotated[
        bool,
        tyro.conf.arg(
            help="Home, then disable motors and exit -- no interactive control loop, no TTY needed"
        ),
    ] = False
    no_telem: Annotated[
        bool,
        tyro.conf.arg(
            help=(
                "Allow arm/home/run when one or more motors never send CAN feedback "
                "(still requires at least one live motor; silent axes keep last/zero pose)"
            )
        ),
    ] = False
    log: Annotated[
        bool, tyro.conf.arg(help="Force msgpack logging (on by default for heuristic and trained)")
    ] = False
    no_log: Annotated[bool, tyro.conf.arg(help="Disable msgpack logging of policy observations")] = False
    log_path: Annotated[
        str | None,
        tyro.conf.arg(
            help="msgpack output path (default: logs/<policy>_policy_YYYYMMDD_HHMMSS.msgpack)"
        ),
    ] = None
    imu_port: str | None = os.environ.get("BALLBOT_IMU_PORT")
    motor_port: str | None = os.environ.get("BALLBOT_MOTOR_PORT")
    motor_temp_limit: float = MOTOR_TEMP_LIMIT_C
    drive_temp_limit: float = DRIVE_TEMP_LIMIT_C
    preflight: Annotated[
        bool, tyro.conf.arg(help="load and validate TorchScript without opening hardware")
    ] = False


def parse_args(default_policy: str | None = None) -> Args:
    default = Args(policy=default_policy) if default_policy else Args()
    # FlagCreatePairsOff: keep bools as plain store-true flags (no auto --no-x pair), since
    # --log/--no-log are both real independent flags here and would otherwise collide with
    # tyro's generated --no-log/--no-no-log.
    return tyro.cli(Annotated[Args, tyro.conf.FlagCreatePairsOff], default=default)


def main(default_policy: str | None = None) -> None:
    args = parse_args(default_policy)
    if not 0.05 <= args.speed <= MAX_COMMAND_SPEED_MPS:
        raise SystemExit(f"--speed must be in [0.05, {MAX_COMMAND_SPEED_MPS}]")
    if not 0.0 <= args.action_scale <= 1.0:
        raise SystemExit("--action-scale must be in [0, 1]")
    if not 0.0 <= args.heuristic_magnitude <= 1.0:
        raise SystemExit("--heuristic-magnitude must be in [0, 1]")
    # A non-finite offset would pass the IMU validity check (it is applied after) and put
    # NaNs straight into the policy observation.
    if not math.isfinite(args.yaw_offset_deg):
        raise SystemExit("--yaw-offset-deg must be finite")
    if not MIN_VEL_LIMIT_RAD_S <= args.vel_limit <= MAX_VEL_LIMIT_RAD_S:
        raise SystemExit(
            f"--vel-limit must be in [{MIN_VEL_LIMIT_RAD_S}, {MAX_VEL_LIMIT_RAD_S}] rad/s"
        )
    if not MIN_VEL_LIMIT_RAD_S <= args.home_vel <= MAX_VEL_LIMIT_RAD_S:
        raise SystemExit(
            f"--home-vel must be in [{MIN_VEL_LIMIT_RAD_S}, {MAX_VEL_LIMIT_RAD_S}] rad/s"
        )
    if args.preflight:
        # Before the trained-policy floors: preflight loads no hardware, so motor limits
        # do not apply.
        policy = TrainedPolicy(Path(args.policy_path), args.action_scale)
        print(
            f"PASS: {policy.policy_path} accepts (1, 3, {POLICY_FRAME_DIM}) "
            "and returns (1, 3)."
        )
        return
    if args.policy == "trained":
        if args.vel_limit < TRAINED_MIN_VEL_LIMIT_RAD_S:
            raise SystemExit(
                f"--vel-limit {args.vel_limit} rad/s is below the trained-policy floor of "
                f"{TRAINED_MIN_VEL_LIMIT_RAD_S} rad/s. At this ceiling the 50 Hz targets are "
                "clipped on nearly every control step and the robot will not move "
                "(measured 0.004 m/s against a 0.300 m/s command at 2.0 rad/s). "
                f"Pass --vel-limit {DEFAULT_CONTROL_VEL_LIMIT_RAD_S:.0f} (the default), or "
                "lower --speed to move more slowly."
            )
        if args.action_scale < TRAINED_MIN_ACTION_SCALE:
            raise SystemExit(
                f"--action-scale {args.action_scale} is below {TRAINED_MIN_ACTION_SCALE}. "
                "This knob rescales the trained action space rather than derating it: below "
                "~0.8 the robot keeps rolling at full speed while losing command tracking. "
                "Use --speed or --vel-limit to move more slowly."
            )
    if args.home_only:
        controller = BallbotController(args)
        try:
            controller.home_and_center()
            print("Homing complete; disabling motors and exiting.")
        finally:
            controller.shutdown()
        return

    if not sys.stdin.isatty():
        raise SystemExit("Interactive TTY required; launch over SSH with `ssh -t`.")

    controller: BallbotController | None = None
    try:
        controller = BallbotController(args)
        controller.run()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        print(f"\nFATAL: {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
    finally:
        if controller is not None:
            try:
                controller.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    main()
