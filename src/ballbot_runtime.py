#!/usr/bin/env python3
"""Shared coordinate, calibration, and policy logic for physical ballbot control."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
CALIBRATION_FILE = PROJECT_ROOT / "config" / "calibration.json"
DEFAULT_POLICY_PATH = PROJECT_ROOT / "policies" / "BallbotVelRingCageShellDR" / "policy_deployed.pt"

MOTOR_IDS = (0x08, 0x07, 0x06)
JOINT_NAMES = (
    "base_link_Slider-5",
    "base_link_Slider-6",
    "base_link_Slider-7",
)

MM_PER_TURN = 100.0
IDLE_POSITION_MM = 114.0
MAX_TRAVEL_MM = 220.0
# Keep normal control away from both hardstops and inside the GL40 feedback
# encoder's documented +/-12.5 rad range (195 mm = 12.25 rad).
SAFE_MIN_TRAVEL_MM = 10.0
SAFE_MAX_TRAVEL_MM = 195.0
ACTION_SCALE_M = 0.106
JOINT_LOWER_M = -0.114
JOINT_UPPER_M = 0.106
HEURISTIC_LOWER_M = -0.104
HEURISTIC_UPPER_M = 0.096
POLICY_HISTORY = 3
POLICY_FRAME_DIM = 23
# Policy rate; must match env_config.yaml, checked by PolicyDeployConfig.validate.
POLICY_HZ = 50.0

# Pose of imu_site in the MuJoCo model Sim_Model (every policies/*/robot.xml carries
# the same site), with a +90 deg rotation about sensor Z so gravity aligns with base -Z
# on this physical board. Position is unchanged from the model.
IMU_POSITION_BASE_M = np.array([0.037411, 0.111365, -0.071461], dtype=np.float64)
IMU_QUAT_BASE_SENSOR_WXYZ = np.array(
    [0.7046051026, -0.4562093290, -0.5402527632, -0.0594276822],
    dtype=np.float64,
)

# Positive simulation-joint motion moves the internal mass in these base-frame directions.
JOINT_POSITIVE_AXIS = {
    "base_link_Slider-5": ("Z", -1),
    "base_link_Slider-6": ("Y", 1),
    "base_link_Slider-7": ("X", -1),
}

# Joint lines compiled from the Sim_Model MuJoCo model (policies/*/robot.xml) at the base
# reference pose. For each slider, q is the closest-point projection of a base-frame
# target onto:
#   anchor + q * axis
SLIDER_AXES_BASE = np.array(
    [
        [0.0, 0.0, -1.0],  # Slider-5
        [0.0, 1.0, 0.0],   # Slider-6
        [-1.0, 0.0, 0.0],  # Slider-7
    ],
    dtype=np.float64,
)
SLIDER_ANCHORS_BASE_M = np.array(
    [
        [0.0, -0.03, 0.0],
        [-0.03, 0.0, 0.0],
        [0.0, 0.0, 0.03],
    ],
    dtype=np.float64,
)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        raise ValueError("Quaternion norm is zero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


R_BASE_SENSOR = quat_wxyz_to_matrix(IMU_QUAT_BASE_SENSOR_WXYZ)


def rotation_z(angle_rad: float) -> np.ndarray:
    """Right-handed rotation about world +Z (yaw)."""
    c = math.cos(float(angle_rad))
    s = math.sin(float(angle_rad))
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def wrap_to_pi(angle_rad: float) -> float:
    """Fold an angle into [-pi, pi)."""
    return float((float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi)


def mm_to_rad(mm: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(mm) * (2.0 * math.pi / MM_PER_TURN)


def rad_to_mm(rad: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(rad) * (MM_PER_TURN / (2.0 * math.pi))


@dataclass(frozen=True)
class ImuState:
    """Base orientation and body rate in the command world frame.

    Both policies take the velocity command in world XY and the orientation as
    `rotation_world_base`, so "which way is forward" is a property of this frame, whose
    yaw origin is whatever heading IMU_QUAT_BASE_SENSOR_WXYZ was captured at -- not the
    direction the robot is set down or driven from. `yaw_reference_rad` re-aims it.
    """

    rotation_world_base: np.ndarray
    angular_velocity_world: np.ndarray

    def with_yaw_reference(self, yaw_reference_rad: float) -> "ImuState":
        """Re-express this state in a world frame whose +X points at `yaw_reference_rad`.

        Yawing the *world* frame (left-multiplying by Rz^T) redefines forward without
        touching gravity: roll, pitch and tilt are preserved exactly. Right-multiplying
        the mount rotation instead would rotate about *base* Z and corrupt tilt whenever
        the robot is not upright.
        """
        yaw_reference_rad = float(yaw_reference_rad)
        if yaw_reference_rad == 0.0:
            return self
        rotation = rotation_z(-yaw_reference_rad)
        return ImuState(
            rotation @ self.rotation_world_base,
            rotation @ self.angular_velocity_world,
        )

    @classmethod
    def from_device(cls, imu: Any, yaw_reference_rad: float = 0.0) -> "ImuState":
        rotation_world_sensor = np.asarray(imu.rotation_matrix, dtype=np.float64)
        angular_velocity_sensor = np.asarray(imu.ang_vel, dtype=np.float64)
        if rotation_world_sensor.shape != (3, 3):
            raise ValueError(f"Expected IMU rotation_matrix shape (3, 3), got {rotation_world_sensor.shape}")
        if angular_velocity_sensor.shape != (3,):
            raise ValueError(f"Expected IMU ang_vel shape (3,), got {angular_velocity_sensor.shape}")
        # The model's imu_site gives sensor orientation relative to the base:
        # R_ws = R_wb R_bs, therefore R_wb = R_ws R_bs^T.
        rotation_world_base = rotation_world_sensor @ R_BASE_SENSOR.T
        angular_velocity_world = rotation_world_sensor @ angular_velocity_sensor
        if not np.isfinite(rotation_world_base).all() or not np.isfinite(angular_velocity_world).all():
            raise ValueError("IMU returned non-finite data")
        if not np.allclose(
            rotation_world_base.T @ rotation_world_base, np.eye(3), atol=1e-2
        ) or np.linalg.det(rotation_world_base) < 0.9:
            raise ValueError("IMU rotation matrix is invalid")
        # Validate the raw matrix first; the yaw rotation is exactly orthonormal and
        # cannot turn a good matrix bad, but it would mask a bad one.
        return cls(rotation_world_base, angular_velocity_world).with_yaw_reference(
            yaw_reference_rad
        )


@dataclass(frozen=True)
class JointMapping:
    motor_id: int
    encoder_sign: int
    joint_to_travel_sign: int

    def validate(self) -> None:
        if self.motor_id not in MOTOR_IDS:
            raise ValueError(f"Unexpected motor ID 0x{self.motor_id:02X}")
        if self.encoder_sign not in (-1, 1):
            raise ValueError("encoder_sign must be -1 or +1")
        if self.joint_to_travel_sign not in (-1, 1):
            raise ValueError("joint_to_travel_sign must be -1 or +1")


@dataclass(frozen=True)
class SliderCalibration:
    joints: dict[str, JointMapping]

    @classmethod
    def load(cls, path: Path = CALIBRATION_FILE) -> "SliderCalibration":
        if not path.exists():
            raise FileNotFoundError(
                f"Slider calibration not found: {path}\n"
                "Run calibrate_sliders.py before controlling the robot."
            )
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if raw.get("schema_version") != 2:
            raise ValueError(
                f"{path} is not schema_version 2. Re-run calibrate_sliders.py; "
                "legacy X/Y/Z calibration is not safe for a trained policy."
            )
        raw_joints = raw.get("sliders", {}).get("joints", {})
        joints = {
            name: JointMapping(
                motor_id=int(raw_joints[name]["motor_id"]),
                encoder_sign=int(raw_joints[name]["encoder_sign"]),
                joint_to_travel_sign=int(raw_joints[name]["joint_to_travel_sign"]),
            )
            for name in JOINT_NAMES
            if name in raw_joints
        }
        calibration = cls(joints)
        calibration.validate()
        return calibration

    def validate(self) -> None:
        if set(self.joints) != set(JOINT_NAMES):
            missing = sorted(set(JOINT_NAMES) - set(self.joints))
            raise ValueError(f"Calibration is missing joints: {missing}")
        motor_ids = []
        for mapping in self.joints.values():
            mapping.validate()
            motor_ids.append(mapping.motor_id)
        if sorted(motor_ids) != sorted(MOTOR_IDS):
            raise ValueError(f"Each motor must be mapped exactly once; got {motor_ids}")

    def save(self, path: Path = CALIBRATION_FILE) -> None:
        self.validate()
        payload = {
            "schema_version": 2,
            "imu": {
                "source": "imu_site of MuJoCo model Sim_Model (policies/*/robot.xml)",
                "position_base_m": IMU_POSITION_BASE_M.tolist(),
                "quat_base_sensor_wxyz": IMU_QUAT_BASE_SENSOR_WXYZ.tolist(),
            },
            "sliders": {
                "motor_ids": list(MOTOR_IDS),
                "idle_position_mm": IDLE_POSITION_MM,
                "joints": {
                    name: {
                        "motor_id": mapping.motor_id,
                        "encoder_sign": mapping.encoder_sign,
                        "joint_to_travel_sign": mapping.joint_to_travel_sign,
                    }
                    for name, mapping in self.joints.items()
                },
            },
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


class MotorJointMapper:
    """Convert between model joint coordinates and physical motor encoder coordinates."""

    def __init__(self, calibration: SliderCalibration):
        self.calibration = calibration
        self.motor_index = {motor_id: idx for idx, motor_id in enumerate(MOTOR_IDS)}

    def joint_state_from_motors(
        self, motor_position_rad: np.ndarray, motor_velocity_rad_s: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        joint_pos = np.zeros(3, dtype=np.float32)
        joint_vel = np.zeros(3, dtype=np.float32)
        for joint_idx, joint_name in enumerate(JOINT_NAMES):
            mapping = self.calibration.joints[joint_name]
            motor_idx = self.motor_index[mapping.motor_id]
            travel_mm = float(rad_to_mm(mapping.encoder_sign * motor_position_rad[motor_idx]))
            travel_vel_mm_s = float(rad_to_mm(mapping.encoder_sign * motor_velocity_rad_s[motor_idx]))
            joint_pos[joint_idx] = (
                (travel_mm - IDLE_POSITION_MM) / 1000.0 / mapping.joint_to_travel_sign
            )
            joint_vel[joint_idx] = (
                travel_vel_mm_s / 1000.0 / mapping.joint_to_travel_sign
            )
        return joint_pos, joint_vel

    def motor_targets_from_joint_position(self, joint_position_m: np.ndarray) -> np.ndarray:
        joint_position_m = np.asarray(joint_position_m, dtype=np.float64)
        if joint_position_m.shape != (3,):
            raise ValueError(f"Expected 3 joint targets, got {joint_position_m.shape}")
        if not np.isfinite(joint_position_m).all():
            raise ValueError("Joint targets must be finite")
        joint_position_m = np.clip(joint_position_m, JOINT_LOWER_M, JOINT_UPPER_M)
        targets = np.zeros(3, dtype=np.float32)
        for joint_idx, joint_name in enumerate(JOINT_NAMES):
            mapping = self.calibration.joints[joint_name]
            motor_idx = self.motor_index[mapping.motor_id]
            travel_mm = (
                IDLE_POSITION_MM
                + mapping.joint_to_travel_sign * float(joint_position_m[joint_idx]) * 1000.0
            )
            travel_mm = float(
                np.clip(travel_mm, SAFE_MIN_TRAVEL_MM, SAFE_MAX_TRAVEL_MM)
            )
            targets[motor_idx] = mapping.encoder_sign * float(mm_to_rad(travel_mm))
        return targets

    def center_motor_targets(self) -> np.ndarray:
        return self.motor_targets_from_joint_position(np.zeros(3, dtype=np.float32))


class HeuristicPolicy:
    """The paper's geometric controller, in the MuJoCo base/joint convention."""

    name = "heuristic"

    def reset(self, _frame: np.ndarray | None = None) -> None:
        pass

    def act(
        self,
        direction_world_xy: np.ndarray,
        controller_magnitude: float,
        imu_state: ImuState,
        _joint_pos: np.ndarray,
        _joint_vel: np.ndarray,
    ) -> np.ndarray:
        direction = np.asarray(direction_world_xy, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        magnitude = float(np.clip(controller_magnitude, 0.0, 1.0))
        if norm < 1e-9 or magnitude <= 0.0:
            return np.zeros(3, dtype=np.float32)
        direction /= norm

        # Place a target point 1.5 m x magnitude out in the requested horizontal
        # world direction, then express that point in the rotating base frame.
        target_world_m = np.array(
            [direction[0] * 1.5 * magnitude, direction[1] * 1.5 * magnitude, 0.0]
        )
        target_base_m = imu_state.rotation_world_base.T @ target_world_m

        # Closest point on slider line i:
        #   line_i(q) = anchor_i + q * axis_i
        #   q_i = axis_i dot (target - anchor_i)
        joint_target = np.array(
            [
                np.dot(
                    SLIDER_AXES_BASE[joint],
                    target_base_m - SLIDER_ANCHORS_BASE_M[joint],
                )
                for joint in range(3)
            ],
            dtype=np.float64,
        )
        return np.clip(
            joint_target, HEURISTIC_LOWER_M, HEURISTIC_UPPER_M
        ).astype(np.float32)


@dataclass(frozen=True)
class PolicyDeployConfig:
    """The trained policy's own description of its contract, from `env_config.yaml`.

    Checked against this module's constants; a mismatch fails at startup.
    """

    history_length: int
    frame_dim: int
    action_dim: int
    action_scale_m: float
    control_hz: float
    term_dims: tuple[tuple[str, int], ...]
    timestep_s: float
    decimation: int

    # Observation term order the runtime assembles in `TrainedPolicy.build_frame`.
    EXPECTED_TERM_ORDER = (
        "angular_velocity",
        "commands_xy",
        "dofPosition",
        "dofVelocity",
        "actions",
        "base_rotation_matrix",
    )

    @classmethod
    def load(cls, path: Path) -> "PolicyDeployConfig":
        import yaml

        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        group = raw.get("policy_obs_group", "actor")
        terms = raw["observation"][group]
        term_dims = tuple((name, int(cfg["term_dim"])) for name, cfg in terms.items())
        histories = {int(cfg["history_length"]) for cfg in terms.values()}
        if len(histories) != 1:
            raise ValueError(
                f"{path}: observation terms disagree on history_length ({sorted(histories)}); "
                "the runtime keeps a single shared history buffer"
            )

        actuators = raw["actuators"]
        scales = {float(a["action_scale"]) for a in actuators}
        if len(scales) != 1:
            raise ValueError(
                f"{path}: per-actuator action_scale values differ ({sorted(scales)}); "
                "the runtime applies one scalar to all three sliders"
            )

        return cls(
            history_length=histories.pop(),
            frame_dim=sum(dim for _, dim in term_dims),
            action_dim=int(raw["action"]["dim"]),
            action_scale_m=scales.pop(),
            control_hz=float(raw["simulation"]["control_freq"]),
            term_dims=term_dims,
            timestep_s=float(raw["simulation"]["timestep"]),
            decimation=int(raw["simulation"]["decimation"]),
        )

    def validate(self, policy_hz: float) -> None:
        """Fail loudly if the artifact disagrees with this module's constants."""
        problems = []
        if tuple(name for name, _ in self.term_dims) != self.EXPECTED_TERM_ORDER:
            problems.append(
                f"observation term order is {[n for n, _ in self.term_dims]}, but "
                f"build_frame assembles {list(self.EXPECTED_TERM_ORDER)}"
            )
        if self.history_length != POLICY_HISTORY:
            problems.append(
                f"history_length {self.history_length} != POLICY_HISTORY {POLICY_HISTORY}"
            )
        if self.frame_dim != POLICY_FRAME_DIM:
            problems.append(
                f"observation frame is {self.frame_dim}D != POLICY_FRAME_DIM {POLICY_FRAME_DIM}"
            )
        if self.action_dim != len(JOINT_NAMES):
            problems.append(f"action dim {self.action_dim} != {len(JOINT_NAMES)} sliders")
        if not math.isclose(self.action_scale_m, ACTION_SCALE_M, rel_tol=1e-6):
            problems.append(
                f"action scale {self.action_scale_m} m != ACTION_SCALE_M {ACTION_SCALE_M} m"
            )
        if not math.isclose(self.control_hz, policy_hz, rel_tol=1e-6):
            problems.append(f"policy rate {self.control_hz} Hz != runtime rate {policy_hz} Hz")
        # control_freq must equal 1/(decimation*timestep); a hand-edited label is rejected.
        derived_hz = 1.0 / (self.decimation * self.timestep_s)
        if not math.isclose(derived_hz, self.control_hz, rel_tol=1e-6):
            problems.append(
                f"control_freq {self.control_hz} Hz disagrees with "
                f"1/(decimation {self.decimation} x timestep {self.timestep_s}) = "
                f"{derived_hz} Hz -- the artifact was edited by hand, re-export it"
            )
        if problems:
            raise RuntimeError(
                "Trained policy artifact does not match this runtime:\n  - "
                + "\n  - ".join(problems)
            )


class TrainedPolicy:
    """Trained TorchScript policy and its 3x23 observation contract."""

    name = "trained"

    def __init__(
        self,
        policy_path: Path = DEFAULT_POLICY_PATH,
        action_scale: float = 1.0,
        policy_hz: float = POLICY_HZ,
        strict: bool = True,
    ):
        if not 0.0 <= action_scale <= 1.0:
            raise ValueError("trained action scale must be in [0, 1]")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for trained control. Run the policy preflight "
                "inside the Orange Pi py312 environment."
            ) from exc
        self.torch = torch
        self.policy_path = Path(policy_path)
        if not self.policy_path.exists():
            raise FileNotFoundError(f"Trained TorchScript policy not found: {self.policy_path}")
        self.model = torch.jit.load(str(self.policy_path), map_location="cpu")
        self.model.eval()
        self.action_scale = float(action_scale)
        self.history: deque[np.ndarray] = deque(maxlen=POLICY_HISTORY)
        self.deploy_config = self._load_deploy_config(policy_hz, strict)
        self._preflight()

    def _load_deploy_config(
        self, policy_hz: float, strict: bool = True
    ) -> "PolicyDeployConfig | None":
        """Validate against the `env_config.yaml` beside the policy, if present.

        Without it only `_preflight`'s shape check runs. `strict=False` warns instead of
        raising; it is for offline diagnostics only, never for driving the robot.
        """
        config_path = self.policy_path.parent / "env_config.yaml"
        if not config_path.exists():
            print(
                f"NOTE: no env_config.yaml beside {self.policy_path.name}; skipping contract "
                "validation. Observation order and action scale are unverified."
            )
            return None
        config = PolicyDeployConfig.load(config_path)
        try:
            config.validate(policy_hz)
        except RuntimeError as exc:
            if strict:
                raise
            print(f"WARNING: proceeding despite a contract mismatch (strict=False):\n{exc}")
        return config

    def _preflight(self) -> None:
        dummy = self.torch.zeros((1, POLICY_HISTORY, POLICY_FRAME_DIM), dtype=self.torch.float32)
        with self.torch.inference_mode():
            output = self.model(dummy)
        if tuple(output.shape) != (1, 3):
            raise RuntimeError(
                f"Expected trained policy output shape (1, 3), got {tuple(output.shape)}"
            )
        if not bool(self.torch.isfinite(output).all()):
            raise RuntimeError("Policy preflight produced non-finite actions")

    @staticmethod
    def build_frame(
        command_world_xy: np.ndarray,
        imu_state: ImuState,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        last_action: np.ndarray,
    ) -> np.ndarray:
        frame = np.concatenate(
            [
                np.asarray(imu_state.angular_velocity_world, dtype=np.float32),
                np.asarray(command_world_xy, dtype=np.float32),
                np.asarray(joint_pos, dtype=np.float32),
                np.asarray(joint_vel, dtype=np.float32),
                np.asarray(last_action, dtype=np.float32),
                np.asarray(imu_state.rotation_world_base, dtype=np.float32).reshape(9),
            ]
        )
        if frame.shape != (POLICY_FRAME_DIM,):
            raise RuntimeError(f"V12 observation frame must be 23D, got {frame.shape}")
        return frame

    def reset(self, frame: np.ndarray | None = None) -> None:
        self.history.clear()
        initial = (
            np.zeros(POLICY_FRAME_DIM, dtype=np.float32)
            if frame is None
            else np.asarray(frame, dtype=np.float32)
        )
        for _ in range(POLICY_HISTORY):
            self.history.append(initial.copy())

    def set_action_scale(self, action_scale: float) -> None:
        self.action_scale = float(np.clip(action_scale, 0.0, 1.0))

    def act_from_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.history:
            self.reset(frame)
        self.history.append(np.asarray(frame, dtype=np.float32))
        obs = np.stack(tuple(self.history), axis=0)[None, ...]
        tensor = self.torch.from_numpy(obs)
        with self.torch.inference_mode():
            action = self.model(tensor).cpu().numpy()[0]
        if not np.isfinite(action).all():
            raise RuntimeError("Policy produced non-finite actions")
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        target = np.clip(
            action * ACTION_SCALE_M * self.action_scale,
            JOINT_LOWER_M,
            JOINT_UPPER_M,
        ).astype(np.float32)
        return target, action

