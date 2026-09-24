"""Definitions vendored from the lab training repository for the ballbot tasks."""
from __future__ import annotations
from dataclasses import dataclass
from mjlab.actuator import BuiltinPositionActuatorCfg
import mujoco
import re
import numpy as np


M_PI = 3.14159265358979323846


NATURAL_FREQ_HIGH = 10.0 * 2.0 * M_PI


DAMPING_RATIO     = 2.0


@dataclass
class MotorSpec:
    """Motor parameters for one group of joints.

    For slide actuators, units differ from the hinge case:
      effort_limit [N], armature [kg], frictionloss [N], joint_friction [N],
      kp [N/m], kd [N/(m/s)].
    """
    effort_limit:  float
    armature:      float
    motor_type:    str                       = ""
    joints:        tuple[str, ...] | None    = None

    stiffness_range: tuple[float, float] = (1, 100)
    damping_range:   tuple[float, float] = (0.5, 5)

    joint_damping:  float | None = 0.0
    joint_friction: float | None = 5.0  # [N] passive friction (slide unit)
    natural_freq:   float        = NATURAL_FREQ_HIGH
    damping_ratio:  float        = DAMPING_RATIO

    kp: float | None = None
    kd: float | None = None

    @property
    def est_kp(self) -> float:
        return round(self.armature * self.natural_freq ** 2, 1)

    @property
    def est_kd(self) -> float:
        return round(2.0 * 1.0 * self.armature * self.natural_freq, 1)

    @property
    def action_scale(self) -> float:
        """Action scale: 0.1 m to utilize leg stroke safely without hitting joint limits."""
        return 0.1

    def to_actuator_cfg(
        self,
        delay_min_lag:     int = 0,
        delay_max_lag:     int = 0,
        delay_update_period: int = 0,
    ) -> BuiltinPositionActuatorCfg:
        return BuiltinPositionActuatorCfg(
            target_names_expr=self.joints,
            stiffness=self.kp,
            damping=self.kd,
            effort_limit=self.effort_limit,
            armature=self.armature,
            frictionloss=self.joint_friction,
            delay_min_lag=delay_min_lag,
            delay_max_lag=delay_max_lag,
            delay_update_period=delay_update_period,
        )


def _apply_joint_properties(spec: mujoco.MjSpec, motors: list[MotorSpec]) -> None:
    """Apply joint_damping and joint_friction from motor specs to matching joints."""
    for joint in spec.joints:
        for motor in motors:
            if any(re.fullmatch(pat, joint.name) for pat in motor.joints):
                if motor.joint_damping is not None:
                    _d    = np.atleast_1d(np.asarray(motor.joint_damping, dtype=np.float64)).ravel()
                    _damp = np.zeros(3, dtype=np.float64)
                    _damp[:len(_d)] = _d
                    joint.damping = _damp
                if motor.joint_friction is not None:
                    joint.frictionloss = motor.joint_friction
                break


SPHERE_BODY = "base_link"
