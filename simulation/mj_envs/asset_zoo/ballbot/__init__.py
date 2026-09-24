"""Ballbot (underwater sphere-slider) configuration for mjlab RL training."""

from .ballbot_constants import (
    get_ballbot_robot_cfg,
    get_action_scale,
    INIT_BASE_Z,
    HULL_COLLISION,
    SPHERE_BODY,
    SLIDER_JOINT_NAMES,
    ACTION_SCALE,
    SHELL_XML_PATH,
    COMPLEX_XML_PATH,
    COMPLEX_HULL_RADIUS,
    get_ballbot_complex_robot_cfg,
)

__all__ = [
    "get_ballbot_robot_cfg",
    "get_action_scale",
    "INIT_BASE_Z",
    "HULL_COLLISION",
    "SPHERE_BODY",
    "SLIDER_JOINT_NAMES",
    "ACTION_SCALE",
    "SHELL_XML_PATH",
    "COMPLEX_XML_PATH",
    "COMPLEX_HULL_RADIUS",
    "get_ballbot_complex_robot_cfg",
]
