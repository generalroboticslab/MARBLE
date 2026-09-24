"""Deployment environment configuration dataclasses.

Defines the typed config structure for the deploy env. Can be constructed
programmatically or loaded from YAML files exported by run.py.

The YAML serialization format is unchanged — existing env_config.yaml files
load directly via DeployEnvCfg.from_yaml().

Assumptions:
- YAML files are exported by run.py:export_policy_for_deployment()
- All observation group names ("actor", "critic", etc.) are dynamic dict keys
- joint_pos uses either exact joint names or regex patterns as keys
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Optional

import yaml


def _from_dict(cls, d: dict):
    """Construct a dataclass from a dict, ignoring unknown keys.

    Extra keys in d are dropped; missing keys use the dataclass defaults.
    This makes loading tolerant of YAML/Python version mismatches.
    """
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class SimulationCfg:
    """Simulation parameters for deployment environment."""
    timestep: float = 0.005
    decimation: int = 4
    control_freq: float = 50.0
    # Enable granular gravity compensation via feedforward torque injection.
    # The solver calculates qfrc_bias (gravity + Coriolis) and applies it to masked joints.
    # - True: Compensate all actuated joints.
    # - False: Disable compensation entirely.
    # - List[str]: List of joint name patterns (regex) to compensate (e.g. ["shoulder", "elbow"]).
    # Default: ["shoulder", "elbow", "wrist"] (compensates arms only to preserve leg stability).
    gravity_compensation: bool | list[str] = field(default_factory=lambda: ["shoulder", "elbow", "wrist"])





@dataclass
class DefaultPoseCfg:
    root_position: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.59])
    # Keys are exact joint names or regex patterns; values are radians
    joint_pos: dict[str, float] = field(default_factory=dict)


@dataclass
class ObsTermCfg:
    history_length: int = 0
    flatten_history_dim: bool = True
    scale: Optional[float] = None
    clip: Optional[float] = None
    delay_hold_prob: float = 0.0
    delay_max_lag: int = 0
    delay_min_lag: int = 0
    term_dim: Optional[int] = None
    # Term-specific keyword arguments forwarded to the observation function.
    # Mirrors Isaac Lab's ObservationTermCfg.params — each observation term class
    # reads its own keys from here (e.g. "period_range", "lin_thresh" for gait_phase;
    # "ee_body_name" for end-effector terms). Exported verbatim from training config
    # and consumed at deployment time by the deploy env.
    params: dict = field(default_factory=dict)


@dataclass
class ActionCfg:
    dim: int = 0
    low_pass_alpha: float = 1.0
    clip_actions: Optional[float] = None
    controlled_joints: list[str] = field(default_factory=list)


@dataclass
class ActuatorCfg:
    joints: list[str] = field(default_factory=list)  # regex patterns matching joint names
    kp: float = 0.0
    kd: float = 0.0
    action_scale: float = 0.25
    effort_limit: Optional[float] = None
    armature: float = 0.0
    joint_damping: float = 0.0
    joint_friction: float = 0.0


@dataclass
class DeployEnvCfg:
    """Complete deployment environment configuration.

    Construct programmatically, load from YAML via from_yaml()/from_dict(),
    or serialize back to YAML via to_yaml().

    observation maps group_name -> (term_name -> ObsTermCfg).
    Group names ("actor", "critic", etc.) vary per task, so they stay as
    dynamic dict keys rather than fixed fields.
    """
    simulation: SimulationCfg = field(default_factory=SimulationCfg)
    default_pose: DefaultPoseCfg = field(default_factory=DefaultPoseCfg)
    observation: dict[str, dict[str, ObsTermCfg]] = field(default_factory=dict)
    action: ActionCfg = field(default_factory=ActionCfg)
    actuators: list[ActuatorCfg] = field(default_factory=list)
    foot_bodies: list[str] = field(default_factory=list)
    obs_dim: Optional[int] = None
    # Observation group the deployed policy consumes. Defaults to
    # "actor" so pre-existing configs (no field) keep their old behavior.
    policy_obs_group: str = "actor"
    # Whether the exported policy's forward_with_estimation yields a valid velocity
    # estimate. The FlashSAC wrapper always defines that method but raises unless
    # the actor has a velocity estimator; this flag is the authoritative signal.
    has_velocity_estimator: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> DeployEnvCfg:
        """Load from a YAML file exported by run.py."""
        with open(path, 'r') as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def from_dict(cls, d: dict) -> DeployEnvCfg:
        """Construct from a plain dict (e.g. parsed YAML)."""
        obs = {
            group_name: {name: _from_dict(ObsTermCfg, term) for name, term in terms.items()}
            for group_name, terms in d.get("observation", {}).items()
        }
        return cls(
            simulation=_from_dict(SimulationCfg, d.get("simulation", {})),
            default_pose=_from_dict(DefaultPoseCfg, d.get("default_pose", {})),
            observation=obs,
            action=_from_dict(ActionCfg, d.get("action", {})),
            actuators=[_from_dict(ActuatorCfg, a) for a in d.get("actuators", [])],
            foot_bodies=d.get("foot_bodies", []),
            obs_dim=d.get("obs_dim"),
            policy_obs_group=d.get("policy_obs_group", "actor"),
            has_velocity_estimator=d.get("has_velocity_estimator", False),
        )

    def to_yaml(self, path: str) -> None:
        """Serialize to YAML."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
