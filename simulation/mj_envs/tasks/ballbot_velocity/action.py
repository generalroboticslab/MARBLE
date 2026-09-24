"""Action term helpers: slider position action with the driver's velocity clamp."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


@dataclass(kw_only=True)
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Joint position action whose TARGET may not slew faster than `vel_limit`.

    Models the motor driver's velocity clamp (`--vel-limit`), which the deployed launch passes
    and which sim previously had no equivalent of: a MuJoCo position actuator clamps force but
    not speed, so a policy could demand — and get — a carriage slew far beyond what the hardware
    will ever produce.

    The limit is applied to the target, not the achieved velocity, because that is what the
    driver does: it rate-limits the commanded position and the servo tracks it. With this
    drive's gains (zeta 8.2) the carriage follows the target closely, so clamping the target
    clamps the motion.
    """

    vel_limit: float  # m/s at the carriage

    def build(self, env: ManagerBasedRlEnv) -> "RateLimitedJointPositionAction":
        return RateLimitedJointPositionAction(self, env)


class RateLimitedJointPositionAction(JointPositionAction):
    """JointPositionAction + per-control-step slew clamp on the processed target."""

    def __init__(self, cfg: RateLimitedJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # Per POLICY step, not per physics substep: the driver receives a new target once per
        # control step, so that is the interval over which the clamp applies.
        self._max_draw = cfg.vel_limit * env.step_dt
        # Own buffer, written only with copy_: cloning an entity-data tensor yields an inference
        # tensor, which cannot then be updated in place outside inference mode (reset runs there).
        self._prev_target = torch.zeros_like(self._raw_actions)
        self._prev_target.copy_(self._entity.data.default_joint_pos[:, self._target_ids])

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._processed_actions = torch.clamp(
            self._processed_actions,
            self._prev_target - self._max_draw,
            self._prev_target + self._max_draw,
        )
        self._prev_target.copy_(self._processed_actions)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        super().reset(env_ids)
        # Seed from where the sliders actually ARE after the reset, so the first commanded step
        # is limited relative to the true carriage position rather than to a stale target.
        idx = slice(None) if env_ids is None else env_ids
        self._prev_target[idx] = self._entity.data.joint_pos[:, self._target_ids][idx]
