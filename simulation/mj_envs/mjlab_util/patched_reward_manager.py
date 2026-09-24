"""Restores the log_metrics protocol dropped from vanilla mjlab's RewardManager.

Problem:
    RewardManager.compute() skips weight=0 terms entirely (`continue`). Some
    class-based reward terms need to log monitoring scalars (e.g. running means)
    even when their weight is 0 — the penalty must not be computed (no gradient,
    no cost), but the metric side-effect still must run every step.

Solution:
    A term opts in by implementing `log_metrics(env: ManagerBasedRlEnv) -> None`.
    Patched compute() calls it in place of `__call__` when weight == 0.0, instead
    of just skipping the term.

Consumers: reward terms that implement log_metrics.

Mechanism: monkeypatches RewardManager.compute directly on the shared class
object (`RewardManager.compute = _patched_compute`), rather than replacing the
class via subclass + dual-rebind. RewardManager's instance shape is unchanged
(no __init__ override needed), so mutating the method in place is enough — every
existing reference to the name `RewardManager` (mjlab.managers.reward_manager,
mjlab.envs.manager_based_rl_env) already points at this same class object, so no
second rebind is required the way ChronologicalObservationManager needs one.
"""

from __future__ import annotations

import torch

from mjlab.managers.reward_manager import RewardManager


def _patched_compute(self: RewardManager, dt: float) -> torch.Tensor:
    self._reward_buf[:] = 0.0
    scale = dt if self._scale_by_dt else 1.0
    for term_idx, (name, term_cfg) in enumerate(
        zip(self._term_names, self._term_cfgs, strict=False)
    ):
        if term_cfg.weight == 0.0:
            self._step_reward[:, term_idx] = 0.0
            # log_metrics protocol: class-based terms may implement log_metrics(env)
            # to log monitoring scalars even when weight=0. Penalty computation is
            # skipped; only the cheap metric side-effect runs.
            if hasattr(term_cfg.func, "log_metrics"):
                term_cfg.func.log_metrics(self._env)
            continue
        value = term_cfg.func(self._env, **term_cfg.params)
        self._check_term_shape(name, value)
        value = value * term_cfg.weight * scale
        # NaN/Inf can occur from corrupted physics state; zero them to avoid policy crash.
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        self._reward_buf += value
        self._episode_sums[name] += value
        self._step_reward[:, term_idx] = value / scale
    return self._reward_buf


RewardManager.compute = _patched_compute
