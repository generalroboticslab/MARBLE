"""Observation terms and buffers for optimized view-on-demand processing.

Defines the ObsTerm base class along with subclasses wrapping individual
MDP functions (from mjlab and task-specific observation modules).
"""

from __future__ import annotations
from typing import Literal
import torch
import numpy as np

from mjlab.envs import mdp
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.utils.noise import noise_cfg, noise_model

from .obs_buffer import HistoryBuffer, GatherDelayBuffer


class ObsTerm:
    """Per-term compute + per-term corrupt + per-term delay + per-term history + (optional) augmentation."""

    def __init__(
        self,
        func,
        params: dict | None = None,
        corrupt=None,
        clip: tuple[float, float] | None = None,
        scale: float | torch.Tensor | None = None,
        delay_min_lag: int = 0,
        delay_max_lag: int = 0,
        delay_per_env: bool = True,
        delay_hold_prob: float = 0.0,
        delay_update_period: int = 0,
        delay_per_env_phase: bool = True,
        history_length: int = 0,
        flatten_history_dim: bool = True,
        augment: dict | None = None,
    ):
        self.func = func
        self.params = params or {}
        self.corrupt_cfg = corrupt
        self.clip = clip
        self.scale = scale
        
        self.delay_min_lag = delay_min_lag
        self.delay_max_lag = delay_max_lag
        self.delay_per_env = delay_per_env
        self.delay_hold_prob = delay_hold_prob
        self.delay_update_period = delay_update_period
        self.delay_per_env_phase = delay_per_env_phase
        
        self.history_length = history_length
        self.flatten_history_dim = flatten_history_dim
        self.augment_fns = augment or {}

        self.env = None
        self.device = None
        self.num_envs = None
        self.dim = None
        
        self.corrupt_model = None
        self.delay_bufs: dict[str, GatherDelayBuffer] = {}
        self.history_bufs: dict[str, HistoryBuffer] = {}
        self.view_flatten_history: dict[str, bool] = {}

    def bind(self, env, num_envs: int, device):
        """Binds the environment, device, and shape configuration."""
        self.env = env
        self.num_envs = num_envs
        self.device = device

        # If self.func is a class type, instantiate it with a mock config and env
        if isinstance(self.func, type):
            class MockCfg:
                def __init__(self, params):
                    self.params = params
            self.func = self.func(MockCfg(self.params), env)

        # Determine feature dimension of the term function
        with torch.no_grad():
            dummy = self.func(env, **self.params)
            self.dim = dummy.shape[-1]

        if self.scale is not None:
            if isinstance(self.scale, (int, float)):
                pass
            elif not isinstance(self.scale, torch.Tensor):
                self.scale = torch.tensor(self.scale, dtype=torch.float32, device=device)
            else:
                self.scale = self.scale.to(device)

        # Initialize corrupt model
        if self.corrupt_cfg is not None:
            if hasattr(self.corrupt_cfg, "class_type"):
                self.corrupt_model = self.corrupt_cfg.class_type(
                    self.corrupt_cfg, num_envs=num_envs, device=device
                )
            else:
                self.corrupt_model = self.corrupt_cfg

    def init_buffers(self, requested_views: set[str], policy_history_len: int | None = None, policy_flatten_history: bool | None = None):
        """Pre-allocate history and delay buffers for requested views."""
        if not hasattr(self, "view_is_corrupt"):
            self.view_is_corrupt: dict[str, bool] = {}

        # Use policy overrides if provided (group-level overrides)
        raw_history_len = policy_history_len if policy_history_len is not None else self.history_length
        history_len = raw_history_len if raw_history_len is not None else 0
        flatten_history = policy_flatten_history if policy_flatten_history is not None else self.flatten_history_dim
        if flatten_history is None:
            flatten_history = True

        for view in requested_views:
            self.view_flatten_history[view] = flatten_history
            self.view_is_corrupt[view] = view.startswith("corrupt") or "aug_corrupt" in view
            # 1. History Buffer
            if history_len > 0:
                self.history_bufs[view] = HistoryBuffer(
                    max_len=history_len,
                    num_envs=self.num_envs,
                    dim=self.dim,
                    device=self.device,
                )
                history_out_dim = history_len * self.dim if flatten_history else history_len
            else:
                history_out_dim = self.dim

            # 2. Delay Buffer
            if self.delay_max_lag is not None and self.delay_max_lag > 0:
                # If history is not flattened, the output of get_window has shape (num_envs, history_length, dim)
                # We flatten it when pushing to/retrieving from the delay buffer
                total_dim = history_len * self.dim if (history_len > 0 and not flatten_history) else history_out_dim
                self.delay_bufs[view] = GatherDelayBuffer(
                    min_lag=self.delay_min_lag,
                    max_lag=self.delay_max_lag,
                    num_envs=self.num_envs,
                    dim=total_dim,
                    device=self.device,
                    per_env=self.delay_per_env,
                    hold_prob=self.delay_hold_prob,
                    update_period=self.delay_update_period,
                    per_env_phase=self.delay_per_env_phase,
                )

    def compute(self, env) -> torch.Tensor:
        """Evaluates raw function. Each view clones before mutating, so no clone here."""
        return self.func(env, **self.params)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        """Resets the state of the inner function, corrupt models, and view buffers."""
        if hasattr(self.func, "reset") and callable(self.func.reset):
            self.func.reset(env_ids=env_ids)
        if self.corrupt_model is not None and hasattr(self.corrupt_model, "reset") and callable(self.corrupt_model.reset):
            self.corrupt_model.reset(env_ids=env_ids)
        for db in self.delay_bufs.values():
            db.reset(env_ids)
        for hb in self.history_bufs.values():
            hb.reset(env_ids)

    def __call__(
        self, env, requested_views: set[str], update_history: bool = True, **_
    ) -> dict[str, torch.Tensor]:
        x = self.compute(env)
        views = {}

        for view in requested_views:
            if view in self.augment_fns:
                continue
            is_corrupt = self.view_is_corrupt.get(view)
            if is_corrupt is None:
                is_corrupt = view.startswith("corrupt") or "aug_corrupt" in view
                self.view_is_corrupt[view] = is_corrupt
            
            # Apply base value (clean or corrupt)
            if is_corrupt:
                if self.corrupt_model is not None:
                    if hasattr(self.corrupt_model, "apply"):
                        val = self.corrupt_model.apply(x.clone())
                    else:
                        val = self.corrupt_model(x.clone())
                else:
                    val = x.clone()
            else:
                val = x.clone()

            if self.clip is not None:
                val = val.clamp(min=self.clip[0], max=self.clip[1])
            if self.scale is not None:
                val = val * self.scale

            # Apply chronological buffer: history -> delay
            if view in self.history_bufs:
                hb = self.history_bufs[view]
                if update_history or not hb.is_initialized:
                    hb.append(val)
                val = hb.get_window()
                if self.view_flatten_history.get(view, self.flatten_history_dim):
                    val = val.reshape(self.num_envs, -1)

            if view in self.delay_bufs:
                db = self.delay_bufs[view]
                orig_shape = val.shape
                val_flat = val.reshape(self.num_envs, -1)
                if update_history or not db.is_initialized:
                    db.append(val_flat)
                val = db.compute(update_lags=update_history).reshape(orig_shape)

            views[view] = val

        # Handle augmentations. Pass a clone: compute() returns the raw (possibly
        # env-aliased) tensor, so cloning here prevents an augment fn that mutates
        # in-place from corrupting env state or other terms reading the same source.
        for aug_name in requested_views:
            if aug_name in self.augment_fns:
                views[aug_name] = self.augment_fns[aug_name](x.clone(), env)

        return views


# ---------------------------------------------------------------------------
# Specific Wrappers
# ---------------------------------------------------------------------------


class LastActionTerm(ObsTerm):
    def __init__(self, **kwargs):
        super().__init__(func=vel_mdp.last_action, **kwargs)


class CommandTerm(ObsTerm):
    def __init__(self, command_name="twist", **kwargs):
        params = kwargs.pop("params", None) or {"command_name": command_name}
        super().__init__(func=vel_mdp.generated_commands, params=params, **kwargs)


