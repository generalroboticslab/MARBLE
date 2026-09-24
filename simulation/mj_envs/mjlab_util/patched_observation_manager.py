"""ChronologicalObservationManager: Fixes mjlab's delay→history temporal scrambling bug.

Problem:
    mjlab applies DelayBuffer BEFORE CircularBuffer, causing history frames to be
    temporally scrambled when sensor delays change. Example:
        Step 8: delay=2 → obs(t=6) → buffer[0]
        Step 9: delay=0 → obs(t=9) → buffer[1]
        Step 10: delay=1 → obs(t=9) → buffer[2]
        Result: [obs(t=6), obs(t=9), obs(t=9)] ❌ NON-CHRONOLOGICAL

    This breaks temporal reasoning (velocity estimation, momentum tracking) and causes
    violent oscillation in trained policies.

Solution:
    Apply CircularBuffer FIRST, then DelayBuffer:
        buffer: [obs(t=8), obs(t=9), obs(t=10)] ✓ chronological
        delay=2 → return buffer from t=[6,7,8] ✓ delayed but chronological

Implementation:
    Forward-compatible wrapper that:
    1. Intercepts cfg during __init__, captures history/delay settings
    2. Zeros out mjlab's internal buffers (history_length=0, delay_max_lag=0)
    3. Forces mjlab to return un-concatenated raw tensors (concatenate_terms=False)
    4. Applies custom buffers in correct order: history → delay
    5. Manually concatenates if original config requested it

This approach treats mjlab as a black box (no copied logic), ensuring forward compatibility.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import copy
import torch

from mjlab.managers.observation_manager import ObservationManager, ObservationGroupCfg
from mjlab.utils.buffers import CircularBuffer, DelayBuffer

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


class ChronologicalObservationManager(ObservationManager):
    """Zero-allocation, view-on-demand custom observation manager for all policies.
    
    Acts as a drop-in replacement for the default ObservationManager in any environment.
    Wraps the optimized ObsBuffer and applies the chronological (history -> delay) flow.
    """

    def __init__(self, cfg: dict[str, ObservationGroupCfg], env: ManagerBasedRlEnv):
        """Intercept config and initialize parent and custom ObsBuffer.

        Args:
            cfg: Original observation configs from environment.
            env: Parent environment (needed for shape determination, device, num_envs).
        """
        # Capture original concatenation settings
        self._original_concat = {}
        self._original_concat_dim = {}

        # Wrap env.cfg.observations using ObservationsProxy to route configuration
        # updates to flat registries dynamically.
        if hasattr(env, "cfg") and env.cfg is not None:
            if not hasattr(env.cfg, "observation_terms"):
                env.cfg.observation_terms = {}
                env.cfg.policy_observations = {}
            from mjlab_util.obs_buffer import ObservationsProxy
            if not isinstance(getattr(env.cfg, "observations", None), ObservationsProxy):
                proxy = ObservationsProxy(env.cfg.observation_terms, env.cfg.policy_observations)
                # Transfer existing groups if observations was a standard dict
                orig_obs = getattr(env.cfg, "observations", {})
                if isinstance(orig_obs, dict):
                    for k, v in list(orig_obs.items()):
                        proxy[k] = v
                env.cfg.observations = proxy

        # Deep copy config to avoid modifying user's original config
        modified_cfg = copy.deepcopy(cfg)

        # Zero out history/delay buffers in parent's config so mjlab doesn't allocate duplicate buffers
        for group_name, group_cfg in modified_cfg.items():
            self._original_concat[group_name] = group_cfg.concatenate_terms
            self._original_concat_dim[group_name] = group_cfg.concatenate_dim
            group_cfg.concatenate_terms = False  # Force dict returns in parent (prevent double concat)

            group_history_length = getattr(group_cfg, "history_length", None)
            group_flatten = getattr(group_cfg, "flatten_history_dim", True)
            group_cfg.history_length = None

            for term_cfg in group_cfg.terms.values():
                if group_history_length is not None:
                    term_cfg.history_length = group_history_length
                    term_cfg.flatten_history_dim = group_flatten

                # Zero buffer settings in parent
                term_cfg.history_length = 0
                term_cfg.delay_min_lag = 0
                term_cfg.delay_max_lag = 0

        # Initialize parent with modified config
        super().__init__(modified_cfg, env)

        # Build optimized custom observation buffer using original config
        # We must resolve SceneEntityCfg objects first so their dimensions are correct during bind()
        resolved_cfg = copy.deepcopy(env.cfg.observations)
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        for group_cfg in resolved_cfg.values():
            for term_cfg in group_cfg.terms.values():
                for value in term_cfg.params.values():
                    if isinstance(value, SceneEntityCfg):
                        value.resolve(env.scene)

        from mjlab_util.obs_buffer import build_obs_buffer_from_config
        self.obs_buffer = build_obs_buffer_from_config(resolved_cfg, env.num_envs, env.device)
        self.obs_buffer.bind(env)

        # Cached values to prevent double-pushing to buffers when compute_group is called multiple times per step
        self._last_ep_len_buf: torch.Tensor | None = None

    def compute(
        self,
        update_history: bool = False,
        env_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Compute all group observations in a single step call and cache the results.

        `env_ids` (mjlab 1.6.0) is ACCEPTED AND IGNORED. Upstream added it so a partial
        reset backfills only the reset envs' buffers without advancing the others' — see
        ObservationManager.compute and CircularBuffer.backfill. Our ObsBuffer already
        scopes the post-reset backfill per env (HistoryBuffer.reset_env_ids), but its ring
        POINTER is a single global scalar, so this call still advances every env's window
        by one frame. That is unchanged pre-1.6.0 behavior, not a new regression; porting
        upstream's non-advancing backfill needs a backfill-only path threaded through
        ObsBuffer.step and ObsTerm, which is a separate change.

        RETURNS VIEWS, NOT COPIES (footgun): every group tensor here is a view into the
        ObsBuffer's persistent ring storage (history groups via get_window() ->
        buffer[:, ptr:ptr+L]; non-concat groups via raw_obs[..., sl]). That storage is
        overwritten IN PLACE on the next step. Any caller that PERSISTS or CARRIES obs across
        steps (e.g. an off-policy replay buffer) MUST clone first, or the stored obs silently
        mutate to a later state. ManagerBasedRlEnvWithFinalObs.step clones obs_buf at the env
        boundary for exactly this reason — see its docstring.

        Freshness guard: the cached _obs_buffer is reused only while the sim state has NOT
        advanced since it was built (same episode_length_buf), mirroring compute_group.
        Without the freshness check, a compute(update_history=False) call -- which
        env_wrapper uses to capture terminal final_obs AFTER physics has advanced -- would
        return the PREVIOUS step's cached reconstruction (o_t) instead of the post-physics
        terminal obs (o_{t+1}). History groups are unaffected (window not advanced when
        update_history=False), but LIVE groups (critic, estimator_target) would be stale,
        feeding a one-step-stale critic obs into the SAC Bellman bootstrap on every timeout
        (runner stores final_obs[critic] as next_critic) -> corrupted value target ->
        from-scratch training collapse while inference (actor group) stays correct. This
        guard was missing after the dfd9d9f obs-manager rewrite; it is the regression fix.
        """
        same_step = (
            self._last_ep_len_buf is not None
            and self._last_ep_len_buf.shape == self._env.episode_length_buf.shape
            and torch.equal(self._last_ep_len_buf, self._env.episode_length_buf)
        )
        if not update_history and self._obs_buffer is not None and same_step:
            return self._obs_buffer

        if update_history:
            self.obs_buffer.step(self._env, update_history=True)
            self._last_ep_len_buf = self._env.episode_length_buf.clone()
        else:
            self.obs_buffer.step(self._env, update_history=False)
            self._last_ep_len_buf = self._env.episode_length_buf.clone()

        obs_buffer = {}
        for group_name in self._group_obs_term_names:
            raw_obs = self.obs_buffer.get(group_name)
            if not self._original_concat[group_name]:
                group_cfg = self.cfg[group_name]
                group_terms = list(group_cfg.terms.keys())
                slices = self.obs_buffer._policy_slices[group_name]
                obs_buffer[group_name] = {
                    term_name: raw_obs[..., sl]
                    for term_name, (_, _, sl) in zip(group_terms, slices)
                }
            else:
                obs_buffer[group_name] = raw_obs

        self._obs_buffer = obs_buffer
        return obs_buffer

    def compute_group(
        self,
        group_name: str,
        update_history: bool = True,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute observation group with corrected history->delay buffer order.

        `env_ids` accepted and ignored — see compute().
        """
        same_step = (
            self._last_ep_len_buf is not None
            and self._last_ep_len_buf.shape == self._env.episode_length_buf.shape
            and torch.equal(self._last_ep_len_buf, self._env.episode_length_buf)
        )
        if update_history:
            if not same_step:
                self.obs_buffer.step(self._env, update_history=True)
                self._last_ep_len_buf = self._env.episode_length_buf.clone()
            else:
                self.obs_buffer.step(self._env, update_history=False)
        else:
            self.obs_buffer.step(self._env, update_history=False)

        raw_obs = self.obs_buffer.get(group_name)
        if not self._original_concat[group_name]:
            group_cfg = self.cfg[group_name]
            group_terms = list(group_cfg.terms.keys())
            slices = self.obs_buffer._policy_slices[group_name]
            return {
                term_name: raw_obs[..., sl]
                for term_name, (_, _, sl) in zip(group_terms, slices)
            }

        return raw_obs

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, torch.Tensor | dict]:
        """Reset buffers for specified environments."""
        # Reset parent's buffers (noise models, etc.)
        result = super().reset(env_ids)
        # Reset custom terms delay and history buffers
        self.obs_buffer.reset(env_ids)
        # Clear cache/steps tracker
        self._obs_buffer = None
        self._last_ep_len_buf = None
        return result


# Monkeypatch mjlab to use ChronologicalObservationManager by default
import mjlab.managers.observation_manager
import mjlab.envs.manager_based_rl_env

mjlab.managers.observation_manager.ObservationManager = ChronologicalObservationManager

# Env construction binds the name imported INTO manager_based_rl_env's namespace
# (manager_based_rl_env.py: `from ...observation_manager import ObservationManager`, then
# `self.observation_manager = ObservationManager(...)`), so THIS rebind is the one that
# takes effect at env build — the source-module rebind above does not. Hard-assert it took:
# a silent fallback to the unpatched base manager would reintroduce the delay->history
# temporal-scrambling bug with no error, only wrong training.
assert hasattr(mjlab.envs.manager_based_rl_env, "ObservationManager"), (
    "mjlab.envs.manager_based_rl_env no longer imports ObservationManager into its namespace; "
    "ChronologicalObservationManager monkeypatch would silently not apply at env construction."
)
mjlab.envs.manager_based_rl_env.ObservationManager = ChronologicalObservationManager

