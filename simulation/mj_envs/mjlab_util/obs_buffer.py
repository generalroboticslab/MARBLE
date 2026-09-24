"""Custom Observation Buffer registry and assembly.

Implements HistoryBuffer, GatherDelayBuffer, PolicyObsSpec,
the proxy config classes, and the main ObsBuffer manager.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Sequence, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from .observation_terms import ObsTerm


@dataclass
class PolicyObsSpec:
    """Configuration for a policy's observation slots and buffer layouts."""
    name: str
    slots: list[tuple[str, str]]  # [("term_name", "view_name"), ...]
    normalizer: Literal["empirical", "identity"] = "empirical"
    history_length: int = 0
    delay: tuple[int, int] = (0, 0)
    flatten_history_dim: bool = True
    concatenate_terms: bool = True
    concatenate_dim: int = -1
    enable_corruption: bool = True


class HistoryBuffer:
    """Contiguous chronological window of length L over 2L pre-allocated buffer.

    Zero allocations or index sorting on retrieval. View slice is contiguous.
    """

    def __init__(self, max_len: int, num_envs: int, dim: int, device, dtype=torch.float32):
        self.max_len = max_len
        self.num_envs = num_envs
        self.dim = dim
        self.device = device
        self.buffer = torch.zeros((num_envs, 2 * max_len, dim), dtype=dtype, device=device)
        self.pointer = 0
        self.num_pushes = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.reset_env_ids = torch.arange(num_envs, device=device)  # initial backfill
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def append(self, x: torch.Tensor):
        self.buffer[:, self.pointer] = x
        self.buffer[:, self.pointer + self.max_len] = x
        
        # Sparse backfill only on resets, avoiding torch.where/syncs on standard steps
        if self.reset_env_ids is not None:
            self.buffer[self.reset_env_ids, :] = x[self.reset_env_ids].unsqueeze(1)
            self.reset_env_ids = None
            
        self.pointer = (self.pointer + 1) % self.max_len
        self.num_pushes += 1
        self._is_initialized = True

    def get_window(self) -> torch.Tensor:
        # Returns a VIEW into self.buffer (the persistent ring), not a copy. The ring is
        # overwritten in place by append(); callers that persist/carry this across steps
        # (e.g. an off-policy replay buffer) MUST clone or the data mutates underneath them.
        # See ManagerBasedRlEnvWithFinalObs.
        return self.buffer[:, self.pointer : self.pointer + self.max_len]

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        """Reset history buffer state for specified env_ids (or all if None)."""
        if env_ids is None:
            self.buffer.fill_(0.0)
            self.num_pushes.fill_(0)
            self.reset_env_ids = torch.arange(self.num_envs, device=self.device)
            self._is_initialized = False
        else:
            if isinstance(env_ids, slice):
                indices = range(*env_ids.indices(self.num_envs))
                env_ids = torch.tensor(list(indices), dtype=torch.long, device=self.device)
            elif not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(list(env_ids), dtype=torch.long, device=self.device)
            
            self.buffer[env_ids] = 0.0
            self.num_pushes[env_ids] = 0
            if self.reset_env_ids is not None:
                self.reset_env_ids = torch.unique(torch.cat([self.reset_env_ids, env_ids]))
            else:
                self.reset_env_ids = env_ids


class GatherDelayBuffer:
    """Stochastic delay buffer utilizing coalesced torch.gather instead of advanced indexing."""

    def __init__(
        self,
        min_lag: int,
        max_lag: int,
        num_envs: int,
        dim: int,
        device,
        per_env: bool = True,
        hold_prob: float = 0.0,
        update_period: int = 0,
        per_env_phase: bool = True,
        generator: torch.Generator | None = None,
    ):
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.num_envs = num_envs
        self.dim = dim
        self.device = device
        self.per_env = per_env
        self.hold_prob = hold_prob
        self.update_period = update_period
        self.per_env_phase = per_env_phase
        self.generator = generator

        self.buf_size = max_lag + 1 if max_lag > 0 else 1
        self.buffer = torch.zeros((num_envs, self.buf_size, dim), device=device)
        self.pointer = 0
        self.num_pushes = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.current_lags = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.reset_env_ids = torch.arange(num_envs, device=device)
        self._is_initialized = False

        if update_period > 0 and per_env_phase:
            self.phase_offsets = torch.randint(
                0, update_period, (num_envs,), dtype=torch.long, device=device, generator=generator
            )
        else:
            self.phase_offsets = torch.zeros(num_envs, dtype=torch.long, device=device)

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        """Reset delay buffer state for specified env_ids (or all if None)."""
        if env_ids is None:
            self.buffer.fill_(0.0)
            self.num_pushes.fill_(0)
            self.current_lags.fill_(0)
            self.step_count.fill_(0)
            self.reset_env_ids = torch.arange(self.num_envs, device=self.device)
            self._is_initialized = False
            if self.update_period > 0 and self.per_env_phase:
                self.phase_offsets = torch.randint(
                    0, self.update_period, (self.num_envs,), dtype=torch.long, device=self.device, generator=self.generator
                )
        else:
            if isinstance(env_ids, slice):
                indices = range(*env_ids.indices(self.num_envs))
                env_ids = torch.tensor(list(indices), dtype=torch.long, device=self.device)
            elif not isinstance(env_ids, torch.Tensor):
                env_ids = torch.tensor(list(env_ids), dtype=torch.long, device=self.device)

            self.buffer[env_ids] = 0.0
            self.num_pushes[env_ids] = 0
            self.current_lags[env_ids] = 0
            self.step_count[env_ids] = 0
            if self.reset_env_ids is not None:
                self.reset_env_ids = torch.unique(torch.cat([self.reset_env_ids, env_ids]))
            else:
                self.reset_env_ids = env_ids
            if self.update_period > 0 and self.per_env_phase:
                new_phases = torch.randint(
                    0, self.update_period, (self.num_envs,), dtype=torch.long, device=self.device, generator=self.generator
                )
                self.phase_offsets[env_ids] = new_phases[env_ids]

    def append(self, x: torch.Tensor):
        self.buffer[:, self.pointer] = x
        if self.reset_env_ids is not None:
            self.buffer[self.reset_env_ids, :] = x[self.reset_env_ids].unsqueeze(1)
            self.reset_env_ids = None
        self.pointer = (self.pointer + 1) % self.buf_size
        self.num_pushes += 1
        self._is_initialized = True

    def set_lags(self, lags: torch.Tensor):
        self.current_lags.copy_(lags.clamp(self.min_lag, self.max_lag))

    def _update_lags(self) -> None:
        if self.update_period > 0:
            phase_adjusted_count = (self.step_count + self.phase_offsets) % self.update_period
            should_update = phase_adjusted_count == 0
        else:
            should_update = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        new_lags = self._sample_lags(should_update)
        self.current_lags = torch.where(should_update, new_lags, self.current_lags)
        self.step_count += 1

    def _sample_lags(self, mask: torch.Tensor) -> torch.Tensor:
        if self.per_env:
            candidate_lags = torch.randint(
                self.min_lag, self.max_lag + 1, (self.num_envs,),
                dtype=torch.long, device=self.device, generator=self.generator
            )
        else:
            shared_lag = torch.randint(
                self.min_lag, self.max_lag + 1, (1,),
                dtype=torch.long, device=self.device, generator=self.generator
            )
            candidate_lags = shared_lag.expand(self.num_envs)

        if self.hold_prob > 0.0:
            should_sample = (
                torch.rand(self.num_envs, dtype=torch.float32, device=self.device, generator=self.generator)
                >= self.hold_prob
            )
            update_mask = mask & should_sample
        else:
            update_mask = mask

        return torch.where(update_mask, candidate_lags, self.current_lags)

    def compute(self, update_lags: bool = True) -> torch.Tensor:
        if not self._is_initialized:
            raise RuntimeError("Buffer not initialized. Call append() first.")
        # Advance the stochastic lag schedule only on real history-advancing steps.
        # Non-history "peek" reads (final-obs snapshot, same-step re-reads) must NOT
        # resample lags or bump step_count, else the delay schedule decouples from
        # true timesteps (extra reads per step => lags advance too fast).
        if update_lags:
            self._update_lags()
            
        if self.buf_size == 1:
            return self.buffer[:, 0].clone()

        valid_lags = torch.minimum(self.current_lags, self.num_pushes - 1).clamp_min(0)
        idx = (self.pointer - 1 - valid_lags) % self.buf_size
        idx_expanded = idx.view(self.num_envs, 1, 1).expand(self.num_envs, 1, self.dim)
        return torch.gather(self.buffer, 1, idx_expanded).squeeze(1)


def configs_are_equivalent(c1, c2) -> bool:
    if getattr(c1, "func", None) != getattr(c2, "func", None):
        return False
    if getattr(c1, "params", {}) != getattr(c2, "params", {}):
        return False
    corrupt1 = getattr(c1, "corrupt", None)
    if corrupt1 is None:
        corrupt1 = getattr(c1, "noise", None)
    corrupt2 = getattr(c2, "corrupt", None)
    if corrupt2 is None:
        corrupt2 = getattr(c2, "noise", None)
    if corrupt1 != corrupt2:
        return False
    if getattr(c1, "clip", None) != getattr(c2, "clip", None):
        return False
    if getattr(c1, "scale", None) != getattr(c2, "scale", None):
        return False
    for attr in [
        "delay_min_lag",
        "delay_max_lag",
        "delay_hold_prob",
        "delay_update_period",
        "delay_per_env",
        "delay_per_env_phase",
        "history_length",
        "flatten_history_dim",
    ]:
        if getattr(c1, attr, None) != getattr(c2, attr, None):
            return False
    return True


class TermsProxy:
    def __init__(self, policy_name: str, observation_terms: dict, spec: PolicyObsSpec):
        self._policy_name = policy_name
        self._observation_terms = observation_terms
        self._spec = spec

    def __setitem__(self, key: str, value):
        self._observation_terms[key] = value
        # Ensure it exists in policy slots
        view_base = "clean" if "critic" in self._policy_name else "corrupt"
        view = f"{view_base}_{self._policy_name}"
        if not any(slot[0] == key for slot in self._spec.slots):
            self._spec.slots.append((key, view))

    def __getitem__(self, key: str):
        return self._observation_terms[key]

    def __delitem__(self, key: str):
        self._spec.slots = [slot for slot in self._spec.slots if slot[0] != key]

    def update(self, other: dict):
        for k, v in other.items():
            self[k] = v

    def keys(self):
        return [slot[0] for slot in self._spec.slots]

    def values(self):
        return [self._observation_terms[name] for name in self.keys()]

    def items(self):
        return [(name, self._observation_terms[name]) for name in self.keys()]

    def __contains__(self, key: str) -> bool:
        return any(slot[0] == key for slot in self._spec.slots)


class PolicyObsGroupProxy:
    def __init__(self, policy_name: str, observation_terms: dict, spec: PolicyObsSpec):
        self._policy_name = policy_name
        self._observation_terms = observation_terms
        self._spec = spec
        self.terms = TermsProxy(policy_name, observation_terms, spec)

    # Group-level settings live on the PolicyObsSpec dataclass; expose them so mjlab's
    # group-major code (which reads/writes group_cfg.history_length etc.) routes through.
    @property
    def history_length(self) -> int:
        return self._spec.history_length

    @history_length.setter
    def history_length(self, val: int):
        self._spec.history_length = val

    @property
    def flatten_history_dim(self) -> bool:
        return self._spec.flatten_history_dim

    @flatten_history_dim.setter
    def flatten_history_dim(self, val: bool):
        self._spec.flatten_history_dim = val

    @property
    def concatenate_terms(self) -> bool:
        return self._spec.concatenate_terms

    @concatenate_terms.setter
    def concatenate_terms(self, val: bool):
        self._spec.concatenate_terms = val

    @property
    def concatenate_dim(self) -> int:
        return self._spec.concatenate_dim

    @concatenate_dim.setter
    def concatenate_dim(self, val: int):
        self._spec.concatenate_dim = val

    @property
    def enable_corruption(self) -> bool:
        return self._spec.enable_corruption

    @enable_corruption.setter
    def enable_corruption(self, val: bool):
        self._spec.enable_corruption = val


class ObservationsProxy(dict):
    def __init__(self, observation_terms: dict | None = None, policy_observations: dict | None = None):
        if policy_observations is None:
            # dataclasses.asdict() reconstructs dict subclasses via type(obj)(pairs);
            # observation_terms is then an iterable of (key, value) pairs, not the terms registry.
            super().__init__(observation_terms or ())
            return
        self._observation_terms = observation_terms
        self._policy_observations = policy_observations
        super().__init__()
        for policy_name, spec in policy_observations.items():
            self[policy_name] = PolicyObsGroupProxy(policy_name, observation_terms, spec)

    def __setitem__(self, key: str, value):
        from mjlab.managers.observation_manager import ObservationGroupCfg
        if isinstance(value, ObservationGroupCfg) or (not isinstance(value, dict) and hasattr(value, "terms")):
            slots = []
            for term_name, term_cfg in value.terms.items():
                reg_name = term_name
                if reg_name in self._observation_terms:
                    existing_cfg = self._observation_terms[reg_name]
                    if not configs_are_equivalent(term_cfg, existing_cfg):
                        reg_name = f"{key}_{term_name}"
                self._observation_terms[reg_name] = term_cfg
                # Check legacy noise config
                noise_val = getattr(term_cfg, "noise", None)
                if noise_val is None:
                    noise_val = getattr(term_cfg, "corrupt", None)
                view_base = "corrupt" if (getattr(value, "enable_corruption", False) and noise_val is not None) else "clean"
                view = f"{view_base}_{key}"
                slots.append((reg_name, view))
            spec = PolicyObsSpec(
                name=key,
                slots=slots,
                normalizer=getattr(value, "normalizer", "empirical"),
                history_length=value.history_length if value.history_length is not None else 0,
                flatten_history_dim=value.flatten_history_dim if value.flatten_history_dim is not None else True,
                concatenate_terms=getattr(value, "concatenate_terms", True),
                concatenate_dim=getattr(value, "concatenate_dim", -1),
                enable_corruption=getattr(value, "enable_corruption", True),
            )
            self._policy_observations[key] = spec
            super().__setitem__(key, PolicyObsGroupProxy(key, self._observation_terms, spec))
        else:
            super().__setitem__(key, value)

    def __deepcopy__(self, memo):
        import copy
        new_terms = copy.deepcopy(self._observation_terms, memo)
        new_policies = copy.deepcopy(self._policy_observations, memo)
        new_proxy = ObservationsProxy(new_terms, new_policies)
        for k, v in self.items():
            if k not in new_proxy:
                new_proxy[k] = copy.deepcopy(v, memo)
        return new_proxy



class ObsBuffer:
    """Per-env-step tensor factory + per-policy concat + per-policy normalizer + replay slot."""

    def __init__(self, terms: dict[str, ObsTerm], policies: dict[str, PolicyObsSpec], num_envs: int, device):
        self.terms = terms
        self.policies = policies
        self.num_envs = num_envs
        self.device = device
        self.env = None

        self.policy_buf: dict[str, torch.Tensor] = {}
        self._policy_slices: dict[str, list[tuple[str, str, slice]]] = {}
        self._term_view_set: dict[str, set[str]] = {}

    def bind(self, env):
        """Binds environment and initializes the sub-buffers."""
        self.env = env
        for tname, term in self.terms.items():
            term.bind(env, self.num_envs, self.device)

        # Collect views and overrides per term
        term_view_cfgs = {}
        for pname, spec in self.policies.items():
            for tname, view in spec.slots:
                term = self.terms[tname]
                # Fallback to term configurations if not overridden by policy
                spec_h = spec.history_length if spec.history_length is not None else 0
                term_h = term.history_length if term.history_length is not None else 0
                spec_f = spec.flatten_history_dim if spec.flatten_history_dim is not None else True
                term_f = term.flatten_history_dim if term.flatten_history_dim is not None else True

                H = spec_h if spec_h > 0 else term_h
                F = spec_f if spec_h > 0 else term_f

                term_view_cfgs.setdefault(tname, {})[view] = (H, F)
                self._term_view_set.setdefault(tname, set()).add(view)

        # Lazy initialize views
        for tname, view_cfgs in term_view_cfgs.items():
            term = self.terms[tname]
            for view, (H, F) in view_cfgs.items():
                term.init_buffers({view}, policy_history_len=H, policy_flatten_history=F)

        # Pre-allocate contiguous policy buffers
        for pname, spec in self.policies.items():
            cursor = 0
            slices = []
            for tname, view in spec.slots:
                term = self.terms[tname]
                H, F = term_view_cfgs[tname][view]
                H = H if H is not None else 0
                F = F if F is not None else True
                if H > 0:
                    term_dim = term.dim
                    if F:
                        D = H * term_dim
                    else:
                        D = term_dim
                else:
                    D = term.dim
                slices.append((tname, view, slice(cursor, cursor + D)))
                cursor += D

            spec_h = spec.history_length if spec.history_length is not None else 0
            spec_f = spec.flatten_history_dim if spec.flatten_history_dim is not None else True
            if spec_h > 0 and not spec_f:
                self.policy_buf[pname] = torch.empty(self.num_envs, spec_h, cursor, device=self.device, dtype=torch.float32)
            else:
                self.policy_buf[pname] = torch.empty(self.num_envs, cursor, device=self.device, dtype=torch.float32)

            self._policy_slices[pname] = slices

    def step(self, env, *, update_history: bool = True) -> None:
        """Computes all registered term views once, writing directly to pre-allocated buffers."""
        term_outputs: dict[str, dict[str, torch.Tensor]] = {}
        for tname, term in self.terms.items():
            requested = self._term_view_set.get(tname, set())
            if requested:
                term_outputs[tname] = term(env, requested, update_history=update_history)

        for pname, spec in self.policies.items():
            buf = self.policy_buf[pname]
            is_3d = (buf.ndim == 3)
            for tname, view, sl in self._policy_slices[pname]:
                val = term_outputs[tname][view]
                if is_3d:
                    buf[:, :, sl] = val
                else:
                    buf[:, sl] = val

    def get(self, policy: str) -> torch.Tensor:
        """Returns the pre-allocated RAW (pre-norm) observation tensor."""
        return self.policy_buf[policy]

    def compute_final_obs(self, env) -> dict[str, torch.Tensor]:
        """Snapshots pre-reset observations without advancing history/delay states."""
        self.step(env, update_history=False)
        return self.policy_buf

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """Resets the history and delay buffers for the specified environment IDs."""
        for term in self.terms.values():
            term.reset(env_ids)


def build_obs_buffer_from_config(
    obs_cfgs: any,
    num_envs: int,
    device,
    term_overrides: dict[str, any] | None = None,
) -> ObsBuffer:
    """Builds an ObsBuffer from a unified flat ObservationsProxy.

    The proxy (set up by ChronologicalObservationManager) exposes the deduplicated
    physical terms in `_observation_terms` and the per-policy view specs in
    `_policy_observations`. Group-major task configs are translated into this flat
    form by the proxy before reaching here.
    """
    from .observation_terms import ObsTerm

    # Explicit check: auto-convert dict to ObservationsProxy for backward compatibility.
    if not (hasattr(obs_cfgs, "_observation_terms") and hasattr(obs_cfgs, "_policy_observations")):
        if isinstance(obs_cfgs, dict):
            proxy = ObservationsProxy({}, {})
            for k, v in obs_cfgs.items():
                proxy[k] = v
            obs_cfgs = proxy
        else:
            raise TypeError(
                f"build_obs_buffer_from_config requires an ObservationsProxy (flat registry) or dict, "
                f"got {type(obs_cfgs).__name__}."
            )

    term_overrides = term_overrides or {}
    terms = {}
    for name, term_cfg in obs_cfgs._observation_terms.items():
        if name in term_overrides:
            terms[name] = term_overrides[name]
            continue
        corrupt_val = getattr(term_cfg, "corrupt", None)
        if corrupt_val is None:
            corrupt_val = getattr(term_cfg, "noise", None)
        term_instance = ObsTerm(
            func=term_cfg.func,
            params=getattr(term_cfg, "params", None),
            corrupt=corrupt_val,
            clip=getattr(term_cfg, "clip", None),
            scale=getattr(term_cfg, "scale", None),
            delay_min_lag=getattr(term_cfg, "delay_min_lag", 0),
            delay_max_lag=getattr(term_cfg, "delay_max_lag", 0),
            delay_hold_prob=getattr(term_cfg, "delay_hold_prob", 0.0),
            delay_update_period=getattr(term_cfg, "delay_update_period", 0),
            delay_per_env=getattr(term_cfg, "delay_per_env", True),
            delay_per_env_phase=getattr(term_cfg, "delay_per_env_phase", True),
            history_length=getattr(term_cfg, "history_length", 0),
            flatten_history_dim=getattr(term_cfg, "flatten_history_dim", True),
        )
        term_instance._resolved_cfg = term_cfg
        terms[name] = term_instance
    return ObsBuffer(terms, obs_cfgs._policy_observations, num_envs, device)

