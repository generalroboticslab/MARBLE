"""Environment wrapper that captures pre-reset observations for off-policy algorithms."""

from __future__ import annotations

import torch

from mjlab.envs import ManagerBasedRlEnv


class ManagerBasedRlEnvWithFinalObs(ManagerBasedRlEnv):
    """ManagerBasedRlEnv that captures final (pre-reset) observations.

    Standard ManagerBasedRlEnv resets terminated envs BEFORE computing observations,
    so terminal state observations are lost. This subclass computes observations
    between termination detection and reset, storing them for off-policy algorithms
    that need pre-reset obs for proper value bootstrapping.

    SNAPSHOT INVARIANT (load-bearing — do not remove the clone in step()):
        `obs_buf` and `final_obs` returned from step() MUST be standalone tensors, not
        views into the observation manager's storage. The new ChronologicalObservationManager
        returns each group's obs as a VIEW into a persistent ring buffer that step() overwrites
        IN PLACE every call (zero-alloc by design). An off-policy runner carries obs across the
        collection loop (`actor_obs = next_actor_obs`) and stores them in a replay buffer; if
        those are views, the NEXT step() silently mutates already-stored transitions — the
        stored "current obs" becomes s_{t+1} while its paired action was chosen for s_t. The
        resulting (s_{t+1}, a_for_s_t, r, s_{t+1}) tuples corrupt the Bellman target -> value
        divergence -> training collapse, while on-policy/inference (which consume obs
        immediately and never carry them) look perfectly fine. The legacy mjlab manager
        returned a freshly-cat'd tensor each step, so this invariant held implicitly; the obs
        rewrite (dfd9d9f) broke it. step() restores it by cloning obs_buf at the env boundary.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_obs: dict[str, torch.Tensor] | None = None
        # Cached reset ids from the most recent step(). Exposed so the runner can
        # recover pre-reset obs without re-running the nonzero kernel. Shape (k,)
        # on CUDA, or empty (0,) on a step with no resets. Recomputed every step.
        self.reset_env_ids: torch.Tensor | None = None

    def step(self, action: torch.Tensor):
        # --- Physics simulation (same as parent) ---
        self.action_manager.process_action(action.to(self.device))

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt=self.physics_dt)

        self.sim.forward()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # --- Termination & reward (same as parent) ---
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # *** CAPTURE PRE-RESET OBSERVATIONS ***
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.reset_env_ids = reset_env_ids
        if len(reset_env_ids) > 0:
            final_obs_all = self.observation_manager.compute(update_history=False)
            self.final_obs = {
                group_name: group_obs[reset_env_ids].clone()
                for group_name, group_obs in final_obs_all.items()
            }
        else:
            self.final_obs = None

        # --- Reset, commands, events, observations (same as parent) ---
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # The new ChronologicalObservationManager returns obs tensors that are VIEWS into its
        # persistent ring storage (zero-alloc by design). The off-policy runner carries obs
        # across the loop boundary (actor_obs = next_actor_obs) and stores them; the NEXT
        # env.step overwrites the ring in place, so a carried view silently mutates to the next
        # state -> every stored transition's "observations" becomes s_{t+1} instead of s_t ->
        # corrupted Bellman tuple -> value divergence -> training collapse (inference is fine,
        # it consumes obs immediately). The old mjlab manager returned a freshly-cat'd tensor,
        # so carrying was safe. Clone at the env boundary to restore that snapshot contract.
        obs_buf = self.observation_manager.compute(update_history=True)
        self.obs_buf = {
            group_name: (
                group_obs.clone() if torch.is_tensor(group_obs)
                else {term: t.clone() for term, t in group_obs.items()}
            )
            for group_name, group_obs in obs_buf.items()
        }

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )

