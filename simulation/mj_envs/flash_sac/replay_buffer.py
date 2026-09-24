"""FlashSAC frame-ring replay buffer (relocated from flash_sac_utils.py).

cpu_state helper kept here (used by the runner for async checkpoint offload). Pure relocation.
"""
from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn


class SimpleReplayBuffer(nn.Module):
    def __init__(
        self,
        n_env: int,
        buffer_size: int,
        n_obs: int,
        n_act: int,
        n_critic_obs: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        device=None,
        n_estimator_target: int = 0,
        frame_ring: bool = False,
        obs_seq: tuple[int, int] | None = None,
        critic_seq: tuple[int, int] | None = None,
    ):
        """
        A simple replay buffer that stores transitions in a circular buffer.
        Supports n-step returns and asymmetric observations.

        frame_ring: when True, store ONE frame per timestep
        per history view (obs_seq/critic_seq give the (L, D) split of n_obs /
        n_critic_obs) and reconstruct the L-frame window at sample time, instead
        of storing the full overlapping window per transition (~L× less VRAM, lossless, same
        sampling distribution). The reconstructed window is byte-identical to the windowed path
        (verified by the Phase-0 equivalence test). Requires flatten=False history groups so each
        stored window is a clean (N, L, D) flattened row. Phase 1 supports n_steps==1 only.
        """
        super().__init__()
        # frame_ring applies to both views, actor and critic (obs_seq/critic_seq required).
        if frame_ring:
            assert obs_seq is not None and critic_seq is not None, "frame_ring needs actor+critic (L,D)"

        self.n_env = n_env
        self.buffer_size = buffer_size
        self.n_obs = n_obs
        self.n_act = n_act
        self.n_critic_obs = n_critic_obs
        self.gamma = gamma
        self.n_steps = n_steps
        self.device = device
        self.n_estimator_target = n_estimator_target
        self.frame_ring = frame_ring
        self.obs_seq = obs_seq
        self.critic_seq = critic_seq

        if frame_ring:
            self._max_L = max(obs_seq[0], critic_seq[0], 0)
            store_len = buffer_size + self._max_L - 1
        else:
            store_len = buffer_size
        self._store_len = store_len

        self.actions = torch.zeros((n_env, store_len, n_act), device=device, dtype=torch.float)
        self.rewards = torch.zeros((n_env, store_len), device=device, dtype=torch.float)
        self.dones = torch.zeros((n_env, store_len), device=device, dtype=torch.uint8)
        self.truncations = torch.zeros((n_env, store_len), device=device, dtype=torch.uint8)
        # History views: fp16 (critic obs are privileged GT ~O(1e3), well within fp16 range, and
        # get EmpiricalNormalization after sampling → half precision harmless; halves VRAM).
        if frame_ring:
            # Frame-ring: one D-frame per timestep per view; window reconstructed at sample time.
            D_a = obs_seq[1]
            D_c = critic_seq[1]
            self.obs_frame = torch.zeros((n_env, store_len, D_a), device=device, dtype=torch.float16)
            self.obs_next_frame = torch.zeros((n_env, store_len, D_a), device=device, dtype=torch.float16)
            self.critic_frame = torch.zeros((n_env, store_len, D_c), device=device, dtype=torch.float16)
            self.critic_next_frame = torch.zeros((n_env, store_len, D_c), device=device, dtype=torch.float16)
            # Per-transition frames-since-reset (capped at the largest view L), derived inside
            # extend from this buffer's own `dones` ring. Reconstruction clamps lags
            # to push_count-1, reproducing CircularBuffer backfill (valid = min(key, num_pushes-1)).
            self.push_count = torch.zeros((n_env, store_len), device=device, dtype=torch.int16)
        else:
            self.observations = torch.zeros((n_env, store_len, n_obs), device=device, dtype=torch.float16)
            self.next_observations = torch.zeros((n_env, store_len, n_obs), device=device, dtype=torch.float16)
            self.critic_observations = torch.zeros((n_env, store_len, n_critic_obs), device=device, dtype=torch.float16)
            self.next_critic_observations = torch.zeros(
                (n_env, store_len, n_critic_obs), device=device, dtype=torch.float16
            )
        if n_estimator_target > 0:
            self.estimator_targets = torch.zeros(
                (n_env, store_len, n_estimator_target), device=device, dtype=torch.float
            )
        self.ptr = 0

        # Precomputed constants for n-step sampling (avoids repeated allocation in sample()).
        self._seq_offsets = torch.arange(n_steps, device=device).view(1, 1, -1)
        self._discounts = torch.pow(torch.tensor(gamma, device=device), torch.arange(n_steps, device=device))

    @staticmethod
    def _newest(window: torch.Tensor, seq: tuple[int, int]) -> torch.Tensor:
        """Newest frame of a flattened (N, L*D) history window → (N, D), fp16.

        Window is the env's flatten=False group flattened: (N, L, D) row-major, frame L-1 newest.
        """
        L, D = seq
        return window.reshape(window.shape[0], L, D)[:, -1, :].to(torch.float16)

    def _fr_window(
        self, frame_ring: torch.Tensor, next_ring: torch.Tensor | None, indices: torch.Tensor, seq: tuple[int, int]
    ):
        """Reconstruct the (B, L*D) history window(s) for sampled `indices` from a frame ring.

        Builds the window oldest→newest directly (slot 0 oldest, L-1 newest) — matches
        circular_buffer.buffer ordering with NO flip. Lags are clamped to push_count-1, exactly
        reproducing env backfill (CircularBuffer valid = min(key, num_pushes-1)):
        a transition at episode step k<L reconstructs [f0,…,f0,f1,…,f_k], not zeros.

        Reconstruction uses `torch.gather` (coalesced, the same kernel the windowed path uses), NOT
        broadcast advanced indexing `frame_ring[env3, pos]`. Both are value-identical, but advanced
        indexing dispatched the slow `aten::index` elementwise kernel (~14× the per-call cost of
        gather in profiling, ~38% of sample time); folding the per-view position into a gather index
        broadcast across D makes the frame read coalesced like the windowed single-slot gather.
        Gather returns distinct storage for replicated frames → safe for the downstream in-place
        per-frame normalizer. Returns (obs, next_obs); next_obs is None when next_ring is None.
        """
        n_env, batch = indices.shape
        L, D = seq
        bs = self._store_len
        out_n = n_env * batch
        # frames-since-reset for this view: min(stored push_count, L). push_count is shared across
        # views (env episode step); the per-view cap differs. Gather the int16 ring directly and keep
        # the lag math in int16 (the final gather index promotes to int64 via `indices` anyway) — the
        # old code cast the full (n_env, store_len) ring to int64, converting ~L× more than the
        # sampled slice and adding a copy per call.
        vc = torch.gather(self.push_count, 1, indices).clamp_(max=L)          # (n_env, batch) int16
        vcm1 = (vc - 1).unsqueeze(-1)                                         # (n_env, batch, 1)

        def gather_window(raw_lag: torch.Tensor) -> torch.Tensor:
            """Gather one (n_env, batch, L, D) window from frame_ring: output slot j ← the frame at
            lag raw_lag[j] from idx, clamped to vc-1 (env backfill). Fold pos into a gather index
            broadcast across D: gather(frame_ring,1,idx)[e,k,d]=frame_ring[e,pos[e,k],d]. expand() is
            stride-0 (no copy); the gathered (n_env, batch*L, D) is contiguous so the reshape is free.
            """
            eff = torch.minimum(raw_lag, vcm1)
            pos = (indices.unsqueeze(-1) - eff) % bs                          # (n_env, batch, L)
            gidx = pos.reshape(n_env, batch * L, 1).expand(n_env, batch * L, D)
            return torch.gather(frame_ring, 1, gidx).reshape(n_env, batch, L, D)

        # current window: slot j (0..L-1) ← lag (L-1-j) — oldest→newest, newest at slot L-1.
        raw_lag = (L - 1 - torch.arange(L, device=self.device, dtype=torch.int16)).view(1, 1, L)
        obs = gather_window(raw_lag).reshape(out_n, L * D).to(torch.float32)

        if next_ring is None:
            return obs, None

        # next window oldest→newest: slots j=0..L-2 ← frame_ring at idx's episode, lag (L-2-j),
        # clamped to idx's vc-1; slot L-1 = next_ring[idx] (terminal-correct). Gather all L slots from
        # frame_ring (slot L-1 a throwaway at lag 0 via clamp(min=0)), then overwrite slot L-1 with the
        # next frame in place. Avoids torch.cat(older, newest), whose full window copy profiled at ~7%
        # of sample time; the in-place write copies only the (n_env, batch, D) newest slot. The
        # clamp(min=0) on slot L-1 also collapses the old L==1 special case.
        raw_lag_n = (L - 2 - torch.arange(L, device=self.device, dtype=torch.int16)).clamp_(min=0).view(1, 1, L)
        nxt = gather_window(raw_lag_n)
        nxt[:, :, -1, :] = torch.gather(next_ring, 1, indices.unsqueeze(-1).expand(n_env, batch, D))
        next_obs = nxt.reshape(out_n, L * D).to(torch.float32)
        return obs, next_obs

    def extend(
        self,
        tensor_dict: TensorDict,
    ):
        observations = tensor_dict["observations"]
        actions = tensor_dict["actions"]
        rewards = tensor_dict["next"]["rewards"]
        dones = tensor_dict["next"]["dones"]
        truncations = tensor_dict["next"]["truncations"]
        next_observations = tensor_dict["next"]["observations"]

        ptr = self.ptr % self._store_len
        self.actions[:, ptr] = actions
        self.rewards[:, ptr] = rewards
        self.dones[:, ptr] = dones
        self.truncations[:, ptr] = truncations
        critic_observations = tensor_dict["critic_observations"]
        next_critic_observations = tensor_dict["next"]["critic_observations"]
        if self.frame_ring:
            # Store the newest frame of each window; reconstruct windows at sample time.
            self.obs_frame[:, ptr] = self._newest(observations, self.obs_seq)
            self.obs_next_frame[:, ptr] = self._newest(next_observations, self.obs_seq)
            self.critic_frame[:, ptr] = self._newest(critic_observations, self.critic_seq)
            self.critic_next_frame[:, ptr] = self._newest(next_critic_observations, self.critic_seq)
            # Frames-since-reset for this slot: 1 if the previous transition for this env ended an
            # episode (its done=1) else previous count + 1; capped at the largest view L. ptr's
            # previous slot is (ptr-1)%buffer_size — the immediately-preceding transition for every
            # env (ring is per-env contiguous). At ptr==0 the prev slot is zero-init → count 1.
            prev = (ptr - 1) % self._store_len
            c = torch.where(self.dones[:, prev] == 1, torch.ones_like(self.push_count[:, prev]), self.push_count[:, prev] + 1)
            self.push_count[:, ptr] = c.clamp_(max=self._max_L)
        else:
            self.observations[:, ptr] = observations.to(torch.float16)
            self.next_observations[:, ptr] = next_observations.to(torch.float16)
            self.critic_observations[:, ptr] = critic_observations.to(torch.float16)
            self.next_critic_observations[:, ptr] = next_critic_observations.to(torch.float16)
        if self.n_estimator_target > 0 and "estimator_target" in tensor_dict.keys():
            self.estimator_targets[:, ptr] = tensor_dict["estimator_target"]
        self.ptr += 1

    @torch.no_grad()
    def sample(self, batch_size: int):
        # we will sample n_env * batch_size transitions

        if self.n_steps == 1:
            valid = min(self.buffer_size, self.ptr)
            if self.frame_ring:
                # Sample the last `valid` transitions, mapped into the extended (store_len) ring so
                # each one's full window is reconstructable (frontier-safe).
                offsets = torch.randint(0, valid, (self.n_env, batch_size), device=self.device)
                indices = (self.ptr - 1 - offsets) % self._store_len
            else:
                indices = torch.randint(0, valid, (self.n_env, batch_size), device=self.device)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)
            actions = torch.gather(self.actions, 1, act_indices).reshape(self.n_env * batch_size, self.n_act)

            rewards = torch.gather(self.rewards, 1, indices).reshape(self.n_env * batch_size)
            dones = torch.gather(self.dones, 1, indices).reshape(self.n_env * batch_size)
            truncations = torch.gather(self.truncations, 1, indices).reshape(self.n_env * batch_size)
            effective_n_steps = torch.ones_like(dones)
            if self.frame_ring:
                observations, next_observations = self._fr_window(self.obs_frame, self.obs_next_frame, indices, self.obs_seq)
                critic_observations, next_critic_observations = self._fr_window(
                    self.critic_frame, self.critic_next_frame, indices, self.critic_seq
                )
            else:
                obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
                observations = torch.gather(self.observations, 1, obs_indices).reshape(self.n_env * batch_size, self.n_obs).to(torch.float32)
                next_observations = torch.gather(self.next_observations, 1, obs_indices).reshape(
                    self.n_env * batch_size, self.n_obs
                ).to(torch.float32)
                # Gather full critic observations
                critic_obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_critic_obs)
                critic_observations = torch.gather(self.critic_observations, 1, critic_obs_indices).reshape(
                    self.n_env * batch_size, self.n_critic_obs
                )
                next_critic_observations = torch.gather(self.next_critic_observations, 1, critic_obs_indices).reshape(
                    self.n_env * batch_size, self.n_critic_obs
                )
        else:
            # Sample base indices. INVARIANT: any temporary truncation flag set below is restored
            # at the end of this method. The method must not early-return between here and that
            # restore — any early exit would leave the buffer in a corrupted state.
            ring = self._store_len
            guard_slot = None
            if self.frame_ring:
                # Sample base among the last valid transitions, mapped into the extended ring so the
                # base history AND the final next-window are frontier-safe. Guard
                # the latest-written transition as truncated (if not done) so an n-step sequence never
                # bootstraps across the ring frontier into wrapped/unwritten slots.
                valid = min(self.buffer_size, self.ptr)
                guard_slot = (self.ptr - 1) % ring
                guard_val = self.truncations[:, guard_slot].clone()
                self.truncations[:, guard_slot] = torch.logical_not(self.dones[:, guard_slot])
                offsets = torch.randint(0, valid, (self.n_env, batch_size), device=self.device)
                indices = (self.ptr - 1 - offsets) % ring
            elif self.ptr >= self.buffer_size:
                current_pos = self.ptr % self.buffer_size
                curr_truncations = self.truncations[:, current_pos - 1].clone()
                self.truncations[:, current_pos - 1] = torch.logical_not(self.dones[:, current_pos - 1])
                indices = torch.randint(0, self.buffer_size, (self.n_env, batch_size), device=self.device)
            else:
                # Buffer not full - ensure n-step sequence doesn't exceed valid data
                max_start_idx = max(1, self.ptr - self.n_steps + 1)
                indices = torch.randint(0, max_start_idx, (self.n_env, batch_size), device=self.device)
            act_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_act)
            actions = torch.gather(self.actions, 1, act_indices).reshape(self.n_env * batch_size, self.n_act)

            # Get base transitions (current window).
            if self.frame_ring:
                observations, _ = self._fr_window(self.obs_frame, None, indices, self.obs_seq)
                critic_observations, _ = self._fr_window(self.critic_frame, None, indices, self.critic_seq)
            else:
                obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
                observations = torch.gather(self.observations, 1, obs_indices).reshape(self.n_env * batch_size, self.n_obs).to(torch.float32)
                critic_obs_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_critic_obs)
                critic_observations = torch.gather(self.critic_observations, 1, critic_obs_indices).reshape(
                    self.n_env * batch_size, self.n_critic_obs
                )

            # Sequential indices for each sample: [n_env, batch_size, n_step]
            all_indices = (indices.unsqueeze(-1) + self._seq_offsets) % ring

            # Gather all rewards and terminal flags
            # Using advanced indexing - result shapes: [n_env, batch_size, n_step]
            all_rewards = torch.gather(self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices)
            all_dones = torch.gather(self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps), 1, all_indices)
            all_truncations = torch.gather(
                self.truncations.unsqueeze(-1).expand(-1, -1, self.n_steps),
                1,
                all_indices,
            )

            # Create masks for rewards *after* first done
            # This creates a cumulative product that zeroes out rewards after the first done
            all_dones_shifted = torch.cat(
                [torch.zeros_like(all_dones[:, :, :1]), all_dones[:, :, :-1]], dim=2
            )  # First reward should not be masked
            done_masks = torch.cumprod(1.0 - all_dones_shifted, dim=2)  # [n_env, batch_size, n_step]
            effective_n_steps = done_masks.sum(2)

            # Apply masks and discounts to rewards
            masked_rewards = all_rewards * done_masks  # [n_env, batch_size, n_step]
            discounted_rewards = masked_rewards * self._discounts.view(1, 1, -1)  # [n_env, batch_size, n_step]

            # Sum rewards along the n_step dimension
            n_step_rewards = discounted_rewards.sum(dim=2)  # [n_env, batch_size]

            # Find index of first done or truncation or last step for each sequence
            first_done = torch.argmax((all_dones > 0).float(), dim=2)  # [n_env, batch_size]
            first_trunc = torch.argmax((all_truncations > 0).float(), dim=2)  # [n_env, batch_size]

            # Handle case where there are no dones or truncations
            no_dones = all_dones.sum(dim=2) == 0
            no_truncs = all_truncations.sum(dim=2) == 0

            # When no dones or truncs, use the last index
            first_done = torch.where(no_dones, self.n_steps - 1, first_done)
            first_trunc = torch.where(no_truncs, self.n_steps - 1, first_trunc)

            # Take the minimum (first) of done or truncation
            final_indices = torch.minimum(first_done, first_trunc)  # [n_env, batch_size]

            # Create indices to gather the final next observations
            final_next_obs_indices = torch.gather(all_indices, 2, final_indices.unsqueeze(-1)).squeeze(
                -1
            )  # [n_env, batch_size]

            # Gather final values (the bootstrap next-obs at the first done/trunc/last step).
            final_dones = self.dones.gather(1, final_next_obs_indices)
            final_truncations = self.truncations.gather(1, final_next_obs_indices)
            if self.frame_ring:
                # Reconstruct the NEXT window at the final index (terminal-correct): lag0 from the
                # next-frame ring, older frames from the current-frame ring within that episode.
                _, next_observations = self._fr_window(
                    self.obs_frame, self.obs_next_frame, final_next_obs_indices, self.obs_seq
                )
                _, next_critic_observations = self._fr_window(
                    self.critic_frame, self.critic_next_frame, final_next_obs_indices, self.critic_seq
                )
            else:
                final_next_observations = self.next_observations.gather(
                    1, final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
                )
                final_next_critic_observations = self.next_critic_observations.gather(
                    1, final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_critic_obs)
                )
                next_critic_observations = final_next_critic_observations.reshape(self.n_env * batch_size, self.n_critic_obs)
                next_observations = final_next_observations.reshape(self.n_env * batch_size, self.n_obs).to(torch.float32)

            # Reshape everything to batch dimension
            rewards = n_step_rewards.reshape(self.n_env * batch_size)
            dones = final_dones.reshape(self.n_env * batch_size)
            truncations = final_truncations.reshape(self.n_env * batch_size)
            effective_n_steps = effective_n_steps.reshape(self.n_env * batch_size)

        out = TensorDict(
            {
                "observations": observations,
                "actions": actions,
                "next": {
                    "rewards": rewards,
                    "dones": dones,
                    "truncations": truncations,
                    "observations": next_observations,
                    "effective_n_steps": effective_n_steps,
                },
            },
            batch_size=self.n_env * batch_size,
        )
        out["critic_observations"] = critic_observations.to(torch.float32)
        out["next"]["critic_observations"] = next_critic_observations.to(torch.float32)
        if self.n_estimator_target > 0:
            est_indices = indices.unsqueeze(-1).expand(-1, -1, self.n_estimator_target)
            out["estimator_target"] = torch.gather(
                self.estimator_targets, 1, est_indices
            ).reshape(self.n_env * batch_size, self.n_estimator_target)

        # Roll back the temporary truncation flag introduced for safe n-step sampling.
        if self.n_steps > 1:
            if self.frame_ring:
                self.truncations[:, guard_slot] = guard_val
            elif self.ptr >= self.buffer_size:
                self.truncations[:, current_pos - 1] = curr_truncations
        return out


def cpu_state(sd):
    # detach & move to host without locking the compute stream
    return {k: v.detach().to("cpu", non_blocking=True) for k, v in sd.items()}
