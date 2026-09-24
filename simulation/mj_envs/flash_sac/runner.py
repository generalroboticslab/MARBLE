"""FlashSAC training runner for mjlab environments.

Adapts the FlashSAC algorithm from holosoma for the mjlab ManagerBasedRlEnv interface.
"""

from __future__ import annotations

import math
import os
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as F
import tqdm
from tensordict import TensorDict
from torch import nn, optim
from torch.amp.autocast_mode import autocast
from torch.optim.lr_scheduler import LambdaLR

from .models import Actor, Critic, SequenceActor, SequenceCritic
from .models import compute_estimator_mse
from .config import FlashSACConfig
from .env_wrapper import ManagerBasedRlEnvWithFinalObs
from .normalization import EmpiricalNormalization
from .replay_buffer import SimpleReplayBuffer, cpu_state
from .normalization import RewardNormalizer
from .logger import FlashSACCompactLogger

torch.set_float32_matmul_precision("high")


def make_seq_normalizer(normalizer: nn.Module, S: int, D: int):
    """Return fn(obs, update=True) that normalizes a flat (B, S*D) history obs per-frame.

    Reshapes (B, S*D) → (B*S, D) so EmpiricalNormalization computes per-frame mean/std shared
    across the S timesteps, then reshapes back. inplace=update ties allocation to intent:
      - collection (update=False): inplace=False — fresh tensor so the pre-store obs is not
        overwritten before rb.extend copies it (overwrite ⇒ double normalization at sample time).
      - training  (update=True):  inplace=True  — safe (rb.sample owns the tensor) and saves a
        large alloc per call.
    """
    def _normalize(obs, update=True):
        B = obs.shape[0]
        return normalizer(obs.view(B * S, D), update=update, inplace=update).view(B, S * D)
    return _normalize


class FlashSACRunner:
    """FlashSAC training runner for mjlab environments."""

    def __init__(
        self,
        env: ManagerBasedRlEnvWithFinalObs,
        cfg: FlashSACConfig,
        log_dir: str,
        device: str,
    ):
        self.env = env
        self.cfg = cfg
        self.device = device
        self.log_dir = log_dir
        self.global_step = 0
        # Checkpoints deliberately exclude replay (it is large and device-local). A
        # load is therefore a model-state continuation, not an exact trajectory
        # continuation: collect a complete fresh per-env ring before TD updates.
        self._replay_warmup_steps = 0
        self.compact_logger = FlashSACCompactLogger(width=90)

    def setup(self) -> None:
        cfg = self.cfg
        device = self.device
        env = self.env

        total_control_steps = cfg.num_learning_iterations * cfg.num_collect_steps
        for name in env.command_manager.active_terms:
            term = env.command_manager.get_term(name)
            if hasattr(term, "set_training_horizon"):
                term.set_training_horizon(total_control_steps)

        # --- Observation & action dimensions ---
        obs_dict, _ = env.reset()
        actor_obs_shape = tuple(obs_dict[cfg.actor_obs_group].shape[1:])
        critic_obs_shape = tuple(obs_dict[cfg.critic_obs_group].shape[1:])
        actor_obs_dim = int(torch.tensor(actor_obs_shape).prod().item())
        critic_obs_dim = int(torch.tensor(critic_obs_shape).prod().item())

        self.actor_obs_shape = actor_obs_shape
        self.critic_obs_shape = critic_obs_shape
        self.actor_obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim

        # An env may expose `policy_action_dim` to train a learner on a SUBSET of the full
        # action vector (the rest filled internally by the env).
        # Falls back to the full manager dim for all standard envs.
        n_act = getattr(env, "policy_action_dim", None) or env.action_manager.total_action_dim
        self.n_act = n_act

        # --- Trivial obs_indices (mjlab already provides concatenated group tensors) ---
        actor_obs_indices = {
            "actor_obs": {"start": 0, "end": actor_obs_dim, "size": actor_obs_dim}
        }
        critic_obs_indices = {
            "critic_obs": {"start": 0, "end": critic_obs_dim, "size": critic_obs_dim}
        }

        # --- Action scaling: let mjlab action manager handle it ---
        action_scale = torch.ones(n_act, device=device)
        action_bias = torch.zeros(n_act, device=device)

        # --- Detect sequence shape for encoder ---
        # actor_obs_shape is (S, D) for history obs; (flat_dim,) for non-history.
        if cfg.use_sequence_encoder and len(actor_obs_shape) == 2:
            seq_S, seq_D = actor_obs_shape
            norm_shape = seq_D      # normalize per-frame, not over full S*D
        else:
            seq_S = seq_D = None
            norm_shape = actor_obs_dim

        # --- Frame-ring replay per-view (L, D) splits ---
        # When frame_ring_history is on, the buffer stores ONE frame per timestep per history view
        # and reconstructs the L-frame window at sample time (~L× less replay VRAM, lossless). It
        # needs each view's (L, D). Both actor and critic degenerate to a 1-frame ring
        # (obs_seq/critic_seq=(1, D)) when they have no history window (flat obs, or
        # flatten_history_dim=True): the buffer stores one frame/step and reconstructs it by
        # identity (the L==1 path), bit-exact to the non-ring storage, zero VRAM saving but zero
        # harm. This makes frame_ring_history safe as an unconditional default — it only pays off
        # when use_sequence_encoder=True gives the actor a real (S, D) history window, but never
        # breaks a flat-obs experiment. The runner still flattens the critic obs to (B, L_c·D_c) at
        # every read (.flatten below) so the Critic MLP and the existing per-position critic
        # normalizer (shape=critic_obs_dim=L_c·D_c) are byte-identical to a flatten=True critic —
        # frame-ring changes only STORAGE, not the network or normalization.
        if cfg.frame_ring_history:
            obs_seq: tuple[int, int] | None = (
                (int(seq_S), int(seq_D)) if seq_S is not None else (1, actor_obs_dim)
            )
            critic_seq: tuple[int, int] | None = (
                (int(critic_obs_shape[0]), int(critic_obs_shape[1])) if len(critic_obs_shape) == 2
                else (1, critic_obs_dim)
            )
        else:
            obs_seq = critic_seq = None

        # --- Temporal-critic (SequenceCritic) shape detect ---
        # When use_critic_sequence_encoder and the critic obs group carries a window (S, D), the
        # critic gets an RMA-CNN over its (L_c, D_c) history (implicit sys-ID of DR latents). Its
        # normalizer switches to per-FRAME (shape D_c) so the encoder input is scaled like the
        # actor's; else the critic stays the 1-frame flat-MLP Markov critic (norm shape L_c·D_c).
        critic_seq_S = critic_seq_D = None
        critic_norm_shape = critic_seq_D if critic_seq_S is not None else critic_obs_dim

        # --- Observation normalization ---
        if cfg.obs_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(
                shape=norm_shape, device=device
            )
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(
                shape=critic_norm_shape, device=device
            )
        else:
            self.obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        # --- Normalize wrapper (per-frame reshape for sequence encoder) ---
        self._normalize_actor_obs: Any
        self._normalize_critic_obs: Any
        if seq_S is not None:
            self._normalize_actor_obs = make_seq_normalizer(self.obs_normalizer, seq_S, seq_D)
        else:
            self._normalize_actor_obs = self.obs_normalizer.forward

        # Critic obs normalizer — inplace to avoid allocating a (B, critic_obs_dim) temp tensor.
        # critic_obs is flat (90 dims vs actor's 1800), so the saving is small (~22 MB per
        # mini-batch), but inplace is correct and consistent with the actor normalizer.
        if cfg.obs_normalization and critic_seq_S is not None:
            # Temporal critic: per-frame normalize the flat (B, L_c·D_c) window (mirror the actor).
            self._normalize_critic_obs = make_seq_normalizer(
                self.critic_obs_normalizer, critic_seq_S, critic_seq_D
            )
        elif cfg.obs_normalization:
            _cnorm = self.critic_obs_normalizer
            def _normalize_critic_obs(obs, update=True):
                return _cnorm(obs, update=update, inplace=True)
            self._normalize_critic_obs = _normalize_critic_obs
        else:
            # obs_normalization=False: normalizer is nn.Identity, return obs unchanged.
            def _normalize_critic_obs(obs, update=True):
                return obs
            self._normalize_critic_obs = _normalize_critic_obs

        # --- Networks ---
        if cfg.use_sequence_encoder and seq_S is not None:
            self.actor = SequenceActor(
                obs_indices=actor_obs_indices,
                obs_keys=["actor_obs"],
                n_act=n_act,
                num_envs=env.num_envs,
                device=device,
                hidden_dim=cfg.actor_hidden_dim,
                log_std_max=cfg.log_std_max,
                log_std_min=cfg.log_std_min,
                use_tanh=cfg.use_tanh,
                use_layer_norm=cfg.use_layer_norm,
                action_scale=action_scale,
                action_bias=action_bias,
                obs_seq_shape=(seq_S, seq_D),
                encoder_type=cfg.encoder_type,
                encoder_embed_dim=cfg.encoder_embed_dim,
                encoder_latent_dim=cfg.encoder_latent_dim,
                use_velocity_estimator=cfg.use_velocity_estimator,
                vel_head_output_dim=cfg.vel_head_output_dim,
            )
        else:
            self.actor = Actor(
                obs_indices=actor_obs_indices,
                obs_keys=["actor_obs"],
                n_act=n_act,
                num_envs=env.num_envs,
                device=device,
                hidden_dim=cfg.actor_hidden_dim,
                log_std_max=cfg.log_std_max,
                log_std_min=cfg.log_std_min,
                use_tanh=cfg.use_tanh,
                use_layer_norm=cfg.use_layer_norm,
                action_scale=action_scale,
                action_bias=action_bias,
            )
        # Temporal critic (SequenceCritic) when the encoder flag is on and the critic obs is a
        # window; else the plain 1-frame Markov Critic (byte-identical to before). qnet + target
        # MUST share the class/arch.
        _critic_cls = SequenceCritic if critic_seq_S is not None else Critic
        _critic_kw = dict(
            obs_indices=critic_obs_indices,
            obs_keys=["critic_obs"],
            n_act=n_act,
            num_atoms=cfg.num_atoms,
            v_min=cfg.v_min,
            v_max=cfg.v_max,
            hidden_dim=cfg.critic_hidden_dim,
            device=device,
            use_layer_norm=cfg.use_layer_norm,
            num_q_networks=cfg.num_q_networks,
        )
        if critic_seq_S is not None:
            _critic_kw.update(
                seq_shape=(critic_seq_S, critic_seq_D),
                encoder_type=cfg.encoder_type,
                encoder_embed_dim=cfg.encoder_embed_dim,
                encoder_latent_dim=cfg.encoder_latent_dim,
            )
        self.qnet = _critic_cls(**_critic_kw)
        self.qnet_target = _critic_cls(**_critic_kw)
        self.qnet_target.load_state_dict(self.qnet.state_dict())

        print(self.actor)
        print(self.qnet)

        # --- Entropy ---
        self.log_alpha = torch.tensor(
            [math.log(cfg.alpha_init)], requires_grad=True, device=device
        )
        # Target entropy. Prefer target_sigma formulation (matches reference FlashSAC):
        #   target_entropy = 0.5 * n_act * log(2πe·σ²)
        # which directly targets policy std ≈ σ. Falls back to ratio for backward compat.
        if cfg.target_sigma > 0.0:
            self.target_entropy = (
                0.5 * n_act * math.log(2.0 * math.pi * math.e * cfg.target_sigma ** 2)
            )
        else:
            self.target_entropy = -n_act * cfg.target_entropy_ratio

        # --- Optimizers ---
        self.q_optimizer = optim.AdamW(
            list(self.qnet.parameters()),
            lr=cfg.critic_learning_rate,
            weight_decay=cfg.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
        self.actor_optimizer = optim.AdamW(
            list(self.actor.parameters()),
            lr=cfg.actor_learning_rate,
            weight_decay=cfg.weight_decay,
            fused=True,
            betas=(0.9, 0.95),
        )
        self.alpha_optimizer = optim.AdamW(
            [self.log_alpha],
            lr=cfg.alpha_learning_rate,
            fused=True,
            betas=(0.9, 0.95),
        )

        # --- LR Scheduler (optional: warmup + cosine decay) ---
        if cfg.lr_warmup_iters > 0 or cfg.lr_decay_iters > 0:
            total_sched = cfg.lr_warmup_iters + cfg.lr_decay_iters
            warmup = cfg.lr_warmup_iters
            min_fac = cfg.lr_min_factor

            def _lr_fn(
                step: int, warmup_iters: int = warmup, total_iters: int = total_sched,
                min_factor: float = min_fac,
            ) -> float:
                if warmup_iters > 0 and step < warmup_iters:
                    return (step + 1) / warmup_iters
                progress = (step - warmup_iters) / max(total_iters - warmup_iters, 1)
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
                return min_factor + (1.0 - min_factor) * cosine

            self.actor_scheduler: LambdaLR | None = LambdaLR(self.actor_optimizer, _lr_fn)
            self.q_scheduler: LambdaLR | None = LambdaLR(self.q_optimizer, _lr_fn)
            self.alpha_scheduler: LambdaLR | None = LambdaLR(self.alpha_optimizer, _lr_fn)
        else:
            self.actor_scheduler = None
            self.q_scheduler = None
            self.alpha_scheduler = None

        # --- Zeta noise repetition (matches reference FlashSAC) ---
        if cfg.use_zeta_noise:
            self.actor.init_zeta_noise(mu=cfg.zeta_mu, max_n=cfg.zeta_max_n)

        # --- Reward normalization (matches reference FlashSAC) ---
        if cfg.normalize_reward:
            self.reward_normalizer: RewardNormalizer | None = RewardNormalizer(
                gamma=cfg.gamma,
                G_max=cfg.normalized_G_max,
                device=device,
                num_envs=env.num_envs,
            )
        else:
            self.reward_normalizer = None

        self._pg_scale_off = torch.zeros((), device=device)
        self._pg_scale_on = torch.ones((), device=device)

        # --- Replay buffer ---
        self.rb = SimpleReplayBuffer(
            n_env=env.num_envs,
            buffer_size=cfg.buffer_size,
            n_obs=actor_obs_dim,
            n_act=n_act,
            n_critic_obs=critic_obs_dim,
            n_steps=cfg.num_steps,
            gamma=cfg.gamma,
            device=device,
            n_estimator_target=cfg.vel_head_output_dim if cfg.use_velocity_estimator else 0,
            frame_ring=cfg.frame_ring_history,
            obs_seq=obs_seq,
            critic_seq=critic_seq,
        )

        # --- Policy shortcut ---
        self.policy = self.actor.explore

        # --- Logging ---
        # logger=="none" (inference/frozen-policy loads) → no SummaryWriter. A writer
        # with the inference-time log_dir (often "." or the cwd) drops empty
        # events.out.tfevents.* stubs at the repo root. add_scalar/close are
        # training-loop only, so a None writer is safe for inference.
        # Imported HERE, not at module scope: torch.utils.tensorboard pulls in the whole
        # TensorFlow package (~2.0 s), which every inference-only frozen-policy load paid
        # for a writer it then discards.
        self.writer = None
        if cfg.logger != "none":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.wandb_run = None
        if cfg.logger == "wandb":
            import wandb

            self.wandb_run = wandb.init(
                project=cfg.wandb_project,
                dir=self.log_dir,
                config=asdict(cfg),
                name=os.path.basename(self.log_dir),
            )

        print(
            f"[FlashSAC] actor_obs_shape={actor_obs_shape}, critic_obs_shape={critic_obs_shape}, "
            f"actor_obs_dim={actor_obs_dim}, critic_obs_dim={critic_obs_dim}, "
            f"n_act={n_act}, num_envs={env.num_envs}"
        )

    @contextmanager
    def _maybe_amp(self):
        amp_dtype = torch.bfloat16 if self.cfg.amp_dtype == "bf16" else torch.float16
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=self.cfg.amp):
            yield

    # ------------------------------------------------------------------
    # Critic update (distributional C51)
    # ------------------------------------------------------------------
    def _distributional_critic_loss(
        self, qnet, qnet_target, data: TensorDict,
        next_actions: torch.Tensor, next_log_probs: torch.Tensor, log_alpha: torch.Tensor,
    ):
        """C51 cross-entropy critic loss.

        Evaluates the stored EXECUTED action on the privileged critic_obs, bootstrapping from
        next_actions/next_log_probs (π on next_obs) and the entropy temperature (log_alpha). ASSUMES an active `_maybe_amp()` context (the caller opens
        it together with the bootstrap-action forward). Returns
        (rewards_mean, qf_loss, target_value_max, target_value_min, atom_boundary_mass).
        """
        cfg = self.cfg
        critic_observations = data["critic_observations"]
        next_critic_observations = data["next"]["critic_observations"]
        actions = data["actions"]
        rewards = data["next"]["rewards"]
        if self.reward_normalizer is not None:
            rewards = self.reward_normalizer.normalize_rewards(rewards)
        dones = data["next"]["dones"].bool()
        truncations = data["next"]["truncations"].bool()
        bootstrap = (truncations | ~dones).float()

        with torch.no_grad():
            discount = cfg.gamma ** data["next"]["effective_n_steps"]
            all_target_dists = qnet_target.projection(
                next_critic_observations,
                next_actions,
                rewards - discount * bootstrap * log_alpha.exp() * next_log_probs,
                bootstrap,
                discount,
            )
            target_values = qnet_target.get_value(all_target_dists)
            if cfg.use_per_q_target:
                target_distributions = all_target_dists
            else:
                min_q_idx = target_values.argmin(dim=0)
                target_distributions = all_target_dists.gather(
                    0, min_q_idx.view(1, -1, 1).expand(1, -1, cfg.num_atoms)
                ).squeeze(0)
            target_value_max = target_values.max()
            target_value_min = target_values.min()
            # Fraction of the C51 return-distribution mass sitting on the two extreme (rail) atoms.
            # Rises when returns exceed [v_min, v_max] and projection clamps onto the edges; a mean
            # well above ~1e-3 means the fixed support is too narrow and the critic is biased.
            atom_boundary_mass = (all_target_dists[..., 0] + all_target_dists[..., -1]).mean()

        q_outputs = qnet(critic_observations, actions)
        critic_log_probs = F.log_softmax(q_outputs, dim=-1)
        if cfg.use_per_q_target:
            critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
        else:
            critic_losses = -torch.sum(target_distributions.unsqueeze(0) * critic_log_probs, dim=-1)
        qf_loss = critic_losses.mean(dim=1).sum(dim=0)
        return rewards.mean(), qf_loss, target_value_max, target_value_min, atom_boundary_mass

    def _update_main_loss(self, data: TensorDict):
        with self._maybe_amp():
            with torch.no_grad():
                next_actions, next_log_probs, _ = self.actor.get_actions_and_log_probs(
                    data["next"]["observations"]
                )
            return self._distributional_critic_loss(
                self.qnet, self.qnet_target, data, next_actions, next_log_probs, self.log_alpha
            )

    # ------------------------------------------------------------------
    # Actor update
    # ------------------------------------------------------------------
    def _update_pol_loss(self, data: TensorDict):
        cfg = self.cfg

        with self._maybe_amp():
            critic_observations = data["critic_observations"]
            actions, log_probs, log_std = self.actor.get_actions_and_log_probs(data["observations"])
            action_std = log_std.detach().exp().mean()
            policy_entropy = -log_probs.mean()

            q_outputs = self.qnet(critic_observations, actions)
            q_probs = F.softmax(q_outputs, dim=-1)
            q_values = self.qnet.get_value(q_probs)
            qf_value = q_values.min(dim=0).values
            actor_loss = (self.log_alpha.exp().detach() * log_probs - qf_value).mean()

            est_loss = torch.tensor(0.0, device=self.device)
            if cfg.use_velocity_estimator and cfg.estimator_target_key in data.keys():
                est_vel = self.actor.get_estimated_velocity(stop_gradient=cfg.estimator_head_stop_gradient)
                est_loss = compute_estimator_mse(est_vel, data[cfg.estimator_target_key])
                actor_loss = actor_loss + cfg.estimator_loss_coef * est_loss

        return actor_loss, log_probs.detach(), policy_entropy, action_std, est_loss

    # ------------------------------------------------------------------
    # Batched sampling
    # ------------------------------------------------------------------
    def _iter_batches(
        self,
        batch_size: int,
        num_updates: int,
        normalize_obs,
        normalize_critic_obs,
        chunk_size: int = 4,
    ):
        """Yield normalized mini-batches, sampling chunk_size at a time.

        Memory model
        ============
        A single rb.sample(N) call allocates ~N × n_env × (actor_obs_dim + critic_obs_dim) × 4 bytes
        on the GPU.  With n_env=4096, actor_obs=1800, critic_obs=90 and batch_size_local=2:
          - chunk_size=1: ~120 MB per gather (32 gathers/iter, 800 kernel launches)
          - chunk_size=2: ~240 MB per gather (16 gathers/iter, 400 kernel launches)
          - chunk_size=4: ~480 MB per gather (8 gathers/iter, 200 kernel launches)  ← default
          - chunk_size=8: ~960 MB per gather → OOM on 24 GB GPU

        The binding constraint is the transition peak: when the generator has yielded
        the last mini-batch of chunk g, the caller's `data` loop variable still holds
        views into chunk g's allocation while chunk g+1 is being sampled.  Peak GPU
        usage = training_state + 2 × chunk_allocation:
          - chunk_size=4: ~23.7–23.9 GB on a 24 GB GPU (0.1–0.3 GB margin)
          - chunk_size=2: ~23.4 GB (safe fallback if chunk_size=4 OOMs)

        Normalizer note
        ===============
        Total Welford samples per iteration is identical to the original single-gather
        approach (same total rows × same seq_S multiplier).  The only difference is
        batching: chunk_size=4 updates the normalizer 16 times per iteration instead of
        2 times, so mini-batch k sees stats updated by k//chunk_size prior chunks.  At
        training iteration T the normalizer has seen ~T × 5.24M samples; each chunk
        contributes <0.01% to the running count → per-batch delta in normalized values
        is <0.001%.

        Args:
            batch_size: local batch size = global_batch_size // n_env (typically 2).
            num_updates: total gradient steps per outer iteration
                         = num_collect_steps × num_updates_per_step.
            normalize_obs: callable(obs, update=True) → normalized obs (actor).
            normalize_critic_obs: callable(obs, update=True) → normalized obs (critic).
            chunk_size: mini-batches per rb.sample() call.  Must divide num_updates.
        """
        assert num_updates % chunk_size == 0, (
            f"sample_chunk_size={chunk_size} must divide total_updates={num_updates}"
        )
        # Rows per mini-batch (n_env × local_batch_size).
        rows_per_minibatch = batch_size * self.env.num_envs

        for _ in range(num_updates // chunk_size):
            # Sample chunk_size mini-batches in one call.
            # rb.sample(N) returns (n_env × N, obs_dim); chunk_size × batch_size = rows per chunk.
            data = self.rb.sample(chunk_size * batch_size)

            # Normalize the entire chunk at once (inplace — no extra allocation).
            # After this, data["observations"] has been modified in place.
            data["observations"] = normalize_obs(data["observations"])
            data["next"]["observations"] = normalize_obs(data["next"]["observations"])
            data["critic_observations"] = normalize_critic_obs(data["critic_observations"])
            data["next"]["critic_observations"] = normalize_critic_obs(
                data["next"]["critic_observations"]
            )

            # Yield one mini-batch at a time as views into the chunk.
            # The caller's loop variable holds a view that keeps `data` alive until
            # the NEXT iteration of this outer for-loop calls rb.sample() — that is
            # the transition peak described above.
            for j in range(chunk_size):
                s, e = j * rows_per_minibatch, (j + 1) * rows_per_minibatch
                batch = TensorDict(
                    {
                        "observations": data["observations"][s:e],
                        "actions": data["actions"][s:e],
                        "next": {
                            "rewards": data["next"]["rewards"][s:e],
                            "dones": data["next"]["dones"][s:e],
                            "truncations": data["next"]["truncations"][s:e],
                            "observations": data["next"]["observations"][s:e],
                            "effective_n_steps": data["next"]["effective_n_steps"][s:e],
                        },
                        "critic_observations": data["critic_observations"][s:e],
                    },
                    batch_size=rows_per_minibatch,
                )
                batch["next"]["critic_observations"] = data["next"]["critic_observations"][s:e]
                if "estimator_target" in data.keys():
                    batch["estimator_target"] = data["estimator_target"][s:e]
                yield batch

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def learn(self) -> None:
        cfg = self.cfg
        device = self.device
        env = self.env

        normalize_obs = self._normalize_actor_obs
        normalize_critic_obs = self._normalize_critic_obs
        if cfg.compile:
            # compile_mode="reduce-overhead" enables cudagraphs on the hot inner loops.
            # First-call capture is slow; subsequent calls skip the per-launch CPU path.
            # See FlashSACConfig.compile_mode for trade-offs.
            _cm = cfg.compile_mode
            update_main_loss = torch.compile(self._update_main_loss, mode=_cm)
            update_pol_loss = torch.compile(self._update_pol_loss, mode=_cm)
            policy = torch.compile(self.policy, mode=_cm)
        else:
            update_main_loss = self._update_main_loss
            update_pol_loss = self._update_pol_loss
            policy = self.policy

        qnet = self.qnet
        qnet_target = self.qnet_target
        rb = self.rb

        # --- Initial reset ---
        obs_dict, _ = env.reset()
        actor_obs = obs_dict[cfg.actor_obs_group].flatten(start_dim=1)
        critic_obs = obs_dict[cfg.critic_obs_group].flatten(start_dim=1)
        est_target = (
            obs_dict[cfg.estimator_target_key].flatten(start_dim=1)
            if cfg.use_velocity_estimator and cfg.estimator_target_key in obs_dict
            else None
        )

        dones = None
        # Placeholder metrics
        policy_entropy = torch.tensor(0.0, device=device)
        action_std = torch.tensor(0.0, device=device)
        actor_loss = torch.tensor(0.0, device=device)
        actor_grad_norm = torch.tensor(0.0, device=device)
        est_loss = torch.tensor(0.0, device=device)

        # Per-env episode tracking (like rsl-rl logger)
        ep_reward_sum = torch.zeros(env.num_envs, device=device)
        ep_length_sum = torch.zeros(env.num_envs, device=device)
        episode_stat_capacity = 10000
        completed_ep_rewards_gpu = torch.empty(episode_stat_capacity, device=device)
        completed_ep_lengths_gpu = torch.empty(episode_stat_capacity, device=device)
        completed_ep_stat_pos = 0
        completed_ep_stat_count = 0

        def _append_episode_stats(dst: torch.Tensor, vals: torch.Tensor, pos: int, count: int) -> tuple[int, int]:
            """Append completed-episode scalars to a fixed GPU ring matching deque(maxlen)."""
            vals = vals.detach().to(dtype=dst.dtype)
            n = vals.numel()
            if n >= episode_stat_capacity:
                dst.copy_(vals[-episode_stat_capacity:])
                return 0, episode_stat_capacity
            end = pos + n
            if end <= episode_stat_capacity:
                dst[pos:end].copy_(vals)
            else:
                first = episode_stat_capacity - pos
                dst[pos:].copy_(vals[:first])
                dst[: end - episode_stat_capacity].copy_(vals[first:])
            return end % episode_stat_capacity, min(episode_stat_capacity, count + n)

        def _episode_stat_mean(src: torch.Tensor, pos: int, count: int) -> float | None:
            """Mean of the current rolling window, preserving chronological deque order."""
            if count == 0:
                return None
            if count < episode_stat_capacity:
                vals = src[:count]
            elif pos == 0:
                vals = src
            else:
                vals = torch.cat((src[pos:], src[:pos]))
            return statistics.mean(vals.cpu().tolist())

        # Metric accumulators. Tensors are accumulated on-GPU (no .item() per step) and
        # converted to float only at logging time — eliminates ~132 GPU-CPU syncs per
        # outer iteration that previously stalled CPU-GPU pipeline overlap.
        metric_sums: dict[str, Any] = {}
        metric_counts: dict[str, int] = {}
        reward_term_sums: dict[str, list[float]] = {}  # "Episode_Reward/*" terms
        extras_gpu_sum: dict[str, torch.Tensor] = {}  # all other extras["log"] keys
        extras_gpu_count: dict[str, int] = {}

        # Put the bar on sys.stdout — the SAME stream the periodic metrics dump uses
        # (logger.py emits via tqdm.write, which defaults to sys.stdout). tqdm.write only
        # clears+redraws bars on its own stream; a bar on sys.stderr cannot be cleared by a
        # stdout dump, so on a shared TTY the bar overwrites the metric lines. Matching the
        # stream lets tqdm.write coordinate cleanly. Disable when stdout is redirected
        # (non-TTY) so carriage-return updates do not pollute training.log.
        pbar = tqdm.tqdm(
            total=cfg.num_learning_iterations,
            initial=self.global_step,
            dynamic_ncols=True,
            mininterval=2.0,
            file=sys.stdout,
            disable=not sys.stdout.isatty(),
        )

        # Timing
        start_time = time.time()
        tot_time = 0.0
        collect_time = 0.0
        learn_time = 0.0
        start_it = self.global_step
        _log_iters: int = 0

        # Batch size is constant; compute once outside the loop.
        batch_size = max(cfg.batch_size // env.num_envs, 1)

        # total gradient steps per outer iteration = num_collect_steps × num_updates.
        # compute once; used by _iter_batches below.
        total_updates = cfg.num_collect_steps * cfg.num_updates

        while self.global_step <= cfg.num_learning_iterations:

            # =============================================================
            # 1. COLLECT — all num_collect_steps env steps before any update
            # =============================================================
            # Collect-all-then-update-all preserves the original UTD ratio and
            # avoids mid-collection policy updates that would change the behavioral
            # distribution of transitions collected within the same outer iteration.
            for _collect_idx in range(cfg.num_collect_steps):
                collect_start = time.time()
                with torch.no_grad(), self._maybe_amp():
                    norm_obs = normalize_obs(actor_obs, update=False)
                    actions = policy(obs=norm_obs, dones=dones)

                (
                    next_obs_dict,
                    rewards,
                    terminated,
                    truncated,
                    extras,
                ) = env.step(actions.float())

                next_actor_obs = next_obs_dict[cfg.actor_obs_group].flatten(start_dim=1)
                next_critic_obs = next_obs_dict[cfg.critic_obs_group].flatten(start_dim=1)
                dones = (terminated | truncated).to(torch.uint8)
                truncations = truncated.to(torch.uint8)

                # Alive reward bonus
                if cfg.alive_reward != 0.0:
                    alive_bonus = cfg.alive_reward * (~terminated.bool()).float() * env.step_dt
                    rewards = rewards + alive_bonus

                # Reward normalization stats (running G_t per env). Buffer stores raw
                # rewards; normalization is applied at batch consumption time.
                if self.reward_normalizer is not None:
                    self.reward_normalizer.update_reward_stats(
                        reward=rewards.float(),
                        terminated=terminated.bool(),
                        truncated=truncated.bool(),
                    )

                # Track per-env episode stats
                ep_reward_sum += rewards
                ep_length_sum += 1
                done_mask = dones.bool()
                if done_mask.any():
                    old_pos = completed_ep_stat_pos
                    old_count = completed_ep_stat_count
                    new_pos, new_count = _append_episode_stats(
                        completed_ep_rewards_gpu,
                        ep_reward_sum[done_mask],
                        old_pos,
                        old_count,
                    )
                    _append_episode_stats(
                        completed_ep_lengths_gpu,
                        ep_length_sum[done_mask],
                        old_pos,
                        old_count,
                    )
                    completed_ep_stat_pos = new_pos
                    completed_ep_stat_count = new_count
                    ep_reward_sum[done_mask] = 0.0
                    ep_length_sum[done_mask] = 0.0

                # Handle final observations for reset envs.
                true_next_actor = next_actor_obs
                true_next_critic = next_critic_obs
                if env.final_obs is not None:
                    # C3: reuse the ids computed in env.step() (avoid a second
                    # nonzero kernel per step; same tensor, bit-exact).
                    reset_ids = env.reset_env_ids
                    assert reset_ids is not None
                    if len(reset_ids) > 0:
                        true_next_actor = next_actor_obs.clone()
                        true_next_critic = next_critic_obs.clone()
                        true_next_actor[reset_ids] = env.final_obs[cfg.actor_obs_group].flatten(start_dim=1)
                        true_next_critic[reset_ids] = env.final_obs[cfg.critic_obs_group].flatten(start_dim=1)

                # Store transition
                transition = TensorDict(
                    {
                        "observations": actor_obs,
                        "actions": actions.float(),
                        "next": {
                            "observations": true_next_actor,
                            "rewards": rewards.float(),
                            "truncations": truncations.long(),
                            "dones": dones,
                        },
                    },
                    batch_size=(env.num_envs,),
                    device=device,
                )
                transition["critic_observations"] = critic_obs
                transition["next"]["critic_observations"] = true_next_critic
                if est_target is not None:
                    transition["estimator_target"] = est_target

                actor_obs = next_actor_obs
                critic_obs = next_critic_obs
                if est_target is not None and cfg.estimator_target_key in next_obs_dict:
                    est_target = next_obs_dict[cfg.estimator_target_key].flatten(start_dim=1)

                rb.extend(transition)

                # Track per-reward-term statistics from extras
                if "log" in extras:
                    log_info = extras["log"]
                    for key, val in log_info.items():
                        if isinstance(val, torch.Tensor):
                            v = val.detach().to(device=device, dtype=torch.float32).reshape(())
                        elif isinstance(val, (int, float)):
                            v = torch.tensor(float(val), device=device)
                        else:
                            continue
                        if key.startswith("Episode_Reward/"):
                            reward_term_sums.setdefault(key, []).append(v.item())
                        else:
                            if key in extras_gpu_sum:
                                extras_gpu_sum[key].add_(v)
                            else:
                                extras_gpu_sum[key] = v.clone()
                            extras_gpu_count[key] = extras_gpu_count.get(key, 0) + 1

                collect_time += time.time() - collect_start

            # =============================================================
            # 2. UPDATE — total_updates gradient steps after all env steps
            # =============================================================
            # _iter_batches samples cfg.sample_chunk_size mini-batches per rb.sample()
            # call, yielding one mini-batch at a time.  This keeps peak GPU allocation
            # at ~2 × chunk_allocation (transition peak) instead of the full
            # total_updates × mini-batch size that a single rb.sample(total_updates)
            # would require.  See FlashSACConfig.sample_chunk_size and
            # _iter_batches docstring for the full memory analysis.
            learn_start = time.time()
            updates_ready = (
                self.global_step > cfg.learning_starts
                and rb.ptr >= self._replay_warmup_steps
            )
            if updates_ready:
                # i counts across ALL total_updates steps; policy_frequency check uses i
                # so actor update cadence matches the original (every policy_frequency
                # critic updates across the full 32-step sequence).
                for i, data in enumerate(self._iter_batches(
                    batch_size,
                    total_updates,
                    normalize_obs,
                    normalize_critic_obs,
                    chunk_size=cfg.sample_chunk_size,
                )):
                    buffer_rewards, qf_loss_raw, qf_max, qf_min, atom_boundary_mass = update_main_loss(data)
                    self.q_optimizer.zero_grad(set_to_none=True)
                    qf_loss_raw.backward()
                    if cfg.max_grad_norm > 0:
                        critic_grad_norm = torch.nn.utils.clip_grad_norm_(qnet.parameters(), max_norm=cfg.max_grad_norm)
                    else:
                        critic_grad_norm = torch.tensor(0.0, device=device)
                    self.q_optimizer.step()
                    qf_loss = qf_loss_raw.detach()

                    alpha_loss = torch.tensor(0.0, device=device)
                    est_loss = torch.tensor(0.0, device=device)
                    should_update_actor = (
                        (i % cfg.policy_frequency == 1)
                        if cfg.num_updates > 1
                        else (self.global_step % cfg.policy_frequency == 0)
                    )
                    if should_update_actor:
                        actor_loss_raw, log_probs_detached, policy_entropy, action_std, est_loss = update_pol_loss(data)
                        self.actor_optimizer.zero_grad(set_to_none=True)
                        actor_loss_raw.backward()
                        if cfg.max_grad_norm > 0:
                            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=cfg.max_grad_norm)
                        else:
                            actor_grad_norm = torch.tensor(0.0, device=device)
                        self.actor_optimizer.step()
                        if cfg.use_autotune:
                            self.alpha_optimizer.zero_grad(set_to_none=True)
                            alpha_loss = (self.log_alpha.exp() * (-log_probs_detached - self.target_entropy)).mean()
                            alpha_loss.backward()
                            self.alpha_optimizer.step()
                        actor_loss = actor_loss_raw.detach()

                    # Accumulate metrics
                    _metrics = {
                        "actor_loss": actor_loss,
                        "qf_loss": qf_loss,
                        "qf_max": qf_max,
                        "qf_min": qf_min,
                        "atom_boundary_mass": atom_boundary_mass,
                        "critic_grad_norm": critic_grad_norm,
                        "actor_grad_norm": actor_grad_norm,
                        "alpha_loss": alpha_loss,
                        "alpha_value": self.log_alpha.exp().detach().mean(),
                        "policy_entropy": policy_entropy,
                        "action_std": action_std,
                        "buffer_rewards": buffer_rewards,
                    }
                    if cfg.use_velocity_estimator:
                        _metrics["estimator_loss"] = est_loss

                    enc = None
                    actor_net = self.actor
                    if hasattr(actor_net, "core") and hasattr(actor_net.core, "backbone"):
                        enc = getattr(actor_net.core.backbone, "encoder", None)
                    elif hasattr(actor_net, "backbone"):
                        enc = getattr(actor_net.backbone, "encoder", None)

                    if enc is not None and hasattr(enc, "last_max_weight") and enc.last_max_weight is not None:
                        _metrics["temporal_max_weight"] = enc.last_max_weight

                    for k, v in _metrics.items():
                        if isinstance(v, torch.Tensor):
                            v = v.detach()
                            if k in metric_sums:
                                metric_sums[k].add_(v)
                            else:
                                metric_sums[k] = v.clone()
                        else:
                            metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
                        metric_counts[k] = metric_counts.get(k, 0) + 1

                    # Polyak update target network
                    with torch.no_grad():
                        src_ps = [p.data for p in qnet.parameters()]
                        tgt_ps = [p.data for p in qnet_target.parameters()]
                        torch._foreach_mul_(tgt_ps, 1.0 - cfg.tau)
                        torch._foreach_add_(tgt_ps, src_ps, alpha=cfg.tau)

            learn_time += time.time() - learn_start

            if updates_ready:
                tot_time = time.time() - start_time
                _log_iters += 1

                # =============================================================
                # 3. LOG
                # =============================================================
                if self.global_step % cfg.logging_interval == 0 and metric_counts:
                    with torch.no_grad():
                        wandb_metrics = {}
                        metric_name_map = {
                            "actor_loss": "Loss/actor_loss",
                            "qf_loss": "Loss/qf_loss",
                            "qf_max": "Policy/q_value_max",
                            "qf_min": "Policy/q_value_min",
                            "atom_boundary_mass": "Policy/atom_boundary_mass",
                            "critic_grad_norm": "Loss/critic_grad_norm",
                            "actor_grad_norm": "Loss/actor_grad_norm",
                            "alpha_loss": "Loss/alpha_loss",
                            "alpha_value": "Policy/alpha",
                            "policy_entropy": "Loss/entropy",
                            "action_std": "Policy/mean_noise_std",
                            "buffer_rewards": "Train/buffer_reward_mean",
                            "estimator_loss": "Loss/estimator_loss",
                            "temporal_max_weight": "Loss/temporal_max_weight",
                        }
                        # Compute averages — single .item() per metric here (not per grad step).
                        metric_avgs = {}
                        for k in metric_sums:
                            val = metric_sums[k]
                            avg = (val.item() if isinstance(val, torch.Tensor) else val) / metric_counts[k]
                            metric_avgs[k] = avg
                            tag = metric_name_map.get(k, f"Train/{k}")
                            self.writer.add_scalar(tag, avg, self.global_step)
                            wandb_metrics[tag] = avg
                        env_rew = rewards.mean().item()
                        self.writer.add_scalar(
                            "Train/env_reward_mean_step", env_rew, self.global_step
                        )
                        wandb_metrics["Train/env_reward_mean_step"] = env_rew
                        # Log mean episode reward/length (matching PPO format).
                        mean_rew = None
                        mean_len = None
                        mean_rew = _episode_stat_mean(
                            completed_ep_rewards_gpu,
                            completed_ep_stat_pos,
                            completed_ep_stat_count,
                        )
                        if mean_rew is not None:
                            self.writer.add_scalar("Train/mean_reward", mean_rew, self.global_step)
                            wandb_metrics["Train/mean_reward"] = mean_rew
                        mean_len = _episode_stat_mean(
                            completed_ep_lengths_gpu,
                            completed_ep_stat_pos,
                            completed_ep_stat_count,
                        )
                        if mean_len is not None:
                            self.writer.add_scalar("Train/mean_episode_length", mean_len, self.global_step)
                            wandb_metrics["Train/mean_episode_length"] = mean_len
                        # Log individual reward terms
                        reward_term_avgs = {}
                        if reward_term_sums:
                            for key, vals in reward_term_sums.items():
                                avg = statistics.mean(vals)
                                reward_term_avgs[key] = avg
                                self.writer.add_scalar(key, avg, self.global_step)
                                wandb_metrics[key] = avg
                        total_steps = self.global_step * cfg.num_collect_steps * env.num_envs
                        fps = total_steps / tot_time if tot_time > 0 else 0.0
                        avg_collect = collect_time / max(_log_iters, 1)
                        avg_learn = learn_time / max(_log_iters, 1)
                        self.writer.add_scalar("Perf/collection_time", avg_collect, self.global_step)
                        self.writer.add_scalar("Perf/learning_time", avg_learn, self.global_step)
                        self.writer.add_scalar("Perf/total_fps", fps, self.global_step)
                        self.writer.add_scalar("Perf/total_env_steps", total_steps, self.global_step)
                        wandb_metrics["Perf/collection_time"] = avg_collect
                        wandb_metrics["Perf/learning_time"] = avg_learn
                        wandb_metrics["Perf/total_fps"] = fps
                        wandb_metrics["Perf/total_env_steps"] = total_steps

                        # Build extras_log: reward terms (interval avg) + running mean of all other extras
                        extras_log: dict[str, float] = dict(reward_term_avgs)
                        for key, total in extras_gpu_sum.items():
                            count = extras_gpu_count.get(key, 1)
                            extras_log[key] = (total / count).item() if count > 0 else 0.0

                        # Write env extras (Curriculum/*, Metrics/*, etc.) to TB + wandb
                        for key, val in extras_log.items():
                            if not key.startswith("Episode_Reward/"):
                                self.writer.add_scalar(key, val, self.global_step)
                                wandb_metrics[key] = val

                        # Log to wandb (after all metrics including extras are collected)
                        if self.wandb_run is not None:
                            import wandb

                            wandb.log(wandb_metrics, step=self.global_step)

                        self.compact_logger.log_flash_sac(
                            it=self.global_step,
                            total_it=cfg.num_learning_iterations,
                            collect_time=avg_collect,
                            learn_time=avg_learn,
                            metric_avgs=metric_avgs,
                            mean_rew=mean_rew,
                            mean_len=mean_len,
                            extras_log=extras_log,
                            total_steps=total_steps,
                            fps=fps,
                            tot_time=tot_time,
                            iter_time=avg_collect + avg_learn,
                            start_it=start_it,
                        )

                    metric_sums.clear()
                    metric_counts.clear()
                    reward_term_sums.clear()
                    extras_gpu_sum.clear()
                    extras_gpu_count.clear()
                    collect_time = 0.0
                    learn_time = 0.0
                    _log_iters = 0

                # =============================================================
                # 4. SAVE
                # =============================================================
                if (
                    cfg.save_interval > 0
                    and self.global_step > 0
                    and self.global_step % cfg.save_interval == 0
                ):
                    self.save(
                        os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt")
                    )

            if self.global_step >= cfg.num_learning_iterations:
                break
            self.global_step += 1
            pbar.update(1)
            if self.actor_scheduler is not None and self.q_scheduler is not None and self.alpha_scheduler is not None:
                self.actor_scheduler.step()
                self.q_scheduler.step()
                self.alpha_scheduler.step()

        # Final save
        self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
        if self.writer is not None:
            self.writer.close()
        if self.wandb_run is not None:
            import wandb

            wandb.finish()
        print(f"[FlashSAC] Training complete at step {self.global_step}")

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    # Checkpoint schema tag. Bump when the nested layout below changes.
    CHECKPOINT_FORMAT = "flash_sac_v2"

    def save(self, path: str) -> None:
        """Write a checkpoint in the nested fresh schema (CHECKPOINT_FORMAT).

        Layout: top-level meta (`format`, `algorithm`, `global_step`, `args`) + grouped
        sub-dicts `networks` / `normalizers` / `optimizers`, plus optional `schedulers`.
        `args` stays top-level (read via from_saved_args).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt: dict[str, Any] = {
            "format": self.CHECKPOINT_FORMAT,
            "algorithm": "flash_sac",
            "global_step": self.global_step,
            "args": asdict(self.cfg),
            "networks": {
                "actor": cpu_state(self.actor.state_dict()),
                "qnet": cpu_state(self.qnet.state_dict()),
                "qnet_target": cpu_state(self.qnet_target.state_dict()),
                "log_alpha": self.log_alpha.detach().cpu(),
            },
            "normalizers": {
                "obs": cpu_state(self.obs_normalizer.state_dict()),
                "critic_obs": cpu_state(self.critic_obs_normalizer.state_dict()),
            },
            "optimizers": {
                "actor": self.actor_optimizer.state_dict(),
                "qnet": self.q_optimizer.state_dict(),
                "alpha": self.alpha_optimizer.state_dict(),
            },
        }
        if self.actor_scheduler is not None:
            ckpt["schedulers"] = {
                "actor": self.actor_scheduler.state_dict(),
                "qnet": self.q_scheduler.state_dict(),
                "alpha": self.alpha_scheduler.state_dict(),
            }
        command_curricula = {
            name: self.env.command_manager.get_term(name).curriculum_state_dict()
            for name in self.env.command_manager.active_terms
            if hasattr(self.env.command_manager.get_term(name), "curriculum_state_dict")
        }
        if command_curricula:
            ckpt["command_curricula"] = command_curricula
        torch.save(ckpt, path)
        print(f"[FlashSAC] Saved checkpoint to {path}")
        if self.wandb_run is not None:
            import wandb

            wandb.save(path, base_path=os.path.dirname(path))

    def load(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        if ckpt.get("format") != self.CHECKPOINT_FORMAT:
            raise ValueError(
                f"{ckpt_path} is not a {self.CHECKPOINT_FORMAT} checkpoint "
                f"(format={ckpt.get('format')!r})."
            )

        nets = ckpt["networks"]
        self.actor.load_state_dict(nets["actor"])
        self.qnet.load_state_dict(nets["qnet"])
        self.qnet_target.load_state_dict(nets["qnet_target"])
        self.log_alpha.data.copy_(nets["log_alpha"].to(self.device))

        norms = ckpt["normalizers"]
        if norms.get("obs"):
            self.obs_normalizer.load_state_dict(norms["obs"])
        if norms.get("critic_obs"):
            self.critic_obs_normalizer.load_state_dict(norms["critic_obs"])

        opt = ckpt["optimizers"]
        self.actor_optimizer.load_state_dict(opt["actor"])
        self.q_optimizer.load_state_dict(opt["qnet"])
        self.alpha_optimizer.load_state_dict(opt["alpha"])

        self.global_step = ckpt.get("global_step", 0)
        # Replay contents, current environment states, and RNG streams are not
        # checkpointed.  Do not update the restored critics from the handful of
        # transitions collected immediately after a load: first replace the full
        # per-env replay ring with trajectories from the restored policy.
        self._replay_warmup_steps = self.rb.buffer_size
        # ManagerBasedRlEnv does not persist its global curriculum counter in a FlashSAC checkpoint.
        # Reconstruct it before the caller's next reset so curriculum-driven commands (notably the
        # mixed-arm home_ratio) resume their trained phase instead of restarting at step zero.
        self.env.common_step_counter = self.global_step * self.cfg.num_collect_steps
        print(f"[FlashSAC] Restored env.common_step_counter={self.env.common_step_counter} ({self.global_step} × {self.cfg.num_collect_steps})")
        for name, state in ckpt.get("command_curricula", {}).items():
            if name not in self.env.command_manager.active_terms:
                continue
            term = self.env.command_manager.get_term(name)
            if hasattr(term, "load_curriculum_state_dict"):
                term.load_curriculum_state_dict(state)
        schedulers = ckpt.get("schedulers")
        if self.actor_scheduler is not None and schedulers:
            self.actor_scheduler.load_state_dict(schedulers["actor"])
            self.q_scheduler.load_state_dict(schedulers["qnet"])
            self.alpha_scheduler.load_state_dict(schedulers["alpha"])
        print(f"[FlashSAC] Loaded checkpoint from {ckpt_path} at step {self.global_step}")
        print(
            "[FlashSAC] Replay reset on load; collecting "
            f"{self._replay_warmup_steps} fresh env steps before updates."
        )

    def get_inference_policy(self, device: str | None = None):
        """Return callable policy for play mode."""
        device = device or self.device
        policy = self.actor.to(device)
        normalizer = self.obs_normalizer
        normalize = self._normalize_actor_obs
        group_key = self.cfg.actor_obs_group
        prep = lambda t: t.flatten(start_dim=1)
        normalizer.to(device)
        policy.eval()
        normalizer.eval()

        def policy_fn(obs):
            if isinstance(obs, dict) or hasattr(obs, "get"):
                obs_tensor = obs.get(group_key, obs.get("actor", obs.get("policy", next(iter(obs.values())))))
            else:
                obs_tensor = obs
            obs_tensor = prep(obs_tensor)
            if self.cfg.obs_normalization:
                normalized = normalize(obs_tensor, update=False)
            else:
                normalized = obs_tensor
            return policy(normalized)[0]

        return policy_fn
