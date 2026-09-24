"""FlashSAC configuration for mjlab environments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FlashSACConfig:
    """Configuration for FlashSAC training."""

    # Training
    num_learning_iterations: int = 25000
    learning_starts: int = 10

    # Network architecture
    actor_hidden_dim: int = 512
    critic_hidden_dim: int = 768
    use_layer_norm: bool = True

    # SAC hyperparameters
    critic_learning_rate: float = 3e-4
    actor_learning_rate: float = 3e-4
    alpha_learning_rate: float = 3e-4

    gamma: float = 0.99
    tau: float = 0.033  # Scaled from 0.125 for 8 updates → 0.033 for 32 (maintains same effective target convergence rate)
    alpha_init: float = 0.01
    use_autotune: bool = True
    target_entropy_ratio: float = 0.5
    # Target entropy via target_sigma (preferred — matches reference FlashSAC).
    # When > 0, overrides target_entropy_ratio: target_entropy = 0.5*n_act*log(2πe·σ²).
    # σ=0.15 → for |A|=22, target_entropy ≈ -10.5 (vs ratio=0.25 giving -5.5).
    # Set to 0.0 to fall back to target_entropy_ratio formulation.
    target_sigma: float = 0.0

    # Replay buffer
    buffer_size: int = 1024  # Per environment
    batch_size: int = 8192  # Global batch size
    num_steps: int = 1  # N-step returns
    # Frame-ring replay: store one frame per timestep per history
    # view and reconstruct the L-frame window at sample time, instead of storing the full overlapping
    # window per transition. ~L× less replay VRAM, lossless, same sampling distribution. Default ON —
    # runner.py degenerates any flat/non-history obs group to a bit-exact 1-frame ring, so this is
    # safe even for experiments without use_sequence_encoder=True (zero savings there, but no harm).
    frame_ring_history: bool = True

    # Update schedule
    num_updates: int = 8  # Updates per env step
    policy_frequency: int = 2  # Actor updates every N critic updates
    num_collect_steps: int = 4  # Env steps collected per iteration (increases data 4x)

    # Distributional critic (C51)
    num_atoms: int = 501
    v_min: float = -20.0
    v_max: float = 20.0
    num_q_networks: int = 2

    # Action handling
    use_tanh: bool = True

    # Optimization
    compile: bool = True
    # torch.compile mode: "default" (no cudagraphs), "reduce-overhead" (cudagraphs on the
    # per-step actor fwd + the learn-step loss fns), "max-autotune" (also picks
    # best matmul). cudagraphs eliminate per-launch CPU overhead on the hot inner loops
    # (actor fwd is called 4× per collect step). Trade-off: first-call warmup is
    # much longer (graphs are captured lazily) and dynamic shapes break capture (we keep
    # shape-static on the fwd side). Default "default" preserves byte-identical numerics
    # to date; flip to "reduce-overhead" to A/B test capture overhead.
    compile_mode: str = "default"
    amp: bool = True
    amp_dtype: str = "bf16"
    weight_decay: float = 0.001
    max_grad_norm: float = 0.0
    # LR schedule (applied to actor, critic, alpha optimizers).
    # linear warmup 0 → base_lr over lr_warmup_iters, then cosine decay to
    # base_lr * lr_min_factor over lr_decay_iters. Both 0 = constant lr.
    lr_warmup_iters: int = 0
    lr_decay_iters: int = 0
    lr_min_factor: float = 0.1

    # Observation normalization
    obs_normalization: bool = True

    # Reward normalization (matches reference FlashSAC).
    # Tracks discounted returns G_t per env, divides batch rewards by max(sqrt(Var[G]), max|G|/G_max).
    # When True, set v_min=-G_max, v_max=G_max so critic support matches normalized scale.
    normalize_reward: bool = False
    normalized_G_max: float = 5.0

    # Log std bounds
    log_std_max: float = 0.0
    log_std_min: float = -5.0

    # Logging & saving
    save_interval: int = 2500
    logging_interval: int = 50
    logger: str = "wandb"  # "wandb" or "tensorboard"
    wandb_project: str = "mjlab"

    # mjlab-specific: observation group names
    actor_obs_group: str = "actor"
    critic_obs_group: str = "critic"

    # Alive reward bonus (added per step for off-policy stability)
    alive_reward: float = 0.0

    # Sequence encoder (RMA-CNN / TCN) for history obs. Disabled by default (flat MLP).
    use_sequence_encoder: bool = False
    encoder_type: str = "rma_cnn"       # "rma_cnn" | "tcn" | "attn_rma_cnn"
    encoder_embed_dim: int = 32
    encoder_latent_dim: int = 128

    # Velocity estimator head on the sequence encoder latent (feature-gated).
    # Requires use_sequence_encoder=True. When enabled, SequenceActor owns a
    # VelocityEstimatorHead and the actor update adds estimator MSE to actor_loss.
    # estimator_target_key must name a key present in the critic obs TensorDict
    # (e.g. "estimator_target") providing ground-truth base_lin_vel (B, 3).
    # Default-off: exact current behavior when use_velocity_estimator=False.
    use_velocity_estimator: bool = False
    estimator_loss_coef: float = 1.0
    estimator_target_key: str = "estimator_target"
    # Output dimension of the velocity estimator head. Default 3 = (vx, vy, vz).
    # Increase to include privileged physical latents (friction, mass scale, payload).
    vel_head_output_dim: int = 3
    # Stop-gradient between the estimator head and the encoder.
    # True (default): estimator MSE gradient stops at the vel_head — encoder is not updated
    #   by the estimator loss (no side effects on locomotion).
    # False: estimator MSE gradient flows back into the encoder, forcing the encoder to
    #   represent the estimator targets (e.g. friction, mass scale).
    estimator_head_stop_gradient: bool = True

    # Per-Q-net target distributions (ablation: matches old fast_sac behavior).
    # When False (default): both Q-nets train against the min-Q distribution (current behavior).
    # When True: each Q-net trains against its own projected distribution (old fast_sac runner).
    use_per_q_target: bool = False

    # Zeta-distributed noise repetition (matches reference FlashSAC).
    # Per env, sample N ~ Zeta(zeta_mu) ∈ [1, zeta_max_n]; reuse same noise ε
    # for N consecutive steps; resample at boundaries. Eliminates IID per-step
    # jitter in replay buffer transitions, preventing critic from learning to
    # value action chatter. PMF for mu=2.0: P(N=1)≈61%, P(N=2)≈15%, mean≈1.85.
    use_zeta_noise: bool = False
    zeta_mu: float = 2.0
    zeta_max_n: int = 16

    # Mini-batches per rb.sample() call.
    # More mini-batches per sample → fewer kernel launches (gather is memory-bandwidth-bound,
    # not compute-bound, so throughput is unchanged) but higher transition-peak GPU memory.
    # Transition peak = 2 × (chunk_size × batch_size_local) rows of obs (old chunk still alive
    # when new chunk is allocated).
    # - 1: 32 gathers/iter, ~23.2 GB peak.
    # - 2: 16 gathers/iter, ~23.4 GB peak (safe fallback on 24 GB GPU).
    # - 4 (default): 8 gathers/iter, ~23.7–23.9 GB peak (tight; reduce to 2 if OOM).
    # - 8: OOM on 24 GB GPU.
    # Must divide total_updates = num_collect_steps × num_updates evenly.
    sample_chunk_size: int = 4

    @classmethod
    def from_saved_args(cls, saved: dict) -> "FlashSACConfig":
        """Build a config from a checkpoint's saved `args`, dropping any unknown keys.

        Checkpoints store current field names directly, so no per-load key remapping is needed.
        """
        valid = cls.__dataclass_fields__
        return cls(**{k: v for k, v in saved.items() if k in valid})
