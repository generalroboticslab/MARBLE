"""FlashSAC network definitions: sequence encoders, RMA estimator core, SAC actor/critic.

Merged from history_encoder.py + rma_estimator.py + flash_sac.py (pure relocation). Definition order
encoders -> estimator -> actor/critic so module-level names resolve without forward references; the
former lazy intra-package imports become same-module references.
"""
# NOTE: deliberately NO `from __future__ import annotations`. PEP 563 stringizes
# class-body annotations, and TorchScript cannot resolve the stringized `Final[bool]`
# (Actor/RMAEstimatorCore constants) — `torch.jit.script` raises
# "Unknown type annotation: 'Final[bool]'" during deployment export. Keeping eager
# annotations lets TorchScript read the real typing.Final/Optional objects. Definition
# order (encoders -> estimator -> actor/critic) avoids forward references, so eager
# annotation evaluation imports cleanly.
import math
from typing import Final, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import EmpiricalNormalization


def _sinusoidal_pe(latent_dim: int, max_len: int = 100) -> torch.Tensor:
    """Fixed sinusoidal positional encoding in channels-first layout (1, latent_dim, max_len).

    Assumes latent_dim is even. Breaks permutation invariance from step 0 without
    requiring any gradient budget — unlike learned zero-init pos_embed which takes
    many iterations to develop a usable temporal signal.
    """
    pe = torch.zeros(1, latent_dim, max_len)
    pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)        # (S, 1)
    div = torch.exp(
        torch.arange(0, latent_dim, 2, dtype=torch.float) * -(math.log(10000.0) / latent_dim)
    )                                                                    # (latent_dim//2,)
    pe[0, 0::2, :] = torch.sin(pos * div).T
    pe[0, 1::2, :] = torch.cos(pos * div).T
    return pe


class RMACNNEncoder(nn.Module):
    """Per-step embedding + 3-layer 1D conv over time. Mirrors RMA adaptation module.

    Args:
        input_dim: Observation dimension per frame.
        embed_dim: Per-step projection dimension (RMA default: 32).
        latent_dim: Output latent dimension.
    """
    def __init__(self, input_dim: int, embed_dim: int = 32, latent_dim: int = 128):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.ELU())
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 32, kernel_size=3, padding=1), nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ELU(),
            nn.Conv1d(32, latent_dim, kernel_size=3, padding=1), nn.ELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Seq, Dim)
        B, S, D = x.shape
        embedded = self.embed(x.reshape(B * S, D)).view(B, S, -1)  # (B, S, embed_dim)
        return self.conv(embedded.permute(0, 2, 1)).mean(dim=-1)    # (B, latent_dim)


class AttentionPoolRMACNNEncoder(nn.Module):
    """RMA-CNN with learned attention pooling over the time axis instead of mean pool.

    Zero-init query: at init all logits=0 → softmax gives uniform weights (1/S each),
    so training starts from the same point as mean-pool baseline.

    pos_enc_type options:
      "none"       — no positional encoding (permutation invariant)
      "learned"    — zero-init nn.Parameter (slow to develop temporal signal)
      "sinusoidal" — fixed sinusoidal buffer (breaks permutation invariance from step 0)

    attn_temperature: softmax temperature τ. logits divided by τ before softmax.
      τ=1.0 (default): standard attention, can sharpen/collapse under DR.
      τ=2.0: keeps weights closer to uniform, reduces collapse risk.

    use_learnable_norm: if True, use nn.RMSNorm(latent_dim) with trainable gamma before
      the attention query. If False (default), use fixed-scale RMSNorm (gamma=1).
      Learnable gamma can amplify logits (→ sharper attention, collapse risk) or shrink
      them (→ uniform, stable) depending on gradient direction.

    Collapse diagnostic: `last_max_weight` (GPU scalar tensor) updated each forward
    with batch-mean of max temporal weight. Uniform → 1/S; collapse → 1.0.
    """
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 32,
        latent_dim: int = 128,
        pos_enc_type: str = "none",
        attn_temperature: float = 1.0,
        use_learnable_norm: bool = False,
    ):
        super().__init__()
        self.pos_enc_type = pos_enc_type
        self.attn_temperature = attn_temperature
        self.embed = nn.Sequential(nn.Linear(input_dim, embed_dim), nn.ELU())
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 32, kernel_size=3, padding=1), nn.ELU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ELU(),
            nn.Conv1d(32, latent_dim, kernel_size=3, padding=1), nn.ELU(),
        )
        self.attn_query = nn.Conv1d(latent_dim, 1, kernel_size=1, bias=False)
        nn.init.zeros_(self.attn_query.weight)

        # nn.RMSNorm normalizes over last dim — apply on (B, S, latent_dim) transpose.
        self.attn_norm = nn.RMSNorm(latent_dim) if use_learnable_norm else None

        if pos_enc_type == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, latent_dim, 100))
        elif pos_enc_type == "sinusoidal":
            self.register_buffer("pos_embed", _sinusoidal_pe(latent_dim))
        else:
            self.pos_embed = None

        self.last_max_weight: torch.Tensor = torch.zeros(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        embedded = self.embed(x.reshape(B * S, D)).view(B, S, -1).permute(0, 2, 1)  # (B, embed_dim, S)
        conv_out = self.conv(embedded)                                               # (B, latent_dim, S)

        if self.pos_embed is not None:
            conv_pos = conv_out + self.pos_embed[:, :, :S]
        else:
            conv_pos = conv_out

        if self.attn_norm is not None:
            # nn.RMSNorm expects channels-last: transpose → norm → transpose back
            normalized = self.attn_norm(conv_pos.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            # Fixed-scale RMSNorm (gamma=1): normalize channels at each timestep
            rms = torch.rsqrt(conv_pos.pow(2).mean(dim=1, keepdim=True) + 1e-6)
            normalized = conv_pos * rms

        logits = self.attn_query(normalized)                                         # (B, 1, S)
        weights = F.softmax(logits / self.attn_temperature, dim=2)                   # (B, 1, S)

        if self.training:
            self.last_max_weight = weights.detach().squeeze(1).max(dim=1).values.mean()

        return (conv_pos * weights).sum(dim=2)                                       # (B, latent_dim)


class TCNEncoder(nn.Module):
    """Dilated 1D TCN with symmetric padding.

    Kernel=5, dilations=[1,2,4]: receptive field = 1+4+8+16 = 29 steps, covering the
    full 25-step history in a single pass. Symmetric padding — we always process a
    complete fixed-size history buffer.
    """
    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=5, dilation=1, padding=2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, dilation=2, padding=4)
        self.conv3 = nn.Conv1d(64, latent_dim, kernel_size=5, dilation=4, padding=8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        return x.mean(dim=-1)  # (B, latent_dim)


_ENCODER_CLASSES = {
    "rma_cnn":                    RMACNNEncoder,
    "attn_rma_cnn":               AttentionPoolRMACNNEncoder,   # no pos enc, τ=1.0
    "attn_pos_rma_cnn":           AttentionPoolRMACNNEncoder,   # learned pos enc, τ=1.0
    "attn_sinpos_rma_cnn":        AttentionPoolRMACNNEncoder,   # sinusoidal pos enc, τ=1.0
    "attn_rma_cnn_t2":            AttentionPoolRMACNNEncoder,   # no pos enc, τ=2.0
    "attn_sinpos_rma_cnn_t2":     AttentionPoolRMACNNEncoder,   # sinusoidal pos enc, τ=2.0
    "attn_sinpos_rma_cnn_norm":   AttentionPoolRMACNNEncoder,   # sinusoidal pos enc, learnable γ
    "tcn":                        TCNEncoder,
}

_ACTIVATION_CLASSES = {
    "elu": nn.ELU,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
}


def build_encoder(
    encoder_type: str,
    input_dim: int,
    embed_dim: int = 32,
    latent_dim: int = 128,
) -> nn.Module:
    """Factory for sequence encoder variants."""
    if encoder_type not in _ENCODER_CLASSES:
        raise ValueError(f"Unknown encoder_type: {encoder_type!r}. Choose from {list(_ENCODER_CLASSES)}")
    cls = _ENCODER_CLASSES[encoder_type]
    if encoder_type == "tcn":
        return cls(input_dim=input_dim, latent_dim=latent_dim)
    _attn_kwargs = {
        "attn_rma_cnn":             dict(pos_enc_type="none",       attn_temperature=1.0),
        "attn_pos_rma_cnn":         dict(pos_enc_type="learned",    attn_temperature=1.0),
        "attn_sinpos_rma_cnn":      dict(pos_enc_type="sinusoidal", attn_temperature=1.0),
        "attn_rma_cnn_t2":          dict(pos_enc_type="none",       attn_temperature=2.0),
        "attn_sinpos_rma_cnn_t2":   dict(pos_enc_type="sinusoidal", attn_temperature=2.0),
        "attn_sinpos_rma_cnn_norm": dict(pos_enc_type="sinusoidal", attn_temperature=1.0,
                                         use_learnable_norm=True),
    }
    if encoder_type in _attn_kwargs:
        return cls(input_dim=input_dim, embed_dim=embed_dim, latent_dim=latent_dim,
                   **_attn_kwargs[encoder_type])
    return cls(input_dim=input_dim, embed_dim=embed_dim, latent_dim=latent_dim)


class SequenceActorBackbone(nn.Module):
    """Shared encoder + MLP trunk for sequence-based actor networks.

    Takes 3D history observations (B, S, D), runs a temporal encoder, concatenates
    the latent with the current frame, then passes through an MLP trunk.
    Returns a feature vector (B, hidden_dims[-1]) that callers attach output heads to.

    Normalization is the caller's responsibility — pass already-normalized obs.

    Args:
        encoder_type: "rma_cnn" | "attn_rma_cnn" | "tcn"
        input_dim: Per-frame observation dimension D.
        embed_dim: Per-step embedding dim for rma_cnn encoders (ignored for tcn).
        latent_dim: Encoder output dimension.
        hidden_dims: MLP trunk hidden layer sizes. No output layer included.
        activation: "elu" | "silu" | "relu"
        use_layer_norm: Pre-activation LayerNorm in trunk.
        device: Tensor device for parameter placement.
    """

    def __init__(
        self,
        encoder_type: str,
        input_dim: int,
        embed_dim: int = 32,
        latent_dim: int = 128,
        hidden_dims: list[int] | tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        use_layer_norm: bool = True,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = build_encoder(encoder_type, input_dim, embed_dim, latent_dim)
        if device is not None:
            self.encoder = self.encoder.to(device)

        if activation not in _ACTIVATION_CLASSES:
            raise ValueError(f"Unknown activation: {activation!r}. Choose from {list(_ACTIVATION_CLASSES)}")
        act_cls = _ACTIVATION_CLASSES[activation]

        # Trunk: encoder latent + current frame → MLP layers (no output layer)
        trunk_input_dim = latent_dim + input_dim
        layers: list[nn.Module] = []
        in_dim = trunk_input_dim
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim, device=device))
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_dim, device=device))
            layers.append(act_cls())
            in_dim = out_dim
        self.trunk = nn.Sequential(*layers)
        self._output_dim = in_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, S, D) normalized history observations.
        Returns:
            (B, output_dim) feature vector.
        """
        latent = self.encoder(obs)                               # (B, latent_dim)
        feat = torch.cat([latent, obs[:, -1, :]], dim=-1)        # (B, latent_dim + D)
        return self.trunk(feat)                                  # (B, output_dim)




class VelocityEstimatorHead(nn.Module):
    """2-layer MLP: encoder latent → estimated state (B, output_dim).

    Args:
        latent_dim: Input dimension (encoder output).
        hidden_dim: Hidden layer width (default 64).
        output_dim: Output state dimension (default 3 for base_lin_vel).
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


def compute_estimator_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and target state (B, output_dim)."""
    return nn.functional.mse_loss(pred, target)


class RMAEstimatorCore(nn.Module):
    """Algorithm-agnostic backbone + optional normalizer + optional velocity head.

    Owns:
        normalizer (optional): EmpiricalNormalization over frame_dim. Present when
            obs_normalization=True (PPO path). Absent (nn.Identity) when False (SAC
            path where the runner owns the normalizer).
        backbone: SequenceActorBackbone (encoder + trunk). Shared between PPO and SAC.
        vel_head (optional): VelocityEstimatorHead. None when vel_head_hidden_dim=None.

    Caches _last_latent and _last_latent_sg after each encode() call so the
    training loop can read them without re-running the encoder.

    Args:
        frame_dim: Per-frame observation dimension.
        encoder_type: "rma_cnn" | "tcn" | "attn_rma_cnn"
        embed_dim: Per-step embedding dim for rma_cnn variants.
        latent_dim: Encoder output / trunk input split dim.
        backbone_hidden_dims: Hidden dims for trunk MLP.
        activation: Activation for backbone ("elu").
        use_layer_norm: Apply LayerNorm in trunk.
        obs_normalization: If True, owns EmpiricalNormalization; if False, identity.
        vel_head_hidden_dim: Hidden dim of vel_head; None = no vel_head.
    """

    obs_normalization: Final[bool]  # TorchScript constant: dead-code-eliminates normalizer.update() when False
    _last_latent: Optional[torch.Tensor]
    _last_latent_sg: Optional[torch.Tensor]

    def __init__(
        self,
        frame_dim: int,
        *,
        encoder_type: str = "rma_cnn",
        embed_dim: int = 32,
        latent_dim: int = 128,
        backbone_hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        use_layer_norm: bool = True,
        obs_normalization: bool = True,
        vel_head_hidden_dim: int | None = 64,
        vel_head_output_dim: int = 3,
    ):
        super().__init__()
        self.frame_dim = frame_dim
        self.latent_dim = latent_dim
        self.obs_normalization = obs_normalization

        if obs_normalization:
            self.normalizer: nn.Module = EmpiricalNormalization(shape=[frame_dim])
        else:
            self.normalizer = nn.Identity()

        self.backbone = SequenceActorBackbone(
            encoder_type=encoder_type,
            input_dim=frame_dim,
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            hidden_dims=backbone_hidden_dims,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )

        self.vel_head: Optional[VelocityEstimatorHead] = (
            VelocityEstimatorHead(latent_dim, vel_head_hidden_dim, vel_head_output_dim)
            if vel_head_hidden_dim is not None
            else None
        )

        self._last_latent: Optional[torch.Tensor] = None
        self._last_latent_sg: Optional[torch.Tensor] = None

    def encode(self, actor_obs: torch.Tensor, update_norm: bool = False) -> torch.Tensor:
        """Normalize per-frame, run encoder + trunk, cache latent.

        Args:
            actor_obs: (B, S, D) raw or pre-normalized observations.
            update_norm: Update normalizer running stats (True during rollout).
        Returns:
            trunk_feat: (B, backbone.output_dim) trunk output.
        """
        if self.obs_normalization:
            B, S, D = actor_obs.shape
            flat = actor_obs.reshape(B * S, D)
            # Single gated call: forward updates running stats iff (self.training and update_norm),
            # then normalizes. Avoids the double-update that a separate .update() + forward() would
            # cause with this normalizer (its forward auto-updates when training). update gating is
            # equivalent to the old explicit-update path (both fire only during rollout/training).
            norm_obs = self.normalizer(flat, update=update_norm).view(B, S, D)
        else:
            norm_obs = actor_obs

        latent = self.backbone.encoder(norm_obs)
        self._last_latent = latent
        self._last_latent_sg = latent.detach()

        feat = torch.cat([latent, norm_obs[:, -1, :]], dim=-1)
        return self.backbone.trunk(feat)

    def get_last_latent(self) -> torch.Tensor:
        # Local var required: TorchScript can't narrow Optional[Tensor] on attribute access.
        lat = self._last_latent
        if lat is None:
            raise RuntimeError("encode() must be called before get_last_latent()")
        return lat

    def get_last_latent_detached(self) -> torch.Tensor:
        # Local var required: TorchScript can't narrow Optional[Tensor] on attribute access.
        sg = self._last_latent_sg
        if sg is None:
            raise RuntimeError("encode() must be called before get_last_latent_detached()")
        return sg

    def estimate_velocity(self, use_detached: bool = True) -> torch.Tensor:
        """Predict state (B, output_dim) from cached latent. Requires vel_head."""
        # Local var required: TorchScript can't narrow Optional[Module] on attribute access.
        head = self.vel_head
        if head is None:
            raise RuntimeError("vel_head not configured (vel_head_hidden_dim=None)")
        latent = self.get_last_latent_detached() if use_detached else self.get_last_latent()
        return head(latent)


def _safe_tanh_log_det_jacobian(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log|det J| for tanh: log(1 - tanh²(x)).

    Equivalent to log(1 - tanh²(x)) but avoids catastrophic cancellation.
    Formula: 2*(log(2) - x - softplus(-2x)).

    When x is large (action saturated): returns -∞ (correct), not log(1e-6) ≈ -13.8.
    The 1e-6 floor causes log_prob to be artificially large when actions saturate,
    inverting the alpha update direction and causing training collapse.
    """
    return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


def concat_obs_groups(
    obs: torch.Tensor, obs_indices: dict[str, dict[str, int]], obs_keys: list[str]
) -> torch.Tensor:
    """Slice and concatenate the named obs groups from a flat obs tensor along the last dim.

    Shared by Actor / Critic / SequenceActor — the obs_indices map each group to a
    [start, end) slice of the concatenated mjlab group tensor.
    """
    return torch.cat(
        [obs[..., obs_indices[k]["start"] : obs_indices[k]["end"]] for k in obs_keys],
        dim=-1,
    )




class Actor(nn.Module):
    def __init__(
        self,
        obs_indices: dict[str, dict[str, int]],
        obs_keys: list[str],
        n_act: int,
        num_envs: int,
        hidden_dim: int,
        log_std_max: float,
        log_std_min: float,
        use_tanh: bool = True,
        use_layer_norm: bool = True,
        device: torch.device | str | None = None,
        action_scale: torch.Tensor | None = None,
        action_bias: torch.Tensor | None = None,
        encoder_obs_key: str | None = None,
        encoder_obs_shape: tuple[int, int, int] | None = None,
    ):
        super().__init__()
        self.obs_indices = obs_indices
        self.obs_keys = obs_keys
        self.n_act = n_act
        self.log_std_max = log_std_max
        self.log_std_min = log_std_min
        self.use_tanh = use_tanh
        self.n_envs = num_envs
        self.device = device
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        self.encoder_obs_key = encoder_obs_key
        self.encoder_obs_shape = encoder_obs_shape

        # Setup the network - this will be overridden in subclasses if needed
        self.setup_network()

        # Register action scaling parameters as buffers
        if action_scale is not None:
            self.register_buffer("action_scale", action_scale.to(device))
        else:
            self.register_buffer("action_scale", torch.ones(n_act, device=device))

        if action_bias is not None:
            self.register_buffer("action_bias", action_bias.to(device))
        else:
            self.register_buffer("action_bias", torch.zeros(n_act, device=device))

    def setup_network(self) -> None:
        """Setup the network architecture. Can be overridden by subclasses."""
        n_obs = sum(self.obs_indices[obs_key]["size"] for obs_key in self.obs_keys)
        self._setup_network_with_input_dim(n_obs)

    def _setup_network_with_input_dim(self, input_dim: int) -> None:
        """Setup network with specific input dimension."""
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim, device=self.device),
            nn.LayerNorm(self.hidden_dim, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2, device=self.device),
            nn.LayerNorm(self.hidden_dim // 2, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4, device=self.device),
            nn.LayerNorm(self.hidden_dim // 4, device=self.device) if self.use_layer_norm else nn.Identity(),
            nn.SiLU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Linear(self.hidden_dim // 4, self.n_act, device=self.device),
        )
        self.fc_logstd = nn.Linear(self.hidden_dim // 4, self.n_act, device=self.device)
        nn.init.constant_(self.fc_mu[0].weight, 0.0)
        nn.init.constant_(self.fc_mu[0].bias, 0.0)
        nn.init.constant_(self.fc_logstd.weight, 0.0)
        nn.init.constant_(self.fc_logstd.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.process_obs(obs)
        x = self.net(x)
        mean = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        if self.use_tanh:
            tanh_mean = torch.tanh(mean)
            action = tanh_mean * self.action_scale + self.action_bias
        else:
            action = mean

        return action, mean, log_std

    def get_actions_and_log_probs(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()

        if self.use_tanh:
            tanh_action = torch.tanh(raw_action)
            action = tanh_action * self.action_scale + self.action_bias

            log_prob = dist.log_prob(raw_action)
            log_prob -= _safe_tanh_log_det_jacobian(raw_action)
            log_prob -= torch.log(self.action_scale + 1e-6)
        else:
            action = raw_action
            log_prob = dist.log_prob(raw_action)

        log_prob = log_prob.sum(1)
        return action, log_prob, log_std

    def init_zeta_noise(self, mu: float, max_n: int) -> None:
        """Initialize zeta-distributed noise repetition state buffers.

        Per env, sample N ~ truncated Zeta(mu) ∈ [1, max_n]; reuse same noise ε
        for N consecutive steps. Eliminates IID per-step jitter in the replay
        buffer, preventing the SAC critic from learning to value chatter.

        Buffer persistence design:
          _zeta_cdf   — persistent=True  (env-agnostic CDF lookup table, valid to checkpoint)
          _zeta_noise — persistent=False (per-env noise, shape (n_envs, n_act); transient)
          _zeta_count — persistent=False (per-env step counter; transient)
          _zeta_n     — persistent=False (per-env sampled repeat length; transient)

        The three non-persistent buffers are deliberately excluded from state_dict().
        Saving them would embed the training n_envs into the checkpoint, causing shape
        mismatches when loading at a different n_envs (e.g. play uses n_envs=4, training
        uses n_envs=4096). Their state is also irrelevant to resume correctness: worst
        case is max_n steps of re-randomization per env before the zeta distribution
        stabilises (mean repeat ~1.85 steps at mu=2.0).

        Reference: `_sample_flashsac_actions` in FlashSAC repo
        (`flash_rl/agents/flashSAC/agent.py:223-257`).
        """
        device = self.action_scale.device
        ns = torch.arange(1, max_n + 1, dtype=torch.float32, device=device)
        pmf = ns ** (-mu)
        pmf = pmf / pmf.sum()
        cdf = torch.cumsum(pmf, dim=0)
        self.register_buffer("_zeta_cdf", cdf)  # persistent (env-agnostic, valid to save)
        self.register_buffer(                   # persistent=False: shape (n_envs, n_act),
            "_zeta_noise",                      # must not enter state_dict (see docstring)
            torch.randn(self.n_envs, self.n_act, device=device),
            persistent=False,
        )
        self.register_buffer(                   # persistent=False: shape (n_envs,)
            "_zeta_count",
            torch.zeros(self.n_envs, dtype=torch.int32, device=device),
            persistent=False,
        )
        self.register_buffer(                   # persistent=False: shape (n_envs,)
            "_zeta_n",
            torch.ones(self.n_envs, dtype=torch.int32, device=device),
            persistent=False,
        )

    def _zeta_sample_n(self, batch_size: int) -> torch.Tensor:
        """Vectorized sample N ~ truncated Zeta from cached CDF, returns (B,) int32."""
        u = torch.rand(batch_size, device=self._zeta_cdf.device).unsqueeze(-1)
        idx = torch.argmax((u < self._zeta_cdf).to(torch.int32), dim=-1)
        return (idx + 1).to(torch.int32)

    @torch.no_grad()
    def explore(
        self, obs: torch.Tensor, dones: torch.Tensor | None = None, deterministic: bool = False
    ) -> torch.Tensor:
        _, mean, log_std = self(obs)
        if deterministic:
            if self.use_tanh:
                tanh_mean = torch.tanh(mean)
                return tanh_mean * self.action_scale + self.action_bias
            return mean

        std = log_std.exp()

        # Zeta-correlated noise: reuse same epsilon for N consecutive steps per env.
        if hasattr(self, "_zeta_cdf"):
            B = mean.shape[0]
            # Force reinit for done envs (start of new episode → fresh noise).
            # Branchless: torch.where handles the empty-mask case cheaply; avoids
            # data-dependent Python conditionals that break torch.compile graph traces.
            if dones is not None:
                done_mask = dones.bool()
                self._zeta_count = torch.where(done_mask, self._zeta_n, self._zeta_count)

            reinit = (self._zeta_count == 0) | (self._zeta_count >= self._zeta_n)
            # Always sample new noise and new N; torch.where selects per-env.
            # Note: randn always consumes RNG (even when reinit is all-False), which
            # changes the RNG stream vs. the guarded version — acceptable for training.
            new_noise = torch.randn(B, self.n_act, device=mean.device)
            new_n = self._zeta_sample_n(B)
            reinit_2d = reinit.unsqueeze(-1)
            self._zeta_noise = torch.where(reinit_2d, new_noise, self._zeta_noise)
            self._zeta_n = torch.where(reinit, new_n, self._zeta_n)
            self._zeta_count = torch.where(reinit, torch.zeros_like(self._zeta_count), self._zeta_count)
            raw_action = mean + std * self._zeta_noise
            self._zeta_count = self._zeta_count + 1
        else:
            dist = torch.distributions.Normal(mean, std)
            raw_action = dist.rsample()

        if self.use_tanh:
            tanh_action = torch.tanh(raw_action)
            action = tanh_action * self.action_scale + self.action_bias
        else:
            action = raw_action

        return action

    def process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return concat_obs_groups(obs, self.obs_indices, self.obs_keys)


class SequenceActor(Actor):
    """Actor using SequenceActorBackbone (RMA-CNN/TCN) for history observations.

    Expects flat (B, S*D) observations as stored in the replay buffer. Internally
    reshapes to (B, S, D) before passing to the backbone. Uses ELU activation to
    match the PPO backbone (diverges from reference FlashSAC SiLU).

    Normalization is applied externally by the runner (per-frame EmpiricalNormalization
    over D dimensions, then flatten back to S*D before passing here).

    When use_velocity_estimator=True, the actor wraps the backbone in RMAEstimatorCore
    and owns a VelocityEstimatorHead. The encoder latent is cached (stop-gradient) so
    the runner can compute estimator MSE during the actor update step. This shares the
    same core as ActorCriticRMAEstimator on the PPO side.

    Args:
        obs_seq_shape: (S, D) — sequence length and per-frame observation dim.
        encoder_type: "rma_cnn" | "tcn" | "attn_rma_cnn"
        encoder_embed_dim: Per-step embedding dim for rma_cnn variants.
        encoder_latent_dim: Encoder output latent dim.
        use_velocity_estimator: Add VelocityEstimatorHead and cache latent.
        vel_head_hidden_dim: Hidden dim for velocity head MLP (default 64).
    """

    _use_velocity_estimator: Final[bool]  # TorchScript constant: dead-code-eliminates unused backbone/core branch

    def __init__(
        self,
        *args,
        obs_seq_shape: tuple[int, int],
        encoder_type: str = "rma_cnn",
        encoder_embed_dim: int = 32,
        encoder_latent_dim: int = 128,
        use_velocity_estimator: bool = False,
        vel_head_hidden_dim: int = 64,
        vel_head_output_dim: int = 3,
        **kwargs,
    ):
        self._obs_seq_shape = obs_seq_shape
        self._encoder_type = encoder_type
        self._encoder_embed_dim = encoder_embed_dim
        self._encoder_latent_dim = encoder_latent_dim
        self._use_velocity_estimator = use_velocity_estimator
        self._vel_head_hidden_dim = vel_head_hidden_dim
        self._vel_head_output_dim = vel_head_output_dim
        super().__init__(*args, **kwargs)

    def setup_network(self) -> None:
        S, D = self._obs_seq_shape

        if self._use_velocity_estimator:
            # obs_normalization=False: runner owns normalizer; core gets pre-normalized input.
            self.core = RMAEstimatorCore(
                frame_dim=D,
                encoder_type=self._encoder_type,
                embed_dim=self._encoder_embed_dim,
                latent_dim=self._encoder_latent_dim,
                backbone_hidden_dims=(self.hidden_dim, self.hidden_dim // 2, self.hidden_dim // 4),
                activation="elu",
                use_layer_norm=self.use_layer_norm,
                obs_normalization=False,
                vel_head_hidden_dim=self._vel_head_hidden_dim,
                vel_head_output_dim=self._vel_head_output_dim,
            ).to(self.device)
            feat_dim = self.core.backbone.output_dim
        else:
            self.backbone = SequenceActorBackbone(
                encoder_type=self._encoder_type,
                input_dim=D,
                embed_dim=self._encoder_embed_dim,
                latent_dim=self._encoder_latent_dim,
                hidden_dims=[self.hidden_dim, self.hidden_dim // 2, self.hidden_dim // 4],
                activation="elu",
                use_layer_norm=self.use_layer_norm,
                device=self.device,
            )
            feat_dim = self.backbone.output_dim

        self.fc_mu = nn.Linear(feat_dim, self.n_act, device=self.device)
        self.fc_logstd = nn.Linear(feat_dim, self.n_act, device=self.device)
        nn.init.zeros_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.zeros_(self.fc_logstd.weight)
        nn.init.zeros_(self.fc_logstd.bias)

    def _to_seq(self, obs: torch.Tensor) -> torch.Tensor:
        """Slice+concat the obs groups and reshape flat (B, S*D) → (B, S, D) for the backbone."""
        S, D = self._obs_seq_shape
        return concat_obs_groups(obs, self.obs_indices, self.obs_keys).view(obs.shape[0], S, D)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self._to_seq(obs)
        if self._use_velocity_estimator:
            # core.encode caches _last_latent_sg (stop-gradient) for estimator MSE.
            x = self.core.encode(seq, update_norm=False)
        else:
            x = self.backbone(seq)

        log_std = torch.tanh(self.fc_logstd(x))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        mean = self.fc_mu(x)

        if self.use_tanh:
            action = torch.tanh(mean) * self.action_scale + self.action_bias
        else:
            action = mean

        return action, mean, log_std

    def get_estimated_velocity(self, stop_gradient: bool = True) -> torch.Tensor:
        """Predict estimator head output (B, output_dim) from cached latent.

        Only valid when use_velocity_estimator=True. Raises otherwise.
        stop_gradient=True (default): no gradient flows into encoder.
        stop_gradient=False: estimator MSE gradient flows into encoder.
        """
        if not self._use_velocity_estimator:
            raise RuntimeError("get_estimated_velocity() requires use_velocity_estimator=True")
        return self.core.estimate_velocity(use_detached=stop_gradient)




class DistributionalQNetwork(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        use_layer_norm: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs + n_act, hidden_dim, device=device),
            nn.LayerNorm(hidden_dim, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2, device=device),
            nn.LayerNorm(hidden_dim // 2, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
            nn.LayerNorm(hidden_dim // 4, device=device) if use_layer_norm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_atoms, device=device),
        )
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], 1)
        x = self.net(x)
        return x  # noqa: RET504

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
        q_support: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
        target_z = target_z.clamp(self.v_min, self.v_max)
        # bin_pos = continuous atom coordinate of each projected target value (C51).
        bin_pos = (target_z - self.v_min) / delta_z
        lower = torch.floor(bin_pos).long()
        upper = torch.ceil(bin_pos).long()

        is_integer = upper == lower
        lower_mask = torch.logical_and((lower > 0), is_integer)
        upper_mask = torch.logical_and((lower == 0), is_integer)

        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)

        next_dist = F.softmax(self(obs, actions), dim=1)
        proj_dist = torch.zeros_like(next_dist)
        offset = (
            torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=device)
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )

        # Additional safety check for indices
        lower_indices = (lower + offset).view(-1)
        upper_indices = (upper + offset).view(-1)
        max_index = proj_dist.numel() - 1

        lower_indices = torch.clamp(lower_indices, 0, max_index)
        upper_indices = torch.clamp(upper_indices, 0, max_index)

        proj_dist.view(-1).index_add_(0, lower_indices, (next_dist * (upper.float() - bin_pos)).view(-1))
        proj_dist.view(-1).index_add_(0, upper_indices, (next_dist * (bin_pos - lower.float())).view(-1))
        return proj_dist


class Critic(nn.Module):
    def __init__(
        self,
        obs_indices: dict[str, dict[str, int]],
        obs_keys: list[str],
        n_act: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        hidden_dim: int,
        use_layer_norm: bool = True,
        num_q_networks: int = 2,
        encoder_obs_key: str | None = None,
        encoder_obs_shape: tuple[int, int, int] | None = None,
        seq_shape: tuple[int, int] | None = None,
        encoder_type: str = "rma_cnn",
        encoder_embed_dim: int = 32,
        encoder_latent_dim: int = 128,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.obs_indices = obs_indices
        self.obs_keys = obs_keys
        self.n_act = n_act
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        if num_q_networks < 1:
            raise ValueError("num_q_networks must be at least 1")
        self.num_q_networks = num_q_networks
        self.encoder_obs_key = encoder_obs_key
        self.encoder_obs_shape = encoder_obs_shape
        # Temporal-critic (SequenceCritic) params: the (L_c, D_c) split of the flat critic window
        # and the encoder hyperparams (reused from the actor's sequence encoder). None on the plain
        # Markov critic.
        self.seq_shape = seq_shape
        self.encoder_type = encoder_type
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_latent_dim = encoder_latent_dim
        self.device = device

        # Setup Q-networks - this will be overridden in subclasses if needed
        self.setup_qnetworks()

        self.register_buffer("q_support", torch.linspace(v_min, v_max, num_atoms, device=device))

    def setup_qnetworks(self) -> None:
        """Setup Q-networks. Can be overridden by subclasses."""
        n_obs = sum(self.obs_indices[obs_key]["size"] for obs_key in self.obs_keys)
        self._setup_qnetworks_with_obs_dim(n_obs)

    def _setup_qnetworks_with_obs_dim(self, n_obs: int) -> None:
        """Setup Q-networks with specific observation dimension."""
        self.qnets = nn.ModuleList(
            [
                DistributionalQNetwork(
                    n_obs=n_obs,
                    n_act=self.n_act,
                    num_atoms=self.num_atoms,
                    v_min=self.v_min,
                    v_max=self.v_max,
                    hidden_dim=self.hidden_dim,
                    use_layer_norm=self.use_layer_norm,
                    device=self.device,
                )
                for _ in range(self.num_q_networks)
            ]
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = self.process_obs(obs)
        outputs = [qnet(x, actions) for qnet in self.qnets]
        return torch.stack(outputs, dim=0)

    def projection(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor,
    ) -> torch.Tensor:
        """Projection operation that includes q_support directly"""
        x = self.process_obs(obs)
        projections = [
            qnet.projection(
                x,
                actions,
                rewards,
                bootstrap,
                discount,
                self.q_support,
                self.q_support.device,
            )
            for qnet in self.qnets
        ]
        return torch.stack(projections, dim=0)

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate value from logits using support"""
        return torch.sum(probs * self.q_support, dim=-1)

    def process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return concat_obs_groups(obs, self.obs_indices, self.obs_keys)




class SequenceCritic(Critic):
    """Distributional critic with a TEMPORAL RMA-CNN encoder over the privileged critic history
    window, the value-net analog of SequenceActor. The critic obs group is a flat (B, L_c*D_c)
    window (reconstructed by the frame-ring buffer); process_obs reshapes it to (B, L_c, D_c),
    runs the shared 1-D-conv-over-time encoder → (B, latent_dim), and the distributional Q-nets
    take that latent (+ action).

    Why (vs the plain Critic on a windowed obs). Under per-episode domain randomization the value
    depends on hidden physical latents (mass, friction) the 1-frame critic cannot observe — a POMDP
    from the critic's input. A temporal encoder lets the critic implicitly system-ID those latents
    from the state response. Flat-concatenating the
    window into one MLP (no temporal inductive bias) regressed; this class supplies the missing
    encoder.

    seq_shape=(L_c, D_c) is required. The encoder (encoder_type/embed/latent) mirrors the actor.

    Q-input = concat(temporal latent, newest raw frame), exactly as SequenceActorBackbone
    (torch.cat([latent, obs[:, -1, :]])). Passing the latent ALONE (an earlier bug) throws away
    the raw 1-frame privileged obs (GT lin/ang vel + contacts) that the plain 1-frame critic feeds
    the Q-net directly — the encoder's mean-pool is a lossy bottleneck (latent 128 < frame 151), so
    latent-only made the critic STRICTLY WEAKER than the 1-frame baseline and the actor collapsed to
    a near-standing policy (2026-07-09 single-seed regress). Concat of
    the newest frame makes the temporal critic strictly dominate the baseline (it can zero the latent
    to recover the 1-frame critic), so history can only add value.
    """

    def setup_qnetworks(self) -> None:
        if self.seq_shape is None:
            raise ValueError("seq_shape=(L_c, D_c) must be provided for SequenceCritic")
        _, d_c = self.seq_shape
        self.encoder = build_encoder(
            self.encoder_type, input_dim=d_c,
            embed_dim=self.encoder_embed_dim, latent_dim=self.encoder_latent_dim,
        ).to(self.device)
        # Q-nets take (temporal latent ++ newest raw frame) (+ action) — mirrors SequenceActorBackbone.
        self._setup_qnetworks_with_obs_dim(self.encoder_latent_dim + d_c)

    def process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: flat (B, L_c*D_c) window → (B, L_c, D_c); encoder → (B, latent); concat newest frame.
        window = concat_obs_groups(obs, self.obs_indices, self.obs_keys)
        l_c, d_c = self.seq_shape
        window = window.view(window.shape[0], l_c, d_c)
        latent = self.encoder(window)
        return torch.cat([latent, window[:, -1, :]], dim=-1)     # (B, latent_dim + D_c)


