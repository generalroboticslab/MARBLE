"""Running-statistics normalizers: empirical obs normalization + reward normalization.

Merged from reward_normalization.py (RunningMeanStd, RewardNormalizer) + the EmpiricalNormalization
superset relocated from flash_sac_utils.py. Pure relocation — no logic changes.
"""
from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


@torch.compile
def _update_reward_stats(
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    G_r: torch.Tensor,
    G_r_max: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    done = torch.logical_or(terminated, truncated).float()
    new_G_r = gamma * (1.0 - done) * G_r + reward
    new_G_r_max = torch.maximum(G_r_max, torch.max(torch.abs(new_G_r)))
    return new_G_r, new_G_r_max


@torch.compile
def _scale_reward(
    rewards: torch.Tensor,
    G_var: torch.Tensor,
    G_r_max: torch.Tensor,
    G_max: float,
    eps: float,
) -> torch.Tensor:
    var_denominator = torch.sqrt(G_var + eps)
    min_required_denominator = G_r_max / G_max
    denominator = torch.maximum(var_denominator, min_required_denominator)
    return rewards / denominator


@torch.compile
def _update_mean_var_count(
    samples: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    running_count: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_mean = torch.mean(samples, dim=0)
    sample_var = torch.var(samples, dim=0, unbiased=False)
    sample_count = float(samples.shape[0])

    delta = sample_mean - running_mean
    total_count = running_count + sample_count
    ratio = sample_count / total_count

    new_mean = running_mean + delta * ratio
    m_a = running_var * (running_count + epsilon)
    m_b = sample_var * sample_count
    M2 = m_a + m_b + torch.square(delta) * running_count * ratio
    new_var = M2 / total_count

    return new_mean, new_var, total_count


class RunningMeanStd:
    """Welford-style running mean/var/count, single-scalar default."""

    def __init__(
        self,
        device: torch.device | str,
        epsilon: float = 1e-4,
        shape: tuple[int, ...] = (),
        dtype: torch.dtype = torch.float32,
    ):
        self.mean = torch.zeros(shape, dtype=dtype, device=device)
        self.var = torch.ones(shape, dtype=dtype, device=device)
        self.count = torch.tensor(0.0, dtype=dtype, device=device)
        self.epsilon = epsilon

    def update(self, x: torch.Tensor) -> None:
        self.mean, self.var, self.count = _update_mean_var_count(
            samples=x,
            running_mean=self.mean,
            running_var=self.var,
            running_count=self.count,
            epsilon=self.epsilon,
        )


class RewardNormalizer:
    """Normalize rewards by running variance of discounted returns G_t.

    Per-step: maintains G_t = γ G_{t-1} + r_t per env, resetting on terminate/truncate.
    Per-batch: divides rewards by max(sqrt(Var[G]), max|G| / G_max).

    The G_max ceiling prevents pathological cases where Var[G] is small but a
    rare large return appeared, which would over-scale rewards.
    """

    def __init__(
        self,
        gamma: float,
        G_max: float,
        device: torch.device | str,
        epsilon: float = 1e-8,
        num_envs: int = 1,
    ):
        self.gamma = gamma
        self.G_r = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.G_r_max = torch.zeros(1, dtype=torch.float32, device=device)
        self.G_rms = RunningMeanStd(shape=(1,), device=device, dtype=torch.float32)
        self.G_max = G_max
        self.epsilon = epsilon
        self.device = device

    def update_reward_stats(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        self.G_r, self.G_r_max = _update_reward_stats(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            G_r=self.G_r,
            G_r_max=self.G_r_max,
            gamma=self.gamma,
        )
        self.G_rms.update(self.G_r.unsqueeze(-1))

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        return _scale_reward(
            rewards=rewards,
            G_var=self.G_rms.var,
            G_r_max=self.G_r_max,
            G_max=self.G_max,
            eps=self.epsilon,
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "G_r": self.G_r.detach().clone(),
            "G_r_max": self.G_r_max.detach().clone(),
            "G_rms_mean": self.G_rms.mean.detach().clone(),
            "G_rms_var": self.G_rms.var.detach().clone(),
            "G_rms_count": self.G_rms.count.detach().clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.G_r = state["G_r"].to(self.device)
        self.G_r_max = state["G_r_max"].to(self.device)
        self.G_rms.mean = state["G_rms_mean"].to(self.device)
        self.G_rms.var = state["G_rms_var"].to(self.device)
        self.G_rms.count = state["G_rms_count"].to(self.device)


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, device=None, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            device: Buffer device. None → leave on default device (`.to(None)` is a no-op).
                    Optional so RMAEstimatorCore can construct it without a device (PPO RMA
                    path); the runner still passes device explicitly.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.device = device
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, center: bool = True, update: bool = True, inplace: bool = False) -> torch.Tensor:
        """Normalize x to zero mean / unit variance using running statistics.

        Args:
            x:       Input tensor, shape (B, *shape).
            center:  If False, skip mean subtraction (divide by std only).
            update:  If True and self.training, update running mean/std with this batch.
            inplace: If True, normalize x in-place and return it — avoids allocating a
                     new output tensor of the same size as x.  Safe when the caller owns
                     x exclusively (e.g. freshly sampled replay-buffer data).  The
                     returned tensor IS x (same storage); callers that rely on the
                     original values must clone first.
                     Memory impact: eliminates one (B, *shape) allocation per call.
                     Profiled at B=131072, shape=(90,): saves ~900 MB with zero speed cost.
        """
        if x.shape[1:] != self._mean.shape[1:]:
            raise ValueError(f"Expected input of shape (*,{self._mean.shape[1:]}), got {x.shape}")

        if self.training and update:
            self.update(x)
        if center:
            if inplace:
                # x.sub_ / div_ reuse x's existing GPU memory; no new tensor allocated.
                # std + eps creates a tiny (1, *shape) buffer — negligible.
                return x.sub_(self._mean).div_(self._std + self.eps)
            return (x - self._mean) / (self._std + self.eps)
        if inplace:
            return x.div_(self._std + self.eps)
        return x / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        if self.until is not None and self.count >= self.until:
            return

        if dist.is_available() and dist.is_initialized():
            # Calculate global batch size arithmetically
            local_batch_size = x.shape[0]
            world_size = dist.get_world_size()
            global_batch_size = world_size * local_batch_size

            # Calculate the stats
            x_shifted = x - self._mean
            local_sum_shifted = torch.sum(x_shifted, dim=0, keepdim=True)
            local_sum_sq_shifted = torch.sum(x_shifted.pow(2), dim=0, keepdim=True)

            # Sync the stats across all processes
            stats_to_sync = torch.cat([local_sum_shifted, local_sum_sq_shifted], dim=0)
            dist.all_reduce(stats_to_sync, op=dist.ReduceOp.SUM)
            global_sum_shifted, global_sum_sq_shifted = stats_to_sync

            # Calculate the mean and variance of the global batch
            batch_mean_shifted = global_sum_shifted / global_batch_size
            batch_var = global_sum_sq_shifted / global_batch_size - batch_mean_shifted.pow(2)
            batch_mean = batch_mean_shifted + self._mean

        else:
            global_batch_size = x.shape[0]
            batch_mean = torch.mean(x, dim=0, keepdim=True)
            batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)

        new_count = self.count + global_batch_size

        # Update mean
        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (global_batch_size / new_count))

        # Update variance
        delta2 = batch_mean - self._mean
        m_a = self._var * self.count
        m_b = batch_var * global_batch_size
        M2 = m_a + m_b + delta2.pow(2) * (self.count * global_batch_size / new_count)
        self._var.copy_(M2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean
