"""Geometric heuristic ballbot policy (sim port of HeuristicPolicy in
ballbot_control/src/ballbot_runtime.py).

Given the world XY velocity command, place the internal 3-slider mass offset along
that direction — expressed in the rolling base frame — so the sphere leans and rolls
toward the command ("hamster counter-rotation"). It is a pure `policy(obs) -> action`
callable consumed by run.py's `--agent experiment` seam; the env keeps its normal
`joint_pos` action term, this just supplies the target.

Fully batched: one einsum + one matmul + one `where`, no per-env python loop and no
host<->device sync, so it scales N=1 -> 4096 unchanged.

Slider axes are derived FROM the model (asset/ball_linear/Sim_Model.xml), NOT from
reward.py's `_SLIDER_TO_OFFSET` — that constant is stale (it matches the retired
UnderwaterRobotSim model, not ball_linear) and would aim the lean the wrong way. See _M and
slider_axes_base() below. The hardware `ballbot_runtime.py::SLIDER_AXES_BASE` now holds the
SAME axes ([[0,0,-1],[0,1,0],[-1,0,0]]); we still re-derive _M from the model here (not copy
the hardware constant) so a model change can't silently drift the two apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.utils.lab_api.math import matrix_from_quat

# Actor obs frame (B, 23) layout, newest history frame:
#   [0:3] angular_velocity, [3:5] commands_xy (WORLD), [5:8] dofPos, [8:11] dofVel,
#   [11:14] actions, [14:23] base_rotation_matrix (R_world_base, row-major; v_w = R @ v_b).
_CMD_SLICE = slice(3, 5)
_ROT_SLICE = slice(14, 23)

# Slider map: s_i = axis_i_base . offset_base, i.e. s = M @ offset_base, where the rows of M
# are each slider's axis in the BASE frame. This is the openloop/hardware projection: place a
# world lean, express it in base (o_b = R^T o_w), closest-point project onto each slider line
# (anchor from the model, see _ANCHORS_SHELL; obs-fallback path assumes anchor=0).
# The axes are derived FROM the ball_linear model, not reward.py: at rest (base_link identity),
#   axis_i_base = R_base^T @ (R_body_i @ jnt_axis_i_local)
# computed via mujoco on asset/ball_linear/Sim_Model.xml gives, for [s5, s6, s7]:
#   s5=[0,0,-1]  s6=[0,1,0]  s7=[-1,0,0]
# (each slider joint sits on a separate 90deg-rotated body base-2/base-1/base, so the raw local
# jnt_axis ~[0,1,0] is misleading — the body rotation is what makes them orthogonal). NOTE:
# reward.py's _SLIDER_TO_OFFSET disagrees with this: it's stale (matches the retired
# UnderwaterRobotSim model, not ball_linear). Its only consumer lean_progress_ballbot is dead
# code (not in any active reward set), so it never affected training. slider_axes_base()
# recomputes M from any env's model and __init__ asserts against this constant, so a future
# model change can't go silent.
_M = torch.tensor([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])

# Each slider LINE in the base frame is anchor_i + q*axis_i, not through the origin. Hardware
# (ballbot_runtime.py::SLIDER_ANCHORS_BASE_M) does the true closest-point projection
# q_i = axis_i . (o_b - anchor_i); the sim matches instead of assuming anchor=0. Rows [s5,s6,s7],
# base frame. The live model's values (slider_anchors_base) are what actually drive the
# projection; the constants below are only the drift guard, and the guard accepts either variant
# because the two models no longer agree:
#   ball_linear_shell — each slider line sits on a single base axis at 28.2 mm, along the
#     SLIDER_ANCHORS_BASE_M directions with the off-axis components zeroed. 28.2 mm = the 26.2 mm
#     rail offset of the current CAD (GRL/Sim/CURRENTLinearball/Sim_Model.xml) plus a 2 mm
#     outward shift. Being exactly perpendicular to its own axis, the along-axis term
#     c_i = axis_i . anchor_i is 0 and q reduces to the plain dot product. NOTE the magnitude no
#     longer matches the hardware SLIDER_ANCHORS_BASE_M (30 mm) — only its directions; bump the
#     hardware constant to 28.2 mm to keep sim and real bit-consistent.
#   ball_linear — older CAD-derived anchors, ~3.3 mm off the above; ~perpendicular, so only the
#     residual along-axis part (<0.6 mm) enters q.
_ANCHORS_SHELL = torch.tensor([[0.0, -0.0282, 0.0], [-0.0282, 0.0, 0.0], [0.0, 0.0, 0.0282]])
_ANCHORS_LINEAR = torch.tensor(
    [[-0.00024, -0.03065, 0.00028], [-0.03144, 0.00058, 0.00025], [-0.00027, 0.00055, 0.03145]]
)

# Ballbot constants (asset_zoo/ballbot/ballbot_constants.py).
ACTION_SCALE = 0.106  # m; joint_pos scale, target_m = raw * ACTION_SCALE
# Heuristic OUTPUT clamp — mirrors hardware HEURISTIC_LOWER/UPPER_M in ballbot_runtime.py. Full
# joint travel is [-0.114, 0.106] (env ctrlrange), but the heuristic keeps a 10 mm/side safety
# buffer off the hard stop, so its own target saturates at [-0.104, 0.096]. The env's joint_pos
# term still applies the full-range clamp downstream — same two-stage structure as hardware.
JOINT_LOWER = -0.104  # m; full lower -0.114 + 10 mm buffer (hardware HEURISTIC_LOWER_M)
JOINT_UPPER = 0.096  # m; full upper 0.106 - 10 mm buffer (hardware HEURISTIC_UPPER_M)
# Command magnitude (m/s) mapping to unit control magnitude: norm/MAX_CMD == hardware's
# controller_magnitude in [0,1]. Matches MAX_LINEAR_VELOCITY_CMD in ballbot_velocity_env_cfg.
MAX_CMD = 0.5  # m/s; ||cmd|| == MAX_CMD -> magnitude 1.0 (full 1.5 m lean point, travel-clamped)
STAND_EPS = 0.05  # m/s; below this ||cmd|| the robot holds still (offset = 0)
# Slider-sign reconciliation. +1 = lean the mass along +command (mass toward command → rolls
# toward it), which is correct for the sim model. Exists only so a hardware port can flip it to
# -1 if the physical slider's positive direction is inverted relative to the model; the sim
# itself needs no flip.
ROLL_SIGN = 1.0


def heuristic_slider_action(
    cmd_w: torch.Tensor,
    R_world_base: torch.Tensor,
    *,
    max_cmd: float = MAX_CMD,
    lean_gain: float = 0.0,
    action_scale: float = ACTION_SCALE,
    joint_lower: float = JOINT_LOWER,
    joint_upper: float = JOINT_UPPER,
    stand_eps: float = STAND_EPS,
    roll_sign: float = ROLL_SIGN,
    _M_dev: torch.Tensor | None = None,
    anchor_offset: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batched geometric heuristic. Single source of truth for the control law.

    Args:
        cmd_w: (B, 2) world-frame XY velocity command.
        R_world_base: (B, 3, 3) base rotation, v_world = R @ v_body.
        max_cmd: command magnitude that maps to full lean (when lean_gain == 0).
        lean_gain: if > 0, use proportional lean L = clamp(lean_gain*||cmd||, 0, upper)
            instead of the saturating default L = 1.5*clamp(||cmd||/max_cmd, 0, 1).
        action_scale/joint_lower/joint_upper: joint_pos scale and travel limits (m).
        stand_eps: below this ||cmd|| the offset is zeroed (hold still).
        roll_sign: slider-sign reconciliation; +1 for the sim model, flip to -1 only for a
            hardware slider whose positive direction is inverted vs the model (see ROLL_SIGN).
        _M_dev: optional precomputed device copy of the inverse-slider matrix.
        anchor_offset: optional (3,) per-slider along-axis anchor constant c_i = axis_i . anchor_i
            (metres) subtracted from the projection so it matches the hardware closest-point
            q_i = axis_i . (o_b - anchor_i). None (default / obs-fallback) assumes anchor=0.

    Returns:
        (B, 3) raw action in [-1, 1] for sliders [s5, s6, s7].
    """
    norm = torch.linalg.vector_norm(cmd_w, dim=-1, keepdim=True)  # (B,1)
    dir_w = cmd_w / (norm + 1e-8)  # (B,2)

    if lean_gain > 0.0:
        lean = torch.clamp(lean_gain * norm, min=0.0, max=joint_upper)  # (B,1)
    else:
        lean = 1.5 * torch.clamp(norm / max_cmd, min=0.0, max=1.0)  # (B,1)

    # World-frame mass offset: lean along (roll_sign * command) direction, no vertical part.
    # roll_sign = +1 in sim (mass toward command); flip only for inverted hardware sliders.
    o_w = torch.cat([roll_sign * lean * dir_w, torch.zeros_like(norm)], dim=-1)  # (B,3)

    # Rotate world offset into the base frame: o_b = R^T @ o_w.
    o_b = torch.einsum("nji,nj->ni", R_world_base, o_w)  # (B,3)

    # Base offset -> slider metres via closest-point projection s_i = axis_i . (o_b - anchor_i)
    # = (o_b @ M.T) - c, where c_i = axis_i . anchor_i (anchor_offset). Clamp to travel.
    M = _M_dev if _M_dev is not None else _M.to(o_b.device, o_b.dtype)
    s = o_b @ M.t()  # (B,3)
    if anchor_offset is not None:
        s = s - anchor_offset
    s = torch.clamp(s, min=joint_lower, max=joint_upper)

    # Hold still under standing commands.
    s = torch.where(norm < stand_eps, torch.zeros_like(s), s)

    return torch.clamp(s / action_scale, min=-1.0, max=1.0)


_SLIDER_JOINTS = ("base_link_Slider-5", "base_link_Slider-6", "base_link_Slider-7")


def slider_axes_base(mj_model) -> torch.Tensor:
    """Return (3,3) rows = each slider's axis in the base_link frame, from the compiled model.

    axis_i_base = R_base^T @ (R_body_i @ jnt_axis_i_local) at rest. This is the authoritative
    slider-projection matrix M (see _M); recomputing it from the live model guards against a
    model change silently invalidating the hardcoded constant.
    """
    import mujoco

    d = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, d)

    def _bid(name):
        return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)

    base = _bid("robot/base_link")
    if base < 0:
        base = _bid("base_link")
    R_base = d.xmat[base].reshape(3, 3)

    rows = []
    for jn in _SLIDER_JOINTS:
        j = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if j < 0:
            j = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + jn)
        R_body = d.xmat[mj_model.jnt_bodyid[j]].reshape(3, 3)
        rows.append(R_base.T @ (R_body @ mj_model.jnt_axis[j]))
    return torch.tensor(rows, dtype=torch.float32)


def slider_anchors_base(mj_model) -> torch.Tensor:
    """Return (3,3) rows = each slider LINE's anchor point in the base_link frame, from the model.

    anchor_i_base = R_base^T @ (xanchor_i - base_xpos) at rest, where xanchor is the joint anchor
    MuJoCo computes. Used for the hardware-matching closest-point projection (see _ANCHORS_SHELL
    / _ANCHORS_LINEAR); re-deriving from the live model guards against a model change
    invalidating the constants.
    """
    import mujoco

    d = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, d)

    def _bid(name):
        return mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)

    base = _bid("robot/base_link")
    if base < 0:
        base = _bid("base_link")
    R_base = d.xmat[base].reshape(3, 3)
    p_base = d.xpos[base]

    rows = []
    for jn in _SLIDER_JOINTS:
        j = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if j < 0:
            j = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "robot/" + jn)
        rows.append(R_base.T @ (d.xanchor[j] - p_base))
    return torch.tensor(rows, dtype=torch.float32)


@dataclass(frozen=True)
class DeployRealismCfg:
    """Sim-only IMU + actuator corruption so "controllable in sim" LOWER-BOUNDS the real robot.

    The heuristic reads ground-truth base rotation (clean) and commands an ideal position slider.
    Real deployment feeds it a noisy, drifting, delayed IMU and a rate-limited motor — all of which
    make control HARDER. Enabling these makes the sim no easier than the hardware, so a controller
    that works in sim transfers. Modeled after the training IMU model
    (additive obs noise + slow additive bias + delay lag):

    - imu_ori_noise_std: per-entry Gaussian on R (the base_rotation_matrix Unoise analog).
    - imu_yaw_drift_std / imu_yaw_bias_max: yaw-bias random walk (rad/step) capped at bias_max —
      the heading drift a mag-less IMU shows on a free-spinning hull; top sim2real breaker on water.
    - imu_latency_steps: whole control-step delay applied to R (IMU + comm latency).
    - slider_vel_limit: slider slew cap (m/s); motor can't jump to the lean target in one step. The
      deployed launch's --vel-limit 20 rad/s = 0.318 m/s (MM_PER_TURN=100 mm/rev).

    All default 0 = clean (no-op); flat-ground/air runs are unchanged unless a cfg is passed.
    """

    imu_ori_noise_std: float = 0.0
    imu_yaw_drift_std: float = 0.0
    imu_yaw_bias_max: float = 0.35
    imu_latency_steps: int = 0
    slider_vel_limit: float = 0.0

    @property
    def imu_active(self) -> bool:
        return self.imu_ori_noise_std > 0 or self.imu_yaw_drift_std > 0 or self.imu_latency_steps > 0


class BallbotHeuristicPolicy:
    """Callable policy(obs) -> action wrapping heuristic_slider_action.

    Reads GROUND-TRUTH base rotation + world command from the env each call. The geometric
    counter-rotation needs a clean base orientation: the actor-obs base_rotation_matrix
    carries +-0.05 per-entry noise plus a history/delay buffer, and a hand-written controller
    (unlike the trained net) cannot filter that — feeding it the noisy obs washes the
    world-fixed lean out to ~0 net motion. Reading ground truth is also deployment-faithful:
    the hardware runtime feeds HeuristicPolicy the clean IMU rotation, not the training obs.

    If constructed without an env (env=None), falls back to slicing the actor obs (used by the
    module smoke test); that path is noisy and not meant for driving the robot.
    """

    _CMD_TERM = "twist"

    def __init__(self, env=None, device=None, *, lean_gain: float = 0.0,
                 realism: DeployRealismCfg | None = None):
        self._env = getattr(env, "unwrapped", env)
        if device is None:
            device = self._env.device if self._env is not None else "cpu"
        self.device = torch.device(device)
        self.lean_gain = float(lean_gain)
        self.rcfg = realism  # None = clean sim (no IMU/actuator corruption)
        self._yaw_bias = None     # (B,) IMU yaw drift bias, lazy-init
        self._R_hist: list[torch.Tensor] = []  # latency buffer of past corrupted R
        self._prev_raw = None     # (B,3) previous raw action, for slider slew limit
        self._M = _M.to(self.device)
        self._anchor_offset = None  # obs-fallback: assume anchor=0
        # Recompute slider axes + anchors from the live model and verify the constants.
        if self._env is not None:
            M_model = slider_axes_base(self._env.sim.mj_model).to(self.device)
            if not torch.allclose(M_model, self._M, atol=1e-3):
                raise ValueError(
                    "Slider axes from model disagree with hardcoded _M "
                    f"(model=\n{M_model}\n constant=\n{self._M}). Update _M."
                )
            self._M = M_model
            A_model = slider_anchors_base(self._env.sim.mj_model).to(self.device)
            if not any(
                torch.allclose(A_model, A.to(self.device), atol=1e-3)
                for A in (_ANCHORS_SHELL, _ANCHORS_LINEAR)
            ):
                raise ValueError(
                    "Slider anchors from model match neither known variant "
                    f"(model=\n{A_model}\n shell=\n{_ANCHORS_SHELL}\n "
                    f"ball_linear=\n{_ANCHORS_LINEAR}). Update the constants."
                )
            # Per-slider along-axis constant c_i = axis_i . anchor_i, subtracted in the projection.
            self._anchor_offset = (M_model * A_model).sum(dim=-1)  # (3,)

    def _reset_mask(self):
        """(B,) bool of envs that just reset this step (episode_length_buf == 0), or None."""
        elb = getattr(self._env, "episode_length_buf", None)
        return (elb == 0) if elb is not None else None

    def _corrupt_imu(self, R: torch.Tensor) -> torch.Tensor:
        """Apply the deploy IMU model to ground-truth R: yaw-drift bias, ori noise, latency."""
        if self.rcfg is None or not self.rcfg.imu_active:
            return R
        B = R.shape[0]
        c = self.rcfg
        if self._yaw_bias is None or self._yaw_bias.shape[0] != B:
            self._yaw_bias = torch.zeros(B, device=R.device, dtype=R.dtype)
            self._R_hist = []
        reset = self._reset_mask()
        if reset is not None:
            self._yaw_bias[reset] = 0.0

        # Yaw-bias random walk, capped: perceived world heading drifts about +Z.
        if c.imu_yaw_drift_std > 0:
            self._yaw_bias += torch.randn(B, device=R.device, dtype=R.dtype) * c.imu_yaw_drift_std
            self._yaw_bias.clamp_(-c.imu_yaw_bias_max, c.imu_yaw_bias_max)
            cy, sy = torch.cos(self._yaw_bias), torch.sin(self._yaw_bias)
            Rz = torch.zeros(B, 3, 3, device=R.device, dtype=R.dtype)
            Rz[:, 0, 0] = cy; Rz[:, 0, 1] = -sy
            Rz[:, 1, 0] = sy; Rz[:, 1, 1] = cy
            Rz[:, 2, 2] = 1.0
            R = Rz @ R

        # Per-entry additive orientation noise (base_rotation_matrix Unoise analog).
        if c.imu_ori_noise_std > 0:
            R = R + torch.randn_like(R) * c.imu_ori_noise_std

        # Whole-step latency: emit R delayed by imu_latency_steps control steps.
        if c.imu_latency_steps > 0:
            self._R_hist.append(R)
            while len(self._R_hist) > c.imu_latency_steps + 1:
                self._R_hist.pop(0)
            R = self._R_hist[0]
        return R

    def _slew_limit(self, raw: torch.Tensor) -> torch.Tensor:
        """Rate-limit the raw slider action so the target moves no faster than the motor can."""
        if self.rcfg is None or self.rcfg.slider_vel_limit <= 0:
            return raw
        max_draw = self.rcfg.slider_vel_limit * self._env.step_dt / ACTION_SCALE  # raw units/step
        reset = self._reset_mask()
        if self._prev_raw is None or self._prev_raw.shape != raw.shape:
            self._prev_raw = raw.detach().clone()
        elif reset is not None:
            self._prev_raw[reset] = raw[reset]
        raw = torch.clamp(raw, self._prev_raw - max_draw, self._prev_raw + max_draw)
        self._prev_raw = raw.detach().clone()
        return raw

    def __call__(self, obs) -> torch.Tensor:
        if self._env is not None:
            robot = self._env.scene["robot"]
            R = matrix_from_quat(robot.data.root_link_quat_w)          # (B,3,3) ground truth
            R = self._corrupt_imu(R)                                    # sim-only deploy IMU model
            cmd_w = self._env.command_manager.get_term(self._CMD_TERM).vel_command_w[:, :2]
        else:  # obs fallback (noisy) — smoke test only
            frame = obs if isinstance(obs, torch.Tensor) else obs["actor"]
            if frame.dim() == 3:  # (B, history, 23) -> newest frame
                frame = frame[:, -1, :]
            cmd_w = frame[:, _CMD_SLICE]
            R = frame[:, _ROT_SLICE].view(frame.shape[0], 3, 3)
        raw = heuristic_slider_action(
            cmd_w, R, lean_gain=self.lean_gain, _M_dev=self._M,
            anchor_offset=self._anchor_offset,
        )
        return self._slew_limit(raw) if self._env is not None else raw


if __name__ == "__main__":
    # Parallelism + units smoke: verify batched shape/range and forward-command alignment.
    def _check(device: str, B: int):
        dev = torch.device(device)
        pol = BallbotHeuristicPolicy(device=dev)  # obs-fallback path (no env)
        obs = {"actor": torch.randn(B, 3, 23, device=dev)}
        # Seed identity rotation + a pure +x command in the newest frame.
        obs["actor"][:, -1, _ROT_SLICE] = torch.eye(3, device=dev).flatten()
        obs["actor"][:, -1, _CMD_SLICE] = torch.tensor([MAX_CMD, 0.0], device=dev)
        a = pol(obs)
        assert a.shape == (B, 3), a.shape
        assert torch.isfinite(a).all()
        assert (a >= -1.0).all() and (a <= 1.0).all()
        assert a.device.type == dev.type
        # Identity R, +x command, lean along +x: s7 = axis7_base . [lean,0,0] with
        # axis7_base=[-1,0,0] => s7 negative (saturates to -1); s5,s6 ~ 0.
        assert (a[:, 2] < -0.5).all(), a[0]
        assert a[:, :2].abs().max() < 1e-4, a[0]
        print(f"[OK] {device} B={B}: action[0]={a[0].tolist()}")

    _check("cpu", 1)
    _check("cpu", 4096)
    if torch.cuda.is_available():
        _check("cuda", 1)
        _check("cuda", 4096)
    print("smoke passed")
