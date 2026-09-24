"""Definitions vendored from the lab training repository for the ballbot tasks."""
from __future__ import annotations
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from mjlab.sim import MujocoCfg
from dataclasses import dataclass
import mujoco
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.action_manager import ActionTerm
from mjlab.utils.lab_api.math import quat_apply
import math
import numpy as np
from mjlab.scene import SceneCfg
from asset_zoo.ballbot.shared import SPHERE_BODY
from mjlab.managers.event_manager import RecomputeLevel


UNDERWATER_DENSITY = 1000.0      # water density kg/m³ (activates MuJoCo inertia/viscous drag — NOT buoyancy)


UNDERWATER_VISCOSITY = 0.001     # dynamic viscosity Pa·s (water at 20°C); near-negligible vs density drag


UNDERWATER_Z_WEIGHT = 0.3        # 3× air baseline (original rationale "buoyancy drift" is moot — see below;


@dataclass
class UnderwaterMujocoCfg(MujocoCfg):
    """MujocoCfg extended with fluid medium parameters.

    Setting density > 0 activates MuJoCo's *global inertia-based* fluid model
    (mjwarp _fluid_force). density/viscosity propagate to GPU via put_model.

    IMPORTANT — no Archimedes buoyancy. The global inertia-based model applies
    only velocity-dependent drag/lift (computed from each body's inertia-equivalent
    box); it does NOT add a static buoyancy force. Measured: free-fall a_z in water
    = −9.81 m/s² (identical to air) → buoyancy = 0. True Archimedes buoyancy would
    require the per-geom ellipsoid model (fluidshape="ellipsoid" + fluidcoef) or an
    explicit applied force. We deliberately omit it: a dense robot (open frame,
    motors) sinks, so net buoyancy is small; the robot rests on the floor at full
    weight, which is the regime we train.

    Force breakdown (measured, base moving along x):
      density drag (1000): ~57 N @ 0.4 m/s, quadratic in v — dominant resistance.
      viscosity drag (0.001): ~0.02 N @ 0.4 m/s — negligible.
    The quadratic drag caps top speed near 0.2 m/s (see UNDERWATER_MAX_VEL_CMD).
    """

    density: float = UNDERWATER_DENSITY
    viscosity: float = UNDERWATER_VISCOSITY

    def apply(self, model: mujoco.MjModel) -> None:
        super().apply(model)
        model.opt.density = self.density
        model.opt.viscosity = self.viscosity


def commands_xy(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
    """XY slice of world-frame velocity command."""
    command_term = env.command_manager.get_term(command_name)
    return command_term.vel_command_w[:, :2]


def base_lin_vel_world(
    env: ManagerBasedRlEnv, asset_cfg=None
) -> torch.Tensor:
    """Base linear velocity in world frame."""
    if asset_cfg is None:
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_w


def base_ang_vel_world(
    env: ManagerBasedRlEnv, asset_cfg=None
) -> torch.Tensor:
    """Base angular velocity in world frame."""
    if asset_cfg is None:
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        asset_cfg = SceneEntityCfg("robot")
    asset = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_w


class WorldFrameVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand with true world-frame vel_command_w.

    Base class sets vel_command_w = vel_command_b at resample time without heading
    rotation. For a robot with random spawn yaw ±π this makes vel_command_w point
    an arbitrary world direction each episode, corrupting the reward signal.

    Fix: rotate vel_command_b by heading_w at resample time so vel_command_w is a
    stable, consistent world-frame target throughout the episode.
    """

    world_frame_command: bool = True  # nav_cmd from GUI is world-frame; receiver must rotate to body

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        heading = self.robot.data.heading_w[env_ids]
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        vx_b = self.vel_command_b[env_ids, 0]
        vy_b = self.vel_command_b[env_ids, 1]
        self.vel_command_w[env_ids, 0] = cos_h * vx_b - sin_h * vy_b
        self.vel_command_w[env_ids, 1] = sin_h * vx_b + cos_h * vy_b
        self.vel_command_w[env_ids, 2] = self.vel_command_b[env_ids, 2]

    def _update_metrics(self) -> None:
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        
        # World-frame XY linear velocity tracking error
        self.metrics["error_vel_xy"] += (
            torch.norm(
                self.vel_command_w[:, :2] - self.robot.data.root_link_lin_vel_w[:, :2], dim=-1
            )
            / max_command_step
        )
        
        # World-frame Z angular velocity (yaw) tracking error
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_w[:, 2] - self.robot.data.root_link_ang_vel_w[:, 2])
            / max_command_step
        )

    def _debug_vis_impl(self, visualizer: object) -> None:
        """Draw velocity command and actual velocity arrows in world frame."""
        import numpy as np
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        cmds_w = self.vel_command_w.cpu().numpy()
        base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        lin_vel_ws = self.robot.data.root_link_lin_vel_w.cpu().numpy()
        ang_vel_ws = self.robot.data.root_link_ang_vel_w.cpu().numpy()

        scale = self.cfg.viz.scale
        z_offset = self.cfg.viz.z_offset

        for batch in env_indices:
            base_pos_w = base_pos_ws[batch]
            cmd_w = cmds_w[batch]
            lin_vel_w = lin_vel_ws[batch]
            ang_vel_w = ang_vel_ws[batch]

            # Skip if robot appears uninitialized (at origin).
            if np.linalg.norm(base_pos_w) < 1e-6:
                continue

            # Base offset on world Z axis
            cmd_lin_from = base_pos_w + np.array([0, 0, z_offset]) * scale

            # Command linear velocity arrow (blue)
            cmd_lin_to = cmd_lin_from + np.array([cmd_w[0], cmd_w[1], 0]) * scale
            visualizer.add_arrow(
                cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
            )

            # Command angular velocity arrow (green)
            cmd_ang_from = cmd_lin_from
            cmd_ang_to = cmd_ang_from + np.array([0, 0, cmd_w[2]]) * scale
            visualizer.add_arrow(
                cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
            )

            # Actual linear velocity arrow (cyan)
            act_lin_from = cmd_lin_from
            act_lin_to = act_lin_from + np.array([lin_vel_w[0], lin_vel_w[1], 0]) * scale
            visualizer.add_arrow(
                act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
            )

            # Actual angular velocity arrow (light green)
            act_ang_from = act_lin_from
            act_ang_to = act_ang_from + np.array([0, 0, ang_vel_w[2]]) * scale
            visualizer.add_arrow(
                act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
            )


class WorldFrameVelocityCommandCfg(UniformVelocityCommandCfg):
    def build(self, env: ManagerBasedRlEnv) -> WorldFrameVelocityCommand:
        return WorldFrameVelocityCommand(self, env)


def base_rotation_matrix(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Computes the base rotation matrix from the root link quaternion.

    Args:
        env: The environment instance.

    Returns:
        torch.Tensor: Flattened 3x3 rotation matrix. Shape: (num_envs, 9).
    """
    asset = env.scene["robot"]
    quat = asset.data.root_link_quat_w  # Shape: (num_envs, 4) in [w, x, y, z]
    rot_matrix = matrix_from_quat(quat)  # Shape: (num_envs, 3, 3)
    return rot_matrix.reshape(env.num_envs, 9)


def track_linear_velocity_relative(
    env: ManagerBasedRlEnv,
    std_rel: float,
    std_min: float,
    command_name: str,
    z_weight: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    jump_gate_command_name: str | None = None,
) -> torch.Tensor:
    """Relative-standard deviation linear velocity tracking reward in world space.

    Calculates tracking reward in the world frame, which is invariant to sphere rolling:
        std_eff = max(std_rel * ||cmd_w_xy||_2, std_min)
        reward = exp(-(||v_w_err_xy||_2^2 + z_weight * v_w_z^2) / std_eff^2)

    Args:
        env: The environment instance.
        std_rel: Relative scaling coefficient of command velocity norm.
        std_min: Minimum standard deviation limit floor.
        command_name: Name of target command inside command manager.
        z_weight: Dimension coordinate weight for Z velocity penalty.
        asset_cfg: Scene entity configuration for robot.
        jump_gate_command_name: If set, name of a jump command term whose `jump_active`
            zeroes the Z-velocity penalty during active jump windows (else the unconditional
            penalty fights every jump attempt). None (default) preserves old behavior.

    Returns:
        torch.Tensor: Shape (num_envs,). Velocity tracking reward.
    """
    robot = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    if command_term is None:
        raise ValueError(f"Command '{command_name}' not found in command_manager.")

    command_w_xy = command_term.vel_command_w[:, :2]
    actual_velocity_w = robot.data.root_link_lin_vel_w

    command_norm = torch.linalg.vector_norm(command_w_xy, dim=1)
    std_effective = torch.clamp(std_rel * command_norm, min=std_min)

    xy_error_squared = torch.sum(torch.square(command_w_xy - actual_velocity_w[:, :2]), dim=1)
    z_error_squared = torch.square(actual_velocity_w[:, 2])
    if jump_gate_command_name is not None:
        jump_active = env.command_manager.get_term(jump_gate_command_name).jump_active.float()
        z_error_squared = z_error_squared * (1.0 - jump_active)

    weighted_error = xy_error_squared + z_weight * z_error_squared
    return torch.exp(-weighted_error / torch.square(std_effective))


def track_linear_velocity_ema(
    env: ManagerBasedRlEnv,
    std_rel: float,
    std_min: float,
    command_name: str,
    z_weight: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Time-averaged (EMA) world-frame linear velocity tracking reward for the ballbot.

    Identical to track_linear_velocity_relative except the XY velocity compared to
    the command is the command term's exponential-moving-average world velocity
    `v_ema_w` (time constant cfg.vel_ema_tau, maintained in BallbotUniformVelocityCommand),
    NOT the instantaneous root velocity. The Z penalty stays instantaneous (bounce/hop
    must be punished in real time, not averaged away).

        std_eff = max(std_rel * ||cmd_w_xy||, std_min)
        reward  = exp(-(||cmd_w_xy - v_ema_w||^2 + z_weight * v_w_z^2) / std_eff^2)

    WHY (open-loop probe, 2026-06-14): the ballbot lean->speed map is DISCONTINUOUS — no
    steady lean yields 0 < v < ~0.66 m/s (lean 0.06 -> ~0 m/s, lean 0.08 -> 0.66 m/s). So
    a steady sub-knee command (e.g. 0.3) is physically unreachable; it exists only as the
    time-average of a burst gait (roll above the bifurcation, coast, repeat). Instantaneous
    tracking actively punishes that gait (alternating 0 / 0.66 vs target 0.3 -> large
    instantaneous error; with std_min=0.1, standing reward 0.018 > bursting 0.003). EMA
    averaging makes a burst gait that nets the command score ~1.0.

    Hack-resistant: v_ema_w is causal and exponentially decaying, so the policy must
    SUSTAIN the correct average throughout the command hold — it cannot be end-loaded
    (sprint at the window edge), and continuous over-speed (v_ema -> 0.66 vs cmd 0.3)
    scores worse than a correctly modulated burst.

    Args:
        env: The environment instance.
        std_rel: Relative scaling coefficient of command velocity norm.
        std_min: Minimum standard deviation floor.
        command_name: Name of target command (must be a BallbotUniformVelocityCommand
            with vel_ema_tau > 0 so v_ema_w is maintained).
        z_weight: Weight on instantaneous Z velocity penalty.
        asset_cfg: Scene entity configuration for robot.

    Returns:
        torch.Tensor: Shape (num_envs,). Time-averaged velocity tracking reward.
    """
    robot = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    if command_term is None:
        raise ValueError(f"Command '{command_name}' not found in command_manager.")

    command_w_xy = command_term.vel_command_w[:, :2]
    v_ema_xy = command_term.v_ema_w
    v_z = robot.data.root_link_lin_vel_w[:, 2]

    command_norm = torch.linalg.vector_norm(command_w_xy, dim=1)
    std_effective = torch.clamp(std_rel * command_norm, min=std_min)

    xy_error_squared = torch.sum(torch.square(command_w_xy - v_ema_xy), dim=1)
    z_error_squared = torch.square(v_z)

    weighted_error = xy_error_squared + z_weight * z_error_squared
    return torch.exp(-weighted_error / torch.square(std_effective))


def joint_pos_limits_slider(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for joints exceeding soft position limits.

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for robot.

    Returns:
        torch.Tensor: Shape (num_envs,). Penalty value (sum of squared violations).
    """
    robot = env.scene[asset_cfg.name]
    joint_positions = robot.data.joint_pos[:, asset_cfg.joint_ids]
    soft_limits = robot.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]

    lower_limit = soft_limits[:, :, 0]
    upper_limit = soft_limits[:, :, 1]

    lower_violation = torch.clamp(lower_limit - joint_positions, min=0.0)
    upper_violation = torch.clamp(joint_positions - upper_limit, min=0.0)

    total_violation = torch.sum(torch.square(lower_violation) + torch.square(upper_violation), dim=1)
    return total_violation


RING_NORMALS_13 = [
    (0, 0, 1), (0, 1, 0), (1, 0, 0),
    (0, 1, 1), (0, 1, -1), (1, 0, 1), (1, 0, -1), (1, 1, 0), (1, -1, 0),
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
]


def ring_segment_layout(n_rings: int, segs_per_ring: int) -> tuple[np.ndarray, np.ndarray]:
    """Ring-cage drag-element layout: per-segment centroid dir + hoop tangent (body-local units).

    The buildable hull is an armillary cage of great-circle rings (NOT a stud-covered sphere);
    each submerged ring ARC is a slender drag rod. Discretizes each of the n_rings octahedral
    great circles (RING_NORMALS_13) into segs_per_ring equal segments and returns, per segment,
    the unit centroid direction ĉ (on the sphere surface) and the unit hoop TANGENT t̂ (in the
    ring plane, ⊥ ĉ and ⊥ the ring normal). Drag opposes flow perpendicular to t̂ — a slender rod
    feels only cross-axis flow; sliding along its length costs ≈0. Single source of truth: the
    force term reads these for drag, the env reads them to place matching visual capsule segments.

    Only n_rings == 13 (octahedral-complete) is supported; see ring_cage_viz.py for the geometry.
    """
    if n_rings != 13:
        raise ValueError(f"n_rings must be 13, got {n_rings}")
    th = (np.arange(segs_per_ring) + 0.5) * (2.0 * np.pi / segs_per_ring)  # segment midpoints
    centroids, tangents = [], []
    for nrm in RING_NORMALS_13:
        n = np.asarray(nrm, float)
        n /= np.linalg.norm(n)
        seed = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
        u = np.cross(n, seed)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)                                            # u,v,n orthonormal
        c = np.cos(th)[:, None] * u + np.sin(th)[:, None] * v         # centroid dirs (unit)
        t = -np.sin(th)[:, None] * u + np.cos(th)[:, None] * v        # hoop tangents (unit)
        centroids.append(c)
        tangents.append(t)
    return (np.concatenate(centroids).astype(np.float32),
            np.concatenate(tangents).astype(np.float32))


class SurfaceFloatForce(ActionTerm):
    """Per-substep hydrostatic buoyancy + drag for a SEALED sphere floating ON water.

    Not a policy action (action_dim=0): it hijacks the ActionTerm hook because
    ``apply_actions`` is the only manager callback that runs every decimation substep
    *before* ``sim.step()`` (manager_based_rl_env.step). Events run once per policy step
    → 40 ms-stale force, wrong for a bobbing float. Each substep it reads the hull state
    and writes a fresh world-frame wrench to ``xfrc_applied`` on ``base_link``.

    Why custom (not MuJoCo's opt.density fluid): that model fills ALL space uniformly
    (no free surface) and adds ZERO Archimedes buoyancy — it cannot float a body at a
    surface. Surface float needs depth-gated buoyancy+drag, computed here.

    Physics (waterline at world z=cfg.water_level_z; sphere radius R; sealed → only the
    HULL is wetted, internal masses dry):
      h    = clamp(R − z_center + water_z, 0, 2R)    submerged cap depth (waterline−bottom)
      Vsub = π·h²·(3R − h)/3                          cap volume
      F_buoy = ρ·g·Vsub  (world +z)                   Archimedes, applied at CoB
      CoB  = directly below the sphere CENTER (a sphere ∩ horizontal halfspace is always
             axisymmetric about the vertical through the center, for any body tilt), at
             z = z_center + ū, ū = ∫u(R²−u²)du / (Vsub/π) over u∈[−R, −R+h].
      τ    = (CoB − CoM) × F_buoy → only the HORIZONTAL CoM-vs-center offset (created by
             the eccentric masses) makes a moment ⇒ correct metacentric righting/capsize.
             A buoyancy point-force at the body origin would lose this and teach fake
             stability ⇒ capsize on the real robot. Hence the explicit torque.
      A_fr = R²·arccos(d/R) − d·√(R²−d²), d=R−h       submerged FRONTAL silhouette (cap projection)
      F_drag = −½·ρ·Cd·A_fr·‖v‖·v                     quadratic, submerged only

    Added mass intentionally OMITTED: at full submersion the sphere's added mass
    (½·ρ·V ≈ 3.6 kg) ≈ the robot mass (3.06 kg); an EXPLICIT added-mass force at this
    ratio is integration-unstable (it belongs in the mass matrix, i.e. implicit, which
    mjlab/mjwarp does not expose). Valid only at low Froude (Fr=v/√(gL)≈0.12 ≪ 0.3 here),
    where wave-making drag — which MuJoCo cannot model — is negligible.
    """

    cfg: "SurfaceFloatForceCfg"

    def __init__(self, cfg: "SurfaceFloatForceCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        body_ids, _ = self._entity.find_bodies(cfg.body_name)
        assert len(body_ids) == 1, f"expected 1 body for {cfg.body_name!r}, got {body_ids}"
        self._body_ids = body_ids
        self._R = cfg.radius
        self._water_z = cfg.water_level_z
        self._rho = cfg.water_density
        self._g = cfg.gravity
        self._cd = cfg.drag_coef
        # Body-local offset from frame origin to the geometric sphere centre. Zero for the
        # idealized robot (origin = centre) → _sphere_center short-circuits to root_link_pos_w.
        off = cfg.center_offset
        self._has_center_offset = any(v != 0.0 for v in off)
        self._center_offset = torch.tensor(off, device=self.device).expand(self.num_envs, 3)
        self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)

    def _sphere_center(self, d) -> torch.Tensor:
        """World position of the geometric sphere centre, (N,3).

        = root_link_pos_w + R(q)·center_offset. For the real CAD body the frame origin is
        NOT the sphere centre (offset along the model long axis), and that centre ORBITS the
        origin as the hull rolls — so the rotation must be applied every substep, not folded
        into a fixed world offset. Idealized robot (zero offset) returns root_link_pos_w.
        """
        if not self._has_center_offset:
            return d.root_link_pos_w
        return d.root_link_pos_w + quat_apply(d.root_link_quat_w, self._center_offset)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        # No policy action: the wrench is computed from sim state in apply_actions.
        pass

    def apply_actions(self) -> None:
        force, torque = self._compute_wrench(self._entity.data)
        self._entity.write_external_wrench_to_sim(
            force[:, None, :], torque[:, None, :], body_ids=self._body_ids
        )

    def _compute_wrench(self, d) -> tuple[torch.Tensor, torch.Tensor]:
        """Buoyancy + CoB restoring torque + wetted quadratic hull drag → (force, torque).

        Split out from apply_actions so subclasses (PaddlewheelSurfaceForce) can ADD pad
        wrenches before the single overwrite write (xfrc_applied is overwrite-semantics).
        """
        R = self._R
        pos = self._sphere_center(d)     # geometric sphere centre (body origin + R(q)·offset), (N,3)
        com = d.root_com_pos_w           # CoM (shifted by eccentric masses), (N,3)
        vel = d.root_com_lin_vel_w       # world linear velocity, (N,3)

        # Submerged cap depth = waterline minus sphere BOTTOM (centre z − R), clamped.
        h = torch.clamp(R - pos[:, 2] + self._water_z, min=0.0, max=2.0 * R)  # (N,)
        vsub_over_pi = h * h * (3.0 * R - h) / 3.0                        # Vsub/π
        f_buoy = self._rho * self._g * math.pi * vsub_over_pi             # (N,) up

        # Cap centroid offset ū (signed, ≤0 below centre); ∫u(R²−u²)du over [−R,−R+h].
        a = -R
        b = -R + h
        num = R * R * (b * b - a * a) / 2.0 - (b**4 - a**4) / 4.0
        eps = 1e-9
        ubar = torch.where(vsub_over_pi > eps, num / (vsub_over_pi + eps), torch.zeros_like(h))

        # Buoyancy force (world +z) at CoB; CoB horizontally at the sphere centre.
        force = torch.zeros(self.num_envs, 3, device=self.device)
        force[:, 2] = f_buoy
        cob = torch.empty_like(pos)
        cob[:, 0] = pos[:, 0]
        cob[:, 1] = pos[:, 1]
        cob[:, 2] = pos[:, 2] + ubar
        r = cob - com
        torque = torch.cross(r, force, dim=-1)   # (r_y·F, −r_x·F, 0): metacentric righting

        # Quadratic drag on the submerged FRONTAL area, applied at CoM (no extra torque modelled).
        # Frontal area = vertical projection of the submerged cap (silhouette seen by horizontal
        # flow), NOT the waterplane disc π(2Rh−h²): that is the HORIZONTAL cut, which mis-orients
        # the drag — 2× too high at half-submersion and →0 when fully submerged. For cap depth h the
        # silhouette is the circular segment of the great-circle disc below the waterline chord at
        # signed distance d=(R−h) from centre: A = R²·arccos(d/R) − d·√(R²−d²), valid h∈[0,2R]
        # (πR²/2 at half, πR² at full, 0 at h=0).
        d = R - h                                                         # chord dist from centre (signed)
        a_front = (R * R * torch.arccos(torch.clamp(d / R, -1.0, 1.0))
                   - d * torch.sqrt(torch.clamp(2.0 * R * h - h * h, min=0.0)))
        a_front = torch.clamp(a_front, min=0.0)                           # (N,)
        speed = torch.linalg.vector_norm(vel, dim=-1)                     # (N,)
        force = force - 0.5 * self._rho * self._cd * (a_front * speed)[:, None] * vel
        return force, torque

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        # Stateless: xfrc_applied is rewritten every substep before sim.step.
        pass


@dataclass(kw_only=True)
class SurfaceFloatForceCfg(ActionTermCfg):
    """Config for SurfaceFloatForce.

    body_name:     hull body the wrench acts on (the sealed sphere; "base_link").
    radius:        sphere radius R [m].
    water_level_z: world-z of the still-water surface [m].
    water_density: ρ [kg/m³]. drag_coef: Cd (smooth sealed sphere ≈ 0.47).
    gravity:       g [m/s²] (magnitude; must match sim gravity).
    center_offset: body-local vector from the body frame origin (base_link / freejoint
        reference) to the GEOMETRIC sphere centre [m]. Default (0,0,0) = origin IS the
        centre (idealized robot). The real CAD authors the frame at the COM, offset along
        the model long axis, so its sphere/cage centre is (0, CAGE_OFFSET, 0). Buoyancy
        depth, CoB, ring/pad placement all reference this true centre, which orbits as the
        hull rolls; using the body origin instead off-centres the cage and forces.
    """

    body_name: str
    radius: float
    water_level_z: float
    water_density: float
    drag_coef: float
    gravity: float
    center_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def build(self, env: ManagerBasedRlEnv) -> SurfaceFloatForce:
        return SurfaceFloatForce(self, env)


class RingCageSurfaceForce(SurfaceFloatForce):
    """SurfaceFloatForce + depth-gated cross-tangent drag on an armillary RING CAGE (propulsion).

    Fin-free counterpart of PaddlewheelSurfaceForce. The buildable hull is a frame of 13
    octahedral great-circle rings around the central sealed buoyancy sphere; the submerged ring
    arcs ARE the paddle elements (no spikes/fins). Each ring is discretized into capsule segments
    (ring_segment_layout); a slender rod segment drags on the flow PERPENDICULAR to its local
    hoop tangent t̂ (motion along the rod ≈ free). Half-submersion rectifies the spin into net
    thrust exactly as the spike model does — only the per-element slide axis differs (tangent t̂
    here vs radial p̂ for spikes). Buoyancy + hull drag UNCHANGED (super()._compute_wrench);
    segments are massless (float setpoint untouched).

    Per segment i, per substep (vectorised over envs × segments):
      c_i  = x_c + R·R(q)·ĉ_i              segment centroid, world (ring on sphere surface)
      t_i^w= R(q)·t̂_i                       hoop tangent, world (unit)
      v_i  = v_com + ω × (c_i − com)        segment velocity through still water
      v⊥   = v_i − (v_i·t_i^w)·t_i^w        cross-TANGENT (perpendicular) flow
      wet  = smoothstep((z_water − c_i.z)/(2·r_tube))
      vent = vent_res + (1−vent_res)/(1 + (Fr/Fr_c)²),  Fr = |v⊥|/√(g·h_sub)  (ventilation knockdown)
      F_i  = −wet·vent·½ρ·Cd_ring·A_seg·|v⊥|·v⊥ · thrust_scale
      τ_i  = (c_i − com) × F_i
    A_seg = 2·r_tube·(2πR/S) (tube diameter × segment arc length), uniform over segments.

    Degenerate spin (hull spins exactly about one ring's normal) makes THAT ring's segments slide
    along their tangent → zero drag, but the other 12 rings (normals at octahedral angles) still
    drag → no global thrust blind spot, so fins are unnecessary. Node overlap (rings cross at
    intersection points) double-counts at discrete points (measure-zero) → ignored, same spirit
    as the spike model's unmodelled pad-pad shadowing (thrust_scale DR covers the lumped error).
    """

    cfg: "RingCageSurfaceForceCfg"

    def __init__(self, cfg: "RingCageSurfaceForceCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        cen, tan = ring_segment_layout(cfg.n_rings, cfg.segs_per_ring)
        # (1,M,3) body-local centroid dirs + hoop tangents, for broadcast over envs.
        self._cen = torch.from_numpy(cen).to(self.device)[None]
        self._tan = torch.from_numpy(tan).to(self.device)[None]
        # Rings sit on the VISIBLE shell surface (ring_radius), NOT the buoyancy radius
        # (cfg.radius) — those differ (shell 0.15 vs buoyancy 0.12; see env cfg RING_RADIUS).
        self._ring_radius = cfg.ring_radius
        self._r_tube = cfg.ring_tube_radius
        arc = 2.0 * math.pi * cfg.ring_radius / cfg.segs_per_ring   # segment arc length
        self._a_seg = 2.0 * cfg.ring_tube_radius * arc             # frontal area per segment
        self._cd_ring = cfg.ring_drag_coef
        # Surface-piercing ventilation knockdown params (see _compute_wrench).
        self._vent_fr_c = cfg.vent_froude_crit
        self._vent_residual = cfg.vent_residual
        # Per-episode lumped thrust-loss multiplier (domain randomization). 1.0 = nominal.
        self._thrust_scale_range = cfg.thrust_scale_range
        self._thrust_scale = torch.ones(self.num_envs, device=self.device)

    def _compute_wrench(self, d) -> tuple[torch.Tensor, torch.Tensor]:
        force, torque = super()._compute_wrench(d)

        x_c = self._sphere_center(d)[:, None, :] # (N,1,3) geometric sphere centre
        com = d.root_com_pos_w[:, None, :]       # (N,1,3)
        v_com = d.root_com_lin_vel_w[:, None, :] # (N,1,3)
        omega = d.root_com_ang_vel_w[:, None, :] # (N,1,3)
        q = d.root_link_quat_w                   # (N,4) wxyz

        m = self._cen.shape[1]
        q_exp = q[:, None, :].expand(-1, m, -1)              # (N,M,4)
        cen = self._cen.expand(self.num_envs, -1, -1)
        tan = self._tan.expand(self.num_envs, -1, -1)

        c_dir = quat_apply(q_exp, cen)                       # (N,M,3) centroid dir, world
        t_w = quat_apply(q_exp, tan)                         # (N,M,3) hoop tangent, world
        c = x_c + self._ring_radius * c_dir                  # segment centroids on shell, world
        rel = c - com
        v = v_com + torch.cross(omega.expand_as(rel), rel, dim=-1)
        v_perp = v - (v * t_w).sum(dim=-1, keepdim=True) * t_w   # cross-tangent flow
        vp_mag = torch.linalg.vector_norm(v_perp, dim=-1)        # (N,M)

        depth = (self._water_z - c[..., 2]) / (2.0 * self._r_tube)  # 0 at surface, 1 deep
        t = torch.clamp(depth, 0.0, 1.0)
        wet = t * t * (3.0 - 2.0 * t)                       # smoothstep, no chatter at surface

        # Ventilation knockdown — make the sim PESSIMISTIC where the smooth wet-gate is least
        # trustworthy. A surface-piercing ring segment moving fast through shallow water entrains
        # air on its suction side, establishing a ventilated cavity that collapses cross-flow drag.
        # Onset scales with the local depth-Froude Fr = |v⊥| / √(g·h_sub) (h_sub = submergence below
        # the still waterline, floored at the tube radius = shallowest meaningful depth, avoids ÷0).
        # Below Fr_c the segment stays fully wetted (vent→1); well above it only a residual
        # base-vented drag survives (vent→vent_residual). Unlike the uniform thrust_scale DR this is
        # a STRUCTURED per-element loss concentrated in the shallow-fast band — exactly the regime a
        # zero-shot policy must not over-trust. Shared by BOTH the ballbot and ball-circle ring-cage
        # bots (same class); deeper segments (large h_sub → Fr→0 → vent→1) are unaffected.
        h_sub = self._water_z - c[..., 2]                                   # (N,M) submergence depth
        h_eff = torch.clamp(h_sub, min=self._r_tube)
        fr = vp_mag / torch.sqrt(self._g * h_eff)
        vent = self._vent_residual + (1.0 - self._vent_residual) / (1.0 + (fr / self._vent_fr_c) ** 2)

        # Quadratic drag opposing the cross-tangent flow, gated by submersion AND ventilation.
        f_seg = -(wet * vent * 0.5 * self._rho * self._cd_ring * self._a_seg * vp_mag)[..., None] * v_perp
        # Lumped per-episode thrust-loss DR: scale the RING wrench only (not buoyancy/hull drag).
        f_seg = f_seg * self._thrust_scale[:, None, None]
        tau_seg = torch.cross(rel, f_seg, dim=-1)

        return force + f_seg.sum(dim=1), torque + tau_seg.sum(dim=1)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        # Resample the per-episode lumped thrust-loss multiplier for the reset envs.
        if self._thrust_scale_range is None:
            return
        lo, hi = self._thrust_scale_range
        if env_ids is None:
            env_ids = slice(None)
        n = self._thrust_scale[env_ids].shape[0]
        self._thrust_scale[env_ids] = torch.rand(n, device=self.device) * (hi - lo) + lo


@dataclass(kw_only=True)
class RingCageSurfaceForceCfg(SurfaceFloatForceCfg):
    """Config for RingCageSurfaceForce (extends SurfaceFloatForceCfg).

    n_rings:          number of great-circle rings (13 = octahedral-complete; only value supported).
    segs_per_ring:    capsule segments per ring (drag-element resolution).
    ring_radius:      radius the rings sit at [m] = the VISIBLE shell surface; distinct from the
        buoyancy radius (cfg.radius) — used for centroid placement, lever arm, and arc length.
    ring_tube_radius: ring tube radius r_tube [m]; A_seg = 2·r_tube·(2πR/segs), gate width 2·r_tube.
    ring_drag_coef:   Cd of a circular cylinder in cross-flow ≈ 1.0–1.2.
    thrust_scale_range: per-episode uniform multiplier (lo,hi) on the ring wrench, modelling the
        lumped UNMODELLED optimistic thrust losses (ring-ring/node shadowing, induced wake,
        added-mass) — all of which only REDUCE effective thrust, so center ≤1.
        None ⇒ scale fixed at 1 (no DR). Scales rings ONLY; buoyancy/hull drag stay deterministic.
    vent_froude_crit: critical local depth-Froude for surface-piercing ventilation onset
        (Fr = |v⊥|/√(g·h_sub)); ~2 per surface-piercing-strut inception data. Below it segments
        stay fully wetted; the cavity establishes well above it. STRUCTURED per-element loss,
        distinct from the uniform thrust_scale DR (see _compute_wrench). Shared by both ring-cage bots.
    vent_residual:    residual wetted-drag fraction of a fully ventilated (base-vented) segment
        (front stagnation survives, base suction lost) ≈ 0.3. vent ∈ [vent_residual, 1].
    """

    n_rings: int
    segs_per_ring: int
    ring_radius: float
    ring_tube_radius: float
    ring_drag_coef: float
    thrust_scale_range: tuple[float, float] | None = None
    vent_froude_crit: float = 2.0
    vent_residual: float = 0.3

    def build(self, env: ManagerBasedRlEnv) -> RingCageSurfaceForce:
        return RingCageSurfaceForce(self, env)


WATER_LEVEL_Z = 1.0          # m, world-z of the still-water surface (1 m water column)


SURFACE_WATER_DENSITY = UNDERWATER_DENSITY   # 1000 kg/m³ (fresh water)


SURFACE_GRAVITY = 9.81       # m/s² (matches sim gravity magnitude)


SPHERE_DRAG_CD = 0.47


N_RINGS = 13              # octahedral-complete (3 four-fold + 6 two-fold + 4 three-fold axes)


SEGS_PER_RING = 24        # capsule segments per ring (15° resolution); 13·24 = 312 drag elements


RING_TUBE_RADIUS = 0.008  # m; tube radius r_tube. A_seg = 2·r_tube·(2πR/segs), gate width 2·r_tube


RING_DRAG_CD = 1.0        # circular cylinder cross-flow


def _add_water_surface_visual(
    scene_cfg: SceneCfg, water_z: float, half_size: float = 5.0
) -> None:
    """Add a half-transparent, non-colliding water-surface plane to the SCENE worldbody.

    Uses SceneCfg.spec_fn (the post-merge hook that mutates the full scene MjSpec) rather
    than the robot entity spec — mjlab strips an entity's worldbody geoms during scene
    merge, so a robot-spec surface never survives. Visual-only (contype=conaffinity=0)
    single-sided PLANE at z=water_z; carries no mass/contact, so dynamics are unchanged
    (physics is in SurfaceFloatForce). A PLANE (not a thin BOX) is used so the viewer shows
    ONE surface — a transparent box renders both its top and bottom faces, reading as a
    double/triple waterline. Lets the viewer show the robot floating at the waterline.
    """
    import mujoco

    def spec_fn(spec: "mujoco.MjSpec") -> None:
        g = spec.worldbody.add_geom()
        g.name = "water_surface_visual"
        g.type = mujoco.mjtGeom.mjGEOM_PLANE
        g.size = [half_size, half_size, 1.0]  # half-extents x/y, grid spacing
        g.pos = [0.0, 0.0, water_z]
        g.rgba = [0.15, 0.45, 0.8, 0.4]  # blue, 40% opacity
        g.contype = 0
        g.conaffinity = 0

    scene_cfg.spec_fn = spec_fn


def _solve_float_depth(radius: float, mass: float, density: float) -> float:
    """Submerged cap depth h where buoyancy balances weight: ρ·Vsub(h) = mass.

    Vsub(h) = π·h²·(3R − h)/3 is monotonic on [0, 2R] → bisection. Used to start each
    episode at float equilibrium (no drop transient).
    """
    import math

    target = mass / density  # displaced volume needed
    lo, hi = 0.0, 2.0 * radius
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        vsub = math.pi * mid * mid * (3.0 * radius - mid) / 3.0
        if vsub < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _ring_rgba(ring_idx: int, alpha: float = 0.2):
    """Per-ring color matching ring_cage_viz.py (RING_NORMALS_13 share its PRINCIPAL+DIAGONAL+BODYDIAG
    order): rings 0/1/2 = R/G/B principal, 3-8 = gray diagonal, 9-12 = magenta body-diagonal."""
    if ring_idx == 0:   return [0.85, 0.10, 0.10, alpha]   # red   (XY)
    if ring_idx == 1:   return [0.10, 0.55, 0.10, alpha]   # green (XZ)
    if ring_idx == 2:   return [0.10, 0.25, 0.85, alpha]   # blue  (YZ)
    if ring_idx <= 8:   return [0.55, 0.55, 0.55, alpha]   # gray  diagonal
    return [0.85, 0.10, 0.85, alpha]                       # magenta body-diagonal


def _torus_mesh(major_r: float, minor_r: float, n_u: int = 64, n_v: int = 20):
    """Triangulated torus in the XY plane (symmetry axis = local z), major/minor radii given.

    Returns (uservert flat float list xyz, userface flat int list tri indices). One canonical mesh
    reused by every ring geom (each rotated z→ring-normal); a torus is the exact swept tube, so 13
    of these replace 13·segs_per_ring overlapping capsules — far fewer translucent surfaces to sort.
    """
    import numpy as np
    u = np.linspace(0.0, 2 * np.pi, n_u, endpoint=False)        # around the major ring
    v = np.linspace(0.0, 2 * np.pi, n_v, endpoint=False)        # around the tube cross-section
    uu, vv = np.meshgrid(u, v, indexing="ij")                   # (n_u, n_v)
    x = (major_r + minor_r * np.cos(vv)) * np.cos(uu)
    y = (major_r + minor_r * np.cos(vv)) * np.sin(uu)
    z = minor_r * np.sin(vv)
    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3)         # (n_u*n_v, 3)
    faces = []
    for i in range(n_u):
        for j in range(n_v):
            a = i * n_v + j
            b = ((i + 1) % n_u) * n_v + j
            c = ((i + 1) % n_u) * n_v + (j + 1) % n_v
            d = i * n_v + (j + 1) % n_v
            faces.extend([a, b, c, a, c, d])                   # two CCW triangles per quad
    return verts.reshape(-1).tolist(), faces


def _quat_z_to(n) -> list[float]:
    """wxyz quaternion rotating local +z onto unit vector n."""
    import numpy as np
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, n)
    s = np.linalg.norm(axis)
    if s < 1e-8:
        return [1.0, 0.0, 0.0, 0.0] if n[2] > 0 else [0.0, 1.0, 0.0, 0.0]
    ang = np.arccos(np.clip(float(np.dot(z, n)), -1.0, 1.0))
    axis /= s
    return [float(np.cos(ang / 2)), *(np.sin(ang / 2) * axis).tolist()]


def _add_ring_torus_visuals(
    robot_cfg, tube_radius: float, radius: float, alpha: float = 0.2,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Add visual-only ring cage as 13 transparent TORUS meshes on base_link (rotate with the hull).

    One canonical torus mesh (major=radius, minor=tube_radius) reused by 13 geoms, each rotated
    z→ring-normal (RING_NORMALS_13). Replaces the 312-capsule visual (see _add_ring_visuals, kept
    as fallback) so transparent intersections sort far better. Physics (RingCageSurfaceForce, 312
    drag segments in actions.py) is UNCHANGED — this is purely cosmetic (contype=conaffinity=0).

    center: body-local geometric sphere centre [m]. Default (0,0,0) = body origin (idealized
        robot). The real CAD frame origin is offset from the cage centre, so the rings must be
        placed at g.pos=center to stay concentric with the collision sphere (which is also at
        that offset); otherwise the cage renders off-centre from the hull.
    """
    import mujoco

    verts, faces = _torus_mesh(radius, tube_radius)
    _base_spec_fn = robot_cfg.spec_fn

    def spec_fn() -> "mujoco.MjSpec":
        spec = _base_spec_fn()
        mesh = spec.add_mesh()
        mesh.name = "ring_torus"
        mesh.uservert = verts
        mesh.userface = faces
        body = spec.body(SPHERE_BODY)
        for i, normal in enumerate(RING_NORMALS_13):
            g = body.add_geom()
            g.name = f"ring_torus_{i}"
            g.type = mujoco.mjtGeom.mjGEOM_MESH
            g.meshname = "ring_torus"
            g.pos = list(center)
            g.quat = _quat_z_to(normal)
            g.rgba = _ring_rgba(i, alpha)
            g.contype = 0
            g.conaffinity = 0
        return spec

    robot_cfg.spec_fn = spec_fn


class DeferredModelFieldsWrapper:
    """Wraps a @requires_model_fields DR function, suppressing its recompute trigger.

    WHY: mjwarp.set_const() costs ~40ms for ALL envs each time any requires_model_fields
    event fires on reset (not just the resetting envs — it's all-or-nothing). This wrapper
    lets the GPU field write happen per-episode (cheap, ~0.1ms) while deferring set_const
    to a PeriodicPhysicsRecompute interval event. Saves ~40ms/step at the cost of derived
    quantities (subtree mass, inv weights) being stale for up to N steps.

    Preserves .model_fields so sim.expand_model_fields() still allocates per-world GPU
    buffers. Sets .recompute = RecomputeLevel.none so event_manager skips set_const.
    """

    def __init__(self, fn):
        self.model_fields = getattr(fn, "model_fields", ())
        self.recompute = RecomputeLevel.none  # suppress set_const after each reset call
        self._fn = fn

    def __call__(self, env, env_ids, **kwargs):
        self._fn(env, env_ids, **kwargs)


class PeriodicPhysicsRecompute:
    """Periodic mode="interval" event that triggers mjwarp.set_const() globally.

    Used with DeferredModelFieldsWrapper. Mass/COM writes happen per-reset (cheap),
    but derived quantities (body_subtreemass, dof_invweight0, etc.) only sync after
    set_const. This event fires every N seconds globally, amortizing the ~40ms cost.

    The event_manager reads .recompute after __call__ and calls
    env.sim.recompute_constants(set_const) automatically — __call__ is a no-op.
    Configure as: mode="interval", is_global_time=True, interval_range_s=(N_s, N_s)
    where N_s is a multiple of num_steps_per_env × control_dt to align with rollout
    boundaries and keep physics constants stable within each training batch.
    """

    recompute = RecomputeLevel.set_const  # event_manager triggers set_const after __call__
    model_fields = ()  # no GPU fields written here

    def __init__(self, cfg, env):
        pass

    def __call__(self, env, env_ids, **kwargs):
        pass  # recompute handled by event_manager via .recompute attribute above


class reward_weight_linear:
    """Piecewise-linear interpolation of a reward weight over curriculum stages.

    Params (cfg.params):
        reward_name:  Single reward key (str). Mutually exclusive with reward_names.
        reward_names: List of reward keys sharing one weight schedule (list[str]).
                      Use when multiple rewards should be ramped in lockstep.
        weight_stages: List of {step, weight} dicts defining the ramp. Weight is held
                       constant before the first stage and after the last stage.
        decimation:   Control steps between updates (default: 480 = 20 iters).
                      Weight changes on 9.6s-of-iter timescales; finer granularity
                      is wasted recomputation.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        names = cfg.params.get("reward_names") or [cfg.params["reward_name"]]
        self.reward_term_cfgs = [env.reward_manager.get_term_cfg(n) for n in names]
        stages = cfg.params["weight_stages"]
        self.decimation = cfg.params.get("decimation", 480)

        self.steps = tuple(s["step"] for s in stages)
        self.weights = tuple(s["weight"] for s in stages)
        self.slopes = tuple(
            (self.weights[i + 1] - self.weights[i]) / (self.steps[i + 1] - self.steps[i])
            for i in range(len(stages) - 1)
        )

        self._last_idx = 0
        self._cached_weight = self.weights[0]
        self._return_tensor = torch.zeros(1, device=env.device)

    def __call__(self, env: ManagerBasedRlEnv, _env_ids, **_kwargs) -> torch.Tensor:
        step = env.common_step_counter
        if step % self.decimation == 0:
            # Fast path: cached interval is still valid
            i = self._last_idx
            if i < len(self.slopes) and self.steps[i] <= step < self.steps[i + 1]:
                self._cached_weight = self.weights[i] + (step - self.steps[i]) * self.slopes[i]
            else:
                # Slow path: search for current interval (runs on stage transitions only)
                weight = None
                for i in range(len(self.slopes)):
                    if self.steps[i] <= step < self.steps[i + 1]:
                        self._last_idx = i
                        weight = self.weights[i] + (step - self.steps[i]) * self.slopes[i]
                        break
                if weight is None:  # before first or after last stage
                    weight = self.weights[-1] if step >= self.steps[-1] else self.weights[0]
                self._cached_weight = weight

            for term_cfg in self.reward_term_cfgs:
                term_cfg.weight = self._cached_weight
            self._return_tensor[0] = self._cached_weight

        return self._return_tensor
