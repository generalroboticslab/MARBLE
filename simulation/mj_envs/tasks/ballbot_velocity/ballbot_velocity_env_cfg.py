"""Ballbot velocity tracking task environment configuration."""

from __future__ import annotations

from asset_zoo.ballbot import get_ballbot_robot_cfg, get_action_scale, SPHERE_BODY
from asset_zoo.ballbot.ballbot_constants import SLIDER_VEL_LIMIT
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from tasks.ballbot_velocity.action import RateLimitedJointPositionActionCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import (
    UniformNoiseCfg as Unoise,
    GaussianNoiseCfg as Gnoise,
    NoiseModelWithAdditiveBiasCfg,
)

from tasks.ballbot_velocity.shared import (
    commands_xy,
    base_lin_vel_world,
    base_ang_vel_world,
    WorldFrameVelocityCommandCfg,
    UnderwaterMujocoCfg,
    UNDERWATER_DENSITY,
    UNDERWATER_VISCOSITY,
    UNDERWATER_Z_WEIGHT,
)
from tasks.ballbot_velocity.shared import base_rotation_matrix
from tasks.ballbot_velocity.shared import (
    track_linear_velocity_relative,
    joint_pos_limits_slider,
)
from tasks.ballbot_velocity.shared import reward_weight_linear

SIMULATION_DT = 0.005
DECIMATION = 4  # 50 Hz control (sim_dt 0.005 * 4); was 8 (25 Hz)

# Command range, PER AXIS. The 0.5 here was a "conservative first run, tune upward once the
# reachable speed is known" placeholder that never got tuned, and it was the binding constraint
# on deployed speed: the policy tracks its command distribution, so a 0.5 ceiling caps it at
# roughly half the open-loop heuristic no matter how good the tracking is.
#
# Sized to the DEPLOYED command envelope, read off the robot (ballpi:~/repo/ballbot_control):
# ballbot_terminal.py MAX_COMMAND_SPEED_MPS = 0.7 is the hardest the operator can ask for, so
# training past 0.7 buys nothing — no hardware command will ever land there. The old 0.5 here
# happened to equal DEFAULT_COMMAND_SPEED_MPS, so the policy was trained exactly to the default
# and had no headroom when the operator turned it up.
#
# KNOWN REMAINING MISMATCH, not fixed here: lin_vel_x and lin_vel_y are sampled independently,
# so sim commands fill a SQUARE and most draws are well inside it (mean magnitude ~0.38 of the
# cap). The robot instead sends `speed` x a unit joystick direction — a CIRCLE of FIXED radius.
# So hardware asks for |v| = 0.5-0.7 on every single step while sim mostly trained on much
# slower commands, i.e. the deployed operating point is the rare case in the training
# distribution. Matching that properly means sampling a fixed magnitude with random heading.
# Raising the cap only widens the overlap; it does not close the gap.
MAX_LINEAR_VELOCITY_CMD = 0.7  # m/s per axis — hardware MAX_COMMAND_SPEED_MPS

# Reward weights.
WEIGHT_TRACK_LIN_VEL = 2.0
WEIGHT_JOINT_VEL_L2 = -0.002
WEIGHT_JOINT_POS_LIMITS = -1.0
WEIGHT_ACTION_RATE_L2 = -0.1
WEIGHT_ACTION_ACC_L2 = -0.01

# Underwater variant — drag-limited reachable speed. Ballbot in air is already
# capped at ~0.15 m/s² sustained accel; added density drag pulls the ceiling lower.
# In practice the policy will top out well below this.
BALLBOT_UNDERWATER_MAX_VEL_CMD = 0.2

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

# RSI (reference-state init) slider joints.
_RSI_SLIDER_JOINTS = ("base_link_Slider-5", "base_link_Slider-6", "base_link_Slider-7")

class BallbotUniformVelocityCommand(UniformVelocityCommand):
    """Velocity command with true world-frame vel_command_w for a freely-rotating sphere.

    Base class sets vel_command_w = vel_command_b at resample time (no heading rotation).
    For bipeds (consistent heading ≈ 0) this approximation holds. For a sphere with
    yaw randomised to ±π at reset, body frame is arbitrary → vel_command_w ends up in
    a random world direction → reward signal is corrupted every episode → standing still
    becomes the best achievable policy.

    Fix: rotate vel_command_b by the sphere's current heading at resample time so that
    vel_command_w is a stable, consistent world-frame target throughout the episode.
    """

    world_frame_command: bool = True  # nav_cmd from GUI is world-frame; receiver must rotate to body

    def __init__(self, cfg: BallbotUniformVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        # EMA of world-frame XY velocity for time-averaged tracking (V12+). Allocated only
        # when enabled (vel_ema_tau > 0) so V1-V11 carry no extra state/compute.
        if cfg.vel_ema_tau > 0.0:
            self.v_ema_w = torch.zeros(self.num_envs, 2, device=self.device)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        # Overwrite the incorrect vel_command_w = vel_command_b copy from base class.
        heading = self.robot.data.heading_w[env_ids]
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        vx_b = self.vel_command_b[env_ids, 0]
        vy_b = self.vel_command_b[env_ids, 1]
        self.vel_command_w[env_ids, 0] = cos_h * vx_b - sin_h * vy_b
        self.vel_command_w[env_ids, 1] = sin_h * vx_b + cos_h * vy_b
        self.vel_command_w[env_ids, 2] = self.vel_command_b[env_ids, 2]

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        """Keep vel_command_b in sync with vel_command_w for current heading.

        Base class only syncs for is_world_env envs (rel_world_envs > 0). Ballbot
        uses rel_world_envs=0, so vel_command_b would stay frozen at reset-time body
        frame. Stale vel_command_b corrupts commands_xy obs and the debug arrow. Fix:
        rotate vel_command_w into current body frame for all envs every step.
        """
        super()._update_command(env_ids)
        heading = self.robot.data.heading_w
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        vx_w = self.vel_command_w[:, 0]
        vy_w = self.vel_command_w[:, 1]
        # Inverse rotation: world → body (R^T)
        self.vel_command_b[:, 0] = cos_h * vx_w + sin_h * vy_w
        self.vel_command_b[:, 1] = -sin_h * vx_w + cos_h * vy_w

        # Time-averaged tracking (V12+): EMA of world-frame XY velocity, updated every step.
        # V13: command-scaled tau. With vel_ema_cmd_ref > 0, the averaging window scales with
        # command magnitude: tau_eff = tau * clamp(||cmd||/ref, max=1), floored at step_dt.
        # cmd~0 -> tau_eff=step_dt -> alpha=1 -> instantaneous (enforces stillness, undoing V12's
        # EMA stillness hole); cmd>=ref -> tau_eff=tau -> full burst averaging (cmd0.3 unchanged).
        if self.cfg.vel_ema_tau > 0.0:
            if self.cfg.vel_ema_cmd_ref > 0.0:
                cmd_norm = torch.linalg.vector_norm(self.vel_command_w[:, :2], dim=1, keepdim=True)
                scale = torch.clamp(cmd_norm / self.cfg.vel_ema_cmd_ref, max=1.0)
                tau_eff = torch.clamp(self.cfg.vel_ema_tau * scale, min=self._env.step_dt)
                alpha = self._env.step_dt / tau_eff
            else:
                alpha = self._env.step_dt / self.cfg.vel_ema_tau
            self.v_ema_w += alpha * (self.robot.data.root_link_lin_vel_w[:, :2] - self.v_ema_w)

    def _debug_vis_impl(self, visualizer) -> None:
        """Draw command and actual velocity arrows in world frame.

        Base class applies local_to_world(vel_command_b), which multiplies by the
        sphere's full 3D rotation matrix (roll + pitch + yaw). A rolling sphere
        accumulates roll/pitch so base_mat_w != R_yaw, causing the command arrow to
        rotate with the ball even though vel_command_w is constant. Fix: render
        vel_command_w directly — it is already in world frame, no rotation needed.
        """
        import numpy as np

        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        cmd_ws = self.vel_command_w.cpu().numpy()
        base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        lin_vel_ws = self.robot.data.root_link_lin_vel_w.cpu().numpy()

        scale = self.cfg.viz.scale
        z_offset = self.cfg.viz.z_offset

        for batch in env_indices:
            base_pos_w = base_pos_ws[batch]
            cmd_w = cmd_ws[batch]
            lin_vel_w = lin_vel_ws[batch]

            if np.linalg.norm(base_pos_w) < 1e-6:
                continue

            origin = base_pos_w + np.array([0.0, 0.0, z_offset * scale])

            # Command linear velocity (blue) — world frame, fixed as sphere rotates.
            visualizer.add_arrow(
                origin,
                origin + np.array([cmd_w[0], cmd_w[1], 0.0]) * scale,
                color=(0.2, 0.2, 0.6, 0.6),
                width=0.015,
            )

            # Actual linear velocity (cyan) — world frame.
            visualizer.add_arrow(
                origin,
                origin + np.array([lin_vel_w[0], lin_vel_w[1], 0.0]) * scale,
                color=(0.0, 0.6, 1.0, 0.7),
                width=0.015,
            )

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

    def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
        """Reference-state init (RSI): start moving envs already on the rolling branch.

        WHY: 9 RL runs (V1-V9, incl. PBS lean-shaping at 2 weights) all stand still. The
        open-loop probe proved a BIFURCATION at lean ~= 0.07 m: below it the ball only
        wobbles (~0.5 m drift), above it it rolls. The true reward is flat below the
        bifurcation and PBS provably cannot reward the transient exploration needed to
        cross it (it telescopes -> pays only for an episode that ENDS leaned). RSI is the
        init-distribution lever the reward cannot be: place the episode past the
        bifurcation so the policy only has to MAINTAIN rolling, never discover the crossing
        from rest. This isolates "is rolling a maintainable optimum?" from "can it be
        discovered?". Reward is unchanged -> not reward hacking (no fake optimum; the
        honest exp(-err^2) still pays ~1 only for actual command-velocity). The one analog
        hazard is coast-farming the free init velocity, but the verdict is eval-from-rest
        (RSI is train-only), which a coast-farmer fails -> cannot fool the gate.

        Runs ONLY on episode reset (this method is the reset-only entry; mid-episode timer
        resamples go through compute()->_resample and are untouched, so velocity is never
        teleported mid-episode). Gated on cfg.rsi_init_lean > 0; standing envs untouched.

        For each moving env: root world XY linear velocity := command; sliders pre-leaned
        to a world offset of magnitude rsi_init_lean along the command direction
        (offset_w = lean * cmd_dir_xy; offset_body = R_yaw^T offset_w; base is yaw-only at
        reset so this is exact).
        """
        extras = super().reset(env_ids)
        # Seed EMA at the fresh command (no startup transient; RSI moving envs truly start
        # at cmd velocity, standing envs at 0 = cmd). super().reset already resampled.
        if self.cfg.vel_ema_tau > 0.0:
            self.v_ema_w[env_ids] = self.vel_command_w[env_ids, :2]
        lean = getattr(self.cfg, "rsi_init_lean", 0.0)
        if lean <= 0.0:
            return extras
        if not hasattr(self, "_rsi_slider_ids"):
            ids, _ = self.robot.find_joints(list(_RSI_SLIDER_JOINTS), preserve_order=True)
            self._rsi_slider_ids = torch.tensor(ids, dtype=torch.long, device=self.device)

        ids = env_ids[~self.is_standing_env[env_ids]]
        if len(ids) == 0:
            return extras

        cmd_w = self.vel_command_w[ids, :2]                                # (M,2) world XY
        cmd_dir = cmd_w / (cmd_w.norm(dim=1, keepdim=True) + 1e-6)

        # Root velocity = command (world XY), vertical zero; preserve pose + ang vel.
        root_pos = self.robot.data.root_link_pos_w[ids]
        root_quat = self.robot.data.root_link_quat_w[ids]
        zeros_z = torch.zeros(len(ids), 1, device=self.device)
        root_lin_vel_w = torch.cat([cmd_w, zeros_z], dim=1)
        root_ang_vel_w = self.robot.data.root_link_ang_vel_w[ids]
        root_state = torch.cat([root_pos, root_quat, root_lin_vel_w, root_ang_vel_w], dim=-1)
        self.robot.write_root_state_to_sim(root_state, ids)

        # Pre-lean sliders: offset_w = lean*cmd_dir (z=0); rotate world->body by yaw.
        heading = self.robot.data.heading_w[ids]
        cos_h, sin_h = torch.cos(heading), torch.sin(heading)
        ox_w = lean * cmd_dir[:, 0]
        oy_w = lean * cmd_dir[:, 1]
        ox_b = cos_h * ox_w + sin_h * oy_w
        oy_b = -sin_h * ox_w + cos_h * oy_w
        # Slider base-frame axes (from model, see heuristic_policy.slider_axes_base):
        #   s5 = [0,0,-1]   s6 = [0,1,0]   s7 = [-1,0,0]
        # Inverse for offset=(ox_b,oy_b,oz_b=0): s5=-oz_b=0, s6=oy_b, s7=-ox_b.
        slider_targets = torch.stack([torch.zeros_like(oy_b), oy_b, -ox_b], dim=1)
        self.robot.write_joint_position_to_sim(
            slider_targets, joint_ids=self._rsi_slider_ids, env_ids=ids
        )
        return extras


class BallbotUniformVelocityCommandCfg(UniformVelocityCommandCfg):
    # RSI init lean (m). 0 = off (default); >0 = init moving envs leaned this far past rest.
    rsi_init_lean: float = 0.0
    # EMA time constant (s) for time-averaged velocity tracking. 0 = off (default).
    vel_ema_tau: float = 0.0
    # Command-norm reference (m/s) for command-scaled EMA tau. 0 = constant tau (default, V12).
    # >0 (V13): tau_eff = tau * clamp(||cmd||/ref, 1), so cmd~0 -> instantaneous tracking.
    vel_ema_cmd_ref: float = 0.0

    def build(self, env: ManagerBasedRlEnv) -> BallbotUniformVelocityCommand:
        return BallbotUniformVelocityCommand(self, env)


class BallbotIntegralErrorVelocityCommand(BallbotUniformVelocityCommand):
    """UniformVelocityCommand that maintains a world-frame bounded integral of tracking error.

    Inherits BallbotUniformVelocityCommand's _resample_command heading-rotation fix so that
    vel_command_w is a true world-frame target (not body-frame-at-resample-time).
    Supplies time pressure to resolve the low-speed/zero-velocity dead zone.
    """
    def __init__(self, cfg: BallbotIntegralErrorVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.integral_error = torch.zeros(self.num_envs, 3, device=self.device)
        self._v_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self._I_max = torch.tensor(cfg.I_max, device=self.device)
        self._dr_bias = torch.tensor(cfg.dr_bias, device=self.device)
        self._dr_noise = torch.tensor(cfg.dr_noise, device=self.device)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        # New command -> debt-free start
        self.integral_error[env_ids] = 0.0
        self._v_bias[env_ids] = (
            torch.rand(len(env_ids), 3, device=self.device) * 2.0 - 1.0
        ) * self._dr_bias

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        super()._update_command(env_ids)
        v_meas = torch.empty(self.num_envs, 3, device=self.device)
        v_meas[:, :2] = self.robot.data.root_link_lin_vel_w[:, :2]
        v_meas[:, 2] = self.robot.data.root_link_ang_vel_w[:, 2]
        v_meas = v_meas + self._v_bias + torch.randn_like(v_meas) * self._dr_noise
        e = self.vel_command_w - v_meas
        self.integral_error = torch.clamp(
            (1.0 - self.cfg.leak) * self.integral_error + e * self._env.step_dt,
            -self._I_max, self._I_max,
        )


from dataclasses import dataclass

@dataclass(kw_only=True)
class BallbotIntegralErrorVelocityCommandCfg(UniformVelocityCommandCfg):
    leak: float = 0.0
    I_max: tuple[float, float, float] = (0.5, 0.5, 0.5)
    dr_bias: tuple[float, float, float] = (0.02, 0.02, 0.005)
    dr_noise: tuple[float, float, float] = (0.05, 0.05, 0.01)

    def build(self, env: ManagerBasedRlEnv) -> BallbotIntegralErrorVelocityCommand:
        return BallbotIntegralErrorVelocityCommand(self, env)


def ballbot_integral_error_obs(env: ManagerBasedRlEnv, command_name: str = "twist") -> torch.Tensor:
    """Observation: XY components of world-frame integral error."""
    return env.command_manager.get_term(command_name).integral_error[:, :2]



def ballbot_velocity_env_cfg(
    play: bool = False,
    slider_mass_scale: float = 1.0,
    xml_path=None,
    num_steps_per_env: int = 4,
    curriculum_decimation: int = 1200,
) -> ManagerBasedRlEnvCfg:
    """Environment configuration for ballbot flat-terrain velocity tracking.

    Robot specifics:
      - 3 prismatic (slide) actuators.
      - No fell-over termination: sphere cannot topple; only time_out fires.
      - No contact sensors: sphere geom IS the hull — base contact is normal operation.
      - Observations and rewards are identical in structure (world-frame tracking).

    Args:
        slider_mass_scale: passed through to get_ballbot_robot_cfg (default 1.0
            = XML as-authored). See that function's docstring for rationale.
        xml_path: override robot MJCF (default None = ball_linear). Pass SHELL_XML_PATH
            for the on-water ring-shell hull.
    """
    robot_cfg = get_ballbot_robot_cfg(slider_mass_scale=slider_mass_scale, xml_path=xml_path)

    scene = SceneCfg(
        terrain=TerrainEntityCfg(terrain_type="plane"),
        num_envs=1 if play else 4096,
        env_spacing=2.5,
        entities={"robot": robot_cfg},
    )

    sim_cfg = SimulationCfg(
        nconmax=50,
        njmax=200,
        contact_sensor_maxmatch=32,
        mujoco=MujocoCfg(
            timestep=SIMULATION_DT,
            iterations=10,
            ls_iterations=20,
            ccd_iterations=150,
        ),
    )

    actions = {
        "joint_pos": RateLimitedJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=get_action_scale(),
            use_default_offset=True,
            vel_limit=SLIDER_VEL_LIMIT,
        )
    }

    commands = {
        "twist": BallbotUniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(2.0, 8.0),
            rel_standing_envs=0.1,
            rel_heading_envs=0.0,
            heading_command=False,
            debug_vis=True,
            ranges=BallbotUniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-MAX_LINEAR_VELOCITY_CMD, MAX_LINEAR_VELOCITY_CMD),
                lin_vel_y=(-MAX_LINEAR_VELOCITY_CMD, MAX_LINEAR_VELOCITY_CMD),
                ang_vel_z=(0.0, 0.0),  # sphere has no actuated yaw DOF
            ),
        )
    }

    # Observation structure. Total per-step: 23 dims.
    #   angular_velocity: 3, commands_xy: 2, dofPosition: 3, dofVelocity: 3,
    #   actions: 3, base_rotation_matrix: 9  →  23 × history_length=3 → (3, 23).
    policy_obs_terms = {
        "angular_velocity": ObservationTermCfg(
            func=base_ang_vel_world,
            noise=Gnoise(std=0.1),
        ),
        "commands_xy": ObservationTermCfg(
            func=commands_xy,
            params={"command_name": "twist"},
        ),
        "dofPosition": ObservationTermCfg(
            func=vel_mdp.joint_pos_rel,
            noise=NoiseModelWithAdditiveBiasCfg(
                noise_cfg=Unoise(n_min=-0.005, n_max=0.005),
                bias_noise_cfg=Unoise(n_min=-0.02, n_max=0.02),
                sample_bias_per_component=True,
            ),
        ),
        "dofVelocity": ObservationTermCfg(
            func=vel_mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "actions": ObservationTermCfg(func=vel_mdp.last_action),
        "base_rotation_matrix": ObservationTermCfg(
            func=base_rotation_matrix,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=policy_obs_terms,
            concatenate_terms=True,
            enable_corruption=True,
            history_length=3,
            flatten_history_dim=False,  # shape (3, 23) → RMA-CNN sequence encoder
        ),
        "critic": ObservationGroupCfg(
            terms=policy_obs_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=3,
            flatten_history_dim=True,  # flat (69,) for critic Q-net
        ),
        "estimator_target": ObservationGroupCfg(
            terms={"base_lin_vel": ObservationTermCfg(func=base_lin_vel_world)},
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=track_linear_velocity_relative,
            weight=WEIGHT_TRACK_LIN_VEL,
            params={
                "std_rel": 0.5,
                "std_min": 0.3,
                "command_name": "twist",
                "z_weight": 0.1,
            },
        ),
        "joint_vel_l2": RewardTermCfg(
            func=vel_mdp.joint_vel_l2,
            weight=WEIGHT_JOINT_VEL_L2,
        ),
        "joint_pos_limits": RewardTermCfg(
            func=joint_pos_limits_slider,
            weight=WEIGHT_JOINT_POS_LIMITS,
        ),
        "action_rate_l2": RewardTermCfg(
            func=vel_mdp.action_rate_l2,
            weight=WEIGHT_ACTION_RATE_L2,
        ),
        "action_acc_l2": RewardTermCfg(
            func=vel_mdp.action_acc_l2,
            weight=WEIGHT_ACTION_ACC_L2,
        ),
    }

    # Ramp action_rate_l2 from -0.05 -> -0.2 over training so the policy can build
    # an initial bang-bang gait before smoothness kicks in.
    reward_curriculum = {
        "action_rate_l2": CurriculumTermCfg(
            func=reward_weight_linear,
            params={
                "reward_name": "action_rate_l2",
                "decimation": curriculum_decimation,
                "weight_stages": [
                    {"step": 500 * num_steps_per_env, "weight": -0.05},
                    {"step": 3000 * num_steps_per_env, "weight": -0.2},
                ],
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(
            func=vel_mdp.time_out,
            time_out=True,
        ),
        # No base_contact termination: sphere hull geom is the contact surface —
        # floor contact is normal operation, not a failure mode.
    }

    from mjlab.viewer import ViewerConfig

    viewer_cfg = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name=SPHERE_BODY,
        distance=3.0,
        elevation=-5.0,
        azimuth=90.0,
    )

    return ManagerBasedRlEnvCfg(
        scene=scene,
        sim=sim_cfg,
        actions=actions,
        commands=commands,
        observations=observations,
        rewards=rewards,
        terminations=terminations,
        decimation=DECIMATION,
        episode_length_s=20.0,
        viewer=viewer_cfg,
        curriculum=reward_curriculum,
    )


def ballbot_underwater_velocity_env_cfg(
    play: bool = False, slider_mass_scale: float = 1.0
) -> ManagerBasedRlEnvCfg:
    """Ballbot underwater velocity tracking environment.

    Density drag (1000 kg/m³) caps the
    sphere's reachable speed well below the in-air ceiling; the cmd range is dropped
    to ±0.2 m/s to track the drag-limited target instead of an unreachable one.
    No buoyancy — MuJoCo's global inertia-based fluid model is DRAG-only (see
    UnderwaterMujocoCfg docstring); ballbot rests on the floor at full weight.

    V1 (slider_mass_scale=1.0) converged to standing-still: open-loop probe
    measured 0.026 m/s at 1x even with optimal
    asymmetric oscillation. slider_mass_scale raises the inertial force ceiling.

    STALE PREMISE, kept for provenance: that probe was set up believing the sliders
    were gravcomp=1 (neutrally buoyant), hence that quasi-static lean produced ZERO
    drive and only inertial reaction (F = M_slider * accel) could move the sphere.
    Both halves were wrong. MuJoCo gravcomp is per-body, so the tag covered only the
    0.1282 kg carriage of a 0.7086 kg slider (1.26 N of 6.95 N), the shell/complex
    hulls never carried it at all, and spec_fn now strips it everywhere. Lean DOES
    drive the sphere. Re-probe before trusting the numbers above.
    """
    cfg = ballbot_velocity_env_cfg(play=play, slider_mass_scale=slider_mass_scale)

    # mjwarp raises NotImplementedError if density/viscosity > 0 with implicitfast/implicit.
    # Fluid model requires euler integrator.
    cfg.sim.mujoco = UnderwaterMujocoCfg(
        timestep=SIMULATION_DT,
        integrator="euler",
        iterations=12,
        ls_iterations=25,
        ccd_iterations=150,
        density=UNDERWATER_DENSITY,
        viscosity=UNDERWATER_VISCOSITY,
    )

    cfg.commands["twist"].ranges.lin_vel_x = (
        -BALLBOT_UNDERWATER_MAX_VEL_CMD,
        BALLBOT_UNDERWATER_MAX_VEL_CMD,
    )
    cfg.commands["twist"].ranges.lin_vel_y = (
        -BALLBOT_UNDERWATER_MAX_VEL_CMD,
        BALLBOT_UNDERWATER_MAX_VEL_CMD,
    )

    cfg.rewards["track_linear_velocity"].params["z_weight"] = UNDERWATER_Z_WEIGHT

    return cfg
