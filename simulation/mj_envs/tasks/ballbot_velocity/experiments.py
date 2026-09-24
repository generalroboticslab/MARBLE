"""Experiment configurations for the Ballbot velocity tracking task."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from utils.experiments import BaseExperiment
from tasks.ballbot_velocity.shared import track_linear_velocity_ema

# Ballbot sealed-hull sphere radius for the on-water ring-cage variant. base_link_geom is a
# mesh; its AABB half-extent ≈0.19 m (probe of the compiled model), total robot mass 4.88 kg →
# a sealed sphere of this radius floats ~17 % submerged at equilibrium. Rings sit tangent
# OUTSIDE the hull (ring_radius = R + tube).
BALLBOT_SHELL_RADIUS = 0.19


class BallbotVel(BaseExperiment):
    """Ballbot velocity tracking baseline.

    FlashSAC config:
    sequence encoder + velocity estimator, C51 distributional critic, n-step
    returns, zeta-noise, PER-style Q-target. No events — fixed initial state,
    no external disturbances.
    """

    task = "ballbot_velocity"
    algo = "flash_sac"
    num_envs = 4096
    num_steps_per_env = 4
    curriculum_decimation: int = 1200

    def build_env_cfg(self, *, play=False, enable_corruption=True, enable_reward_curriculum=True):
        # Air base. Ring-cage / paddle variants add water in configure() over this cfg; the
        # fully-submerged underwater variant overrides build_env_cfg (BallbotVelUnderwaterV1).
        from tasks.ballbot_velocity.ballbot_velocity_env_cfg import ballbot_velocity_env_cfg
        return ballbot_velocity_env_cfg(
            play=play,
            num_steps_per_env=self.num_steps_per_env,
            curriculum_decimation=self.curriculum_decimation,
        )

    def flash_sac_configure(self, sac_cfg: object) -> None:
        super().flash_sac_configure(sac_cfg)
        sac_cfg.num_collect_steps = self.num_steps_per_env
        sac_cfg.num_updates = 3

        sac_cfg.use_sequence_encoder = True
        sac_cfg.use_velocity_estimator = True

        sac_cfg.log_std_min = -3.0
        sac_cfg.normalize_reward = True
        sac_cfg.normalized_G_max = 5.0
        sac_cfg.v_min = -5.0
        sac_cfg.v_max = 5.0
        sac_cfg.num_atoms = 101
        sac_cfg.target_sigma = 0.15

        sac_cfg.sample_chunk_size = 2
        sac_cfg.num_steps = 3
        sac_cfg.alpha_init = 0.01
        sac_cfg.alpha_learning_rate = 2e-4
        sac_cfg.tau = 0.01
        sac_cfg.use_zeta_noise = True
        sac_cfg.zeta_mu = 2.0
        sac_cfg.zeta_max_n = 16
        sac_cfg.buffer_size = 256
        sac_cfg.batch_size = 4096
        sac_cfg.use_per_q_target = True


class BallbotVelRingCage(BallbotVel):
    """Ballbot velocity, floating ON water inside a 13-RING armillary cage (fin-free).

    Built ENTIRELY
    in configure() — no new env-cfg function or run.py task — by mutating the AIR ballbot_velocity
    cfg (task stays "ballbot_velocity", default implicitfast sim; the custom force term writes
    xfrc_applied, so no euler integrator needed). configure() runs after base_env builds the cfg
    but before ManagerBasedRlEnv is constructed, so the action-term + spec mutations take effect.

    Transform applied:
      - add a translucent, non-colliding water surface plane at z=WATER_LEVEL_Z;
      - respawn the hull at float equilibrium (no drop transient);
      - install RingCageSurfaceForce as the "buoyancy" action term: Archimedes buoyancy +
        CoB metacentric restoring torque + wetted quadratic hull drag + depth-gated
        cross-tangent drag on the submerged arcs of 13 octahedral great-circle rings;
      - add the matching 13 transparent torus ring visuals (massless, contype=0).

    Hull modeled as a SEALED sphere R=BALLBOT_SHELL_RADIUS=0.19 m (base_link_geom mesh AABB
    half-extent ≈0.19; total mass 4.88 kg → ~17 % submerged at equilibrium). Rings sit tangent
    OUTSIDE the hull (ring_radius = R + tube). Rings are drag-only + visual (massless) → the
    float setpoint is unchanged. Reward/obs/command/FlashSAC config inherited unchanged.

    Gate 0 (RESOLVED 2026-06-24): the ballbot drives by shifting
    internal SLIDER masses, NOT by continuously spinning the hull. The ring cage rectifies HULL SPIN into thrust, so the open risk was that a
    slider-driven hull might not spin appreciably. Probe of the trained policy: hull spins ω≈1.8–2.8
    rad/s (slider oscillation induces sustained spin) and rings produce the dominant horizontal
    thrust (~0.4–0.6 N) vs ~0.2–0.4 N restoring drag → net forward, tracks ~0.17–0.27 m/s. Same
    propulsion regime as the spinny, NOT an inertial-swimming artifact. CAVEAT: the net margin is
    thin (~0.2 N) and the thrust is harvested partly in the surface-piercing regime — see the
    ventilation knockdown in RingCageSurfaceForce; zero-shot deploy still needs tank calibration.
    """

    # Effective spherical buoyancy radius of the hull. Default = the ball_linear sealed sphere.
    # The shell variant (taller/heavier hull) overrides this so float depth + ring geometry match.
    shell_radius = BALLBOT_SHELL_RADIUS

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)

        R = self.shell_radius
        from asset_zoo.ballbot import SPHERE_BODY
        from tasks.ballbot_velocity.shared import RingCageSurfaceForceCfg
        from tasks.ballbot_velocity.shared import (
            _solve_float_depth,
            _add_water_surface_visual,
            _add_ring_torus_visuals,
            WATER_LEVEL_Z,
            SURFACE_WATER_DENSITY,
            SURFACE_GRAVITY,
            SPHERE_DRAG_CD,
            N_RINGS,
            SEGS_PER_RING,
            RING_TUBE_RADIUS,
            RING_DRAG_CD,
        )

        ring_radius = R + RING_TUBE_RADIUS  # tube tangent-outside the hull

        robot_cfg = env.scene.entities["robot"]
        total_mass = float(robot_cfg.spec_fn().compile().body_mass.sum())
        h_eq = _solve_float_depth(R, total_mass, SURFACE_WATER_DENSITY)
        robot_cfg.init_state.pos = (0.0, 0.0, WATER_LEVEL_Z + (R - h_eq))

        _add_water_surface_visual(env.scene, WATER_LEVEL_Z)

        env.actions["buoyancy"] = RingCageSurfaceForceCfg(
            entity_name="robot",
            body_name=SPHERE_BODY,
            radius=R,
            water_level_z=WATER_LEVEL_Z,
            water_density=SURFACE_WATER_DENSITY,
            drag_coef=SPHERE_DRAG_CD,
            gravity=SURFACE_GRAVITY,
            n_rings=N_RINGS,
            segs_per_ring=SEGS_PER_RING,
            ring_radius=ring_radius,
            ring_tube_radius=RING_TUBE_RADIUS,
            ring_drag_coef=RING_DRAG_CD,
            thrust_scale_range=getattr(self, "thrust_scale_range", None),
        )
        _add_ring_torus_visuals(robot_cfg, RING_TUBE_RADIUS, ring_radius)


class BallbotVelRingCageDR(BallbotVelRingCage):
    """Ring-cage + per-episode thrust-scale domain randomization (zero-shot robustness).

    Single change vs BallbotVelRingCage: thrust_scale_range=(0.6, 1.0) per-episode multiplier on
    the ring wrench, modelling the lumped UNMODELLED optimistic losses (ring-ring/node shadowing,
    induced wake, added-mass) that only REDUCE real thrust. Scales rings only; buoyancy/hull drag stay deterministic.
    (Ventilation is no longer lumped here — RingCageSurfaceForce models it structurally via the
    depth-Froude knockdown; this scalar hedges only the remaining magnitude losses.)
    """

    thrust_scale_range = (0.6, 1.0)


def _add_drive_contact_dr(env: ManagerBasedRlEnvCfg) -> None:
    """Randomize the actuator and the ground contact, in place on `env.events`.

    These four terms used to live inline in BallbotVelRingCageFlatDRLatency.configure only, so
    every water-lineage class trained against a PINNED drive: exactly 15.708 N, exactly nominal
    kp/kd, exactly 0.2 N of rail friction. That was an inheritance accident, not a decision —
    BallbotVelRingCageShellDR.configure predates the block — and it confounded every land/water
    comparison, since the two differed in DR coverage as well as in medium. Hoisted here so both
    lineages call one definition and cannot drift apart again.

    effort_limits / pd_gains / rail_frictionloss are properties of the drive and do not care
    whether the hull is wet. hull_friction is arguably land-specific, but both lineages roll on
    a contact patch and both name that geom `base_link_geom`, so it is included rather than
    special-cased; drop it here if a water variant ever stops touching the floor.
    """
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.tasks.velocity import mdp as vel_mdp
    from asset_zoo.ballbot.ballbot_constants import EFFORT_PEAK_SCALE

    env.events.update({
        # The sphere's single contact patch is its entire drivetrain, so this is the
        # highest-leverage physics term on this robot. Startup mode because a given
        # floor's friction does not change between episodes. Axis 0 (sliding) only —
        # geom_friction defaults to default_axes=[0], leaving torsional/rolling pinned.
        "hull_friction": EventTermCfg(
            mode="startup",
            func=vel_mdp.dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=(r"^base_link_geom$",)),
                "operation": "abs",
                "ranges": (0.2, 0.6),
            },
        ),
        # Wide: the deployed reference filter is a hand-tuned
        # second-order smoother, not a calibrated match to these gains.
        "pd_gains": EventTermCfg(
            mode="reset",
            func=vel_mdp.dr.pd_gains,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": (0.8, 1.2),
                "kd_range": (0.8, 1.2),
            },
        ),
        # Spans the trusted torque band: _MOTOR_SPEC.effort_limit is 0.50 N·m reflected
        # (31.42 N) and EFFORT_PEAK_SCALE = 0.60/0.50, so the upper bound is 0.60 N·m =
        # 37.70 N. Set from how the drive is actually run, not from the datasheet.
        "effort_limits": EventTermCfg(
            mode="reset",
            func=vel_mdp.dr.effort_limits,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "effort_limit_range": (1.0, EFFORT_PEAK_SCALE),
            },
        ),
        # Was (0.0, 0.4) N, which is ~35x too low. Hardware shows a slider sometimes holding
        # itself against gravity and sometimes not, so real breakaway sits at roughly the
        # slider's own axial weight: moving mass 0.7086 kg -> 6.95 N (measured per slider off
        # the compiled model, not the 3-slider total). The band is deliberately WIDE rather
        # than pinned at 6.95: breakaway is inferred from a qualitative "sometimes slides"
        # observation, never measured. A narrow high band would let the policy lean on
        # Coulomb friction as free damping and then chatter on a rail that happens to be
        # slicker; spanning 1-7 N forces a gait that is smooth across the whole bracket.
        # Replace with a measured value once a breakaway ramp is run.
        "rail_frictionloss": EventTermCfg(
            mode="reset",
            func=vel_mdp.dr.dof_frictionloss,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"base_link_Slider-\d+",)),
                "operation": "abs",
                "ranges": (1.0, 7.0),
            },
        ),
    })


class BallbotVelRingCageShellDR(BallbotVelRingCageDR):
    """Ring-cage + thrust DR, TRAINED on the ring-SHELL hull (RL sibling of the heuristic).

    Two changes vs BallbotVelRingCageDR:
      1. robot MJCF = asset/ball_linear_shell/Sim_Model.xml (true 5.43 kg shell mass/inertia,
         identical slider axes + joint names), with shell_radius = 0.193 (sealed hull 386 mm
         dia) so float depth and the ring geometry match the real hull.
      2. collision = ANALYTIC ring-cage sphere, not the shell mesh. The shell mesh is an open
         armillary cage; MuJoCo collides meshes by convex hull, so the compiled contact is an
         arbitrary hull blob whose radius/centre are whatever the CAD export happened to be —
         not the ring envelope the RingCageSurfaceForce drag model assumes. Replace it with a
         sphere of radius shell_radius + RING_TUBE_RADIUS (= the ring_radius passed to the drag
         term), leaving the mesh visual-only. Density 0: the geom must not perturb total mass, which
         BallbotVelRingCage.configure() uses to solve float equilibrium.

    Contact is near-idle on water (hull floats ~17 % submerged) but is load-bearing on
    beach/wave-run-up contact and on any respawn transient, so it must be the ring envelope.
    """

    shell_radius = 0.193

    def build_env_cfg(self, *, play=False, enable_corruption=True, enable_reward_curriculum=True):
        import mujoco
        from dataclasses import replace
        from mjlab.utils.spec_config import CollisionCfg
        from tasks.ballbot_velocity.ballbot_velocity_env_cfg import ballbot_velocity_env_cfg
        from tasks.ballbot_velocity.shared import RING_TUBE_RADIUS
        from asset_zoo.ballbot import SHELL_XML_PATH, SPHERE_BODY

        cfg = ballbot_velocity_env_cfg(play=play, xml_path=SHELL_XML_PATH)
        robot_cfg = cfg.scene.entities["robot"]
        r_collision = self.shell_radius + RING_TUBE_RADIUS

        # Wrap (not replace) the asset spec_fn: it recenters the body frame on the hull AABB
        # centre first, so a geom appended here at pos=0 sits exactly at the shell centre.
        inner_spec_fn = robot_cfg.spec_fn

        def spec_fn() -> "mujoco.MjSpec":
            spec = inner_spec_fn()
            spec.body(SPHERE_BODY).add_geom(
                name="base_link_collision",
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[r_collision, 0.0, 0.0],
                density=0.0,             # massless — float-depth solve reads total body mass
                group=3,                 # collision group; hidden in default render
                rgba=[0.5, 0.5, 0.5, 0.5],
            )
            return spec

        robot_cfg.spec_fn = spec_fn
        # disable_other_geoms defaults True → the shell mesh drops to contype/conaffinity 0.
        robot_cfg.collisions = (
            CollisionCfg(
                geom_names_expr=(r"^base_link_collision$",),
                contype=1,
                conaffinity=1,
                condim=4,
                priority=0,
                friction=(0.3, 0.002, 0.00005),
            ),
        )

        # Sensor delay. Same sharing-of-term-dict precedent as FlatDRLatency: writes hit both
        # actor and critic views.
        terms = cfg.observations["actor"].terms
        for name in DELAYED_OBS_TERMS:
            terms[name] = replace(
                terms[name],
                delay_min_lag=OBS_DELAY_STEPS[0],
                delay_max_lag=OBS_DELAY_STEPS[1],
                delay_update_period=OBS_DELAY_UPDATE_PERIOD,
            )
            assert terms[name].delay_max_lag == OBS_DELAY_STEPS[1]

        # Command delay, applied to this env's actuator cfgs so the shared ballbot asset
        # constants stay untouched for every other ballbot task.
        robot = cfg.scene.entities["robot"]
        robot.articulation = replace(
            robot.articulation,
            actuators=tuple(
                replace(
                    actuator,
                    delay_min_lag=ACTUATOR_DELAY_STEPS[0],
                    delay_max_lag=ACTUATOR_DELAY_STEPS[1],
                    delay_update_period=ACTUATOR_DELAY_UPDATE_PERIOD,
                )
                for actuator in robot.articulation.actuators
            ),
        )
        assert all(
            a.delay_max_lag == ACTUATOR_DELAY_STEPS[1]
            for a in cfg.scene.entities["robot"].articulation.actuators
        )
        return cfg

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        """Stronger action smoothness rewards to match the runtime virtual-trajectory PD smoother on hardware."""
        super().configure(env, agent)

        from mjlab.managers.event_manager import EventTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        from mjlab.tasks.velocity import mdp as vel_mdp
        from tasks.ballbot_velocity.shared import DeferredModelFieldsWrapper, PeriodicPhysicsRecompute
        from asset_zoo.ballbot import SPHERE_BODY

        # Domain randomization (mass + COM + damping).
        # Deferred mass/COM writes are batched and flushed
        # by the physics_recompute helper below (480 * DECIMATION(4) * SIM_DT(0.005) = 9.6 s).
        env.events.update({
            "body_mass": EventTermCfg(
                mode="interval",
                is_global_time=True,
                interval_range_s=(9.6, 9.6),
                func=DeferredModelFieldsWrapper(vel_mdp.dr.body_mass),
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=(r".*",)),
                    "operation": "scale",
                    "ranges": (0.85, 1.15),
                },
            ),
            "com_displacement": EventTermCfg(
                mode="interval",
                is_global_time=True,
                interval_range_s=(9.6, 9.6),
                func=DeferredModelFieldsWrapper(vel_mdp.dr.body_com_offset),
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=(SPHERE_BODY,)),
                    "operation": "add",
                    "ranges": (-0.01, 0.01),
                },
            ),
            "joint_damping": EventTermCfg(
                mode="reset",
                func=vel_mdp.dr.dof_damping,
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(r"base_link_Slider-\d+",)),
                    "operation": "add",
                    "ranges": (0.0, 0.01),
                },
            ),
            "physics_recompute": EventTermCfg(
                mode="interval",
                is_global_time=True,
                interval_range_s=(9.6, 9.6),
                func=PeriodicPhysicsRecompute,
            ),
        })
        _add_drive_contact_dr(env)
        assert {"hull_friction", "pd_gains", "effort_limits", "rail_frictionloss"} <= set(env.events)

        # Bump accel-penalty so the policy doesn't trade one big jerk for a stream of smaller ones.
        # action_rate_l2 is now handled by the reward curriculum in build_env_cfg.
        env.rewards["action_acc_l2"].weight = -0.05
        assert env.rewards["action_acc_l2"].weight == -0.05


# Command latency, in PHYSICS timesteps (SIMULATION_DT = 0.005 s), applied every substep.
# ballbot_control logs of 2026-08-04 measured target -> slider at 280-337 ms, but most of that
# was a deployment bug (the reference filter integrated a hardcoded 10 ms dt while the loop ran
# at 10.6-16.5 ms) and is being fixed in software. What remains is transport: CAN readback
# measured at 26-56 ms, plus one control step of target pipeline. The two candidate shipped
# plants are ~3 control steps (policy target straight to the drive) and ~6 (software reference
# filter retained), so 4-16 physics steps = 20-80 ms covers both.
ACTUATOR_DELAY_STEPS = (4, 16)
ACTUATOR_DELAY_UPDATE_PERIOD = 4

# Sensor latency, in CONTROL steps (20 ms each at DECIMATION=4): age of the CAN joint feedback
# and the IMU frame when the policy reads them.
OBS_DELAY_STEPS = (0, 2)
OBS_DELAY_UPDATE_PERIOD = 6
# commands_xy and actions are known exactly on hardware, so they stay undelayed.
DELAYED_OBS_TERMS = ("dofPosition", "dofVelocity", "angular_velocity", "base_rotation_matrix")


class BallbotVelRingCageFlatDRLatency(BallbotVel):
    """Flatground sibling of BallbotVelRingCageDR. The single canonical flat-ground class —
    superseded the historical FlatDR (no DR) and FlatDRV2 (joint_pos_limits -5.0) siblings.

    Same BallbotVel FlashSAC defaults as the ring-cage experiment, but builds the plain
    ballbot_velocity env on the ground plane (no water/ring-cage transform). Adds:
      - command latency DR (ACTUATOR_DELAY_STEPS 4-16 physics steps)
      - sensor latency DR (OBS_DELAY_STEPS 0-2 control steps)
      - body_mass + com_displacement + joint_damping randomization (DeferredModelFieldsWrapper,
        physics_recompute every 9.6 s)
      - action_rate_l2 curriculum (-0.05 -> -0.2 over 500-3000 env steps)
      - action_acc_l2 static bump (-0.05)

    Two distinct latencies, two distinct knobs — conflating them is the easy mistake here:
      - command latency is actuator-side, units of PHYSICS timesteps, applied every substep
      - sensor latency is observation-side, units of CONTROL steps, applied per env step

    Physics DR covers the mismatches that were actually measured on the robot: slider force
    spans the motor's 0.25 N·m continuous rating to its 0.68 N·m transient peak (15.71-42.73 N
    reflected), and rail friction is a hand-set 0.2 N carrying a "TUNE to hardware" note that
    was never actioned. Hull friction matters more on this robot than on a legged one — a
    rolling sphere has exactly one contact patch and it is the entire drivetrain.

    Hardware zero-shot transfer motivated this class (ballbot_control logs 2026-08-04:
    0.29-0.44 m/s command error against 0.03-0.10 in sim, 23 m of path for 3 m of net
    displacement under the FlatDR-perfect-plant prior).
    """

    def build_env_cfg(self, *, play=False, enable_corruption=True, enable_reward_curriculum=True):
        from dataclasses import replace

        cfg = super().build_env_cfg(
            play=play,
            enable_corruption=enable_corruption,
            enable_reward_curriculum=enable_reward_curriculum,
        )

        # Sensor delay. ballbot_velocity_env_cfg hands the SAME term dict to the actor and the
        # critic group, so writing into it delays both views. Deliberately symmetric,
        # no separate undelayed critic view.
        terms = cfg.observations["actor"].terms
        for name in DELAYED_OBS_TERMS:
            terms[name] = replace(
                terms[name],
                delay_min_lag=OBS_DELAY_STEPS[0],
                delay_max_lag=OBS_DELAY_STEPS[1],
                delay_update_period=OBS_DELAY_UPDATE_PERIOD,
            )
            assert terms[name].delay_max_lag == OBS_DELAY_STEPS[1]

        # Command delay, applied to this env's actuator cfgs so the shared ballbot asset
        # constants stay untouched for every other ballbot task.
        robot = cfg.scene.entities["robot"]
        robot.articulation = replace(
            robot.articulation,
            actuators=tuple(
                replace(
                    actuator,
                    delay_min_lag=ACTUATOR_DELAY_STEPS[0],
                    delay_max_lag=ACTUATOR_DELAY_STEPS[1],
                    delay_update_period=ACTUATOR_DELAY_UPDATE_PERIOD,
                )
                for actuator in robot.articulation.actuators
            ),
        )
        assert all(
            a.delay_max_lag == ACTUATOR_DELAY_STEPS[1]
            for a in cfg.scene.entities["robot"].articulation.actuators
        )
        return cfg

    def configure(self, env, agent):
        super().configure(env, agent)
        # joint_pos_limits stays at the env-cfg default -1.0, matching the aquatic classes.
        # The historical -5.0 bump here was preemptive: the -1.0 term logged 0.0000 @15000,
        # so it never engaged, and the stronger weight risked clipping slider exploration.
        from mjlab.managers.event_manager import EventTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        from mjlab.tasks.velocity import mdp as vel_mdp
        from tasks.ballbot_velocity.shared import DeferredModelFieldsWrapper, PeriodicPhysicsRecompute
        from asset_zoo.ballbot import SPHERE_BODY
        from asset_zoo.ballbot.ballbot_constants import EFFORT_PEAK_SCALE

        # events defaults to {"reset_scene_to_default": ...}; update, never replace.
        _add_drive_contact_dr(env)
        env.events.update({
            # Domain randomization (mass + COM + damping).
            # Deferred mass/COM writes are batched and flushed
            # by the physics_recompute helper below (480 * DECIMATION(4) * SIM_DT(0.005) = 9.6 s).
            "body_mass": EventTermCfg(
                mode="interval",
                is_global_time=True,
                interval_range_s=(9.6, 9.6),
                func=DeferredModelFieldsWrapper(vel_mdp.dr.body_mass),
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=(r".*",)),
                    "operation": "scale",
                    "ranges": (0.85, 1.15),
                },
            ),
            "com_displacement": EventTermCfg(
                mode="interval",
                is_global_time=True,
                interval_range_s=(9.6, 9.6),
                func=DeferredModelFieldsWrapper(vel_mdp.dr.body_com_offset),
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=(SPHERE_BODY,)),
                    "operation": "add",
                    "ranges": (-0.01, 0.01),
                },
            ),
            "joint_damping": EventTermCfg(
                mode="reset",
                func=vel_mdp.dr.dof_damping,
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(r"base_link_Slider-\d+",)),
                    "operation": "add",
                    "ranges": (0.0, 0.01),
                },
            ),
            "physics_recompute": EventTermCfg(
                mode="interval",
                is_global_time=True,
                interval_range_s=(9.6, 9.6),
                func=PeriodicPhysicsRecompute,
            ),
        })
        assert {"hull_friction", "pd_gains", "effort_limits", "rail_frictionloss",
                "body_mass", "com_displacement", "joint_damping", "physics_recompute"} <= set(env.events)

        # Bump accel-penalty so the policy doesn't trade one big jerk for a stream of smaller ones.
        # action_rate_l2 is now handled by the reward curriculum in build_env_cfg.
        env.rewards["action_acc_l2"].weight = -0.05
        assert env.rewards["action_acc_l2"].weight == -0.05


def _use_complex_hull(cfg: ManagerBasedRlEnvCfg, collision_radius: float) -> None:
    """Swap the robot in `cfg` to the refined full-assembly hull, in place.

    Retargets only the three EntityCfg fields that describe the ROBOT (spec, spawn height,
    collision) and leaves everything the caller already configured on it — notably the
    actuator latency DR, which lives in `articulation.actuators` — untouched. Replacing the
    whole EntityCfg instead would silently drop that.
    """
    from asset_zoo.ballbot import get_ballbot_complex_robot_cfg

    complex_cfg = get_ballbot_complex_robot_cfg(collision_radius)
    robot = cfg.scene.entities["robot"]
    robot.spec_fn = complex_cfg.spec_fn
    robot.init_state = complex_cfg.init_state
    robot.collisions = complex_cfg.collisions


class BallbotVelComplexFlatDRLatency(BallbotVelRingCageFlatDRLatency):
    """LAND half of the ball_linear_complex pair — flat ground, refined full-assembly hull.

    Single change vs BallbotVelRingCageFlatDRLatency: the robot MJCF becomes
    `asset/ball_linear_complex/Sim_Model.xml` (with its 342 welded parts fused and an
    analytic 0.194 m hull sphere installed — see get_ballbot_complex_robot_cfg). Every
    reward, command, latency and physics-DR knob is inherited unchanged, so this is a
    like-for-like asset retrain against the 2026-08-29 land baseline, not a new lever.

    What actually changes in the physics: total mass 5.2579 kg (shell 5.2698, ball_linear
    ~4.88) and a mass distribution derived from 342 individually-modelled parts rather than
    one lumped inertial — i.e. a truthful inertia tensor and COM. Hull radius 0.194 vs
    ball_linear's 0.19.

    The inherited `hull_friction` DR event targets `^base_link_geom$`, which is why the
    analytic sphere reuses that name — the event fires here exactly as it does on ball_linear.
    """

    def build_env_cfg(self, *, play=False, enable_corruption=True, enable_reward_curriculum=True):
        from asset_zoo.ballbot import COMPLEX_HULL_RADIUS

        cfg = super().build_env_cfg(
            play=play,
            enable_corruption=enable_corruption,
            enable_reward_curriculum=enable_reward_curriculum,
        )
        _use_complex_hull(cfg, COMPLEX_HULL_RADIUS)
        return cfg


class BallbotVelRingCageComplexDR(BallbotVelRingCageShellDR):
    """WATER half of the ball_linear_complex pair — 13-ring cage, refined full-assembly hull.

    Single change vs BallbotVelRingCageShellDR: the robot MJCF becomes ball_linear_complex.
    `shell_radius` moves 0.193 -> 0.194 to match the measured hull, so the float-depth solve
    and the ring geometry track the real surface; the collision sphere stays at
    `shell_radius + RING_TUBE_RADIUS` (the ring envelope the drag term models), per the
    parent's reasoning. Thrust-scale DR, latency DR, mass/COM/damping DR all inherited.

    READ BEFORE LAUNCHING: ballbot-on-water was declared
    physically dead at buildable hardware — slider TRAVEL is the only lever that ever broke
    0.1 m/s and it is capped by the sphere radius. 0.194 m is not materially bigger than the
    0.19 m that verdict was reached on, so this run is an asset refresh (the prior checkpoint
    is physics-stale), NOT a retest of a rejected lever. Expect `error_vel_xy` near the
    2026-08-29 shell figure of 0.749; treat a large improvement as suspicious, not as news.
    """

    shell_radius = 0.194

    def build_env_cfg(self, *, play=False, enable_corruption=True, enable_reward_curriculum=True):
        from tasks.ballbot_velocity.shared import RING_TUBE_RADIUS

        cfg = super().build_env_cfg(
            play=play,
            enable_corruption=enable_corruption,
            enable_reward_curriculum=enable_reward_curriculum,
        )
        _use_complex_hull(cfg, self.shell_radius + RING_TUBE_RADIUS)
        return cfg


class BallbotVelRingCageComplexSmooth(BallbotVelRingCageComplexDR):
    """WATER sibling of BallbotVelComplexFlatSmooth, but at -0.35, NOT the land class's -0.6.

    action_rate_l2 end weight -0.2 -> -0.35. The land run proved -0.6 is past the useful point:
    sim tracking looked untouched (1.504 vs 1.509) but on hardware the slider visibly slowed,
    i.e. sim's track_linear_velocity does not see the cost that matters. -0.35 splits the
    interval so there is a middle rung between the -0.2 parent and a setting already known to
    over-damp. Treat -0.6 as the upper bound of the bracket, not as a candidate.

    Water is the more fragile side of this lever: ring-cage thrust comes from sustained hull
    SPIN, so the gait is inherently higher action-rate than a rolling one, and over-penalizing
    rate here removes the propulsion mechanism rather than merely smoothing it.

    Same curriculum trap as the land class — see BallbotVelComplexFlatSmooth.
    """

    action_rate_end_weight = -0.35

    def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg) -> None:
        super().configure(env, agent)

        stages = env.curriculum["action_rate_l2"].params["weight_stages"]
        assert stages[-1]["weight"] == -0.2, f"unexpected parent end weight: {stages}"
        stages[-1]["weight"] = self.action_rate_end_weight


class BallbotVelHeuristic(BallbotVel):
    """Geometric heuristic controller on the flat-ground ballbot env — NOT trained.

    Sim port of HeuristicPolicy (ballbot_control/src/ballbot_runtime.py): places the
    3-slider mass offset along the world velocity command (in the rolling base frame) so
    the sphere leans and rolls toward it. Supplies a scripted policy(obs)->action via the
    build_policy seam; run with `--agent experiment`, no checkpoint needed:

        python mj_envs/run.py play --task BallbotVelHeuristic --agent experiment

    Inherits BallbotVel's flat ballbot_velocity env (no water). Optional
    `lean_gain` attribute switches the lean law to proportional (see heuristic_slider_action).
    """

    lean_gain = 0.0

    def build_policy(self, env):
        from tasks.ballbot_velocity.heuristic_policy import BallbotHeuristicPolicy
        return BallbotHeuristicPolicy(env, lean_gain=self.lean_gain)


class BallbotVelComplexHeuristic(BallbotVelHeuristic):
    """Geometric heuristic controller on LAND with the refined ball_linear_complex hull.

    Complex-hull sibling of BallbotVelShellHeuristic: same scripted HeuristicPolicy, flat
    ground, but the robot MJCF becomes asset/ball_linear_complex/Sim_Model.xml with its 342
    welded parts fused and the analytic 0.194 m hull sphere installed. Run with
    --agent experiment, no checkpoint:

        python mj_envs/run.py play --task BallbotVelComplexHeuristic --agent experiment

    Uses _use_complex_hull (not a bare xml_path swap like the shell variant) because the
    complex export needs the fusion + analytic-sphere spec_fn from
    get_ballbot_complex_robot_cfg; loading its raw MJCF gives 399 bodies and 8 cap meshes.
    Slider joint names/axes are unchanged from ball_linear, so the heuristic re-derives its
    _M from this model and the constant holds.
    """

    def build_env_cfg(self, *, play=False, enable_corruption=True, enable_reward_curriculum=True):
        from asset_zoo.ballbot import COMPLEX_HULL_RADIUS

        cfg = super().build_env_cfg(
            play=play,
            enable_corruption=enable_corruption,
            enable_reward_curriculum=enable_reward_curriculum,
        )
        _use_complex_hull(cfg, COMPLEX_HULL_RADIUS)
        return cfg
