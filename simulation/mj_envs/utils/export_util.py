import os
import re
import torch
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg



class DeployedPolicyRMAEstimator(torch.nn.Module):
    """Deployment wrapper for ActorCriticRMAEstimator.

    Same forward() as DeployedPolicyHistory plus forward_with_estimation()
    which returns (actions, est_vel) for the high-level navigation policy.

    Args:
        actor_critic: Trained ActorCriticRMAEstimator instance.
    """

    def __init__(self, actor_critic: torch.nn.Module):
        super().__init__()
        self.normalizer  = actor_critic.actor_obs_normalizer       # type: ignore[attr-defined]
        self.backbone    = actor_critic.backbone                   # type: ignore[attr-defined]
        self.vel_head    = actor_critic.vel_head                   # type: ignore[attr-defined]
        self.actor_head  = actor_critic.actor                      # type: ignore[attr-defined]

    def _encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (trunk_feat, encoder_latent). Latent needed for vel_head."""
        B, S, D = obs.shape
        norm = self.normalizer(obs.reshape(B * S, D)).view(B, S, D)
        latent = self.backbone.encoder(norm)                       # type: ignore[operator]
        feat = torch.cat([latent, norm[:, -1, :]], dim=-1)
        trunk_feat = self.backbone.trunk(feat)                     # type: ignore[operator]
        return trunk_feat, latent

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        trunk_feat, _ = self._encode(obs)
        return self.actor_head(trunk_feat)

    @torch.jit.export
    def forward_with_estimation(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (actions, est_vel). est_vel feeds the high-level navigation policy."""
        trunk_feat, latent = self._encode(obs)
        return self.actor_head(trunk_feat), self.vel_head(latent)




class DeployedPolicyFlashSAC(torch.nn.Module):
    """Deployed policy for FlashSAC: normalizes obs, calls actor, returns the squashed
    deterministic action (tanh(mean)*scale + bias) — parity with the in-process inference path.

    Replicates FlashSACRunner._normalize_actor_obs: flatten any sequence dims first,
    normalize per-frame as (B*S, D), then pass flat (B, S*D) to actor — matching
    training data flow exactly regardless of whether obs arrives as (B,S,D) or (B,S*D).
    """

    def __init__(self, actor: torch.nn.Module, obs_normalizer: torch.nn.Module):
        super().__init__()
        self.actor = actor
        self.obs_normalizer = obs_normalizer
        # Store per-frame dim from actor if it's a SequenceActor; None for plain Actor.
        seq_shape = getattr(actor, "_obs_seq_shape", None)
        self.seq_S: int = seq_shape[0] if seq_shape is not None else 0
        self.seq_D: int = seq_shape[1] if seq_shape is not None else 0

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        flat = obs.flatten(start_dim=1)  # (B, S*D) for sequence; no-op for (B, D)
        if self.seq_S > 0:
            normalized = self.obs_normalizer(flat.view(B * self.seq_S, self.seq_D), update=False).view(B, self.seq_S * self.seq_D)
        else:
            normalized = self.obs_normalizer(flat, update=False)
        action, mean, _ = self.actor(normalized)  # actor returns (action, mean, log_std)
        # Return the squashed deterministic action (= tanh(mean)*scale + bias), NOT the
        # raw pre-tanh `mean`. This matches the in-process inference path
        # (FlashSACRunner.get_inference_policy → actor(obs)[0]); returning `mean` here broke
        # export↔runtime parity (e.g. mean 2.0 vs action tanh(2.0)≈0.964).
        return action

    @torch.jit.export
    def forward_with_estimation(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = obs.shape[0]
        flat = obs.flatten(start_dim=1)
        if self.seq_S > 0:
            normalized = self.obs_normalizer(flat.view(B * self.seq_S, self.seq_D), update=False).view(B, self.seq_S * self.seq_D)
        else:
            normalized = self.obs_normalizer(flat, update=False)
        action, mean, _ = self.actor(normalized)  # caches latent in actor.core for estimator
        return action, self.actor.get_estimated_velocity()  # squashed action, not raw mean (parity)




def load_exported_policy(export_dir: Path, device: str, actor_obs_groups: list[str] | None = None) -> Callable:
    """Load exported TorchScript policy.

    Args:
        export_dir: Directory containing policy_deployed.pt.
        device: Target device string.
        actor_obs_groups: TensorDict keys to concatenate (in order) for the actor input.
            Defaults to ["actor"]. Pass runner.alg.actor.obs_groups for the standard path.
    """
    policy_path = export_dir / "policy_deployed.pt"
    if not policy_path.exists():
        raise FileNotFoundError(f"Exported policy not found: {policy_path}")

    print(f"[INFO] Loading exported policy: {policy_path}")
    policy = torch.jit.load(str(policy_path), map_location=device)
    policy.eval()

    groups = actor_obs_groups or ["actor"]

    def _call(obs):
        # Viewer returns a TensorDict; exported policy expects a flat tensor.
        if not isinstance(obs, torch.Tensor):
            obs = torch.cat([obs[g] for g in groups], dim=-1)
        return policy(obs)

    return _call


def _build_obs_groups(env: ManagerBasedRlEnv, env_cfg: ManagerBasedRlEnvCfg) -> dict:
    """Build deploy-config obs_groups by walking env_cfg.observations and querying term dims from ObservationManager."""
    from deploy.deploy_config import ObsTermCfg
    obs_groups = {}
    for group_name, group_cfg in env_cfg.observations.items():
        # Group-level history_length overrides term-level (matching mjlab behavior)
        group_history_length = getattr(group_cfg, 'history_length', None)
        group_flatten_history_dim = getattr(group_cfg, 'flatten_history_dim', True)
        terms = {}
        for obs_name, sensor_cfg in group_cfg.terms.items():
            effective_history = group_history_length if group_history_length is not None else getattr(sensor_cfg, 'history_length', 0)
            effective_flatten = group_flatten_history_dim if group_history_length is not None else getattr(sensor_cfg, 'flatten_history_dim', True)
            term_dim = None
            try:
                obs_manager = getattr(env.unwrapped, "observation_manager", None)
                if obs_manager is not None:
                    active_terms = obs_manager.active_terms.get(group_name, [])
                    if obs_name in active_terms:
                        idx = active_terms.index(obs_name)
                        term_dims = obs_manager.group_obs_term_dim.get(group_name, [])
                        if idx < len(term_dims):
                            dim = term_dims[idx]
                            term_dim = dim[0] if isinstance(dim, tuple) else dim
            except Exception as e:
                print(f"[WARNING] Could not determine term_dim for {obs_name}: {e}")
            # ObservationsProxy dedups terms colliding across groups (same name, different
            # config) by renaming the flat-registry key to f"{group}_{term}" (see
            # obs_buffer.ObservationsProxy.__setitem__). Export is already group-nested, so
            # strip that dedup prefix back off to keep the term key group-relative.
            term_name = obs_name
            group_prefix = f"{group_name}_"
            if term_name.startswith(group_prefix):
                term_name = term_name[len(group_prefix):]
            terms[term_name] = ObsTermCfg(
                scale=sensor_cfg.scale,
                clip=sensor_cfg.clip,
                delay_hold_prob=sensor_cfg.delay_hold_prob,
                delay_max_lag=sensor_cfg.delay_max_lag,
                delay_min_lag=sensor_cfg.delay_min_lag,
                history_length=effective_history,
                flatten_history_dim=effective_flatten,
                term_dim=term_dim,
            )
        obs_groups[group_name] = terms
    return obs_groups


def _build_actuator_list(mj_model, articulation, action_scale: dict) -> list:
    """Build ActuatorCfg list from MuJoCo model joint properties and robot articulation spec."""
    import re
    import mujoco
    from deploy.deploy_config import ActuatorCfg

    # Map joint name -> (damping, frictionloss)
    joint_props = {}
    for i in range(mj_model.njnt):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            dof = mj_model.jnt_dofadr[i]
            joint_props[name] = (float(mj_model.dof_damping[dof]), float(mj_model.dof_frictionloss[dof]))

    def _props_for_pattern(pattern: str) -> dict:
        for name, (damping, friction) in joint_props.items():
            bare = name.split("/", 1)[-1] if "/" in name else name
            if re.fullmatch(pattern, bare):
                props = {}
                if damping != 0.0:
                    props["joint_damping"] = damping
                if friction != 0.0:
                    props["joint_friction"] = friction
                return props
        return {}

    actuators = []
    for actuator_cfg in articulation.actuators:
        base = getattr(actuator_cfg, "base_cfg", actuator_cfg)
        patterns = list(actuator_cfg.target_names_expr)
        jp = _props_for_pattern(patterns[0])
        actuators.append(ActuatorCfg(
            joints=patterns,
            # Velocity actuators (BuiltinVelocityActuatorCfg, e.g. a gimbal drive)
            # carry no position gain — kp absent → 0; kd holds the velocity gain kv (damping).
            kp=float(getattr(base, "stiffness", 0.0)),
            kd=float(base.damping),
            action_scale=float(action_scale.get(patterns[0], 0.25)),
            effort_limit=float(base.effort_limit) if base.effort_limit else None,
            armature=float(base.armature) if base.armature is not None else 0.0,
            joint_damping=jp.get("joint_damping", 0.0),
            joint_friction=jp.get("joint_friction", 0.0),
        ))
    return actuators


def export_policy_for_deployment(runner, output_dir: Path, env: ManagerBasedRlEnv, env_cfg: ManagerBasedRlEnvCfg, task_name: str = "ballbot_velocity", deployed_model=None, clip_actions: float | None = None, policy_obs_group: str = "actor"):
    """Export policy + normalizer + config for deployment.

    policy_obs_group names the obs group the exported policy consumes (the group the
    deploy runtime must assemble and feed). Defaults to "actor".
    """
    from deploy.deploy_config import DeployEnvCfg, SimulationCfg, DefaultPoseCfg, ActionCfg
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export TorchScript policy
    policy_path = output_dir / "policy_deployed.pt"
    torch.jit.script(deployed_model.eval().to('cpu')).save(str(policy_path))

    # Does the exported policy expose a *working* forward_with_estimation? The FlashSAC
    # wrapper always defines that method but raises unless its actor has a velocity
    # estimator (_use_velocity_estimator); RMA-estimator wrappers always estimate.
    has_velocity_estimator = bool(
        getattr(getattr(deployed_model, "actor", None), "_use_velocity_estimator", False)
        or type(deployed_model).__name__ == "DeployedPolicyRMAEstimator"
    )

    # 2. Extract Robot-specific constants
    if "ballbot" in task_name.lower():
        # Ballbot has no keyframe and no feet. The slider action scale comes from its own
        # constants: the generic fallback action_scale=0.25 would be a 2.36x error against the
        # real ACTION_SCALE of 0.106 m.
        from asset_zoo.ballbot.ballbot_constants import (
            get_action_scale as _ballbot_action_scale,
            SLIDER_JOINT_NAMES,
        )
        action_scale = _ballbot_action_scale()
        # Stand-in for an asset-zoo HOME_KEYFRAME; only .pos/.joint_pos are read.
        # Ballbot's home pose is all sliders at 0 with the hull at its init height.
        home_keyframe = SimpleNamespace(
            pos=tuple(env_cfg.scene.entities["robot"].init_state.pos),
            joint_pos={name: 0.0 for name in SLIDER_JOINT_NAMES},
        )
        foot_bodies = []
    else:
        raise ValueError(f"no deploy constants for task {task_name!r}")

    # 3. Resolve Dimensions & Controlled Joints
    unwrapped = env.unwrapped
    # Use the live robot's articulation, not a hardcoded import: it carries the
    # actuators actually built for this run (incl. camera gimbal actuators for
    # head_camera="actuated"), matching the grafted joints in robot.xml.
    articulation = unwrapped.scene["robot"].cfg.articulation
    action_dim = unwrapped.action_space.shape[-1]
    
    try:
        dims = getattr(unwrapped, "observation_manager").group_obs_dim[policy_obs_group]
        obs_dim = sum(d[0] if isinstance(d, tuple) else d for d in dims) if isinstance(dims, list) else dims
    except (AttributeError, KeyError, TypeError):
        shape = unwrapped.observation_space.spaces[policy_obs_group].shape
        obs_dim = shape[-1] if len(shape) > 1 else shape[0]

    controlled_joints = []
    try:
        # Collect target names from ALL active action terms.
        for term_name in unwrapped.action_manager.active_terms:
            term = unwrapped.action_manager.get_term(term_name)
            if getattr(term, "action_dim", 0) > 0:
                names = getattr(term, 'target_names', getattr(term, '_target_names', []))
                controlled_joints.extend([n.split('/')[-1] for n in names])
    except Exception as e:
        print(f"[WARNING] Could not get controlled_joints from action manager: {e}")

    # 4. Build and save deployment configuration
    deploy_cfg = DeployEnvCfg(
        simulation=SimulationCfg(
            timestep=env_cfg.sim.mujoco.timestep,
            control_freq=1.0 / (env_cfg.decimation * env_cfg.sim.mujoco.timestep),
            decimation=env_cfg.decimation,
        ),
        default_pose=DefaultPoseCfg(
            root_position=list(home_keyframe.pos),
            joint_pos={k: float(v) for k, v in home_keyframe.joint_pos.items()},
        ),
        observation=_build_obs_groups(env, env_cfg),
        action=ActionCfg(
            dim=action_dim,
            low_pass_alpha=getattr(list(env_cfg.actions.values())[0], 'alpha', 1.0),
            clip_actions=float(clip_actions) if (clip_actions is not None and clip_actions is not False) else None,
            controlled_joints=controlled_joints,
        ),
        actuators=_build_actuator_list(unwrapped.sim.mj_model, articulation, action_scale),
        foot_bodies=foot_bodies,
        obs_dim=obs_dim,
        policy_obs_group=policy_obs_group,
        has_velocity_estimator=has_velocity_estimator,
    )

    deploy_cfg.to_yaml(f"{output_dir}/env_config.yaml")

    # 5. Persist robot-only XML so deployment models runtime-grafted geometry
    #    (e.g. the analytic hull sphere) that the bare MJCF lacks.
    #    The live robot Entity spec is the single source of truth — what trained.
    #    The deploy env adds its own ground/light/actuators, so the
    #    robot-only spec is exactly what it expects.
    #    A fresh spec from cfg.spec_fn() is used, not unwrapped.scene["robot"].spec:
    #    the live entity spec is attached-by-reference into the scene parent and
    #    cannot be compiled/serialized standalone.
    #    All mesh paths are written RELATIVE to robot.xml so the file is portable
    #    across machines (on-robot deploy): meshdir relative to the run dir, and the
    #    camera meshes (grafted with absolute paths outside meshdir) rebased relative
    #    to meshdir. The deploy env resolves the relative meshdir against robot.xml's
    #    own dir at load time. to_xml() validates by opening meshes relative to CWD,
    #    so we serialize with the absolute meshdir, then rewrite the emitted paths to
    #    relative (string rewrite, not spec edit, to keep validation working).
    robot_xml_path = None
    try:
        robot_spec = unwrapped.scene["robot"].cfg.spec_fn()
        xml_str = robot_spec.to_xml()
        robot_xml_path = output_dir / "robot.xml"
        robot_xml_path.write_text(xml_str)
    except Exception as e:
        robot_xml_path = None
        print(f"[WARNING] Could not export robot.xml (deploy will fall back to bare XML): {e}")

    robot_xml_msg = f"\nRobot XML exported to: {robot_xml_path}" if robot_xml_path else ""
    print(f"\n{'='*60}\nPolicy exported to: {policy_path}\nConfig exported to: {output_dir / 'env_config.yaml'}{robot_xml_msg}\nObs dim: {obs_dim}, Action dim: {action_dim}\n{'='*60}\n")
