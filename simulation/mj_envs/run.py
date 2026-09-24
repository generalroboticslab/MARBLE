#!/usr/bin/env python
"""Train or play the MARBLE (ballbot) velocity-tracking policies.

Two subcommands, `train` and `play`, both taking `--task <ClassName>`: an experiment class in
`tasks/ballbot_velocity/experiments.py`. See `simulation/README.md` for the task list.

    python mj_envs/run.py play --task BallbotVelComplexFlatDRLatency
    python mj_envs/run.py train --task BallbotVelComplexFlatDRLatency
    python mj_envs/run.py play --task BallbotVelComplexFlatDRLatency \
        --checkpoint runs/<task>/<run>/model_<iter>.pt --export_policy

`play` without `--checkpoint` loads the latest run in `runs/<task>/` with at least 500 iterations,
otherwise the shipped export in `../policies/<task>/`.
"""

from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.native.viewer import PlotCfg
from mjlab.utils.wrappers import VideoRecorder
from mjlab.utils.torch import configure_torch_backends
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
    RslRlPpoAlgorithmCfg as BaseRslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
import tyro
import torch
import gymnasium as gym
import copy
import os
import platform
import sys
import threading
import time
import re
import random
import glfw
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Annotated, Callable, Literal, Union

sys.path.insert(0, str(Path(__file__).parent))
# sys.path.insert(0, str(Path(__file__).parent.parent))

from mjlab.tasks.velocity import mdp as vel_mdp
from utils.experiments import BaseExperiment, apply_experiment
from utils.reward_table import RewardTablePrinter
from mjlab_util.patched_observation_manager import ChronologicalObservationManager
import mjlab_util.patched_reward_manager  # noqa: F401 (monkeypatch side-effect)
import mjlab_util.patched_viewer  # noqa: F401 (monkeypatch side-effect)
from utils.file_logger import FileLogger
from utils.export_util import (
    load_exported_policy,
    export_policy_for_deployment,
)

def banner(title: str):
    line = "=" * 60
    print(f"\n{line}\n{title}\n{line}\n")


def _apply_global_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[INFO] Global seed: {seed}")


def _setup_arch_cache_build_dirs() -> tuple[Path, Path]:
    arch = platform.machine().lower() or "unknown"
    cache_root = Path("cache") / arch
    build_root = Path("build") / arch
    cache_root.mkdir(parents=True, exist_ok=True)
    build_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(cache_root / "torch_extensions"))
    os.environ.setdefault("IK_NATIVE_BUILD_DIR", str(build_root))
    os.environ.setdefault("LEGGED_ARCH", arch)
    return cache_root, build_root


# --- Config ---

@dataclass
class BaseConfig:
    """Shared configuration."""
    num_envs: int | None = None
    device: str = "cuda:0"
    video: bool = False
    video_length: int = 200
    checkpoint: str | None = None
    task: str = "ballbot_velocity"
    enable_corruption: bool = True
    verbose: bool = False  # print live per-term reward table to terminal every display_interval s


@dataclass
class TrainConfig(BaseConfig):
    """Training configuration."""
    seed: int | None = None
    max_iterations: int | None = None
    run_name: str | None = None
    wandb_run_path: str | None = None
    video_interval: int = 2000
    enable_nan_guard: bool = False
    log_interval: int = 10
    algo: Literal["flash_sac"] = "flash_sac"
    torchrunx_log_dir: str | None = None
    gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
    log_to_file: bool = True  # Write training logs to {experiment_folder}/training.log
    file_log_flush_interval: int = 5  # Flush file log every N iterations
    flash_sac_learning_starts: int | None = None
    flash_sac_num_updates: int | None = None
    flash_sac_hp_override: list[str] = field(default_factory=list)  # ["key=val", ...] FlashSAC HP overrides for sweeps, applied AFTER flash_sac_configure


@dataclass
class PlayConfig(BaseConfig):
    """Play/evaluation configuration."""
    seed: int | None = None
    num_envs: int = 4
    device: str = "cuda:0"
    video_length: int = 500
    agent: Literal["zero", "random", "trained", "exported", "experiment"] = "trained"
    algo: Literal["flash_sac"] = "flash_sac"
    viewer: Literal["auto", "native", "viser"] = "auto"
    video_height: int | None = None
    video_width: int | None = None
    camera: int | str | None = None
    export_policy: bool = False
    export_dir: str = "deploy/runs"
    play_after_export: bool = False
    cmd_vel_x: float | None = None
    cmd_vel_y: float | None = None
    cmd_ang_z: float | None = None
    verbose: bool = True # always print reward table unless --no-verbose
    speed: float = 1.0
    """Playback speed multiplier (mjlab's ``BaseViewer.SPEED_MULTIPLIERS``: 1/32 .. 8; the nearest
    listed value is used). The only pacing knob: ``BaseViewer._step_physics`` advances sim time by
    ``dt * _time_multiplier`` off the wall clock, so 1.0 is real time. Replaces the former
    ``--realtime`` flag, which only ever selected a render cadence -- sim speed was 1x either way
    (measured 0.99x with it off). Same field ``curobo_reach_verify.py --speed`` drives."""
    vsync: bool = False  # If False, try to disable VSync for high frame rates (Linux: vblank_mode=0). Defaults off, which is what --realtime's removal preserves: it used to force this off whenever it was absent, i.e. always in practice.
    sim_dt: float | None = None  # Override physics timestep (e.g. 0.01 for 100 Hz)
    decimation: int | None = None  # Override LL control decimation



@dataclass
class RslRlPpoAlgorithmCfg(BaseRslRlPpoAlgorithmCfg):
    """Local config with tuned defaults for velocity tracking."""
    entropy_coef: float = 0.0
    desired_kl: float = 0.02


def create_agent_cfg(task_name: str, max_iterations: int = 30000, run_name: str | None = None, clip_actions: float | bool | None = True) -> RslRlOnPolicyRunnerCfg:
    """Create PPO agent configuration tuned for velocity tracking."""

    experiment_name = "ballbot_velocity"
    use_sns = "smooth" in task_name.lower()
    alg_overrides = {}

    if use_sns:
        experiment_name = f"{experiment_name}_smooth"

    activation = "mish" if use_sns else "elu"

    actor_cfg = RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation=activation,
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )

    critic_cfg = RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation=activation,
        obs_normalization=True,
        distribution_cfg=None,
    )

    algorithm_cfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        **alg_overrides,
    )

    return RslRlOnPolicyRunnerCfg(
        experiment_name=experiment_name,
        run_name=run_name or "",
        max_iterations=max_iterations,
        save_interval=3000,
        num_steps_per_env=24,
        clip_actions=clip_actions,
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
        actor=actor_cfg,
        critic=critic_cfg,
        algorithm=algorithm_cfg,
    )

def apply_play_overrides(cfg: ManagerBasedRlEnvCfg, play_cfg: PlayConfig) -> None:
    """Modify environment config for play mode (episode length, terrain, commands)."""
    # cfg.episode_length_s = int(1e9)

    if cfg.events is not None:
        cfg.events.pop("push_robot", None)
        # Interval DR events (body_mass, com_displacement, ee_*, physics_recompute) would fire
        # periodically during play — stalls the render loop and is unwanted visually.
        # If corruption is enabled, DR is applied once at env startup instead.
        cfg.events.pop("physics_recompute", None)
        for key in ("body_mass", "com_displacement", "ee_payload"):
            if key not in cfg.events:
                continue
            if play_cfg.enable_corruption:
                cfg.events[key].mode = "startup"
            else:
                cfg.events.pop(key)

    # Terrain is deliberately NOT overridden here. Play used to force curriculum=False plus a
    # 5x5 grid with a 10 m border, which rebuilt the world as a random-difficulty sample and
    # silently hid what training actually sees: the curriculum layout is one column per
    # sub-terrain type with difficulty ramping along rows, so the tiles you inspect in play
    # were neither the same types nor the same difficulties. Terrain bugs (a missing uphill,
    # a too-easy mix) are exactly what play is used to catch, so play now renders the training
    # generator verbatim. Trade-off accepted: bigger grid = slower first-time terrain bake,
    # and no border wall to stop a robot walking off the edge.

    # Command velocity overrides
    if "twist" in cfg.commands:
        if any(v is not None for v in [play_cfg.cmd_vel_x, play_cfg.cmd_vel_y, play_cfg.cmd_ang_z]):
            twist = cfg.commands["twist"]
            vx, vy, wz = play_cfg.cmd_vel_x or 0.0, play_cfg.cmd_vel_y or 0.0, play_cfg.cmd_ang_z or 0.0
            twist.ranges.lin_vel_x = (vx, vx)
            twist.ranges.lin_vel_y = (vy, vy)
            twist.ranges.ang_vel_z = (wz, wz)
            twist.rel_standing_envs = 0.0
            print(f"[INFO] Command: vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}")


def make_keyboard_callback(env):
    """Create keyboard callback for velocity control and disturbances."""

    vel_step = 0.1
    VEL_KEYS = {
        glfw.KEY_KP_8: (0, vel_step), glfw.KEY_KP_5: (0, -vel_step),
        glfw.KEY_KP_4: (1, vel_step), glfw.KEY_KP_6: (1, -vel_step),
        glfw.KEY_KP_7: (2, vel_step), glfw.KEY_KP_9: (2, -vel_step),
    }

    # Interative Pushes (K, L, [, ], Numpad /, *)
    # Applied as velocity impulses to the base
    PUSH_KEYS = {
        glfw.KEY_K: {"x": (0.5, 0.5)},              # K: push forward (+x)
        glfw.KEY_L: {"x": (-0.5, -0.5)},             # L: push backward (-x)
        glfw.KEY_LEFT_BRACKET: {"y": (0.5, 0.5)},    # [: push left (+y)
        glfw.KEY_RIGHT_BRACKET: {"y": (-0.5, -0.5)}, # ]: push right (-y)
        glfw.KEY_KP_DIVIDE: {"yaw": (1.0, 1.0)},     # KP_DIVIDE: rotate CCW (+yaw)
        glfw.KEY_KP_MULTIPLY: {"yaw": (-1.0, -1.0)}, # KP_MULTIPLY: rotate CW (-yaw)
    }

    def callback(key: int):
        try:
            # 1. Handle disturbances
            if key in PUSH_KEYS:
                env_ids = torch.arange(env.num_envs, device=env.device)
                vel_mdp.push_by_setting_velocity(env.unwrapped, env_ids, velocity_range=PUSH_KEYS[key])
                # If policy is push-aware, update its internal timer
                if hasattr(env.unwrapped, "_last_push_step"):
                    env.unwrapped._last_push_step[:] = float(env.unwrapped.common_step_counter)
                
                axis = next(iter(PUSH_KEYS[key]))
                val  = next(iter(PUSH_KEYS[key].values()))[0]
                print(f"[PUSH] Applied {axis}={val:+.1f} disturbance")
                return

            # 2. Handle velocity commands
            cmd_manager = env.unwrapped.command_manager
            if "twist" not in cmd_manager._terms:
                return

            cmd = cmd_manager._terms["twist"].command
            if key == glfw.KEY_KP_0:  # KP_0: reset
                cmd[:, :] = 0.0
            elif key in VEL_KEYS:
                idx, delta = VEL_KEYS[key]
                cmd[:, idx] += delta
            else:
                return

            cmd[:, 0].clamp_(-3.0, 3.0)
            cmd[:, 1].clamp_(-2.0, 2.0)
            cmd[:, 2].clamp_(-2.0, 2.0)
            print(f"[CMD] vx={cmd[0, 0]:+.2f}, vy={cmd[0, 1]:+.2f}, wz={cmd[0, 2]:+.2f}")
        except Exception as e:
            print(f"[ERROR] Keyboard callback: {e}")

    return callback


# --- Environment ---



class _SilentCurriculumDict(dict):
    """Dict that silently ignores missing-key access.
    Experiment configure() can freely write curriculum params without checking
    whether a term exists — safe in play mode where reward-curriculum terms are
    stripped. Each missing access returns a fresh SimpleNamespace(params={}).
    """
    def __missing__(self, key):
        from types import SimpleNamespace
        return SimpleNamespace(params={})


def base_env(cfg: BaseConfig, exp: BaseExperiment | None = None, enable_reward_curriculum: bool = True, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Common environment config setup for train and evaluation.

    Experiment-only dispatch: the env is built by exp.build_env_cfg, which reads structural
    params (randomize_arms, n, num_steps_per_env, ...) from the experiment itself. A bare task
    name resolves to exp=None (apply_experiment silent) and is rejected here.
    """
    configure_torch_backends()

    if exp is None:
        raise ValueError(
            f"No experiment resolved for task={cfg.task!r}. Entry points are experiment names "
            f"only — run `--task list` to see registered experiments."
        )

    env_cfg = exp.build_env_cfg(
        play=play,
        enable_corruption=cfg.enable_corruption,
        enable_reward_curriculum=enable_reward_curriculum,
    )
    # Isolate this build's config from any module-level templates build_env_cfg may share
    # (e.g. the commands dict). exp.configure mutates manager dicts in place, and some mutations
    # are not idempotent, so a second build in the same process would see the first's edits.
    # Deep-copying here makes every build independent.
    env_cfg = copy.deepcopy(env_cfg)
    if hasattr(env_cfg, "seed") and getattr(cfg, "seed", None) is not None:
        env_cfg.seed = cfg.seed
    env_cfg.scene.num_envs = cfg.num_envs
    if "actor" in env_cfg.observations:
        env_cfg.observations["actor"].enable_corruption = cfg.enable_corruption
    # Wrap so experiment configure() can write to any curriculum term safely
    # (missing terms are silently ignored, e.g. in play mode without curriculum)
    env_cfg.curriculum = _SilentCurriculumDict(env_cfg.curriculum)
    return env_cfg

# --- Train ---


def train_flash_sac(cfg: TrainConfig, env_cfg: ManagerBasedRlEnvCfg, log_dir: Path, exp=None):
    """Train with FlashSAC algorithm."""
    from flash_sac.env_wrapper import ManagerBasedRlEnvWithFinalObs
    from flash_sac.runner import FlashSACRunner
    from flash_sac.config import FlashSACConfig

    env = ManagerBasedRlEnvWithFinalObs(cfg=env_cfg, device=cfg.device)

    sac_cfg = FlashSACConfig(
        num_learning_iterations=cfg.max_iterations,
        gamma=0.99,
        v_min=-200.0,
        v_max=1000.0,
        num_atoms=501,
        target_entropy_ratio=0.25,
        alpha_init=0.1,
        tau=0.05,
        policy_frequency=2,
        buffer_size=64,
        batch_size=4096,
        learning_starts=(
            cfg.flash_sac_learning_starts
            if cfg.flash_sac_learning_starts is not None
            else FlashSACConfig.learning_starts
        ),
        num_updates=(
            cfg.flash_sac_num_updates
            if cfg.flash_sac_num_updates is not None
            else FlashSACConfig.num_updates
        ),
        use_sequence_encoder=False,
        encoder_type="rma_cnn",
        encoder_embed_dim=32,
        encoder_latent_dim=128,
    )

    if exp is not None:
        exp.flash_sac_configure(sac_cfg)

    # HP sweep overrides. Applied AFTER flash_sac_configure so they win over the experiment
    # class; each value coerced to the live attr's runtime type. Unknown key raises (fail-fast).
    for key_value in cfg.flash_sac_hp_override:
        key, _, raw_value = key_value.partition("=")
        current_value = getattr(sac_cfg, key)
        new_value = (raw_value.lower() in ("1", "true", "yes")) if isinstance(current_value, bool) else type(current_value)(raw_value)
        setattr(sac_cfg, key, new_value)
        print(f"[INFO] FlashSAC HP override (post-configure): {key}={new_value}")

    if cfg.flash_sac_learning_starts is not None:
        print(f"[INFO] FlashSAC override: learning_starts={cfg.flash_sac_learning_starts}")
    if cfg.flash_sac_num_updates is not None:
        print(f"[INFO] FlashSAC override: num_updates={cfg.flash_sac_num_updates}")
    runner = FlashSACRunner(env, sac_cfg, str(log_dir), cfg.device)
    runner.setup()

    if cfg.checkpoint:
        print(f"[INFO] Resuming FlashSAC from checkpoint: {cfg.checkpoint}")
        runner.load(cfg.checkpoint)
    else:
        print("[INFO] Starting FlashSAC from scratch")

    banner(f"Training FlashSAC: {cfg.max_iterations} iters | {env_cfg.scene.num_envs} envs | {cfg.device}")

    runner.learn()
    env.close()
    print(f"\n[INFO] FlashSAC training complete! Logs: {log_dir}")


# --- Compact Logger Runner ---



def run_train(cfg: TrainConfig, log_dir: Path):
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        device = cfg.device
        rank = 0
        local_rank = 0
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
        device = f"cuda:{local_rank}"

    cfg.device = device

    # Setup file logging (rank 0 only)
    file_logger = None
    if cfg.log_to_file and rank == 0:
        log_file = log_dir / "training.log"
        file_logger = FileLogger(log_file)
        file_logger.__enter__()
        print(f"[INFO] Logging to: {log_file}")

    # apply_experiment already called in train(); cfg.task is now the base task but
    # cfg.experiment holds the original experiment name. Use it so multi-GPU workers
    # (spawned by torchrunx) and single-GPU paths both resolve the same experiment.
    _exp_name = getattr(cfg, "experiment", None) or cfg.task
    exp = apply_experiment(_exp_name, cfg, silent=True)

    if cfg.num_envs is None:
        cfg.num_envs = getattr(exp, "num_envs", None) or 4096

    if cfg.max_iterations is None:
        cfg.max_iterations = getattr(exp, "max_iterations", None) or 15000

    env_cfg = base_env(cfg, exp=exp, enable_reward_curriculum=True)
    agent_cfg = create_agent_cfg(cfg.task, cfg.max_iterations, cfg.run_name)
    # Set wandb experiment/run name from log_dir so it shows as
    # "<task>/<timestamp>[_run_name]" (e.g. "BallbotVelComplexFlatDRLatency/2026-03-21_14-00-00").
    agent_cfg.experiment_name = log_dir.parent.name
    agent_cfg.run_name = log_dir.name
    
    # Set wandb run name to include experiment prefix without changing local run_name.
    # We monkey-patch wandb.init because rsl_rl's WandbSummaryWriter hardcodes the
    # run name to the last component of the log directory, overriding WANDB_NAME.
    if rank == 0:
        try:
            import wandb
            _orig_init = wandb.init
            def wandb_init_patched(*args, **kwargs):
                kwargs["name"] = f"{agent_cfg.experiment_name}_{agent_cfg.run_name}"
                return _orig_init(*args, **kwargs)
            wandb.init = wandb_init_patched
        except ImportError:
            pass

    if exp is not None:
        exp.configure(env_cfg, agent_cfg)

    # Offset seeds for diverse experience collection in multi-GPU
    if hasattr(env_cfg, "seed") and env_cfg.seed is not None:
        env_cfg.seed += local_rank
    if hasattr(agent_cfg, "seed") and agent_cfg.seed is not None:
        agent_cfg.seed += local_rank

    if cfg.enable_nan_guard:
        env_cfg.sim.nan_guard.enabled = True
        if rank == 0:
            print("[INFO] NaN guard enabled")

    if rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)
    train_flash_sac(cfg, env_cfg, log_dir, exp=exp)
    return


def train(cfg: TrainConfig):
    """Train velocity tracking."""
    from mjlab.utils.gpu import select_gpus

    _apply_global_seed(cfg.seed)

    # Resolve experiment early so cfg.task becomes the base task.
    # run_label is the user-facing name (experiment or base task) used for log dirs.
    run_label = cfg.task
    apply_experiment(cfg.task, cfg, silent=True)

    log_root = Path("runs") / run_label
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    algo_tag = f"_{cfg.algo}"
    dir_name = f"{timestamp}{algo_tag}"
    if cfg.run_name:
        dir_name += f"_{cfg.run_name}"
    log_dir = log_root / dir_name
    
    selected_gpus, num_gpus = select_gpus(cfg.gpu_ids)

    if selected_gpus is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    os.environ["MUJOCO_GL"] = "egl"

    if num_gpus <= 1:
        print(f"[INFO] Log dir: {log_dir}")
        run_train(cfg, log_dir)
    else:
        import torchrunx
        import logging
        logging.basicConfig(level=logging.INFO)

        if "TORCHRUNX_LOG_DIR" not in os.environ:
            if cfg.torchrunx_log_dir is not None:
                os.environ["TORCHRUNX_LOG_DIR"] = cfg.torchrunx_log_dir
            else:
                os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

        print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
        print(f"[INFO] Log dir: {log_dir}")
        torchrunx.Launcher(
            hostnames=["localhost"],
            workers_per_host=num_gpus,
            backend=None,
            copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
        ).run(run_train, cfg, log_dir)




_MJ_ENVS_DIR = Path(__file__).resolve().parent
REPO_ROOT = _MJ_ENVS_DIR.parent

# Two distinct "deploy/runs" trees, so neither anchor may be swapped for the other:
#   REPO_ROOT/deploy/runs    committed, seed-scoped; the checkpoints a fresh clone gets
#                            (see .gitignore's `!deploy/runs/`). Read by class default_ckpt.
#   _MJ_ENVS_DIR/deploy/runs flat, no seed dir; written by --export_policy, read by
#                            deploy/test_deployment.py's own _SCRIPT_DIR/"runs" lookup.


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a repo-root-relative path literal (a class `default_ckpt`) to an absolute one.

    Those literals are written relative to the REPO ROOT ("deploy/runs/<task>/...",
    "runs/<task>/..."), but bare Path() resolves against the CWD. Launched from anywhere
    but the repo root, a valid default checkpoint silently became a missing file and fell
    through to the runs/ search — which for classes with no runs of their
    own then raised "No checkpoint >= 500 iters found". Absolute paths pass through.
    """
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def default_export_dir(task: str, export_dir: str | Path = "deploy/runs") -> Path:
    """Resolve the deployment export directory for `task`.

    A relative `export_dir` anchors at mj_envs/, not the CWD, so every entrypoint targets
    the same tree no matter where it was launched from. NOT the repo root: this is the
    export tree test_deployment.py reads. Absolute paths (--checkpoint-dir) pass through.

    Sole owner of this path math: callers that pass a bare "deploy/runs" resolve it
    CWD-relative and drop the `task` leaf, which lands exports on top of each other in
    the shared root.
    """
    path = Path(export_dir)
    return path if path.is_absolute() else _MJ_ENVS_DIR / path / task


def _load_trained_policy(
    cfg: PlayConfig, env, agent_cfg, resume_path: Path,
    export_dir: Path | None, env_cfg, device: str,
) -> "Callable | None":
    """Load trained policy from checkpoint.

    `export_dir` None means derive it from `cfg` via default_export_dir — the right
    choice for every caller that has no reason to override it.

    Returns policy callable, or None if export-only mode (env already closed).
    """
    if export_dir is None:
        export_dir = default_export_dir(cfg.task, cfg.export_dir)
    def _maybe_export_and_reload(runner_or_none, policy_callable, deployed_model=None, reload_groups=None):
        if not cfg.export_policy:
            return policy_callable

        export_policy_for_deployment(
            runner_or_none, export_dir, env, env_cfg, cfg.task,
            deployed_model=deployed_model, clip_actions=agent_cfg.clip_actions,
            policy_obs_group=(reload_groups[0] if reload_groups else "actor"),
        )
        if not cfg.play_after_export:
            env.close()
            return None

        actor_obs_groups = reload_groups or (runner_or_none.alg.actor.obs_groups if runner_or_none else None)
        return load_exported_policy(export_dir, device, actor_obs_groups=actor_obs_groups)
    from flash_sac.runner import FlashSACRunner
    from flash_sac.config import FlashSACConfig
    ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    sac_cfg = FlashSACConfig.from_saved_args(saved_args)
    sac_cfg.logger = "none"  # skip wandb.init in play mode
    runner = FlashSACRunner(env.unwrapped, sac_cfg, str(resume_path.parent), device)
    runner.setup()
    runner.load(str(resume_path))
    print(f"[INFO] FlashSAC policy loaded from {resume_path.name}")
    from utils.export_util import DeployedPolicyFlashSAC
    policy = runner.get_inference_policy(device=device)
    deployed = DeployedPolicyFlashSAC(runner.actor, runner.obs_normalizer)
    reload_groups = None
    return _maybe_export_and_reload(None, policy, deployed_model=deployed, reload_groups=reload_groups)

# --- Play ---

def find_latest_checkpoint(task: str, min_it: int = 500, algo: str | None = None, fallback: bool = True) -> str | None:
    """Find the MOST-TRAINED checkpoint in runs/<task>/ with >= min_it iterations.

    Ranked by (iteration, mtime): highest iteration wins, and mtime breaks ties between runs
    that reached the same iteration (so re-training a task still resolves to the newest run).

    Iteration must outrank mtime, not the other way round. Ranking by mtime first meant any
    ACTIVELY TRAINING run hijacked resolution for its whole duration -- it rewrites a checkpoint
    every few hundred iterations, so its freshly-written model_0002500.pt beat a finished
    model_0015000.pt from an earlier run and `play` silently loaded a barely-trained policy.
    Observed 2026-08-02 on GridGaitInitStartNearZero while its seed-1 confirm run was in flight.

    Consequence to know: a finished high-iteration run now outranks a NEWER shorter run of the
    same task. That is the desired default (most-trained wins); pass --checkpoint explicitly when
    a specific shorter/older run is wanted.

    Scans only run-root level (not wandb/ subdirs) to avoid broken symlinks.
    algo filters by substring match in run directory name.
    If fallback=True and no checkpoint exists in runs/<task>/, resolves via the experiment
    class: `fallback_checkpoint` (explicit target class, preferred) else
    `fallback_checkpoint_to_parent=True` (walk the MRO). This is the single resolution site --
    callers pass a task name and get the alias handling for free.
    """
    repo_root = Path(__file__).resolve().parent.parent
    root = repo_root / "runs" / task
    ckpts = []
    if root.exists():
        for run_dir in root.iterdir():
            if not run_dir.is_dir():
                continue
            if algo and algo not in run_dir.name:
                continue
            for p in run_dir.glob("model_*.pt"):
                try:
                    st = p.stat()
                except (FileNotFoundError, OSError):
                    continue
                m = re.search(r"model_(\d+)", p.name)
                if m and (it := int(m.group(1))) >= min_it:
                    ckpts.append((it, st.st_mtime, str(p)))

    if ckpts:
        return max(ckpts)[2]

    if fallback:
        try:
            from utils.experiments import _load_all_experiments, _REGISTRY
            _load_all_experiments()
            cls = _REGISTRY.get(task)
        except Exception:
            cls = None
        if cls is not None:
            # Explicit target wins: an alias names the concrete class it was trained under.
            # Preferred over the MRO walk, which can only find ANCESTORS -- a lane class on a
            # sibling branch (or any run dir off the alias's own base chain) is unreachable by it.
            tgt = cls.__dict__.get("fallback_checkpoint")
            if tgt is not None:
                name = tgt.__name__ if isinstance(tgt, type) else tgt
                print(f"[INFO] No checkpoint in runs/{task}, alias targets runs/{name}/...")
                return find_latest_checkpoint(name, min_it, algo, fallback=False)
            # Legacy opt-in: walk the full MRO, not just bases[0]. bases[0] stopped at the first
            # ancestor with ANY run dir, which was often a less-trained intermediate class.
            if cls.__dict__.get("fallback_checkpoint_to_parent", False):
                for anc in cls.__mro__[1:]:
                    if anc.__name__ in ("BaseExperiment", "object"):
                        continue
                    if ckpt := find_latest_checkpoint(anc.__name__, min_it, algo, fallback=False):
                        return ckpt

    return None


def play(cfg: PlayConfig):
    """Play/evaluate trained policies."""
    _apply_global_seed(cfg.seed)

    run_label = cfg.task
    exp = apply_experiment(cfg.task, cfg, silent=True)

    env_cfg = base_env(cfg, exp=exp, enable_reward_curriculum=False, play=True)
    agent_cfg = create_agent_cfg(cfg.task)

    if exp is not None:
        exp.configure(env_cfg, agent_cfg)

    torch.set_float32_matmul_precision('high')

    device = cfg.device
    print(f"[INFO] Device: {device}")

    if cfg.agent == "trained" and not cfg.checkpoint:
        # Per-experiment default ckpt (classes with no runs of their own). Overrides the "find latest in runs/<task>" default below when set; falls back
        # to that path if the override file is missing.
        default_ckpt = getattr(exp, "default_ckpt", None) if exp is not None else None
        default_ckpt = resolve_repo_path(default_ckpt) if default_ckpt else None
        if default_ckpt and default_ckpt.is_file():
            cfg.checkpoint = str(default_ckpt)
            print(f"[INFO] Using class default ckpt: {cfg.checkpoint}")
        else:
            if default_ckpt:
                print(f"[WARN] Class default ckpt declared but missing: {default_ckpt}")
            print(f"[INFO] Searching latest policy in runs/{run_label} (algo={cfg.algo})...")
            cfg.checkpoint = find_latest_checkpoint(run_label, 500, cfg.algo)
            if not cfg.checkpoint:
                # runs/ is gitignored, so a fresh clone has none; fall back to the shipped
                # export in ../policies/<task>/. Detected as TorchScript below, which flips
                # agent to "exported".
                committed = REPO_ROOT.parent / "policies" / run_label / "policy_deployed.pt"
                if not committed.is_file():
                    raise ValueError(
                        f"No checkpoint >= 500 iters found in runs/{run_label} and no shipped "
                        f"export at {committed}")
                cfg.checkpoint = str(committed)
            print(f"[INFO] Auto-selected: {cfg.checkpoint}")

    # Auto-detect TorchScript policy_deployed.pt -> use exported mode
    if cfg.checkpoint and Path(cfg.checkpoint).name == "policy_deployed.pt":
        cfg.agent = "exported"
        cfg.export_dir = str(Path(cfg.checkpoint).resolve().parent)
        print(f"[INFO] Detected TorchScript policy, using exported mode")

    apply_play_overrides(env_cfg, cfg)

    if cfg.sim_dt is not None:
        env_cfg.sim.mujoco.timestep = cfg.sim_dt
        print(f"[INFO] Physics override: sim_dt={cfg.sim_dt} ({1/cfg.sim_dt:.0f} Hz)")
    if cfg.decimation is not None:
        env_cfg.decimation = cfg.decimation
        print(f"[INFO] Physics override: decimation={cfg.decimation}")

    if cfg.video_height:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width:
        env_cfg.viewer.width = cfg.video_width

    # Create environment
    is_trained = cfg.agent in ("trained", "exported")
    render_mode = "rgb_array" if (is_trained and cfg.video) else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    # --- INJECT CHRONOLOGICAL OBSERVATION FIX ---
    # Replace mjlab's observation manager with patched version that applies
    # history buffers BEFORE delay buffers (prevents temporal scrambling)
    env.observation_manager = ChronologicalObservationManager(env_cfg.observations, env)
    env._configure_gym_env_spaces()
    env.observation_manager.reset()
    print("[INFO] Injected ChronologicalObservationManager (history→delay order)")
    # --------------------------------------------

    # Pick a random (row, col) per env before the first reset. mjlab's curriculum fallback pins
    # `terrain_types` to column 0 when num_envs < num_cols, so play (num_envs=1, num_cols=8) would
    # otherwise always spawn at the left edge of the terrain. training is unaffected because it
    # already uses num_envs >= num_cols. randomize_env_origins early-returns for plane terrain.
    terrain = getattr(env.scene, "terrain", None)
    if terrain is not None:
        terrain.randomize_env_origins(
            torch.arange(env.num_envs, device=env.device)
        )

    export_dir = default_export_dir(run_label, cfg.export_dir)

    log_dir = None
    if cfg.agent == "trained":
        resume_path = Path(cfg.checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        log_dir = resume_path.parent
        print(f"[INFO] Checkpoint: {resume_path}")

    # Video recording — wrap the RAW env, not RslRlVecEnvWrapper. VideoRecorder subclasses
    # ManagerBasedRlEnv and skips __init__, so inherited methods it doesn't override (e.g.
    # get_observations) run with self=VideoRecorder and fall through its __getattr__ to
    # self._wrapped_env. RslRlVecEnvWrapper only proxies a curated attr subset (no
    # observation_manager), so wrapping it instead raises AttributeError. Wrapping the raw
    # env keeps that fallback valid, since raw env has observation_manager directly.
    if is_trained and cfg.video:
        video_folder = (log_dir or export_dir) / "videos" / "play"
        env = VideoRecorder(
            env, video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )
        print(f"[INFO] Recording video to: {video_folder}")

    # Create agent config (used for clip_actions and policy loading)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Create policy
    if cfg.agent == "zero":
        action_shape = env.unwrapped.action_space.shape
        def policy(obs): return torch.zeros(action_shape, device=env.unwrapped.device)
        print("[INFO] Using zero policy")
    elif cfg.agent == "random":
        action_shape = env.unwrapped.action_space.shape
        def policy(obs): return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1
        print("[INFO] Using random policy")
    elif cfg.agent == "experiment":
        build = getattr(exp, "build_policy", None)
        policy = build(env) if build else None
        if policy is None:
            raise ValueError(f"--agent experiment: {cfg.task} defines no build_policy()")
        print("[INFO] Using experiment-supplied policy")
    elif cfg.agent == "exported":
        policy = load_exported_policy(export_dir, device)
    else:  # trained
        policy = _load_trained_policy(cfg, env, agent_cfg, resume_path, export_dir, env_cfg, device)
        if policy is None:
            return

    # Resolve viewer
    if cfg.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        viewer_type = "native" if has_display else "viser"
    else:
        viewer_type = cfg.viewer

    # Print controls
    banner(
        f"Agent: {cfg.agent} | Envs: {cfg.num_envs} | Viewer: {viewer_type}\n"
        f"{'='*60}\n"
        "Numpad: 8/5=fwd/back, 4/6=left/right, 7/9=turn, 0=reset\n"
        "Keys (Disturb): K/L=Fwd/Back, [/]=Left/Right, NP_//NP_*=Yaw\n"
        "Mouse: left=rotate, right=pan, scroll=zoom\n"
        "Keys (Viewer): SPACE=pause, ENTER=reset, I/C/J/N=viz toggles"
    )

    # Run
    env.reset()
    if cfg.enable_corruption:
        # Startup DR events wrote mass/COM but deferred set_const; sync derived quantities now.
        from mjlab.managers.event_manager import RecomputeLevel
        env.unwrapped.sim.recompute_constants(RecomputeLevel.set_const)

    keyboard_cb = make_keyboard_callback(env)

    # The keyboard owns the command, so stop random resampling.
    _twist_terms = env.unwrapped.command_manager._terms
    if "twist" in _twist_terms:
        _twist_terms["twist"].resample = lambda env_ids: None

    # Render cadence only -- sim pacing is --speed. Uncapped, i.e. the viewer draws every tick;
    # this is what play has always defaulted to, and the removed --realtime flag was the only way
    # to lower it. Note the side effect if that ever needs revisiting: BaseViewer._step_physics
    # reuses frame_time as its per-tick step deadline and DROPS the leftover budget on a timeout,
    # so a high frame rate is a *tight* deadline -- at --speed 8 this costs ~3% of sim time
    # (measured 7.72x here against 7.99x at frame_rate=60).
    frame_rate = 999
    # Native only: --vsync acts through glfw.swap_interval on THIS process's GL context, which only
    # the MuJoCo viewer creates. viser renders in a
    # browser, so the flag cannot reach them and announcing it there would be a lie.
    if not cfg.vsync and viewer_type == "native":
        print(f"[INFO] VSync disabled. If FPS is still capped at 60, run with: vblank_mode=0 python mj_envs/run.py ...")

    if viewer_type == "native":
        # Bump plot limits: default max_viewports=12/max_rows_per_col=6 caps at 12;
        # up to 17 reward terms, so 9 rows × 2 cols = 18 slots.
        plot_cfg = PlotCfg(max_viewports=20, max_rows_per_col=10)
        viewer_obj = NativeMujocoViewer(env, policy, frame_rate=frame_rate, key_callback=keyboard_cb, plot_cfg=plot_cfg)
    else:
        viewer_obj = ViserPlayViewer(env, policy, frame_rate=frame_rate)

    if cfg.speed != 1.0:
        # Same state the native viewer's own speed keys drive, set up front -- which is the only way
        # to reach it under the viser viewer, whose key handling mjlab never wires
        # to the speed actions. Snap to the supported ladder rather than writing an arbitrary
        # multiplier, so the native viewer's on-screen speed label stays truthful.
        nearest = min(viewer_obj.SPEED_MULTIPLIERS, key=lambda m: abs(m - cfg.speed))
        viewer_obj._speed_index = viewer_obj.SPEED_MULTIPLIERS.index(nearest)
        viewer_obj._time_multiplier = nearest
        print(f"[INFO] Playback speed {nearest}x (physics timestep unchanged; only wall-clock pacing)")

    if cfg.verbose:
        RewardTablePrinter(env.unwrapped, run_label, viewer=viewer_obj).start()

    # Try to disable VSync programmatically if requested
    if not cfg.vsync:
        try:
            if glfw.get_current_context():
                glfw.swap_interval(0)
                print("[INFO] VSync disabled via glfw.swap_interval(0)")
        except Exception as e:
            print(f"[DEBUG] Could not disable VSync via glfw: {e}")

    # Watchdog daemon: force-exit when viewer closes, even if viewer.run() hangs.
    # mujoco.viewer.Handle.sync() can block after the GLFW window closes because
    # the viewer thread exits while the main thread is mid-sync, leaving an
    # internal mutex in an indeterminate state.
    def _watchdog():
        ever_running = False
        while True:
            running = viewer_obj.is_running()
            if running:
                ever_running = True
            elif ever_running:
                os._exit(0)
            time.sleep(0.05)

    threading.Thread(target=_watchdog, daemon=True).start()
    viewer_obj.run()

    print("[INFO] Done!")
    os._exit(0)

# --- CLI ---

Commands = Union[
    Annotated[TrainConfig, tyro.conf.subcommand("train")],
    Annotated[PlayConfig, tyro.conf.subcommand("play")],
]


def main():
    cache_root, build_root = _setup_arch_cache_build_dirs()
    print(f"[INFO] Arch cache dir: {cache_root}")
    print(f"[INFO] Arch build dir: {build_root}")
    cfg = tyro.cli(Commands)
    if isinstance(cfg, TrainConfig):
        train(cfg)
    else:
        play(cfg)


if __name__ == "__main__":
    main()
