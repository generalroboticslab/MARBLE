"""Experiment base class with zero-maintenance auto-discovery.

Subclass BaseExperiment anywhere in tasks/*/experiments.py and it's
automatically available via --experiment on the CLI. No imports, no
registry, no decorator needed.

Class name (CamelCase) maps directly to CLI name:
  BallbotVelComplexFlatDRLatency  →  BallbotVelComplexFlatDRLatency

Usage in tasks/my_task/experiments.py::

    from utils.experiments import BaseExperiment
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.rl import RslRlOnPolicyRunnerCfg

    class MyTaskZeroVel(BaseExperiment):
        task = "my_task"
        def configure(self, env: ManagerBasedRlEnvCfg, agent: RslRlOnPolicyRunnerCfg):
            env.commands["twist"].ranges.lin_vel_x = (0.0, 0.0)
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

if TYPE_CHECKING:
    from run import BaseConfig

# Auto-populated by __init_subclass__ as classes are defined.
_REGISTRY: dict[str, type[BaseExperiment]] = {}
_loaded = False  # guard for one-time directory scan


class BaseExperiment:
    """Base class for all experiments. Subclass to create a new one.

    Class attributes are applied to run_cfg *before* the env is built (structural).
    Override configure() to tune PPO env/agent config after the env is built.
    Override flash_sac_configure() to tune FlashSACConfig before training starts.

    Structural defaults like num_envs belong here when an experiment needs a
    non-global training default.

    Setting algo = "flash_sac" makes --task <ExperimentName> automatically use
    FlashSAC without requiring --algo flash_sac on the CLI.
    """
    task: str                         # required — which task config to load
    algo: str | None = None           # if set, overrides cfg.algo (e.g. "flash_sac")
    num_envs: int | None = None       # structural default; resolved only if CLI omits it
    max_iterations: int | None = None # structural default; resolved only if CLI omits it
    randomize_arms: bool = True
    observe_com: bool = True
    observe_full_joints: bool = False
    fallback_checkpoint_to_parent: bool = False
    fallback_checkpoint: str | type | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Auto-register every subclass the moment it is defined."""
        super().__init_subclass__(**kwargs)
        _REGISTRY[cls.experiment_name()] = cls

    @classmethod
    def experiment_name(cls) -> str:
        """Return class name as CLI name (e.g., BallbotVelRingCageComplexDR)."""
        return cls.__name__

    def build_env_cfg(
        self,
        *,
        play: bool = False,
        enable_corruption: bool = True,
        enable_reward_curriculum: bool = True,
    ) -> ManagerBasedRlEnvCfg:
        """Build this experiment's environment cfg. Override per task-family base experiment.

        Reads structural params (n, num_steps_per_env, thrust_scale_range, slider_mass_scale,
        plane_flag, curriculum_decimation, randomize_arms) from self as class attributes;
        subclasses tune them by overriding the attrs, not this method. enable_corruption /
        enable_reward_curriculum are runtime flags from the run config — only some
        factories consume them; other factories post-set corruption on observations['actor'].

        This is the single source for task→env-cfg dispatch (replaces the former
        run.get_env_cfg if/elif chain). Bare task names are no longer an entry point; every
        env is built from a resolved experiment.
        """
        raise NotImplementedError(
            f"{type(self).__name__} (task={getattr(self, 'task', '?')!r}) must define build_env_cfg"
        )

    def configure(
        self,
        env: ManagerBasedRlEnvCfg,
        agent: RslRlOnPolicyRunnerCfg,
    ) -> None:
        """Override to tune MDP config after env is built (PPO path)."""

    def flash_sac_configure(self, sac_cfg: object) -> None:
        """Override to tune FlashSACConfig before FlashSAC training starts."""

    def build_policy(self, env):
        """Optional: return a callable policy(obs)->action for `--agent experiment`.

        Default None = experiment supplies no scripted policy.
        """
        return None


def _load_all_experiments() -> None:
    """One-time scan: import every tasks/*/experiments.py to trigger registration."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    tasks_dir = Path(__file__).parent.parent / "tasks"
    for exp_file in sorted(tasks_dir.glob("*/experiments.py")):
        module = f"tasks.{exp_file.parent.name}.experiments"
        importlib.import_module(module)


def apply_experiment(name: str, cfg: BaseConfig, silent: bool = False) -> BaseExperiment | None:
    """Look up experiment by name, apply task to cfg, return instance.

    Args:
        name: Experiment class name, or "list" to print all registered experiments.
        cfg: Run config; cfg.task is set from the experiment's task attribute.
        silent: If True, return None instead of raising when name is not a known experiment.
                Use this when resolving --task, which may be a plain task name.
    """
    _load_all_experiments()

    if name == "list":
        print(f"\n{'='*60}\n Registered experiments ({len(_REGISTRY)})\n{'='*60}")
        for exp_name in sorted(_REGISTRY):
            cls = _REGISTRY[exp_name]
            # Find nearest experiment parent (skip BaseExperiment/object)
            parent = next(
                (
                    c.__name__
                    for c in cls.__mro__[1:]
                    if isinstance(c, type)
                    and issubclass(c, BaseExperiment)
                    and c not in (BaseExperiment, object)
                ),
                None,
            )
            task_val = getattr(cls, "task", "?")
            doc = (cls.__doc__ or "").strip().split("\n")[0]
            parent_str = f"  (extends: {parent})" if parent else ""
            print(f"  {exp_name}{parent_str}  task={task_val!r}")
            if doc:
                print(f"    └─ {doc}")
        print(f"{'='*60}\n")
        raise SystemExit(0)

    if name not in _REGISTRY:
        if silent:
            return None
        available = "\n  ".join(sorted(_REGISTRY))
        raise ValueError(f"Experiment '{name}' not found.\nAvailable:\n  {available}")

    exp = _REGISTRY[name]()
    cfg.experiment = name
    cfg.task = exp.task
    cfg.algo = exp.algo

    print(f"[INFO] Experiment '{name}': task={exp.task!r}" + (f" algo={exp.algo!r}"))
    return exp


def experiment_for_task(task: str) -> BaseExperiment:
    """Resolve a bare task string to its base-variant experiment instance.

    Compatibility helper for scripts that key on task names (via
    run.get_env_cfg). Picks the base-most experiment (shortest MRO) among those declaring
    this task, so default structural params (n=6, thrust_scale_range=None, ...) match the
    pre-refactor factory-default call. The main train/play path resolves the experiment by
    name instead and does not use this.
    """
    _load_all_experiments()
    matches = [c for c in _REGISTRY.values() if getattr(c, "task", None) == task]
    if not matches:
        available = ", ".join(sorted({getattr(c, "task", "?") for c in _REGISTRY.values()}))
        raise ValueError(f"No experiment defines task={task!r}. Known tasks: {available}")
    cls = min(matches, key=lambda c: len(c.__mro__))
    return cls()
