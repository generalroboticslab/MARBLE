#!/usr/bin/env python3
"""Run the deployed policy code against the exported MuJoCo plant, without hardware.

Uses `ballbot_runtime.TrainedPolicy`, the class the robot runs, in CPU MuJoCo. A wrong
observation order, frame convention or action scale shows up as lost tracking.
`--emulate-vel-limit` and `--travel-band` add the hardware's velocity ceiling and travel
clamp, which the simulated plant lacks. `--robot-xml` needs a model whose meshes resolve;
the shipped policies/*/robot.xml files do not.

Usage:
    python3 scripts/policy_sim_check.py \\
        --robot-xml /path/to/BallbotVelComplexFlatDRLatency/robot.xml \\
        --policy-path policies/BallbotVelComplexFlatDRLatency/policy_deployed.pt \\
        --command 0.3 0.0

    # sweep the hardware velocity ceiling
    python3 scripts/policy_sim_check.py ... --emulate-vel-limit 2 5 10 20 30
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import mujoco  # noqa: E402

from ballbot_runtime import (  # noqa: E402
    ImuState,
    JOINT_NAMES,
    MM_PER_TURN,
    POLICY_FRAME_DIM,
    TrainedPolicy,
)


# Defaults; main() replaces them from the artifact's env_config.yaml.
SIM_DT = 0.005
DECIMATION = 8
POLICY_DT = SIM_DT * DECIMATION
SETTLE_S = 0.5

# Reachable slider band under each zeroing scheme, in metres of offset from centre.
#   sim           - the trained plant: full mechanical range, no margin.
#   rezeroed      - encoder zero at the 114 mm centre; 10 mm margin at each hardstop.
#   hardstop-zero - the released mapper: zero at the 0 mm hardstop, so SAFE_MAX_TRAVEL_MM
#                   (195 mm, inside the encoder's +/-12.5 rad) caps the top at +81 mm.
TRAVEL_BANDS = {
    "sim": (-0.114, 0.106),
    "rezeroed": (-0.104, 0.096),
    "hardstop-zero": (-0.104, 0.081),
}

# ballbot_terminal runs its motor-reference loop at 100 Hz.
CONTROL_HZ = 100.0
TICKS_PER_POLICY_STEP = int(round(CONTROL_HZ * POLICY_DT))
SUBSTEPS_PER_TICK = DECIMATION // TICKS_PER_POLICY_STEP


def adopt_artifact_timing(deploy_config) -> None:
    """Take the simulation timing from the artifact's env_config.yaml."""
    global SIM_DT, DECIMATION, POLICY_DT, TICKS_PER_POLICY_STEP, SUBSTEPS_PER_TICK
    if deploy_config is None:
        return
    SIM_DT = float(deploy_config.timestep_s)
    DECIMATION = int(deploy_config.decimation)
    POLICY_DT = SIM_DT * DECIMATION
    TICKS_PER_POLICY_STEP = max(1, int(round(CONTROL_HZ * POLICY_DT)))
    SUBSTEPS_PER_TICK = max(1, DECIMATION // TICKS_PER_POLICY_STEP)


def build_model(robot_xml: Path) -> mujoco.MjModel:
    """Compile the exported robot with a ground plane added.

    The exported spec deliberately carries no floor (the scene supplies one during
    training), so the hull would free-fall without this.
    """
    spec = mujoco.MjSpec.from_file(str(robot_xml))
    spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        pos=[0.0, 0.0, 0.0],
        contype=1,
        conaffinity=1,
    )
    return spec.compile()


class SimPlant:
    """Exported ballbot model plus the state accessors the observation frame needs."""

    def __init__(self, robot_xml: Path):
        self.model = build_model(robot_xml)
        self.data = mujoco.MjData(self.model)
        self.base_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.base_bid < 0:
            raise RuntimeError("base_link body not found in robot XML")

        self.qpos_adr = np.empty(3, dtype=int)
        self.dof_adr = np.empty(3, dtype=int)
        self.ctrl_idx = np.empty(3, dtype=int)
        for idx, joint_name in enumerate(JOINT_NAMES):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                raise RuntimeError(f"joint {joint_name} not found in robot XML")
            self.qpos_adr[idx] = self.model.jnt_qposadr[jid]
            self.dof_adr[idx] = self.model.jnt_dofadr[jid]
            aid = next(
                a for a in range(self.model.nu) if self.model.actuator_trnid[a, 0] == jid
            )
            self.ctrl_idx[idx] = aid

        self._vel6 = np.zeros(6, dtype=np.float64)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        # Sphere centre one radius up; the exported spec recentres the hull on the body
        # origin, so this rests the shell on the plane without a drop transient.
        self.data.qpos[2] = 0.19
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    @property
    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[self.qpos_adr].astype(np.float32)

    @property
    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[self.dof_adr].astype(np.float32)

    @property
    def rotation_world_base(self) -> np.ndarray:
        return self.data.xmat[self.base_bid].reshape(3, 3).copy()

    def base_velocity_world(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (linear, angular) world-frame velocity of base_link.

        mj_objectVelocity with flg_local=0 is used rather than reading qvel[3:6]
        directly: a free joint reports its angular velocity in the body frame, while
        the trained observation term is `root_link_ang_vel_w` (world).
        mjData.cvel convention: [:3] angular, [3:] linear.
        """
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_bid, self._vel6, 0
        )
        return self._vel6[3:].copy(), self._vel6[:3].copy()

    def imu_state(self) -> ImuState:
        _, ang_vel_w = self.base_velocity_world()
        return ImuState(self.rotation_world_base, ang_vel_w)


def rad_s_to_m_s(vel_limit_rad_s: float) -> float:
    """Motor velocity ceiling (rad/s) to slider travel speed (m/s) via the lead screw."""
    return vel_limit_rad_s * MM_PER_TURN / (2.0 * math.pi) / 1000.0


def run_episode(
    plant: SimPlant,
    policy: TrainedPolicy,
    command_xy: np.ndarray,
    duration_s: float,
    vel_limit_rad_s: float | None,
    travel_clip_m: tuple[float, float] | None = None,
    interp: str = "zoh",
) -> dict:
    plant.reset()
    policy.reset()

    slew_cap_m_per_step = (
        rad_s_to_m_s(vel_limit_rad_s) * POLICY_DT if vel_limit_rad_s else None
    )

    # Settle with the sliders centred so the first policy frame sees a resting hull.
    for _ in range(int(SETTLE_S / SIM_DT)):
        mujoco.mj_step(plant.model, plant.data)

    last_action = np.zeros(3, dtype=np.float32)
    applied_target = np.zeros(3, dtype=np.float32)
    previous_target = np.zeros(3, dtype=np.float32)

    vel_samples: list[np.ndarray] = []
    demanded_step_delta: list[np.ndarray] = []
    achieved_slider_speed: list[np.ndarray] = []
    saturated_steps = 0
    clipped_steps = 0
    n_policy_steps = int(duration_s / POLICY_DT)

    for _ in range(n_policy_steps):
        frame = TrainedPolicy.build_frame(
            command_xy,
            plant.imu_state(),
            plant.joint_pos,
            plant.joint_vel,
            last_action,
        )
        if frame.shape != (POLICY_FRAME_DIM,):
            raise RuntimeError(f"frame must be {POLICY_FRAME_DIM}D, got {frame.shape}")

        target, action = policy.act_from_frame(frame)
        last_action = action

        demanded_step_delta.append(np.abs(target - applied_target))
        if slew_cap_m_per_step is not None:
            step = np.clip(
                target - applied_target, -slew_cap_m_per_step, slew_cap_m_per_step
            )
            if np.any(np.abs(target - applied_target) > slew_cap_m_per_step + 1e-12):
                saturated_steps += 1
            applied_target = applied_target + step
        else:
            applied_target = target.copy()

        if travel_clip_m is not None:
            lower, upper = travel_clip_m
            if np.any(applied_target < lower - 1e-12) or np.any(applied_target > upper + 1e-12):
                clipped_steps += 1
            applied_target = np.clip(applied_target, lower, upper).astype(np.float32)

        if interp == "zoh":
            # Simulation semantics: apply the new target immediately, hold for the step.
            plant.data.ctrl[plant.ctrl_idx] = applied_target
            for _ in range(DECIMATION):
                mujoco.mj_step(plant.model, plant.data)
                achieved_slider_speed.append(np.abs(plant.joint_vel))
        else:
            # terminal-alpha: the per-tick blend an earlier ballbot_terminal used.
            reference = previous_target.copy()
            for tick in range(TICKS_PER_POLICY_STEP):
                alpha = min(tick / TICKS_PER_POLICY_STEP, 1.0)
                reference = reference + alpha * (applied_target - reference)
                plant.data.ctrl[plant.ctrl_idx] = reference
                for _ in range(SUBSTEPS_PER_TICK):
                    mujoco.mj_step(plant.model, plant.data)
                    achieved_slider_speed.append(np.abs(plant.joint_vel))
        previous_target = applied_target.copy()

        lin_vel_w, _ = plant.base_velocity_world()
        vel_samples.append(lin_vel_w[:2].copy())

    vel = np.asarray(vel_samples)
    # Judge the second half only: the first seconds are the start-up transient, and the
    # ballbot reaches command speed by bursting rather than settling instantly.
    steady = vel[len(vel) // 2 :]
    demanded = np.asarray(demanded_step_delta)
    achieved = np.asarray(achieved_slider_speed)

    return {
        "mean_speed": float(np.linalg.norm(steady, axis=1).mean()),
        "mean_vel_xy": steady.mean(axis=0),
        "cmd_error": float(np.linalg.norm(steady.mean(axis=0) - command_xy)),
        # Target delta per policy step converted to the motor rate it implies.
        "demand_p50_rad_s": _delta_to_rad_s(np.percentile(demanded, 50)),
        "demand_p95_rad_s": _delta_to_rad_s(np.percentile(demanded, 95)),
        "demand_max_rad_s": _delta_to_rad_s(demanded.max()),
        "achieved_p95_rad_s": _speed_to_rad_s(np.percentile(achieved, 95)),
        "achieved_max_rad_s": _speed_to_rad_s(achieved.max()),
        "saturated_frac": saturated_steps / max(n_policy_steps, 1),
        "clipped_frac": clipped_steps / max(n_policy_steps, 1),
    }


def _delta_to_rad_s(delta_m: float) -> float:
    """Per-policy-step target delta (m) to the motor speed (rad/s) needed to track it."""
    return _speed_to_rad_s(delta_m / POLICY_DT)


def _speed_to_rad_s(speed_m_s: float) -> float:
    return speed_m_s * 1000.0 / MM_PER_TURN * 2.0 * math.pi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot-xml", required=True, type=Path)
    parser.add_argument("--policy-path", required=True, type=Path)
    parser.add_argument(
        "--command", nargs=2, type=float, default=[0.3, 0.0], metavar=("VX", "VY")
    )
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument(
        "--emulate-vel-limit",
        nargs="*",
        type=float,
        default=None,
        metavar="RAD_S",
        help="Motor velocity ceilings to sweep. Omit for the unlimited simulation plant.",
    )
    parser.add_argument(
        "--travel-band",
        choices=sorted(TRAVEL_BANDS),
        default=None,
        help=(
            "Emulate a hardware travel clamp on the commanded target. "
            "'sim' = the trained plant, 'rezeroed' = encoder zero at the 114 mm centre, "
            "'hardstop-zero' = the released mapper (zero at the 0 mm hardstop, capped by "
            "the encoder's +/-12.5 rad range)."
        ),
    )
    parser.add_argument(
        "--interp",
        nargs="*",
        choices=("zoh", "terminal-alpha"),
        default=["zoh"],
        help=(
            "How the policy target reaches the drives. 'zoh' matches simulation and the "
            "default terminal; 'terminal-alpha' models an older terminal that blended "
            "toward each target over 100 Hz ticks."
        ),
    )
    parser.add_argument(
        "--action-scale",
        nargs="*",
        type=float,
        default=[1.0],
        help=(
            "Values of ballbot_terminal's --action-scale / '[' ']' knob to sweep. "
            "1.0 is the trained action space; lower values shrink it."
        ),
    )
    args = parser.parse_args()

    command_xy = np.asarray(args.command, dtype=np.float32)
    plant = SimPlant(args.robot_xml)
    # strict=False: must open mislabelled artifacts; this never drives a robot.
    policy = TrainedPolicy(args.policy_path, action_scale=1.0, strict=False)
    adopt_artifact_timing(policy.deploy_config)

    limits: list[float | None] = [None]
    if args.emulate_vel_limit:
        limits = list(args.emulate_vel_limit)

    bands = [args.travel_band] if args.travel_band else list(TRAVEL_BANDS)
    print(
        f"\ncommand=({command_xy[0]:+.2f},{command_xy[1]:+.2f}) m/s  "
        f"duration={args.duration:.0f}s  policy={args.policy_path.name}\n"
    )
    header = (
        f"{'interp':>14}  {'travel band':>14}  {'vel_limit':>9}  {'a_scale':>7}  "
        f"{'speed':>7}  {'cmd_err':>7}  {'slider p95':>10}  {'sat':>5}  {'clip':>5}"
    )
    print(header)
    print("-" * len(header))
    for interp in args.interp:
        for band in bands:
            for limit in limits:
                for scale in args.action_scale:
                    policy.set_action_scale(scale)
                    result = run_episode(
                        plant, policy, command_xy, args.duration, limit,
                        TRAVEL_BANDS[band], interp,
                    )
                    label = "none" if limit is None else f"{limit:.1f}"
                    print(
                        f"{interp:>14}  {band:>14}  {label:>9}  {scale:7.2f}  "
                        f"{result['mean_speed']:7.3f}  {result['cmd_error']:7.3f}  "
                        f"{result['achieved_p95_rad_s']:10.1f}  "
                        f"{result['saturated_frac']:5.2f}  {result['clipped_frac']:5.2f}"
                    )
    print(
        "\nspeed/cmd_err in m/s (second half of episode); demand and slider columns in "
        "motor rad/s; sat = fraction of policy steps hitting the velocity ceiling, "
        "clip = fraction hitting the travel band.\n"
    )


if __name__ == "__main__":
    main()
