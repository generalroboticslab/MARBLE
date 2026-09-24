#!/usr/bin/env python3
"""Summarise a run log written by `ballbot_terminal.py` (`PolicyRunLogger`).

Reports per log:
1. Policy step rate and stalls.
2. Actuator lag per joint: `target_m` vs `virtual_pos` vs measured joint position.
   Simulation has no transport delay, so any lag here is sim-to-real gap.
3. Command tracking, with body velocity from the no-slip rolling constraint
   v = R * (omega_y, -omega_x) (there is no odometry).

Usage:
    python3 scripts/analyze_policy_log.py logs/trained_policy_*.msgpack
    python3 scripts/analyze_policy_log.py --self-check
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import msgpack
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ballbot_runtime import JOINT_NAMES, POLICY_FRAME_DIM  # noqa: E402

# Rolling hull (shell) radius of the trained plant, in metres; the default for --radius.
SHELL_RADIUS_M = 0.19

# Slice layout of the 23-element frame, in the order TrainedPolicy.build_frame packs it.
# `obs_terms` in the log header records the same thing; check_frame_layout asserts they agree.
OBS_SLICES = {
    "ang_vel_world": slice(0, 3),
    "cmd_world_xy": slice(3, 5),
    "joint_pos": slice(5, 8),
    "joint_vel": slice(8, 11),
    "last_action": slice(11, 14),
    "rotation_world_base": slice(14, 23),
}

# The physical slider band. A joint reading outside this means homing failed and the run
# is not interpretable.
JOINT_LIMIT_M = 0.11


def load_log(path: Path) -> tuple[dict, list[dict], list[dict]]:
    """Split a log into (header, policy steps, per-tick timing records)."""
    with path.open("rb") as handle:
        unpacker = msgpack.Unpacker(handle, raw=False)
        header = next(unpacker)
        steps: list[dict] = []
        ticks: list[dict] = []
        for obj in unpacker:
            if obj.get("type") == "tick":
                ticks.append(obj)
            else:
                steps.append(obj)
    if header.get("type") != "header":
        raise ValueError(f"{path} does not start with a header object")
    return header, steps, ticks


def check_frame_layout(header: dict) -> None:
    """Fail loudly if the log was written by a runtime with a different frame order."""
    if int(header.get("frame_dim", 0)) != POLICY_FRAME_DIM:
        raise ValueError(
            f"log frame_dim {header.get('frame_dim')} != runtime {POLICY_FRAME_DIM}"
        )
    logged = [term.split("[")[0] for term in header.get("obs_terms", [])]
    if logged and logged != list(OBS_SLICES):
        raise ValueError(f"log obs_terms {logged} != expected {list(OBS_SLICES)}")


def column(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def obs_term(rows: list[dict], name: str) -> np.ndarray:
    return column(rows, "obs")[:, OBS_SLICES[name]]


# Minimum overlap before trusting a lag; a few samples correlate near +/-1 for any data.
MIN_LAG_OVERLAP = 30

# Below this the lag fits noise (hardware traces score 0.5-0.91, pure noise 0.12-0.27).
MIN_LAG_CORR = 0.3


def estimate_lag(reference: np.ndarray, response: np.ndarray, max_lag: int = 25) -> tuple[int, float]:
    """Whole-sample lag maximising correlation of `response` against `reference`.

    Returns (lag_in_samples, correlation). Returns (0, 0.0) when the answer would not mean
    anything: a flat (zero-variance) signal, or a trace too short to leave MIN_LAG_OVERLAP
    samples of overlap at the shift being tested.
    """
    n = min(len(reference), len(response))
    # Never test a shift that leaves fewer than MIN_LAG_OVERLAP paired samples.
    usable = min(max_lag, n - MIN_LAG_OVERLAP)
    best_lag, best_corr = 0, -1.0
    for lag in range(max(usable, 0) + 1):
        shifted_ref = reference[: n - lag]
        shifted_res = response[lag:n]
        if shifted_ref.std() < 1e-12 or shifted_res.std() < 1e-12:
            continue
        corr = float(np.corrcoef(shifted_ref, shifted_res)[0, 1])
        if corr > best_corr:
            best_lag, best_corr = lag, corr
    return best_lag, max(best_corr, 0.0)


def rolling_velocity(ang_vel_world: np.ndarray, radius: float = SHELL_RADIUS_M) -> np.ndarray:
    """World-frame body velocity of a sphere rolling without slip: R * (w_y, -w_x)."""
    return radius * np.stack([ang_vel_world[:, 1], -ang_vel_world[:, 0]], axis=1)


def report_timing(header: dict, steps: list[dict], ticks: list[dict]) -> None:
    t = column(steps, "t_s")
    dt_ms = np.diff(t) * 1000.0
    settled = dt_ms[dt_ms < 100.0]  # a stall is a separate fault, not a rate measurement
    target_ms = 1000.0 / float(header["policy_hz"])

    print("  policy rate")
    if settled.size == 0:
        # Every period was a stall; report it rather than fail on an empty array.
        print(
            f"    NO settled steps: every one of {dt_ms.size} periods exceeded 100 ms "
            f"(median {np.median(dt_ms):.0f} ms, max {dt_ms.max() / 1000.0:.1f} s). "
            "The control loop was not running."
        )
        return
    print(
        f"    dt: min {settled.min():.1f}  median {np.median(settled):.1f}  "
        f"mean {settled.mean():.1f}  p99 {np.percentile(settled, 99):.1f} ms "
        f"(target {target_ms:.1f})"
    )
    print(
        f"    achieved {1000.0 / settled.mean():.1f} Hz vs POLICY_HZ "
        f"{float(header['policy_hz']):.1f}"
    )
    stalls = int((dt_ms >= 100.0).sum())
    if stalls:
        print(f"    {stalls} stall(s) over 100 ms, max {dt_ms.max() / 1000.0:.1f} s")

    if ticks:
        for key, label in (
            ("loop_dt", "loop period"),
            ("t_imu", "imu read"),
            ("t_policy", "policy inference"),
            ("t_serial", "serial write"),
            ("t_total", "tick total"),
        ):
            if key not in ticks[0]:
                continue
            values = column(ticks, key) * 1000.0
            print(
                f"    {label:<17} mean {values.mean():6.2f}  p95 {np.percentile(values, 95):6.2f}  "
                f"max {values.max():7.2f} ms"
            )
    else:
        # This log records only policy steps. The gate fires on the first loop tick at
        # or past the policy period, so dt = period + U(0, loop_period) and the mean
        # overshoots by half a loop period. Invert that to recover the loop rate.
        loop_ms = 2.0 * (settled.mean() - target_ms)
        # Guard against 1000/~0 when the policy sat exactly on target: the overshoot model
        # has nothing to invert, which is itself the good news.
        if loop_ms > 0.1:
            print(
                f"    inferred loop period {loop_ms:.1f} ms ({1000.0 / loop_ms:.0f} Hz) "
                "-- this log has no per-tick timing records"
            )
        else:
            print(
                "    policy rate is on target, so the loop period cannot be inferred from "
                "it -- this log has no per-tick timing records"
            )


def report_tracking(steps: list[dict]) -> None:
    target = column(steps, "target_m")
    virtual = column(steps, "virtual_pos")
    actual = obs_term(steps, "joint_pos")
    dt_s = float(np.median(np.diff(column(steps, "t_s"))))

    print("  actuator lag (simulation applies the target with no transport delay)")
    print(
        f"    {'joint':<22}{'tgt->virt':>12}{'virt->act':>12}{'tgt->act':>12}"
        f"{'r':>7}{'amp kept':>10}"
    )
    for index, name in enumerate(JOINT_NAMES):
        lag_tv, _ = estimate_lag(target[:, index], virtual[:, index])
        lag_va, _ = estimate_lag(virtual[:, index], actual[:, index])
        lag_ta, corr_ta = estimate_lag(target[:, index], actual[:, index])
        amplitude = actual[:, index].std() / max(target[:, index].std(), 1e-12)
        # A weak correlation means the lag is whatever noise happened to line up; say so
        # rather than let a confident-looking millisecond figure into the record.
        flag = "" if corr_ta >= MIN_LAG_CORR else "  <- r too low, lag not meaningful"
        print(
            f"    {name.replace('base_link_', ''):<22}"
            f"{lag_tv * dt_s * 1000:9.0f} ms{lag_va * dt_s * 1000:9.0f} ms"
            f"{lag_ta * dt_s * 1000:9.0f} ms{corr_ta:7.2f}{amplitude:10.2f}{flag}"
        )

    excursion = np.abs(actual).max()
    if excursion > JOINT_LIMIT_M:
        worst = JOINT_NAMES[int(np.argmax(np.abs(actual).max(axis=0)))]
        print(
            f"    WARNING: {worst} reached {excursion:.3f} m, outside the "
            f"+/-{JOINT_LIMIT_M:.3f} m band -- homing failed, run is not interpretable"
        )


def report_command(steps: list[dict], radius: float) -> None:
    command = obs_term(steps, "cmd_world_xy")
    velocity = rolling_velocity(obs_term(steps, "ang_vel_world"), radius)
    t = column(steps, "t_s")

    speed = np.linalg.norm(command, axis=1)
    moving = speed > 1e-6
    if not moving.any():
        print("  command tracking: no nonzero command in this log")
        return
    vel, cmd = velocity[moving], command[moving]
    unit = cmd / np.linalg.norm(cmd, axis=1)[:, None]
    along = np.sum(vel * unit, axis=1)
    across = np.abs(vel[:, 0] * unit[:, 1] - vel[:, 1] * unit[:, 0])

    dt = np.diff(t)
    path = float(np.sum(np.linalg.norm(velocity[:-1], axis=1) * dt))
    net = float(np.linalg.norm(np.sum(velocity[:-1] * dt[:, None], axis=0)))

    print("  command tracking (velocity from the rolling constraint, no odometry)")
    print(f"    commanded      {speed[moving].mean():.3f} m/s")
    print(f"    along command  {along.mean():+.3f} m/s  (ratio {along.mean() / speed[moving].mean():.2f})")
    print(f"    across command {across.mean():.3f} m/s")
    print(f"    cmd error      {np.linalg.norm(vel - cmd, axis=1).mean():.3f} m/s")
    print(f"    path {path:.1f} m for {net:.1f} m net displacement")

    print(f"    {'commanded direction':<24}{'n':>6}{'achieved':>20}{'|v|':>8}{'heading err':>13}")
    directions, inverse, counts = np.unique(
        np.round(unit, 3), axis=0, return_inverse=True, return_counts=True
    )
    for index, (direction, count) in enumerate(zip(directions, counts)):
        if count < 40:
            continue
        mean_v = vel[inverse == index].mean(axis=0)
        error = np.degrees(
            np.arctan2(mean_v[1], mean_v[0]) - np.arctan2(direction[1], direction[0])
        )
        error = (error + 180.0) % 360.0 - 180.0
        print(
            f"    [{direction[0]:+.2f},{direction[1]:+.2f}]{'':<13}{count:6d}"
            f"    [{mean_v[0]:+.3f},{mean_v[1]:+.3f}]{np.linalg.norm(mean_v):8.3f}"
            f"{error:+12.0f}d"
        )


def analyze(path: Path, radius: float) -> None:
    header, steps, ticks = load_log(path)
    check_frame_layout(header)
    if len(steps) < 10:
        print(f"\n{path.name}: only {len(steps)} policy steps, skipping")
        return
    duration = column(steps, "t_s")[-1] - column(steps, "t_s")[0]
    print(
        f"\n{path.name}  schema {header.get('schema_version')}  "
        f"{len(steps)} steps  {duration:.1f} s  "
        f"vel_limit {steps[0]['vel_limit_rad_s']:.0f} rad/s  "
        f"stiffness {steps[0]['stiffness']:.0f}  damping {steps[0]['damping']:.0f}"
    )
    report_timing(header, steps, ticks)
    report_tracking(steps)
    report_command(steps, radius)


def _self_check() -> None:
    """Verify the two estimators the conclusions rest on, without hardware."""
    t = np.arange(0, 20, 0.02)

    # A pure delay is recovered exactly, and survives noise plus a scale change.
    for delay in (0, 1, 5, 13):
        reference = np.sin(2 * np.pi * 0.7 * t) + 0.4 * np.sin(2 * np.pi * 1.9 * t)
        response = np.roll(reference, delay) * 0.7
        response += 0.01 * np.random.default_rng(0).standard_normal(len(t))
        lag, corr = estimate_lag(reference, response)
        assert lag == delay, f"estimate_lag returned {lag}, expected {delay}"
        assert corr > 0.95, f"correlation {corr:.3f} too low for a clean delay"

    # A flat signal has no defined lag and must not produce a nan.
    lag, corr = estimate_lag(np.zeros(100), np.zeros(100))
    assert (lag, corr) == (0, 0.0), f"flat signal gave {(lag, corr)}"

    # Rolling constraint: spinning about +y rolls toward +x at R*omega, and about
    # +x toward -y. Spin about the vertical is pure yaw and must not translate.
    omega = 3.0
    for gyro, expected in (
        ((0.0, omega, 0.0), (SHELL_RADIUS_M * omega, 0.0)),
        ((omega, 0.0, 0.0), (0.0, -SHELL_RADIUS_M * omega)),
        ((0.0, 0.0, omega), (0.0, 0.0)),
    ):
        got = rolling_velocity(np.array([gyro]))[0]
        assert np.allclose(got, expected), f"rolling_velocity({gyro}) = {got}, want {expected}"

    # The frame layout the slices assume must cover the frame exactly, with no overlap.
    covered = sorted(i for s in OBS_SLICES.values() for i in range(s.start, s.stop))
    assert covered == list(range(POLICY_FRAME_DIM)), "OBS_SLICES does not tile the frame"

    print("analyze_policy_log self-check passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", type=Path, help="msgpack logs to analyze")
    parser.add_argument(
        "--radius",
        type=float,
        default=SHELL_RADIUS_M,
        help="hull radius in metres for the rolling-velocity estimate",
    )
    parser.add_argument(
        "--self-check", action="store_true", help="verify the estimators and exit"
    )
    args = parser.parse_args()

    if args.self_check:
        _self_check()
        return
    if not args.logs:
        parser.error("give at least one log, or --self-check")
    for path in args.logs:
        analyze(path, args.radius)
    print()


if __name__ == "__main__":
    main()
