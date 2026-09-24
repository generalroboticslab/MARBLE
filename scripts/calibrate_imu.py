#!/usr/bin/env python3
"""Verify the fixed IMU placement (imu_site of the Sim_Model MuJoCo model) against live TM171 data."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "hardware_bindings"))

from ballbot_runtime import (  # noqa: E402
    IMU_POSITION_BASE_M,
    IMU_QUAT_BASE_SENSOR_WXYZ,
    JOINT_LOWER_M,
    JOINT_UPPER_M,
    R_BASE_SENSOR,
    SLIDER_ANCHORS_BASE_M,
    SLIDER_AXES_BASE,
    ImuState,
)
from hardware_bindings.imu.py_imu import IMU  # noqa: E402

# Rolling hull radius; the base origin is the shell centre, one radius above ground.
# Keep equal to SHELL_RADIUS_M in scripts/analyze_policy_log.py.
SHELL_RADIUS_M = 0.19


def run_text(imu: IMU) -> None:
    last_counter = -1
    while True:
        counter = int(imu.counter)
        if counter == last_counter:
            time.sleep(0.002)
            continue
        last_counter = counter
        state = ImuState.from_device(imu)
        with np.printoptions(precision=3, suppress=True):
            print(
                f"\rbase +X world={state.rotation_world_base[:, 0]} "
                f"+Y world={state.rotation_world_base[:, 1]} "
                f"+Z world={state.rotation_world_base[:, 2]} "
                f"omega_w={state.angular_velocity_world}",
                end="",
                flush=True,
            )
        time.sleep(0.02)


def run_capture_upright(imu: IMU, samples: int) -> None:
    """Solve for the base<-sensor mount rotation from a held upright reference pose.

    R_ws = R_wb R_bs, so holding the robot at the reference pose (R_wb = I) gives
    R_bs = R_ws directly. Gravity only pins roll and pitch, so this also *defines*
    the current heading as zero yaw.
    """
    from scipy.spatial.transform import Rotation

    print()
    print("Hold the robot at the reference pose: shell on the ground, chassis")
    print("vertical, +Y pointing the direction you want to call 'forward'.")
    print("That heading becomes the world yaw origin. Keep it still.")
    input("Press ENTER to capture...")

    rotations: list[np.ndarray] = []
    gyro: list[float] = []
    last_counter = -1
    deadline = time.monotonic() + 30.0
    while len(rotations) < samples:
        if time.monotonic() > deadline:
            raise RuntimeError("timed out collecting samples; is the IMU streaming?")
        counter = int(imu.counter)
        if counter == last_counter:
            time.sleep(0.001)
            continue
        last_counter = counter
        rotation_world_sensor = np.asarray(imu.rotation_matrix, dtype=np.float64)
        if rotation_world_sensor.shape != (3, 3):
            continue
        gyro.append(float(np.linalg.norm(np.asarray(imu.ang_vel))))
        rotations.append(rotation_world_sensor)

    # Mean rotation: average then project back onto SO(3).
    stack = np.stack(rotations)
    u, _, vt = np.linalg.svd(stack.mean(axis=0))
    correction = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)))])
    rotation_base_sensor = u @ correction @ vt

    spread = max(
        float(np.linalg.norm(Rotation.from_matrix(rotation_base_sensor.T @ r).as_rotvec()))
        for r in rotations
    )
    print(f"\ncaptured {len(rotations)} samples")
    print(f"  median gyro during capture : {float(np.median(gyro)):.4f} rad/s")
    print(f"  sample spread about mean   : {math.degrees(spread):.3f} deg")
    if float(np.median(gyro)) > 0.05 or math.degrees(spread) > 1.0:
        print("  WARNING: the robot moved during capture. Re-run while holding it still.")

    x, y, z, w = Rotation.from_matrix(rotation_base_sensor).as_quat()
    quat = np.array([w, x, y, z])
    if quat[0] < 0:
        quat = -quat

    # The heading at capture is arbitrary, and R_bs = Rz(heading) R_bs_true leaves
    # row 2 untouched. So base +Z expressed in sensor axes is the only part gravity
    # actually determines, and the only honest way to compare against CAD.
    def up_in_sensor(rotation: np.ndarray) -> np.ndarray:
        return rotation[2, :]

    def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
        return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))

    cad = Rotation.from_quat([0.0594277, -0.704605, -0.540253, 0.456209])  # xyzw
    captured_up = up_in_sensor(rotation_base_sensor)
    print("\n  base +Z in sensor axes (independent of capture heading):")
    print(f"    captured           {np.round(captured_up, 4)}")
    print(
        f"    shipped constant   {np.round(up_in_sensor(R_BASE_SENSOR), 4)}"
        f"   ({angle_deg(captured_up, up_in_sensor(R_BASE_SENSOR)):7.3f} deg off)"
    )
    # A remount in a 90 deg increment about the board normal is the likely failure
    # mode: the shipped constant is already CAD times a hand-patched +90 deg about Z.
    print("\n  vs CAD imu_site * rotz(k*90deg):")
    for k in range(4):
        candidate = up_in_sensor((cad * Rotation.from_euler("z", 90.0 * k, degrees=True)).as_matrix())
        print(f"    k={k} ({90 * k:3d} deg): {np.round(candidate, 4)}   ({angle_deg(captured_up, candidate):7.3f} deg off)")

    print("\nPaste into src/ballbot_runtime.py as IMU_QUAT_BASE_SENSOR_WXYZ:\n")
    print("IMU_QUAT_BASE_SENSOR_WXYZ = np.array(")
    print(f"    [{quat[0]:.10f}, {quat[1]:.10f}, {quat[2]:.10f}, {quat[3]:.10f}],")
    print("    dtype=np.float64,")
    print(")")
    print("\nNothing was written; update the constant by hand, then re-run --viz to confirm")
    print("tilt reads ~0 at the reference pose.")


def run_viz(imu: IMU, port: int) -> None:
    import viser
    from scipy.spatial.transform import Rotation

    def mat_to_wxyz(matrix: np.ndarray) -> np.ndarray:
        x, y, z, w = Rotation.from_matrix(matrix).as_quat()
        return np.array([w, x, y, z])

    server = viser.ViserServer(host="0.0.0.0", port=port)

    server.scene.add_grid("/grid", width=2.0, height=2.0, cell_size=0.1, plane="xy")
    # The hull rolls independently of the chassis and we have no ball encoder, so
    # it stays fixed here purely as a size reference.
    server.scene.add_icosphere(
        "/shell",
        radius=SHELL_RADIUS_M,
        color=(120, 140, 160),
        wireframe=True,
        subdivisions=2,
        position=(0.0, 0.0, SHELL_RADIUS_M),
    )
    # World vertical through the shell centre; the gap between this and the base
    # +Z arrow is the tilt.
    server.scene.add_line_segments(
        "/up_reference",
        points=np.array([[[0.0, 0.0, SHELL_RADIUS_M], [0.0, 0.0, SHELL_RADIUS_M + 0.34]]]),
        colors=(110, 110, 110),
        line_width=2.0,
    )

    base = server.scene.add_frame(
        "/base",
        axes_length=0.16,
        axes_radius=0.005,
        position=(0.0, 0.0, SHELL_RADIUS_M),
    )
    server.scene.add_box("/base/chassis", color=(70, 80, 95), dimensions=(0.11, 0.11, 0.15))
    # Reference pose (docs/SETUP.md): +Y forward, +Z up.
    server.scene.add_arrows(
        "/base/forward",
        points=np.array([[[0.0, 0.0, 0.0], [0.0, 0.30, 0.0]]]),
        colors=(60, 220, 110),
        shaft_radius=0.006,
        head_radius=0.018,
        head_length=0.04,
    )
    server.scene.add_arrows(
        "/base/up",
        points=np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.30]]]),
        colors=(90, 160, 255),
        shaft_radius=0.006,
        head_radius=0.018,
        head_length=0.04,
    )
    server.scene.add_line_segments(
        "/base/sliders",
        points=np.stack(
            [
                np.stack(
                    [
                        SLIDER_ANCHORS_BASE_M[i] + JOINT_LOWER_M * SLIDER_AXES_BASE[i],
                        SLIDER_ANCHORS_BASE_M[i] + JOINT_UPPER_M * SLIDER_AXES_BASE[i],
                    ]
                )
                for i in range(3)
            ]
        ),
        colors=(255, 170, 60),
        line_width=4.0,
    )
    # Fixed sensor pose on the chassis; its triad is visibly tilted against the
    # base triad, which is exactly the offset R_BASE_SENSOR removes.
    server.scene.add_frame(
        "/base/imu",
        axes_length=0.08,
        axes_radius=0.0035,
        position=tuple(IMU_POSITION_BASE_M.tolist()),
        wxyz=tuple(IMU_QUAT_BASE_SENSOR_WXYZ.tolist()),
    )
    server.scene.add_icosphere("/base/imu/dot", radius=0.012, color=(255, 180, 40))
    server.scene.add_label("/base/imu/label", "IMU", position=(0.04, 0.04, 0.04))

    server.gui.add_markdown(
        "Large triad + solid body = **robot base** (`rotation_world_base`).\n\n"
        "Small triad at the orange dot = **IMU**, drawn at its real mounting "
        "rotation. Grey line is world vertical.\n\n"
        "Reference pose: **+Y forward**, **+Z up**."
    )
    readout = server.gui.add_markdown("")

    last_counter = -1
    last_scene = 0.0
    last_gui = 0.0
    last_sample = time.monotonic()
    while True:
        counter = int(imu.counter)
        now = time.monotonic()
        if counter == last_counter:
            if now - last_sample > 2.0:
                readout.content = "**IMU telemetry is stale.**"
                last_sample = now
            time.sleep(0.001)
            continue
        last_counter = counter
        last_sample = now

        try:
            state = ImuState.from_device(imu)
        except ValueError:
            continue
        rotation = state.rotation_world_base

        if now - last_scene >= 1.0 / 60.0:
            base.wxyz = mat_to_wxyz(rotation)
            last_scene = now

        if now - last_gui >= 0.1:
            roll, pitch, yaw = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
            tilt = math.degrees(math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0))))
            omega = state.angular_velocity_world
            readout.content = (
                f"### Tilt from vertical: {tilt:5.2f} deg\n\n"
                f"| | |\n|---|---|\n"
                f"| roll / pitch / yaw [deg] | {roll:7.2f} {pitch:7.2f} {yaw:7.2f} |\n"
                f"| omega world [rad/s] | {omega[0]:6.3f} {omega[1]:6.3f} {omega[2]:6.3f} |\n"
                f"| qos | {int(imu.qos)} |\n"
                f"| temperature [C] | {int(imu.temperature)} |\n"
                f"| update rate [Hz] | {int(imu.updateRate)} |\n"
            )
            last_gui = now


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display live orientation using the fixed MuJoCo IMU transform"
    )
    parser.add_argument("--imu-port", default=os.environ.get("BALLBOT_IMU_PORT"))
    parser.add_argument(
        "--viz", action="store_true", help="serve a live 3D view of the base orientation"
    )
    parser.add_argument("--viz-port", type=int, default=8080)
    parser.add_argument(
        "--capture-upright",
        action="store_true",
        help="solve IMU_QUAT_BASE_SENSOR_WXYZ from a held upright reference pose",
    )
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()

    print(f"Fixed IMU position in base [m]: {IMU_POSITION_BASE_M}")
    print(f"Fixed IMU quaternion base<-sensor [wxyz]: {IMU_QUAT_BASE_SENSOR_WXYZ}")
    print(
        "No runtime IMU calibration is written; the model's imu_site "
        "(src/ballbot_runtime.py) is authoritative."
    )

    imu = IMU(port_name=args.imu_port)
    time.sleep(1.0)
    try:
        if args.capture_upright:
            run_capture_upright(imu, args.samples)
        elif args.viz:
            run_viz(imu, args.viz_port)
        else:
            run_text(imu)
    except KeyboardInterrupt:
        print()
    finally:
        imu.shutdown()


if __name__ == "__main__":
    main()
