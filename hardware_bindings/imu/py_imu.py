import gc
import importlib.util
import pathlib
from functools import lru_cache

import numpy as np
from tqdm import trange


@lru_cache(maxsize=None)
def _load_ext(name):
    path = pathlib.Path(__file__).parent / f"{name}.abi3.so"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is not built. Run ./scripts/prepare_orangepi.sh first (see docs/SETUP.md)."
        )
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




imu_nanobind = _load_ext("imu_nanobind")


class IMU (imu_nanobind.IMU):
    """
    IMU wrapper with optional rotation offset for mounting compensation.

    Args:
        port_name: Serial port name (e.g., "/dev/ttyACM0"). Auto-detect if None.
        rotation_offset: 3x3 numpy rotation matrix to apply to all orientation data.
                        E.g., 180° about Z: np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    """
    def __init__(self, port_name=None, rotation_offset: np.ndarray = None):
        if port_name is not None:
            super().__init__(port_name)
        else:
            super().__init__()
            self.findPortNameByDescription() # use `lsusb` to get port name

        self.should_print = False

        # Set rotation offset in C++ if provided
        if rotation_offset is not None:
            self.rotation_offset = np.asarray(rotation_offset, dtype=np.float32)

        self.run()

    def shutdown(self):
        self.close()
        gc.collect()
        print("IMU closed")

if __name__ == "__main__":
    # Try the package import first (works when hardware_bindings is importable as a
    # package). Fall back to common.publisher for direct runs
    # (python hardware_bindings/imu/py_imu.py): common/ is a subpackage of
    # hardware_bindings/, so add that directory (not the project root) to sys.path.
    import sys

    try:
        from hardware_bindings.common.publisher import DataPublisher
    except ImportError:
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from common.publisher import DataPublisher

    _repo_root = pathlib.Path(__file__).resolve().parents[2]
    _src = _repo_root / "src"
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
    from ballbot_runtime import (  # noqa: E402
        IMU_POSITION_BASE_M,
        IMU_QUAT_BASE_SENSOR_WXYZ,
        R_BASE_SENSOR,
        ImuState,
    )

    import time
    import viser
    from viser import uplot
    from scipy.spatial.transform import Rotation as R

    imu = IMU()
    publisher = DataPublisher('udp://localhost:9870', encoding="msgpack", broadcast=False)

    server = viser.ViserServer(host="0.0.0.0", port=8080)
    print("Viser server running at http://localhost:8080")
    print(
        f"IMU mount in base: pos={IMU_POSITION_BASE_M} m, "
        f"quat_bs_wxyz={IMU_QUAT_BASE_SENSOR_WXYZ}"
    )

    server.scene.add_frame("/world", axes_length=0.5, axes_radius=0.008)
    # Robot base orientation used by control (R_wb = R_ws R_bs^T), at world origin.
    base_frame = server.scene.add_frame("/base", axes_length=0.35, axes_radius=0.008)
    server.scene.add_label("/base/label", "Base", position=(0.0, 0.0, 0.40))
    # Fixed IMU pose on the robot (Sim_Model imu_site + board +90° Z).
    imu_on_base = server.scene.add_frame(
        "/base/imu",
        axes_length=0.18,
        axes_radius=0.005,
        position=tuple(IMU_POSITION_BASE_M.tolist()),
        wxyz=tuple(IMU_QUAT_BASE_SENSOR_WXYZ.tolist()),
    )
    server.scene.add_label("/base/imu/label", "IMU mount", position=(0.05, 0.05, 0.05))
    server.scene.add_icosphere(
        "/base/imu/marker",
        radius=0.015,
        color=(255, 180, 40),
        position=(0.0, 0.0, 0.0),
    )
    # Live sensor frame in world at the transformed IMU position.
    imu_world = server.scene.add_frame("/imu_world", axes_length=0.22, axes_radius=0.005)
    server.scene.add_label("/imu_world/label", "IMU (live)", position=(0.05, 0.05, 0.05))
    imu_world_marker = server.scene.add_icosphere(
        "/imu_world/marker",
        radius=0.012,
        color=(80, 180, 255),
        position=(0.0, 0.0, 0.0),
    )
    # Line from base origin to live IMU position.
    imu_offset_line = server.scene.add_spline_catmull_rom(
        "/imu_offset",
        positions=np.zeros((2, 3)),
        line_width=3.0,
        color=(255, 180, 40),
    )

    frames = {
        'quat':   server.scene.add_frame("/raw/quat",   axes_length=0.20, axes_radius=0.004),
        'euler':  server.scene.add_frame("/raw/euler",  axes_length=0.18, axes_radius=0.004),
        'matrix': server.scene.add_frame("/raw/matrix", axes_length=0.16, axes_radius=0.003),
    }
    for name in frames:
        server.scene.add_label(f"/raw/{name}/label", f"raw {name}", position=(0.22, 0, 0))

    gravity_spline = server.scene.add_spline_catmull_rom(
        "/raw/matrix/gravity", positions=np.zeros((2, 3)), line_width=4.0, color=(255, 80, 255))
    gravity_direct_spline = server.scene.add_spline_catmull_rom(
        "/raw/matrix/gravity_direct", positions=np.zeros((2, 3)), line_width=4.0, color=(200, 100, 100))

    plot_history, plot_idx = 200, [0]
    history = {k: np.zeros((plot_history, 3)) for k in ['acc', 'ang_vel', 'mag']}

    def make_xyz_plot(gui, title):
        gui.add_markdown(f"### {title}")
        init = (np.array([0.0]),) * 4
        series = (uplot.Series(label="t"), uplot.Series(label="X", stroke="red"),
                  uplot.Series(label="Y", stroke="green"), uplot.Series(label="Z", stroke="blue"))
        return gui.add_uplot(data=init, series=series)

    server.gui.add_markdown(
        "**Base** = control orientation (`R_wb`). "
        "**IMU mount** = fixed site pose on the base. "
        "**IMU (live)** = measured sensor frame at that offset in world."
    )
    plots = {
        'acc':     make_xyz_plot(server.gui, "Acceleration"),
        'ang_vel': make_xyz_plot(server.gui, "Angular Velocity"),
        'mag':     make_xyz_plot(server.gui, "Magnetometer (compass)"),
    }

    def xyzw_to_wxyz(q):
        return np.array([q[3], q[0], q[1], q[2]])

    def mat_to_wxyz(mat: np.ndarray) -> np.ndarray:
        q_xyzw = R.from_matrix(np.asarray(mat, dtype=np.float64)).as_quat()
        return xyzw_to_wxyz(q_xyzw)

    counter = 0
    try:
        for _ in trange(100000):
            while counter == imu.counter:
                time.sleep(1e-5)
            counter = imu.counter

            data ={
                "timeStamp": imu.timeStamp, "qos": imu.qos, "temperature": imu.temperature,
                "updateRate": imu.updateRate, "raw_acc": imu.raw_acc, "ang_vel": imu.ang_vel,
                "world_space_ang_vel": imu.world_space_ang_vel, "raw_mag": imu.raw_mag,
                "quat_xyzw": imu.quat_xyzw, "euler": imu.euler, "gravity_vec": imu.gravity_vec,
                "rotation_matrix": imu.rotation_matrix
            }
            publisher.publish({"sensor": data})

            R_ws = np.asarray(imu.rotation_matrix, dtype=np.float64)
            try:
                state = ImuState.from_device(imu)
                R_wb = state.rotation_world_base
            except ValueError:
                # Fall back if a sample fails the orthogonality check.
                R_wb = R_ws @ R_BASE_SENSOR.T

            base_frame.wxyz = mat_to_wxyz(R_wb)
            # Live IMU pose in world: p = R_wb p_base, R = R_ws
            p_imu_world = R_wb @ IMU_POSITION_BASE_M
            imu_world.position = tuple(p_imu_world.tolist())
            imu_world.wxyz = mat_to_wxyz(R_ws)
            imu_offset_line.positions = np.array([[0.0, 0.0, 0.0], p_imu_world])

            frames['quat'].wxyz   = xyzw_to_wxyz(imu.quat_xyzw)
            frames['euler'].wxyz  = xyzw_to_wxyz(R.from_euler('xyz', imu.euler, degrees=True).as_quat())
            frames['matrix'].wxyz = mat_to_wxyz(R_ws)
            gravity_direct_spline.positions = np.array([[0, 0, 0], imu.direct_gravity_vec * 0.3])
            gravity_spline.positions        = np.array([[0, 0, 0], imu.gravity_vec * 0.3])

            idx = plot_idx[0] % plot_history
            history['acc'][idx] = imu.raw_acc
            history['ang_vel'][idx] = imu.ang_vel
            history['mag'][idx] = imu.raw_mag
            plot_idx[0] += 1
            if plot_idx[0] % 10 == 0:
                n = min(plot_idx[0], plot_history)
                roll = (idx + 1) if plot_idx[0] >= plot_history else 0
                t = np.arange(n, dtype=np.float64)
                for key in plots:
                    data = np.roll(history[key], -roll, axis=0)[:n]
                    plots[key].data = (t, data[:, 0], data[:, 1], data[:, 2])

    except KeyboardInterrupt:
        print("KeyboardInterrupt")

    imu.shutdown()

