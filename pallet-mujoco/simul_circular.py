import sys
import time
import math
import tty
import termios
from typing import Tuple

import mujoco
from mujoco import MjModel, MjData, mjtObj, mj_name2id
from mujoco.viewer import launch_passive


def load_sim(model_path: str) -> Tuple[MjModel, MjData]:
    """
    Load a MuJoCo model and create its data.

    Args:
        model_path: Path to the MJCF XML file.

    Returns:
        A tuple (model, data) for simulation.
    """
    model = MjModel.from_xml_path(model_path)
    data = MjData(model)
    return model, data


def wait_key() -> None:
    """
    Wait for a single keypress on stdin, without echo.
    Works on Linux/macOS.
    """
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def watch_circular(
    model: MjModel,
    data: MjData,
    radius: float,
    period: float,
    run_time: float = 5.0
) -> None:
    """
    Move the mocap pallet around a circle, orienting it tangentially,
    and log its kinematics.
    """
    # find body and mocap slot
    body_id = mj_name2id(model, mjtObj.mjOBJ_BODY, "pallet")
    mocap_id = model.body_mocapid[body_id]

    omega = 2 * math.pi / period
    viewer = launch_passive(model, data)
    dt = model.opt.timestep

    # show the initial frame once
    viewer.sync()
    input("Scene is up — press Enter to start the circular motion…")

    start_wall = time.time()

    while viewer.is_running() and (time.time() - start_wall) < run_time:
        t = data.time

        # position on circle
        x = radius * math.cos(omega * t)
        y = radius * math.sin(omega * t)
        # data.mocap_pos[mocap_id] = (x, y, 0.05)

        # orientation tangent to circle: yaw = ω·t + π/2
        psi = omega * t  # + math.pi / 2
        qw = math.cos(psi / 2)
        qz = math.sin(psi / 2)

        data.ctrl[0] = x
        data.ctrl[1] = y
        data.ctrl[2] = psi

        # data.mocap_quat[mocap_id] = (qw, 0.0, 0.0, qz)

        mujoco.mj_step(model, data)
        viewer.sync()

        # print kinematics
        pos = data.qpos[0:2]
        vel = data.qvel[0:2]
        acc = data.qacc[0:2]
        print(
            f"t={t:.3f} s | "
            f"pos=({pos[0]:.3f}, {pos[1]:.3f}) m | "
            f"vel=({vel[0]:.3f}, {vel[1]:.3f}) m/s | "
            f"acc=({acc[0]:.3f}, {acc[1]:.3f}) m/s²"
        )

        # throttle to real time
        elapsed = time.time() - start_wall
        if t > elapsed:
            time.sleep(t - elapsed)

    print("Simulation done – close window to exit.")
    while viewer.is_running():
        time.sleep(0.1)
    viewer.close()


if __name__ == "__main__":
    model, data = load_sim("pallet_bricks_column.xml")
    watch_circular(model, data, radius=3.0, period=10.0, run_time=30.0)
