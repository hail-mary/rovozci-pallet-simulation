import sys
import time
import math
import platform
if platform.system() == 'Windows':
    import msvcrt
else:
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
    Works on both Windows and Unix-like systems.
    """
    if platform.system() == 'Windows':
        msvcrt.getch()
    else:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

# state variables
acceleration = 0
velocity = 0
last_x = 0
last_y = 0
last_psi = 0
last_t = None

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

    global last_t, last_psi, last_x, last_y

    # find body and mocap slot
    body_id = mj_name2id(model, mjtObj.mjOBJ_BODY, "pallet")
    mocap_id = model.body_mocapid[body_id]

    omega = 2 * math.pi / period
    viewer = launch_passive(model, data)
    dt = model.opt.timestep

    # ovwerride the computation with your scenario
    def state_function_circular(t: float, dt: float) -> Tuple[float, float, float]:
        x = radius * math.cos(omega * t)
        y = radius * math.sin(omega * t)
        psi = omega * t  # + math.pi / 2
        return x, y, psi

    def state_function_accel(t: float, dt: float, accel: float) -> Tuple[float, float, float]:
        global velocity, acceleration, last_x, last_y, last_psi
        velocity += dt * accel
        acceleration = accel
        x = last_x + velocity * dt
        y = last_y
        psi = last_psi
        return x, y, psi

    def state_function(t: float, dt: float) -> Tuple[float, float, float]:
        #return state_function_circular(t)
        return state_function_accel(t, dt, 0.3)
    
    # end override

    # show the initial frame once
    viewer.sync()
    input("Scene is up — press Enter to start the circular motion…")

    start_wall = time.time()

    while viewer.is_running() and (time.time() - start_wall) < run_time:
        t = data.time
        if last_t is None:
            last_t = t
        dtt = t - last_t

        # position on circle
        
        x, y, psi = state_function(t, dt)
        last_x, last_y, last_psi = x, y, psi
        print('t:', t, 'x:', x, 'y:', y, 'psi:', psi, 'vel:', velocity, 'acc:', acceleration, 'dt:', dt, 'dtt:', dtt)

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
        """
        print(
            f"t={t:.3f} s | "
            f"pos=({pos[0]:.3f}, {pos[1]:.3f}) m | "
            f"vel=({vel[0]:.3f}, {vel[1]:.3f}) m/s | "
            f"acc=({acc[0]:.3f}, {acc[1]:.3f}) m/s²"
        )
        """

        # throttle to real time
        elapsed = time.time() - start_wall
        if t > elapsed:
            print('sleeping', t - elapsed)
            time.sleep(t - elapsed)

    print("Simulation done – close window to exit.")
    while viewer.is_running():
        time.sleep(0.1)
    viewer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Playback trajectory in MuJoCo scene.'
    )
    parser.add_argument(
        '--scene',
        type=str,
        default="scene.xml",
        help='Path to MuJoCo XML scene file.'
    )
    parser.add_argument(
        '--radius',
        type=float,
        default=3.0,
        help='Simulation step size in seconds (default: 0.05).'  
    )    
    parser.add_argument(
        '--period',
        type=float,
        default=5.0
    )    
    parser.add_argument(
        '--runtime',
        type=float,
        default=30.0
    )    


    args = parser.parse_args()
    model, data = load_sim(args.scene)
    watch_circular(model, data, radius=args.radius, period=args.period, run_time=args.runtime)
