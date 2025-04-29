#!/usr/bin/env python3
"""
Play back a precomputed trajectory in MuJoCo.
Loads scene and trajectory table, then drives actuators at fixed timestep.
"""

import sys
import time
import math
import platform
from typing import Tuple

import pandas as pd
import numpy as np
import mujoco
from mujoco import MjModel, MjData, mjtObj, mj_name2id
from mujoco.viewer import launch_passive


def load_sim(model_path: str) -> Tuple[MjModel, MjData]:
    """
    Load a MuJoCo model and create its data.

    Args:
        model_path: Path to the MJCF XML file.
    Returns:
        Tuple(model, data) for simulation.
    """
    model = MjModel.from_xml_path(model_path)
    data = MjData(model)
    return model, data


def wait_key() -> None:
    """
    Wait for a single keypress without echo (cross-platform).
    """
    if platform.system() == 'Windows':
        import msvcrt
        msvcrt.getch()
    else:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def check_initial_overlap(model: MjModel, data: MjData) -> None:
    """
    Perform one forward pass and report any contact penetrations.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
    """
    mujoco.mj_forward(model, data)
    ncon = data.ncon
    if ncon == 0:
        print("No initial contacts detected.")
        return

    print(f"Initial contacts: {ncon}")
    for i in range(ncon):
        dist = data.contact[i].dist
        if dist < 0.0 and dist > 0.001:
            print(f"  Contact {i}: penetration = {dist:.4f} m")


def warmup_simulation(
    model: MjModel,
    data: MjData,
    steps: int,
    timestep: float
) -> None:
    """
    Run a warm-up period to let contacts settle under gravity.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        steps: Number of zero-control steps to run.
        timestep: Simulation step size in seconds.
    """
    model.opt.timestep = timestep
    # zero all controls during warm-up
    data.ctrl[:] = 0.0
    for _ in range(steps):
        mujoco.mj_step(model, data)


def follow_trajectory(
    model: MjModel,
    data: MjData,
    df: pd.DataFrame,
    timestep: float,
    warmup_steps: int = 200
) -> None:
    """
    Drive the mocap pallet along the trajectory from the table.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        df: DataFrame with trajectory columns (time, x, y, psi, ...).
        timestep: Simulation step size in seconds.
    """
    # Prepare time lookup
    times = df['time'].to_numpy()
    # Find IDs
    body_id = mj_name2id(model, mjtObj.mjOBJ_BODY, 'pallet')
    mocap_id = model.body_mocapid[body_id]
    # Launch viewer
    viewer = launch_passive(model, data)
    # Override internal timestep
    model.opt.timestep = timestep
    viewer.sync()

    print("Checking initial overlap...")
    check_initial_overlap(model, data)
    print("Done!")

    warmup_simulation(model, data, warmup_steps, timestep)
    viewer.sync()

    print('Scene loaded. Press any key to start playback...')
    wait_key()

    start_wall = time.time()
    step_index = 0
    total_steps = 0
    while viewer.is_running():
        sim_start = time.time()
        t = data.time
        # Find next row index
        idx = np.searchsorted(times, t, side='left')
        if idx >= len(df):
            break
        row = df.iloc[idx]
        print(row)
        # Send to actuators
        data.ctrl[0] = float(row['x'])
        data.ctrl[1] = float(row['y'])
        data.ctrl[2] = float(row['psi'])
        # Step simulation
        mujoco.mj_step(model, data)
        viewer.sync()
        # Timing control
        sim_elapsed = time.time() - sim_start
        if sim_elapsed < timestep:
            time.sleep(timestep - sim_elapsed)
        else:
            print(f"Warning: step took {sim_elapsed:.3f}s > timestep {timestep:.3f}s")
        total_steps += 1
    print(f'Trajectory playback done: {total_steps} steps.')
    print('Close window to exit.')
    while viewer.is_running():
        time.sleep(0.1)
    viewer.close()


def main() -> None:
    """
    CLI entrypoint.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description='Playback trajectory in MuJoCo scene.'
    )
    parser.add_argument(
        '--scene',
        required=True,
        help='Path to MuJoCo XML scene file.'
    )
    parser.add_argument(
        '--table',
        required=True,
        help='CSV file with trajectory table (time,x,y,psi,...).' 
    )
    parser.add_argument(
        '--timestep',
        type=float,
        default=0.05,
        help='Simulation step size in seconds (default: 0.05).'  
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=200,
        help='Simulation warmup steps.'  
    )
    args = parser.parse_args()

    # Load simulation and table
    model, data = load_sim(args.scene)
    df = pd.read_csv(args.table)
    # Playback
    follow_trajectory(model, data, df, args.timestep, args.warmup)


if __name__ == '__main__':
    main()
