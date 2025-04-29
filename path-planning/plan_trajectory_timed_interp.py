#!/usr/bin/env python3
"""
Path planning module with CLI demo.
Supports spatial or time-based sampling (--timestep).
Optionally linearly interpolate velocities (--interp).
Can save planned path to CSV with --out.
Plots trajectory when --plot is specified:
  - subplot 1: spatial path.
  - subplot 2: velocity vs time.
  - subplot 3: velocity vs distance.
"""

import argparse
import math
import csv
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import yaml
from pyclothoids.clothoid import SolveG2
import matplotlib.pyplot as plt


@dataclass
class Waypoint:
    """
    Dataclass representing a waypoint.
    """
    x: float
    y: float
    yaw: float
    curvature: float
    desired_velocity: float
    name: str = ""


def plan_velocity_profile(
    path_length: int,
    start_velocity: float,
    end_velocity: float,
    step_size: float,
    max_velocity: float,  # unused when interp=False
    accel: float,
    interp: bool = False,
) -> List[float]:
    """
    Plan velocities: either accel/decel at max_accel between waypoints, or linear interp if interp=True.
    Acceleration per time remains constant at +accel or -accel.
    """
    if path_length < 2:
        return [start_velocity] * path_length
    if interp:
        return np.linspace(start_velocity, end_velocity, path_length).tolist()
    # Determine acceleration sign (positive or negative)
    a = accel if end_velocity >= start_velocity else -accel
    velocities: List[float] = []
    for i in range(path_length):
        # distance traveled along segment
        s = step_size * i
        # v^2 = v0^2 + 2*a*s
        v2 = start_velocity**2 + 2 * a * s
        v = math.sqrt(v2) if v2 > 0 else 0.0
        velocities.append(v)
    # overwrite final value to exactly match target
    velocities[-1] = end_velocity
    return velocities
def plan_route(
    waypoints: List[Waypoint],
    step_size: float,
    max_velocity: float,
    max_accel: float,
    plan_velocities: bool = True,
    interp_velocities: bool = False,
) -> Tuple[
    List[float], List[float], List[float], List[float], List[float], List[float]
]:
    """
    Plan a route using clothoids.
    Returns x, y, theta, curvature, velocity, waypoint distances.
    """
    x_path, y_path, theta_path, kappa_path = [], [], [], []
    vel_profile, wp_dist = [], [0.0]
    cum = 0.0
    for a, b in zip(waypoints, waypoints[1:]):
        curves = SolveG2(
            a.x, a.y, a.yaw, a.curvature,
            b.x, b.y, b.yaw, b.curvature,
        )
        seg_x, seg_y, seg_th, seg_k = [], [], [], []
        for c in curves or []:
            if c is None:
                continue
            s_vals = np.arange(0, c.length, step_size)
            for s in s_vals:
                seg_x.append(c.X(s))
                seg_y.append(c.Y(s))
                seg_th.append(c.Theta(s))
                seg_k.append(
                    c.KappaStart + (c.KappaEnd - c.KappaStart) * (s / c.length)
                )
        if seg_x:
            lx, ly = seg_x[-1], seg_y[-1]
        else:
            lx, ly = a.x, a.y
        dx, dy = b.x - lx, b.y - ly
        dist = math.hypot(dx, dy)
        if dist > 1e-3:
            n = max(int(dist / step_size), 1)
            for j in range(1, n + 1):
                t = j / n
                seg_x.append(lx + t * dx)
                seg_y.append(ly + t * dy)
                seg_th.append(a.yaw + t * (b.yaw - a.yaw))
                seg_k.append(a.curvature + t * (b.curvature - a.curvature))
        x_path += seg_x; y_path += seg_y
        theta_path += seg_th; kappa_path += seg_k
        if plan_velocities:
            vel_profile += plan_velocity_profile(
                len(seg_x), a.desired_velocity, b.desired_velocity,
                step_size, max_velocity, max_accel, interp_velocities
            )
        seg_len = sum(
            math.hypot(seg_x[i] - seg_x[i-1], seg_y[i] - seg_y[i-1])
            for i in range(1, len(seg_x))
        )
        cum += seg_len; wp_dist.append(cum)
    f = waypoints[-1]
    if not x_path or abs(x_path[-1] - f.x) > 1e-3 or abs(y_path[-1] - f.y) > 1e-3:
        x_path.append(f.x); y_path.append(f.y)
        theta_path.append(f.yaw); kappa_path.append(f.curvature)
        vel_profile.append(f.desired_velocity if plan_velocities else 0.0)
    return x_path, y_path, theta_path, kappa_path, vel_profile, wp_dist


def compute_path_length(x: List[float], y: List[float]) -> float:
    """Compute total path length."""
    return sum(
        math.hypot(x[i] - x[i-1], y[i] - y[i-1])
        for i in range(1, len(x))
    )


def sample_trajectory_in_time(
    x: List[float],
    y: List[float],
    theta: List[float],
    velocity: List[float],
    wp_dist: List[float],
    delta_time: float,
    max_accel: float
) -> List[Dict[str, float]]:
    """Resample trajectory at equal time intervals and set acceleration = ±max_accel."""
    # distances & cumulative distance
    dists = [0.0] + [
        math.hypot(x[i] - x[i-1], y[i] - y[i-1])
        for i in range(1, len(x))
    ]
    cum_dist = np.cumsum(dists)
    # times along trajectory
    times = np.cumsum([
        d / v if v > 0 else 0.0
        for d, v in zip(dists, velocity)
    ])
    total_t = times[-1] if len(times) > 0 else 0.0
    t_samples = list(np.arange(0.0, total_t, delta_time))
    if not t_samples or t_samples[-1] < total_t:
        t_samples.append(total_t)
    # interpolate state
    xs = np.interp(t_samples, times, x)
    ys = np.interp(t_samples, times, y)
    psis = np.interp(t_samples, times, theta)
    vs = np.interp(t_samples, times, velocity)
    # compute distance-from-start
    dist_start = np.interp(t_samples, times, cum_dist)
    # build rows
    rows: List[Dict[str, float]] = []
    prev_v = vs[0]
    for i, t in enumerate(t_samples):
        v = vs[i]
        # determine commanded acceleration
        if v > prev_v + 1e-8:
            a_cmd = max_accel
        elif v < prev_v - 1e-8:
            a_cmd = -max_accel
        else:
            a_cmd = 0.0
        prev_v = v
        ds = float(dist_start[i])
        idx = __import__('bisect').bisect_right(wp_dist, ds) - 1
        rows.append({
            'time': float(t),
            'x': float(xs[i]),
            'y': float(ys[i]),
            'psi': float(psis[i]),
            'velocity_x': float(v * math.cos(psis[i])),
            'velocity_y': float(v * math.sin(psis[i])),
            'velocity': float(v),
            'acceleration': a_cmd,
            'distance_from_start': ds,
            'distance_from_last_waypoint': ds - wp_dist[max(idx, 0)],
        })
    return rows


def plot_trajectory(
    x: List[float], y: List[float], velocity: List[float],
    waypoints: List[Waypoint]
) -> None:
    """Plot spatial path, velocity vs time, and velocity vs distance."""
    # cumulative distance
    dists = [0.0] + [math.hypot(x[i] - x[i-1], y[i] - y[i-1])
                     for i in range(1, len(x))]
    cum_dist = np.cumsum(dists)
    # compute times
    times = np.cumsum([0.0] + [d / v if v > 0 else 0.0 for d, v in zip(dists[1:], velocity[1:])])
    # figure
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 12))
    # spatial path
    ax0.plot(x, y, 'b.-', label='Path')
    ax0.plot([wp.x for wp in waypoints], [wp.y for wp in waypoints], 'ro', label='Waypoints')
    ax0.set_title('Spatial Path')
    ax0.axis('equal'); ax0.legend(); ax0.grid(True)
    # velocity vs time
    ax1.plot(times, velocity, 'g.-', label='Vel vs Time')
    ax1.set_title('Velocity Profile')
    ax1.set_xlabel('Time [s]'); ax1.set_ylabel('Velocity [m/s]')
    ax1.legend(); ax1.grid(True)
    # velocity vs distance
    ax2.plot(cum_dist, velocity, 'r.-', label='Vel vs Distance')
    ax2.set_xlabel('Distance [m]'); ax2.set_ylabel('Velocity [m/s]')
    ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.show()


def load_waypoints_from_yaml(file_path: str) -> List[Waypoint]:
    """Load waypoints from a YAML file."""
    with open(file_path) as f:
        data: Dict[str, Any] = yaml.safe_load(f)
    wps = data.get('waypoints', data)
    return [Waypoint(
        x=wp['x'], y=wp['y'],
        yaw=wp.get('yaw', 0.0), curvature=wp.get('curvature', 0.0),
        desired_velocity=wp.get('desired_velocity', 0.1), name=wp.get('name', '')
    ) for wp in wps]


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description='Plan trajectory with optional interp/time sampling.'
    )
    parser.add_argument('yaml_file', help='YAML file with waypoints.')
    parser.add_argument('--step_size', type=float, default=0.1, help='Spatial step (m).')
    parser.add_argument('--max_velocity', type=float, default=0.12, help='Max velocity (m/s).')
    parser.add_argument('--max_accel', type=float, default=0.5, help='Max accel (m/s²).')
    parser.add_argument('--no_velocity_planning', action='store_true', help='Disable velocity planning.')
    parser.add_argument('--interp', action='store_true', help='Linear interp between waypoint speeds.')
    parser.add_argument('--timestep', type=float, nargs='?', const=0.05, help='Time step (s) for time sampling.')
    parser.add_argument('--out', help='CSV file to save output.')
    parser.add_argument('--plot', action='store_true', help='Plot results.')
    args = parser.parse_args()

    waypoints = load_waypoints_from_yaml(args.yaml_file)
    x, y, th, kappa, vel, wp_dist = plan_route(
        waypoints, args.step_size, args.max_velocity,
        args.max_accel, not args.no_velocity_planning, args.interp
    )

    if args.timestep is not None:
        rows = sample_trajectory_in_time(x, y, th, vel, wp_dist, args.timestep, args.max_accel)
        if args.out:
            with open(args.out, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader(); writer.writerows(rows)
        else:
            for r in rows:
                print(r)

    length = compute_path_length(x, y)
    print(f'Trajectory length: {length:.2f} m')
    print(f'Number of points: {len(x)}')
    if args.plot:
        plot_trajectory(x, y, vel, waypoints)


if __name__ == '__main__':
    main()
