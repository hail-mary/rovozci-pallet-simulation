#!/usr/bin/env python3
"""
Path planning module with CLI demo.
Supports spatial sampling (default) or time-based sampling (--timestep).
Can save planned path to CSV with --out.
"""

import argparse
import math
import csv
import bisect
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
import yaml
from pyclothoids.clothoid import SolveG2
from dataclasses import replace  # noqa: F401
from matplotlib import pyplot as plt  # noqa: F401


@dataclass
class Waypoint:
    """
    Dataclass representing a waypoint.
    Attributes:
        x (float): X-coordinate.
        y (float): Y-coordinate.
        yaw (float): Orientation angle in radians.
        curvature (float): Curvature at the waypoint.
        desired_velocity (float): Desired velocity at the waypoint.
        name (str): Optional name.
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
    max_velocity: float,
    accel: float,
) -> List[float]:
    """
    Plan a velocity profile for a segment using constant acceleration.
    """
    if path_length < 2:
        return [0.0] * path_length
    start_velocity = min(start_velocity, max_velocity)
    end_velocity = min(end_velocity, max_velocity)
    min_v = 0.1
    if abs(start_velocity) < min_v:
        start_velocity = min_v
    # distances for accel/decel
    d_acc = (max_velocity - start_velocity) / accel
    d_dec = (max_velocity - end_velocity) / accel
    n_acc = round(d_acc / step_size)
    n_dec = round(d_dec / step_size)
    if n_acc + n_dec > path_length:
        return np.linspace(start_velocity, end_velocity, path_length).tolist()
    v = [0.0] * path_length
    for i in range(n_acc):
        if i < path_length:
            v[i] = start_velocity + accel * (i + 1) * step_size
    for i in range(n_acc, path_length - n_dec):
        v[i] = max_velocity
    for i in range(path_length - n_dec, path_length):
        v[i] = max_velocity - accel * (i - (path_length - n_dec)) * step_size
    v[-1] = end_velocity
    return v


def plan_route(
    waypoints: List[Waypoint],
    step_size: float,
    max_velocity: float,
    max_accel: float,
    plan_velocities: bool = True,
) -> Tuple[
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
]:
    """
    Plan a route using clothoids.
    Returns x, y, theta, curvature, velocity, and cumulative waypoint distances.
    """
    x_path: List[float] = []
    y_path: List[float] = []
    theta_path: List[float] = []
    curvature_path: List[float] = []
    velocity_profile: List[float] = []
    cum_wp_dist: List[float] = [0.0]
    cum_dist = 0.0

    for i in range(len(waypoints) - 1):
        s = waypoints[i]
        e = waypoints[i + 1]
        curves = SolveG2(
            s.x, s.y, s.yaw, s.curvature,
            e.x, e.y, e.yaw, e.curvature,
        )
        seg_x: List[float] = []
        seg_y: List[float] = []
        seg_theta: List[float] = []
        seg_kappa: List[float] = []

        for c in curves:
            if c is None:
                continue
            ss = np.arange(0, c.length, step_size)
            for d in ss:
                seg_x.append(c.X(d))
                seg_y.append(c.Y(d))
                seg_theta.append(c.Theta(d))
                seg_kappa.append(
                    c.KappaStart + (c.KappaEnd - c.KappaStart) * (d / c.length)
                )

        # connect if gap
        if seg_x:
            last = (seg_x[-1], seg_y[-1])
        else:
            last = (s.x, s.y)
        dx = e.x - last[0]
        dy = e.y - last[1]
        dist = math.hypot(dx, dy)
        if dist > 1e-3:
            n = max(int(dist / step_size), 1)
            for j in range(1, n + 1):
                t = j / n
                seg_x.append(last[0] + t * dx)
                seg_y.append(last[1] + t * dy)
                seg_theta.append(s.yaw + t * (e.yaw - s.yaw))
                seg_kappa.append(s.curvature + t * (e.curvature - s.curvature))

        # extend paths
        x_path.extend(seg_x)
        y_path.extend(seg_y)
        theta_path.extend(seg_theta)
        curvature_path.extend(seg_kappa)

        # velocity
        if plan_velocities:
            v = plan_velocity_profile(
                len(seg_x), s.desired_velocity, e.desired_velocity,
                step_size, max_velocity, max_accel
            )
            velocity_profile.extend(v)

        # update cum waypoint distance
        seg_len = compute_path_length(seg_x, seg_y)
        cum_dist += seg_len
        cum_wp_dist.append(cum_dist)

    # final waypoint
    f = waypoints[-1]
    if not x_path or (
        abs(x_path[-1] - f.x) > 1e-3 or abs(y_path[-1] - f.y) > 1e-3
    ):
        x_path.append(f.x)
        y_path.append(f.y)
        theta_path.append(f.yaw)
        curvature_path.append(f.curvature)
        velocity_profile.append(f.desired_velocity if plan_velocities else 0.0)

    return (
        x_path,
        y_path,
        theta_path,
        curvature_path,
        velocity_profile,
        cum_wp_dist,
    )


def compute_path_length(x: List[float], y: List[float]) -> float:
    """
    Compute the total length of a path.
    """
    length = 0.0
    for i in range(1, len(x)):
        length += math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
    return length


def sample_trajectory_in_time(
    x: List[float],
    y: List[float],
    theta: List[float],
    velocity: List[float],
    wp_dist: List[float],
    delta_time: float,
) -> List[Dict[str, float]]:
    """
    Resample trajectory at equal time intervals.
    Returns list of dicts with keys:
    time, x, y, psi, velocity_x, velocity_y, velocity,
    acceleration, distance_from_start, distance_from_last_waypoint.
    """
    # spatial distances & times
    dists = [0.0] + [
        math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
        for i in range(1, len(x))
    ]
    cum_dist = np.cumsum(dists)
    times = np.cumsum([
        d / v if v > 0 else 0.0
        for d, v in zip(dists, velocity)
    ])
    total_t = times[-1] if times.size else 0.0
    t_samples = np.arange(0.0, total_t, delta_time).tolist()
    if not t_samples or t_samples[-1] < total_t:
        t_samples.append(total_t)

    xs = np.interp(t_samples, times, x)
    ys = np.interp(t_samples, times, y)
    psis = np.interp(t_samples, times, theta)
    vs = np.interp(t_samples, times, velocity)

    # kinematics
    vx = vs * np.cos(psis)
    vy = vs * np.sin(psis)
    acc = [0.0] + [
        (vs[i] - vs[i - 1]) / delta_time
        for i in range(1, len(vs))
    ]
    dist_start = np.interp(t_samples, times, cum_dist)

    # distance from last waypoint
    rows: List[Dict[str, float]] = []
    for i, t in enumerate(t_samples):
        ds = float(dist_start[i])
        idx = bisect.bisect_right(wp_dist, ds) - 1
        d_last = ds - wp_dist[max(idx, 0)]
        rows.append({
            "time": float(t),
            "x": float(xs[i]),
            "y": float(ys[i]),
            "psi": float(psis[i]),
            "velocity_x": float(vx[i]),
            "velocity_y": float(vy[i]),
            "velocity": float(vs[i]),
            "acceleration": float(acc[i]),
            "distance_from_start": ds,
            "distance_from_last_waypoint": d_last,
        })
    return rows


def load_waypoints_from_yaml(file_path: str) -> List[Waypoint]:
    """
    Load waypoints from a YAML file.
    """
    with open(file_path, "r") as f:
        data: Dict[str, Any] = yaml.safe_load(f)
    wps = data.get("waypoints", data)
    return [
        Waypoint(
            x=wp["x"],
            y=wp["y"],
            yaw=wp.get("yaw", 0.0),
            curvature=wp.get("curvature", 0.0),
            desired_velocity=wp.get("desired_velocity", 0.1),
            name=wp.get("name", ""),
        )
        for wp in wps
    ]


def main() -> None:
    """
    CLI demo for the path planning module.
    """
    parser = argparse.ArgumentParser(
        description="Path Planning: spatial or time-based sampling."
    )
    parser.add_argument(
        "yaml_file",
        type=str,
        help="YAML file with waypoints.",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.1,
        help="Spatial step size (m).",
    )
    parser.add_argument(
        "--max_velocity",
        type=float,
        default=0.12,
        help="Max velocity (m/s).",
    )
    parser.add_argument(
        "--max_accel",
        type=float,
        default=0.5,
        help="Max accel (m/s^2).",
    )
    parser.add_argument(
        "--no_velocity_planning",
        action="store_true",
        help="Disable velocity planning.",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        nargs="?",
        const=0.05,
        help="Time step (s) for time-based sampling.",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="CSV file to save the planned path.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot spatial trajectory.",
    )
    args = parser.parse_args()

    waypoints = load_waypoints_from_yaml(args.yaml_file)
    x, y, theta, kappa, velocity, wp_dist = plan_route(
        waypoints,
        args.step_size,
        args.max_velocity,
        args.max_accel,
        plan_velocities=not args.no_velocity_planning,
    )

    if args.timestep is not None:
        rows = sample_trajectory_in_time(
            x, y, theta, velocity, wp_dist, args.timestep
        )
        if args.out:
            with open(args.out, "w", newline="") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=list(rows[0].keys())
                )
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved time-sampled path to {args.out}")
        else:
            for r in rows:
                print(r)

    length = compute_path_length(x, y)
    print(f"Planned trajectory length: {length:.2f} m")
    print("Number of points:", len(x))
    if args.plot:
        cum_d = np.cumsum(
            [0.0] + [
                math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
                for i in range(1, len(x))
            ]
        )
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        axs[0].plot(x, y, "b.-", label="Path")
        axs[0].plot(
            [wp.x for wp in waypoints],
            [wp.y for wp in waypoints],
            "ro", label="Waypoints"
        )
        axs[0].axis("equal")
        axs[0].legend()
        axs[1].plot(cum_d, velocity, "g.-", label="Velocity")
        axs[1].legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
