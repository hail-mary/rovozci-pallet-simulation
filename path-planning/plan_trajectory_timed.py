#!/usr/bin/env python3
"""
Path planning module with CLI demo.
Supports spatial sampling (default) or time-based sampling (--timestep).
Can save planned path to CSV with --out.
Plots trajectory when --plot is specified.
"""

import argparse
import math
import csv
import bisect
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
    Attributes:
        x: X-coordinate.
        y: Y-coordinate.
        yaw: Orientation angle in radians.
        curvature: Curvature at the waypoint.
        desired_velocity: Desired velocity at the waypoint.
        name: Optional name.
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
    Plan a velocity profile for a segment using constant acceleration/deceleration.
    """
    if path_length < 2:
        return [0.0] * path_length
    start_velocity = min(start_velocity, max_velocity)
    end_velocity = min(end_velocity, max_velocity)

    #if abs(start_velocity) < 0.1:
    #    start_velocity = 0.1
    
    d_acc = (max_velocity - start_velocity) / accel
    d_dec = (max_velocity - end_velocity) / accel
    n_acc = round(d_acc / step_size)
    n_dec = round(d_dec / step_size)
    if n_acc + n_dec > path_length:
        return np.linspace(start_velocity, end_velocity, path_length).tolist()
    velocities: List[float] = [0.0] * path_length
    for i in range(n_acc):
        if i < path_length:
            velocities[i] = start_velocity + accel * (i + 1) * step_size
    for i in range(n_acc, path_length - n_dec):
        velocities[i] = max_velocity
    for i in range(path_length - n_dec, path_length):
        velocities[i] = max_velocity - accel * (i - (path_length - n_dec)) * step_size
    velocities[-1] = end_velocity
    return velocities


def generate_segment(
    start: Waypoint,
    end: Waypoint,
    step_size: float,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Generate spatial segment between two waypoints using clothoid curves.
    """
    seg_x: List[float] = []
    seg_y: List[float] = []
    seg_theta: List[float] = []
    seg_kappa: List[float] = []
    curves = SolveG2(
        start.x, start.y, start.yaw, start.curvature,
        end.x, end.y, end.yaw, end.curvature,
    )
    for c in curves or []:
        if c is None:
            continue
        s_vals = np.arange(0, c.length, step_size)
        for s in s_vals:
            seg_x.append(c.X(s))
            seg_y.append(c.Y(s))
            seg_theta.append(c.Theta(s))
            seg_kappa.append(
                c.KappaStart + (c.KappaEnd - c.KappaStart) * (s / c.length)
            )
    last_x, last_y = (seg_x[-1], seg_y[-1]) if seg_x else (start.x, start.y)
    dx, dy = end.x - last_x, end.y - last_y
    dist = math.hypot(dx, dy)
    if dist > 1e-3:
        n = max(int(dist / step_size), 1)
        for j in range(1, n + 1):
            t = j / n
            seg_x.append(last_x + t * dx)
            seg_y.append(last_y + t * dy)
            seg_theta.append(start.yaw + t * (end.yaw - start.yaw))
            seg_kappa.append(start.curvature + t * (end.curvature - start.curvature))
    return seg_x, seg_y, seg_theta, seg_kappa


def plan_route(
    waypoints: List[Waypoint],
    step_size: float,
    max_velocity: float,
    max_accel: float,
    plan_velocities: bool = True,
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float]]:
    """Plan a route using clothoids and return path data and waypoint distances."""
    x_path: List[float] = []
    y_path: List[float] = []
    theta_path: List[float] = []
    curvature_path: List[float] = []
    velocity_profile: List[float] = []
    cum_wp_dist: List[float] = [0.0]
    cum_dist = 0.0
    for s_wp, e_wp in zip(waypoints, waypoints[1:]):
        seg_x, seg_y, seg_theta, seg_kappa = generate_segment(s_wp, e_wp, step_size)
        x_path.extend(seg_x)
        y_path.extend(seg_y)
        theta_path.extend(seg_theta)
        curvature_path.extend(seg_kappa)
        if plan_velocities:
            velocity_profile.extend(
                plan_velocity_profile(
                    len(seg_x),
                    s_wp.desired_velocity,
                    e_wp.desired_velocity,
                    step_size,
                    max_velocity,
                    max_accel,
                )
            )
        seg_len = compute_path_length(seg_x, seg_y)
        cum_dist += seg_len
        cum_wp_dist.append(cum_dist)
    final_wp = waypoints[-1]
    if not x_path or (
        abs(x_path[-1] - final_wp.x) > 1e-3
        or abs(y_path[-1] - final_wp.y) > 1e-3
    ):
        x_path.append(final_wp.x)
        y_path.append(final_wp.y)
        theta_path.append(final_wp.yaw)
        curvature_path.append(final_wp.curvature)
        velocity_profile.append(final_wp.desired_velocity if plan_velocities else 0.0)
    return x_path, y_path, theta_path, curvature_path, velocity_profile, cum_wp_dist


def compute_path_length(x: List[float], y: List[float]) -> float:
    """Compute the total length of a path."""
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
      time, x, y, psi, velocity_x, velocity_y,
      velocity, acceleration, distance_from_start,
      distance_from_last_waypoint.
    """
    dists = [0.0] + [
        math.hypot(x[i] - x[i - 1], y[i] - y[i - 1]) for i in range(1, len(x))
    ]
    cum_dist = np.cumsum(dists)
    times = np.cumsum([d / v if v > 0 else 0.0 for d, v in zip(dists, velocity)])
    total_t = times[-1] if len(times) > 0 else 0.0
    t_samples = list(np.arange(0.0, total_t, delta_time))
    if not t_samples or t_samples[-1] < total_t:
        t_samples.append(total_t)
    xs = np.interp(t_samples, times, x)
    ys = np.interp(t_samples, times, y)
    psis = np.interp(t_samples, times, theta)
    vs = np.interp(t_samples, times, velocity)
    vx = vs * np.cos(psis)
    vy = vs * np.sin(psis)
    acc = [0.0] + [(vs[i] - vs[i - 1]) / delta_time for i in range(1, len(vs))]
    dist_start = np.interp(t_samples, times, cum_dist)
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
    """Load waypoints from a YAML file."""
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
        ) for wp in wps
    ]


@dataclass
class PlannedTrajectory:
    """
    Class representing a planned trajectory.
    Properties: length, x, y, theta, curvature, and velocity.
    """
    x: List[float]
    y: List[float]
    theta: List[float]
    curvature: List[float]
    velocity: List[float]
    length: float = 0.0

    def __post_init__(self) -> None:
        self.length = compute_path_length(self.x, self.y)


def plot_trajectory(
    trajectory: PlannedTrajectory,
    waypoints: List[Waypoint],
) -> None:
    """
    Plot the planned trajectory, original waypoints, and velocity profile.
    """
    cum_dist = np.cumsum(
        [0.0] + [
            math.hypot(
                trajectory.x[i] - trajectory.x[i - 1],
                trajectory.y[i] - trajectory.y[i - 1],
            ) for i in range(1, len(trajectory.x))
        ]
    )
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(trajectory.x, trajectory.y, "b.-", label="Planned Path")
    axs[0].plot(
        [wp.x for wp in waypoints], [wp.y for wp in waypoints],
        "ro", label="Waypoints"
    )
    axs[0].set_title("Planned Trajectory")
    axs[0].set_xlabel("X [m]")
    axs[0].set_ylabel("Y [m]")
    axs[0].axis("equal")
    axs[0].legend()
    axs[0].grid(True)
    axs[1].plot(cum_dist, trajectory.velocity, "g.-", label="Velocity Profile")
    axs[1].set_title("Velocity Profile")
    axs[1].set_xlabel("Cumulative Distance [m]")
    axs[1].set_ylabel("Velocity [m/s]")
    axs[1].legend()
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """CLI demo for the path planning module."""
    parser = argparse.ArgumentParser(
        description="Path Planning: spatial or time-based sampling."
    )
    parser.add_argument(
        "yaml_file", type=str, help="YAML file with waypoints."
    )
    parser.add_argument(
        "--step_size", type=float, default=0.1,
        help="Spatial step size (m)."
    )
    parser.add_argument(
        "--max_velocity", type=float, default=0.12,
        help="Max velocity (m/s)."
    )
    parser.add_argument(
        "--max_accel", type=float, default=0.5,
        help="Max accel (m/s^2)."
    )
    parser.add_argument(
        "--no_velocity_planning",
        action="store_true",
        help="Disable velocity planning.",
    )
    parser.add_argument(
        "--timestep", type=float, nargs="?", const=0.05,
        help="Time step (s) for time-based sampling."
    )
    parser.add_argument(
        "--out", type=str,
        help="CSV file to save the planned path."
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Plot the planned trajectory."
    )
    args = parser.parse_args()

    waypoints = load_waypoints_from_yaml(args.yaml_file)
    max_desired = max(wp.desired_velocity for wp in waypoints)
    effective_max_vel = max(args.max_velocity, max_desired)
    x, y, theta, kappa, velocity, wp_dist = plan_route(
        waypoints,
        args.step_size,
        effective_max_vel,
        args.max_accel,
        plan_velocities=not args.no_velocity_planning,
    )
    if args.timestep is not None:
        rows = sample_trajectory_in_time(x, y, theta, velocity, wp_dist, args.timestep)
        if args.out:
            with open(args.out, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Saved time-sampled path to {args.out}")
        else:
            for r in rows:
                print(r)
    length = compute_path_length(x, y)
    print(f"Planned trajectory length: {length:.2f} m")
    print("Number of points in spatial path:", len(x))
    if args.plot:
        trajectory = PlannedTrajectory(x, y, theta, kappa, velocity)
        plot_trajectory(trajectory, waypoints)


if __name__ == "__main__":
    main()
