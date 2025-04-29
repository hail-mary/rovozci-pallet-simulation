#!/usr/bin/env python3
"""
Path planning module with CLI demo.
Reads waypoints definitions from a YAML file, performs path planning using clothoid curves,
and provides a PlannedTrajectory class containing properties:
    - length: Total length of the trajectory.
    - x: List of x coordinates.
    - y: List of y coordinates.
    - theta: List of orientation angles (radians).
    - curvature: List of curvature values.
    - velocity: List of velocity values.
Optionally, it can plot the planned trajectory, original waypoints, and velocity profile using matplotlib.
"""

# Usage example:
#
#  python plan_trajectory.py demo_trajectory.yaml --step_size 0.1 --max_velocity 0.3 --max_accel 0.1 --plot

import argparse
import math
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import yaml
from pyclothoids.clothoid import SolveG2
from dataclasses import dataclass


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
        name (str): Optional name of the waypoint.
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

    If the segment is too short for the computed acceleration and deceleration steps,
    a linear interpolation between start and end velocities is used.

    Velocities are clamped to max_velocity.

    Args:
        path_length (int): Number of steps along the segment.
        start_velocity (float): Starting velocity in m/s.
        end_velocity (float): Ending velocity in m/s.
        step_size (float): Distance between consecutive steps.
        max_velocity (float): Maximum velocity allowed.
        accel (float): Acceleration/deceleration rate (m/s^2).

    Returns:
        List[float]: Velocity values for each step.
    """
    if path_length < 2:
        return [0.0] * path_length

    # Clamp the velocities to the maximum allowed.
    start_velocity = min(start_velocity, max_velocity)
    end_velocity = min(end_velocity, max_velocity)

    velocities: List[float] = [0.0] * path_length
    min_velocity: float = 0.1
    if abs(start_velocity) < min_velocity:
        start_velocity = min_velocity

    accel_distance: float = (max_velocity - start_velocity) / accel
    decel_distance: float = (max_velocity - end_velocity) / accel

    accel_steps: int = round(accel_distance / step_size)
    decel_steps: int = round(decel_distance / step_size)

    # If the computed acceleration and deceleration steps exceed the segment length,
    # fallback to a simple linear interpolation.
    if accel_steps + decel_steps > path_length:
        return np.linspace(start_velocity, end_velocity, path_length).tolist()

    # Acceleration phase.
    for i in range(accel_steps):
        if i < path_length:
            velocities[i] = start_velocity + accel * (i + 1) * step_size

    # Constant velocity phase.
    for i in range(accel_steps, path_length - decel_steps):
        velocities[i] = max_velocity

    # Deceleration phase.
    for i in range(path_length - decel_steps, path_length):
        velocities[i] = max_velocity - accel * (i - (path_length - decel_steps)) * step_size

    velocities[-1] = end_velocity
    return velocities


def plan_route(
    waypoints: List[Waypoint],
    step_size: float,
    max_velocity: float,
    max_accel: float,
    plan_velocities: bool = True,
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
    """
    Plan a route based on a list of waypoints using clothoid curves.

    For each consecutive pair of waypoints, SolveG2 is used to compute the clothoid curves.
    The function samples the curves with the given step_size and linearly interpolates between
    segments if needed. Optionally, it computes a velocity profile for each segment.

    Args:
        waypoints (List[Waypoint]): List of waypoints.
        step_size (float): Interpolation step size in meters.
        max_velocity (float): Maximum velocity allowed.
        max_accel (float): Maximum acceleration/deceleration (m/s^2).
        plan_velocities (bool, optional): Flag to perform velocity planning (default True).

    Returns:
        Tuple containing:
            - x (List[float]): x coordinates.
            - y (List[float]): y coordinates.
            - theta (List[float]): Orientation angles (radians).
            - curvature (List[float]): Curvature values.
            - velocity (List[float]): Velocity values.
    """
    x_path: List[float] = []
    y_path: List[float] = []
    theta_path: List[float] = []
    curvature_path: List[float] = []
    velocity_profile: List[float] = []

    for i in range(len(waypoints) - 1):
        start_wp = waypoints[i]
        end_wp = waypoints[i + 1]

        start_x = start_wp.x
        start_y = start_wp.y
        end_x = end_wp.x
        end_y = end_wp.y

        start_yaw = start_wp.yaw
        end_yaw = end_wp.yaw

        start_curvature = start_wp.curvature
        end_curvature = end_wp.curvature

        # Compute clothoid curves between start and end waypoints.
        clothoid_curves = SolveG2(
            start_x,
            start_y,
            start_yaw,
            start_curvature,
            end_x,
            end_y,
            end_yaw,
            end_curvature,
        )

        segment_x: List[float] = []
        segment_y: List[float] = []
        segment_theta: List[float] = []
        segment_curvature: List[float] = []

        for curve in clothoid_curves:
            if curve is None:
                continue
            s_values = np.arange(0, curve.length, step_size)
            for s in s_values:
                segment_x.append(curve.X(s))
                segment_y.append(curve.Y(s))
                segment_theta.append(curve.Theta(s))
                # Interpolate curvature linearly between kappaStart and kappaEnd.
                curvature_val = curve.KappaStart + (curve.KappaEnd - curve.KappaStart) * (s / curve.length)
                segment_curvature.append(curvature_val)

        # Connect last point to the end waypoint if necessary.
        if segment_x:
            last_point = (segment_x[-1], segment_y[-1])
        else:
            last_point = (start_x, start_y)
        dx = end_x - last_point[0]
        dy = end_y - last_point[1]
        distance = math.hypot(dx, dy)
        if distance > 1e-3:
            num_steps = max(int(distance / step_size), 1)
            for j in range(1, num_steps + 1):
                t = j / num_steps
                x_interp = last_point[0] + t * dx
                y_interp = last_point[1] + t * dy
                segment_x.append(x_interp)
                segment_y.append(y_interp)
                theta_interp = start_yaw + t * (end_yaw - start_yaw)
                curvature_interp = start_curvature + t * (end_curvature - start_curvature)
                segment_theta.append(theta_interp)
                segment_curvature.append(curvature_interp)

        x_path.extend(segment_x)
        y_path.extend(segment_y)
        theta_path.extend(segment_theta)
        curvature_path.extend(segment_curvature)

        if plan_velocities:
            seg_len = len(segment_x)
            seg_velocities = plan_velocity_profile(
                seg_len, start_wp.desired_velocity, end_wp.desired_velocity, step_size, max_velocity, accel=max_accel
            )
            velocity_profile.extend(seg_velocities)

    # Append final waypoint if not already included.
    final_wp = waypoints[-1]
    if not x_path or (abs(x_path[-1] - final_wp.x) > 1e-3 or abs(y_path[-1] - final_wp.y) > 1e-3):
        x_path.append(final_wp.x)
        y_path.append(final_wp.y)
        theta_path.append(final_wp.yaw)
        curvature_path.append(final_wp.curvature)
        if plan_velocities:
            velocity_profile.append(final_wp.desired_velocity)
        else:
            velocity_profile.append(0.0)

    return x_path, y_path, theta_path, curvature_path, velocity_profile


def compute_path_length(x: List[float], y: List[float]) -> float:
    """
    Compute the total length of a path given x and y coordinates.

    Args:
        x (List[float]): List of x coordinates.
        y (List[float]): List of y coordinates.

    Returns:
        float: Total path length.
    """
    length = 0.0
    for i in range(1, len(x)):
        length += math.hypot(x[i] - x[i - 1], y[i] - y[i - 1])
    return length


class PlannedTrajectory:
    """
    Class representing a planned trajectory.
    Contains properties: length, x, y, theta, curvature, and velocity.
    """

    def __init__(
        self,
        x: List[float],
        y: List[float],
        theta: List[float],
        curvature: List[float],
        velocity: List[float],
    ) -> None:
        """
        Initialize the PlannedTrajectory.

        Args:
            x (List[float]): List of x coordinates.
            y (List[float]): List of y coordinates.
            theta (List[float]): List of orientation angles (radians).
            curvature (List[float]): List of curvature values.
            velocity (List[float]): List of velocity values.
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.curvature = curvature
        self.velocity = velocity
        self.length = compute_path_length(x, y)


def load_waypoints_from_yaml(file_path: str) -> List[Waypoint]:
    """
    Load waypoints from a YAML file.

    Args:
        file_path (str): Path to the YAML file containing waypoints definitions.

    Returns:
        List[Waypoint]: List of waypoints.
    """
    with open(file_path, "r") as f:
        data: Dict[str, Any] = yaml.safe_load(f)
    waypoints_data = data.get("waypoints", data)
    waypoints: List[Waypoint] = []
    for wp in waypoints_data:
        waypoint = Waypoint(
            x=wp["x"],
            y=wp["y"],
            yaw=wp.get("yaw", 0.0),
            curvature=wp.get("curvature", 0.0),
            desired_velocity=wp.get("desired_velocity", 0.1),
            name=wp.get("name", ""),
        )
        waypoints.append(waypoint)
    return waypoints


def plot_trajectory(
    trajectory: PlannedTrajectory, waypoints: List[Waypoint]
) -> None:
    """
    Plot the planned trajectory, original waypoints, and velocity profile using matplotlib.

    Args:
        trajectory (PlannedTrajectory): The planned trajectory.
        waypoints (List[Waypoint]): List of original waypoints.
    """
    cum_dist = np.cumsum(
        [0]
        + [
            math.hypot(trajectory.x[i] - trajectory.x[i - 1], trajectory.y[i] - trajectory.y[i - 1])
            for i in range(1, len(trajectory.x))
        ]
    )
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot planned path and waypoints.
    axs[0].plot(trajectory.x, trajectory.y, "b.-", label="Planned Path")
    wp_x = [wp.x for wp in waypoints]
    wp_y = [wp.y for wp in waypoints]
    axs[0].plot(wp_x, wp_y, "ro", label="Waypoints")
    axs[0].set_title("Planned Trajectory")
    axs[0].set_xlabel("X [m]")
    axs[0].set_ylabel("Y [m]")
    axs[0].axis("equal")
    axs[0].legend()
    axs[0].grid(True)

    # Plot velocity profile.
    axs[1].plot(cum_dist, trajectory.velocity, "g.-", label="Velocity Profile")
    axs[1].set_title("Velocity Profile")
    axs[1].set_xlabel("Cumulative Distance [m]")
    axs[1].set_ylabel("Velocity [m/s]")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    CLI demo for the path planning module.
    Reads waypoints from a YAML file, performs path planning, and optionally plots the results.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Path Planning CLI Demo: Reads waypoints from a YAML file, plans a trajectory, "
            "and optionally plots it."
        )
    )
    parser.add_argument(
        "yaml_file",
        type=str,
        help="Path to the YAML file containing waypoints definitions.",
    )
    parser.add_argument(
        "--step_size",
        type=float,
        default=0.1,
        help="Interpolation step size in meters (default: 0.1).",
    )
    parser.add_argument(
        "--max_velocity",
        type=float,
        default=0.12,
        help="Maximum velocity (default: 0.12 m/s).",
    )
    parser.add_argument(
        "--max_accel",
        type=float,
        default=0.5,
        help="Maximum acceleration/deceleration (default: 0.5 m/s^2).",
    )
    parser.add_argument(
        "--no_velocity_planning",
        action="store_true",
        help="Disable velocity planning.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the planned trajectory and velocity profile.",
    )
    args = parser.parse_args()

    waypoints = load_waypoints_from_yaml(args.yaml_file)
    x, y, theta, curvature, velocity = plan_route(
        waypoints,
        args.step_size,
        args.max_velocity,
        args.max_accel,
        plan_velocities=not args.no_velocity_planning,
    )
    trajectory = PlannedTrajectory(x, y, theta, curvature, velocity)

    print(f"Planned trajectory length: {trajectory.length:.2f} m")
    print("Number of points in trajectory:", len(trajectory.x))

    if args.plot:
        plot_trajectory(trajectory, waypoints)


if __name__ == "__main__":
    main()
