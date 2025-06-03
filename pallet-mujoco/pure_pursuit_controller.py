import mujoco
import mujoco.viewer
import numpy as np
import math
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class Waypoint:
    x: float
    y: float
    theta: float

class PurePursuitController:
    def __init__(self, lookahead_distance: float = 1.0, max_steering_angle: float = 90.0):
        self.lookahead_distance = lookahead_distance
        self.max_steering_angle = max_steering_angle
        self.path: List[Waypoint] = []
        self.current_waypoint_idx = 0
        
    def set_path(self, path: List[Waypoint]):
        """Set the reference path to follow."""
        self.path = path
        self.current_waypoint_idx = 0
        
    def find_lookahead_point(self, current_x: float, current_y: float) -> Tuple[float, float]:
        """Find the lookahead point on the path."""
        if not self.path:
            return current_x, current_y
            
        # Find the closest point on the path
        min_dist = float('inf')
        closest_idx = 0
        
        for i in range(len(self.path)):
            dx = self.path[i].x - current_x
            dy = self.path[i].y - current_y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
                
        # Look ahead from the closest point
        lookahead_idx = closest_idx
        while lookahead_idx < len(self.path) - 1:
            dx = self.path[lookahead_idx + 1].x - current_x
            dy = self.path[lookahead_idx + 1].y - current_y
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > self.lookahead_distance:
                break
            lookahead_idx += 1
            
        return self.path[lookahead_idx].x, self.path[lookahead_idx].y
        
    def compute_control(self, current_x: float, current_y: float, current_theta: float) -> Tuple[float, float, float]:
        """Compute control inputs for the pallet jack."""
        if not self.path:
            return 0.0, 0.0, 0.0
            
        # Find lookahead point
        lookahead_x, lookahead_y = self.find_lookahead_point(current_x, current_y)
        
        # Calculate alpha (angle between current heading and lookahead point)
        dx = lookahead_x - current_x
        dy = lookahead_y - current_y
        alpha = math.atan2(dy, dx) - current_theta
        
        # Normalize alpha to [-pi, pi]
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        
        # Calculate steering angle (pure pursuit formula)
        steering_angle = math.atan2(2.0 * math.sin(alpha), self.lookahead_distance)
        
        # Limit steering angle
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        
        # Calculate forward velocity (constant for now)
        forward_velocity = 1.0
        
        # Calculate lateral velocity (for side movement)
        lateral_velocity = 0.0  # Pure pursuit typically doesn't use lateral movement
        
        return forward_velocity, lateral_velocity, steering_angle

def main():
    # Load the model
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    
    # Create controller
    controller = PurePursuitController(lookahead_distance=1.0, max_steering_angle=90.0)
    
    # Create a simple circular path
    path = []
    radius = 2.0
    num_points = 50
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        theta = angle + math.pi/2  # Tangent to circle
        path.append(Waypoint(x, y, theta))
    
    controller.set_path(path)
    
    # Start the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Get current state
            current_x = data.qpos[0]
            current_y = data.qpos[1]
            current_theta = data.qpos[2]
            
            # Compute control inputs
            vx, vy, steer = controller.compute_control(current_x, current_y, current_theta)
            
            # Apply control inputs
            data.ctrl[0] = vx  * 500# drive_x
            data.ctrl[1] = vy  * 500# drive_y
            data.ctrl[2] = steer *10 # steer
            
            # Step simulation
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # Print current state
            print(f"Position: ({current_x:.2f}, {current_y:.2f}), Heading: {math.degrees(current_theta):.2f}°")
            print(f"Controls: vx={vx:.2f}, vy={vy:.2f}, steer={math.degrees(steer):.2f}°")

if __name__ == "__main__":
    main() 