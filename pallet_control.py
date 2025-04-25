import mujoco
import mujoco.viewer
import numpy as np
import time
import keyboard
from mujoco import mjtObj

def get_hinge_joint_angle(model, data, joint_name):
    joint_id = model.mj_name2id(model, mjtObj.mjOBJ_JOINT, joint_name)
    qpos_address = model.jnt_qposadr[joint_id]
    return float(data.qpos[qpos_address])

def main():
    # Load the model
    model = mujoco.MjModel.from_xml_path("pallet-mujoco/pallet_bricks_column.xml")
    data = mujoco.MjData(model)

    # Control parameters
    move_speed = 8.0  # Speed for forward/backward movement
    turn_speed = 8.0  # Speed for rotation
    slide_speed = 8.0  # Speed for left/right movement

    # Initialize control values
    control = np.zeros(model.nu)

    # Start the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # get current pallet position and heading

            current_x = data.joint("pallet_slide_x").qpos
            current_y = data.joint("pallet_slide_y").qpos
            current_phi = data.joint("pallet_hinge_z").qpos

            heading_vec = np.array([np.cos(current_phi), np.sin(current_phi)])

            # Handle keyboard input
            if keyboard.is_pressed('up'):  # Left
                control[0] = heading_vec[0]*move_speed
                control[1] = heading_vec[1]*move_speed
            elif keyboard.is_pressed('down'):  # Right
                control[0] = - heading_vec[0]*move_speed
                control[1] = - heading_vec[1]*move_speed
            else:
                control[1] = 0
                control[0] = 0

            if keyboard.is_pressed('q'):  # Turn left
                control[2] = turn_speed
            elif keyboard.is_pressed('e'):  # Turn right
                control[2] = -turn_speed
            else:
                control[2] = 0

            # Apply control
            data.ctrl[:] = control

            # print(f"qpos: {data.qpos}, qvel: {data.qvel}")
            print(data.joint("pallet_hinge_z").qpos, data.joint("pallet_slide_x").qpos, data.joint("pallet_slide_y").qpos)

            # Step the simulation
            mujoco.mj_step(model, data)

            # Update the viewer
            viewer.sync()

            # Small delay to control simulation speed
            time.sleep(0.01)

if __name__ == "__main__":
    main() 