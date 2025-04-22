import mujoco
import mujoco.viewer
import numpy as np
import time
import keyboard

def main():
    # Load the model
    model = mujoco.MjModel.from_xml_path("pallet-mujoco/pallet_bricks_column.xml")
    data = mujoco.MjData(model)

    # Control parameters
    move_speed = 5.0  # Speed for forward/backward movement
    turn_speed = 5.0  # Speed for rotation
    slide_speed = 5.0  # Speed for left/right movement

    # Initialize control values
    control = np.zeros(model.nu)

    # Start the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Handle keyboard input
            if keyboard.is_pressed('right'):  # Forward
                control[0] = move_speed
            elif keyboard.is_pressed('left'):  # Backward
                control[0] = -move_speed
            else:
                control[0] = 0

            if keyboard.is_pressed('up'):  # Left
                control[1] = -slide_speed
            elif keyboard.is_pressed('down'):  # Right
                control[1] = slide_speed
            else:
                control[1] = 0

            if keyboard.is_pressed('q'):  # Turn left
                control[2] = turn_speed
            elif keyboard.is_pressed('e'):  # Turn right
                control[2] = -turn_speed
            else:
                control[2] = 0

            # Apply control
            data.ctrl[:] = control

            # Step the simulation
            mujoco.mj_step(model, data)

            # Update the viewer
            viewer.sync()

            # Small delay to control simulation speed
            time.sleep(0.01)

if __name__ == "__main__":
    main() 