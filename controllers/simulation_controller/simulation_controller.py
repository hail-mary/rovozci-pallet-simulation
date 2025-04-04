"""simulation_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot
import math

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getDevice('motorname')
#  ds = robot.getDevice('dsname')
#  ds.enable(timestep)
steer_motor = robot.getDevice("rovozci_motor_steering")
wheel_motor = robot.getDevice("rovozci_motor_wheel")

# Main loop:
# - perform simulation steps until Webots is stopping the controller
counter = 0

while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    t = counter / 20
    angle = math.sin(-t) * math.pi / 8
    #steer_motor.setPosition(float('inf'))
    steer_motor.setPosition(angle)
    wheel_motor.setPosition(float('inf'))
    wheel_motor.setVelocity(3.0)
    counter += 1
    print(counter, angle)
    # print('step')

# Enter here exit cleanup code.
