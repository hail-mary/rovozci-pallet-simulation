import pychrono.core as chrono
import numpy as np
from typing import List


def create_bricks(system: chrono.ChSystemNSC) -> List[chrono.ChBody]:
    """
    Create a stack of 80 bricks arranged 5 high, 4 by 4 on each layer.

    Args:
        system (chrono.ChSystemNSC): The Chrono physical system.

    Returns:
        List[chrono.ChBody]: List of brick bodies added to the system.
    """
    bricks: List[chrono.ChBody] = []
    # Brick half-sizes (m)
    sx, sy, sz = 0.15, 0.125, 0.10
    # Spacing between bricks
    dsx, dsy, dsz = 2 * sx, 2 * sy, 2 * sz
    # Base position
    base = np.array([0.75, -0.375, 0.25])

    for layer in range(5):
        for row in range(4):
            for col in range(4):
                brick = chrono.ChBody()
                brick.SetMass(2.0)
                pos = base + np.array([col * dsx, row * dsy, layer * dsz])
                brick.SetPos(chrono.ChVectorD(*pos))
                brick.SetBodyFixed(False)

                box_shape = chrono.ChBoxShape()
                box_shape.GetBoxGeometry().Size = chrono.ChVectorD(sx, sy, sz)
                brick.AddAsset(box_shape)

                system.Add(brick)
                bricks.append(brick)

    return bricks


def setup_system() -> tuple[chrono.ChSystemNSC, chrono.ChLinkMotorLinearSpeed,
                           chrono.ChLinkMotorLinearSpeed, chrono.ChLinkMotorRotationSpeed]:
    """
    Initialize the Chrono NSC system, create ground, pallet jack, bricks, and actuators.

    Returns:
        System object and the three motors (drive_x, drive_y, steer).
    """
    system = chrono.ChSystemNSC()
    system.Set_G_acc(chrono.ChVectorD(0, 0, -9.81))

    # Ground
    ground = chrono.ChBody()
    ground.SetBodyFixed(True)
    system.Add(ground)

    # Pallet jack platform
    platform = chrono.ChBody()
    platform.SetMass(50.0)
    platform.SetPos(chrono.ChVectorD(0, 0, 0.1))
    system.Add(platform)

    # drive_x (prismatic along chassis local X)
    jx = chrono.ChLinkLockPrismatic()
    jx.Initialize(platform, ground,
                  chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0.1),
                                     chrono.Q_from_AngAxis(0, chrono.ChVectorD(1, 0, 0))))
    system.Add(jx)
    mx = chrono.ChLinkMotorLinearSpeed()
    mx.Initialize(jx, chrono.ChFrameD(), chrono.ChFrameD())
    mx.SetSpeedFunction(chrono.ChFunction_Const(0))
    system.Add(mx)

    # drive_y (prismatic along chassis local Y)
    jy = chrono.ChLinkLockPrismatic()
    jy.Initialize(platform, ground,
                  chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0.1),
                                     chrono.Q_from_AngAxis(np.pi / 2, chrono.ChVectorD(0, 0, 1))))
    system.Add(jy)
    my = chrono.ChLinkMotorLinearSpeed()
    my.Initialize(jy, chrono.ChFrameD(), chrono.ChFrameD())
    my.SetSpeedFunction(chrono.ChFunction_Const(0))
    system.Add(my)

    # steer (revolute about Z)
    js = chrono.ChLinkLockRevolute()
    js.Initialize(platform, ground,
                  chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0.1),
                                     chrono.Q_from_AngAxis(0, chrono.ChVectorD(0, 0, 1))))
    system.Add(js)
    ms = chrono.ChLinkMotorRotationSpeed()
    ms.Initialize(js, chrono.ChFrameD(), chrono.ChFrameD())
    ms.SetSpeedFunction(chrono.ChFunction_Const(0))
    system.Add(ms)

    # Create bricks
    create_bricks(system)

    return system, mx, my, ms


def circle_motion_control(system: chrono.ChSystemNSC,
                          motor_x: chrono.ChLinkMotorLinearSpeed,
                          motor_y: chrono.ChLinkMotorLinearSpeed,
                          motor_steer: chrono.ChLinkMotorRotationSpeed,
                          radius: float = 2.0,
                          speed: float = 0.2,
                          duration: float = 20.0,
                          step: float = 0.01) -> None:
    """
    Run simulation applying motor commands for circular motion.

    Args:
        system (chrono.ChSystemNSC): The Chrono system.
        motor_x (chrono.ChLinkMotorLinearSpeed): Motor for drive_x joint.
        motor_y (chrono.ChLinkMotorLinearSpeed): Motor for drive_y joint.
        motor_steer (chrono.ChLinkMotorRotationSpeed): Motor for steer joint.
        radius (float): Circle radius in meters.
        speed (float): Linear speed in m/s.
        duration (float): Total simulation time in seconds.
        step (float): Time step for integration in seconds.
    """
    omega = speed / radius
    time = 0.0
    while time < duration:
        # Desired velocities
        vx = speed * np.cos(omega * time)
        vy = speed * np.sin(omega * time)
        steer_rate = omega  # constant steering rate for circular path

        motor_x.SetSpeedFunction(chrono.ChFunction_Const(vx))
        motor_y.SetSpeedFunction(chrono.ChFunction_Const(vy))
        motor_steer.SetSpeedFunction(chrono.ChFunction_Const(steer_rate))

        system.DoStepDynamics(step)
        time += step


if __name__ == "__main__":
    chrono.SetChronoDataPath("")  # adjust if needed
    system, motor_x, motor_y, motor_steer = setup_system()
    circle_motion_control(system, motor_x, motor_y, motor_steer)
