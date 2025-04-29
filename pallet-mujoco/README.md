## Pallet-Bricks Sliding Simulation Template

This MuJoCo model defines a simple simulation of a sliding pallet carrying a column of bricks. The pallet is driven along a circular path via a motor actuator, and frictional contact between pallet and bricks is used to keep the bricks from slipping under slow motion.

### Model Components

- **Ground Plane** (`floor`): A static plane at z=0.  
- **Pallet** (`body name="pallet"`):  
  - Positioned at z=0.05 m above the ground.  
  - Single **slide** joint (`pallet_slide`) along the X-axis.  
  - **Motor** actuator drives this joint.  

- **Bricks** (`body name="brick_0"` through `brick_4`):  
  - Five dynamic bodies, each with a **free** joint.  
  - Box geometries sized 0.30 × 0.25 × 0.20 m (full dimensions).  
  - High density (13 000 kg/m³) and friction (0.8) parameters to prevent slipping.  

### Friction Settings

These values ensure that, under slow motor‑driven motion, contact forces transmit motion to the bricks without sliding.

### Running the Simulation

1. **Install MuJoCo** (v2.3 or later) and the Python bindings:  
```bash
pip install mujoco
```

2. **Launch the model** in the MuJoCo viewer:  
```bash
python simul_circular.py

# use mouse wheel to zoom out the scene
# and in the console, press any key to start the simulation
```

3. **Control the motor**:  
   - In the viewer’s control panel, adjust `ctrl[0]` (the motor driving `pallet_slide`) to slide the pallet.  
   - Typical range: ±1 (applies force along ±X).

### Customization

- **Sliding Axis**: Change the `axis` attribute of `<joint name="pallet_slide" …>` to modify the direction of motion.  
- **Actuator Gain**: Adjust the `<actuator><motor gear="…"/></actuator>` `gear` value to increase or decrease driving force.  
- **Friction**: Tweak the `friction` parameters on the pallet and bricks for more or less slip.  
- **Brick Count/Arrangement**: Add or reposition `<body name="brick_*" …>` elements for larger stacks or grids.


