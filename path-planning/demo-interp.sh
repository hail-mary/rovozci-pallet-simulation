#!/bin/bash

# Example how to run the path planning script with velocity linear interpolation.

python plan_trajectory_timed_interp.py demo_trajectory.yaml --plot --out planned-with-times.csv --timestep 0.05 --max_accel 0.1 --interp