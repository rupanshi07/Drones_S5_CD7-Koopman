#!/bin/bash
# Prints the drone ("x3") pose from the live pose topic. Gazebo must be running:
#   gz sim -r quadcopter.sdf
# Press Ctrl+C to stop.
gz topic -e -t /world/quadcopter/pose/info | grep -A 12 'name: "x3"'