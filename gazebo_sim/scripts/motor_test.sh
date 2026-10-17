#!/bin/bash
# Needs Gazebo running in another window:  gz sim -r quadcopter.sdf
# A single message gets lost, so send repeatedly (about 5 s).
# Note: maxRotVelocity is 800 rad/s, so 1000 is clipped to full power.
SPEED=${1:-1000}
for i in $(seq 1 50); do
  gz topic -t /X3/gazebo/command/motor_speed -m gz.msgs.Actuators \
    -p "velocity:[$SPEED, $SPEED, $SPEED, $SPEED]"
  sleep 0.1
done
# stop the motors
gz topic -t /X3/gazebo/command/motor_speed -m gz.msgs.Actuators -p 'velocity:[0, 0, 0, 0]'