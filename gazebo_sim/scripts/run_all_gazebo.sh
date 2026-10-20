#!/bin/bash
WORLD=/mnt/c/S5/Drones/koopman-mpc-drone/gazebo_sim/worlds/quadcopter_wrench.sdf
DIR=/mnt/c/S5/Drones/koopman-mpc-drone/gazebo_sim/scripts
for c in pid lqr; do
  for s in 1 2 3 4 5 6; do
    echo "=== $c scenario $s ==="
    gz sim -s -r "$WORLD" > /tmp/gz_${c}_${s}.log 2>&1 &
    sleep 8
    python3 "$DIR/run_scenario_gazebo.py" $c $s | grep -v "^t="
    pkill -f "gz sim" ; pkill -f gz-sim
    sleep 3
  done
done
echo "ALL DONE"
