#!/usr/bin/env bash
# [터미널 1] 시뮬 스택(param/mujoco/can_dummy/hand_dummy/bridge_sim/controller sim)을 띄운다.
# MuJoCo는 일시정지 상태로 시작한다. bringup.sh 가 resume 시킨다.
exec docker exec -it igris-sim bash -lc '
source /opt/ros/jazzy/setup.bash
source /igris/ws/install/setup.bash
cd /tmp
exec ros2 launch igris_c_controller simulation.launch.py \
  config_file:=/robros/configs/robot.yaml'
