#!/usr/bin/env bash
# [터미널 4-A] 진짜 GMR 경로: Redis -> ROS2 브릿지.
# PICO + xrobotoolkit_sdk 로 GMR 리타게팅(retarget_teleop.sh)이 Redis에 쓰고 있어야
# 정책이 돈다. GMR이 없으면 브릿지는 기본 자세만 시딩하는데, 그 프레임에는
# keybody_pos_local 이 비어 있어 obs 구성이 실패한다(action_hz=0).
TTY_FLAG=""; [ -t 0 ] && [ -t 1 ] && TTY_FLAG="-it"
exec docker exec $TTY_FLAG igris-sim bash -lc '
source /opt/ros/jazzy/setup.bash
export CYCLONEDDS_URI=file:///robros/configs/cyclonedds/cyclonedds_local.xml
redis-cli ping >/dev/null 2>&1 || redis-server --daemonize yes --bind 0.0.0.0 --protected-mode no
cd /igris/retargeting/GMR
exec python3 scripts/redis_to_ros2_bridge.py \
  --robot robros_igris_c_v2 --redis_host 127.0.0.1 --redis_port 6379 --poll_hz 100'
