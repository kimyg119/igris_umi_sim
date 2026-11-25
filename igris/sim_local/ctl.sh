#!/usr/bin/env bash
# 제어 명령 한 방.
#   ./ctl.sh sim    <pause|resume|reset|reload>
#   ./ctl.sh torque <on|off>
#   ./ctl.sh mode   <low|lowwalk|hold|stop|walk|home|preset:NAME>
# TTY가 아닐 때(파이프/스크립트) -it 을 붙이면 docker exec 가 실패하므로 분기한다.
TTY_FLAG=""
[ -t 0 ] && [ -t 1 ] && TTY_FLAG="-it"
exec docker exec $TTY_FLAG igris-sim bash -lc "
source /opt/ros/jazzy/setup.bash
source /igris/ws/install/setup.bash
source /igris/sim_local/tools_ws/install/setup.bash
exec ros2 run igris_sim_ctl igris_ctl $* \
  --cyclonedds-xml /robros/configs/cyclonedds/cyclonedds_local.xml"
