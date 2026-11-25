#!/usr/bin/env bash
# 컨테이너에 셸을 하나 붙인다. ROS/워크스페이스는 자동 source.
exec docker exec -it igris-sim bash -lc '
source /opt/ros/jazzy/setup.bash
source /igris/ws/install/setup.bash
[ -f /igris/gmt_ws/install/setup.bash ] && source /igris/gmt_ws/install/setup.bash
cd /igris && exec bash'
