#!/usr/bin/env bash
# [터미널 2] GMT 정책 노드(igris_c_gmt_public).
#
# taskset이 중요하다: onnx_runner.cpp가 ONNX Runtime 스레드 수를 지정하지 않아
# 코어 수만큼 스레드를 띄우고 스핀 대기한다. 32코어 머신에서는 20+ 코어를 먹어
# MuJoCo와 컨트롤러를 굶긴다(제어 루프 300Hz -> 1.8Hz). 코어 4개로 묶으면 충분하다.
CFG="${1:-/igris/sim_local/params_teleop.yaml}"   # 리플레이로 되돌리려면 params_sim.yaml 을 인자로
exec docker exec -it igris-sim bash -lc "
source /opt/ros/jazzy/setup.bash
source /igris/ws/install/setup.bash
source /igris/gmt_ws/install/setup.bash
export CYCLONEDDS_URI=file:///robros/configs/cyclonedds/cyclonedds_local.xml
N=\$(nproc); CORES=\"\$((N-4))-\$((N-1))\"
echo \"ONNX 스레드를 코어 \$CORES 로 제한\"
exec taskset -c \$CORES ros2 run igris_c_gmt_public igris_c_gmt_public_node \
  --config $CFG --namespace igris_c_001"
