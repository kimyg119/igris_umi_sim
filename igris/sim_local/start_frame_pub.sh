#!/usr/bin/env bash
# [터미널 4-B] 헤드셋 없이 텔레옵 경로를 검증하는 테스트 소스.
# 녹화 모션 ONNX를 GMR 페이로드로 변환해 같은 토픽으로 발행한다.
# gmt_public 입장에서는 진짜 GMR과 구분되지 않는다(motion_mode=live).
MOTION="${1:-/igris/igris_c_gmt_public/data/onnx/recorded_standstill.onnx}"
TTY_FLAG=""; [ -t 0 ] && [ -t 1 ] && TTY_FLAG="-it"
exec docker exec $TTY_FLAG igris-sim bash -lc "
source /opt/ros/jazzy/setup.bash
export CYCLONEDDS_URI=file:///robros/configs/cyclonedds/cyclonedds_local.xml
exec python3 /igris/sim_local/tools/gmr_frame_pub.py --motion '$MOTION'"
