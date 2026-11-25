#!/usr/bin/env bash
# 시뮬 스택 컨테이너를 띄운다(백그라운드). 이후 ./sh.sh 로 셸을 여러 개 붙인다.
set -euo pipefail
IG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

docker rm -f igris-sim >/dev/null 2>&1 || true
# MuJoCo 뷰어 창은 기본 꺼짐. 보고 싶으면:  IGRIS_SIM_GUI=1 ./up.sh
# (DISPLAY가 들어가면 igris_c_mujoco가 GLFW 창을 열고, 창을 닫아도
#  launch respawn=True 가 2초 뒤 되살린다 — X를 아예 안 주는 게 끄는 방법)
# igris_c_mujoco 는 GLFW 필수(헤드리스 불가) — GUI 꺼짐일 땐 가상 디스플레이(Xvfb :99)로 창을 숨긴다.
GUI_ARGS=(-e DISPLAY=:99)
if [ "${IGRIS_SIM_GUI:-0}" = "1" ]; then
  xhost +local:root >/dev/null 2>&1 || true
  GUI_ARGS=(-e DISPLAY="${DISPLAY:-:0}" -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix)
fi

# GPU 는 있으면 쓴다. nvidia-container-toolkit 이 없는 서버에서 --gpus all 은 docker run 자체를 실패시킨다
# (MuJoCo 뷰어는 mesa llvmpipe 로 떨어져도 돈다).
GPU_ARGS=()
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
  GPU_ARGS=(--gpus all -e NVIDIA_DRIVER_CAPABILITIES=all)
fi

docker run -d --name igris-sim \
  --shm-size=2g \
  -p 127.0.0.1:6379:6379 \
  --add-host=host.docker.internal:host-gateway \
  ${GPU_ARGS[@]+"${GPU_ARGS[@]}"} \
  --ulimit memlock=-1 --ulimit rtprio=99 --cap-add=SYS_NICE \
  ${GUI_ARGS[@]+"${GUI_ARGS[@]}"} \
  -v "$IG":/igris \
  -v "$IG/configs":/robros/configs \
  -v "$IG/sim_local/robot.yaml":/robros/configs/robot.yaml:ro \
  -v "$IG/sim_local/cyclonedds_local.xml":/robros/configs/cyclonedds/cyclonedds_local.xml:ro \
  igris-sim:jazzy sleep infinity >/dev/null

if [ "${IGRIS_SIM_GUI:-0}" != "1" ]; then
  # 가상 디스플레이 기동 (이미지에 없으면 설치)
  docker exec igris-sim bash -lc 'which Xvfb >/dev/null 2>&1 || { apt-get update -qq >/dev/null && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq xvfb >/dev/null 2>&1; }' 
  docker exec -d igris-sim Xvfb :99 -screen 0 1280x720x24
fi

echo "컨테이너 igris-sim 기동."
echo "  셸 열기: $IG/sim_local/sh.sh"
echo "  내리기 : $IG/sim_local/down.sh"
