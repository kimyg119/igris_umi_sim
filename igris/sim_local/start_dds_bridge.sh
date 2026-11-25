#!/usr/bin/env bash
# DDS<->ZMQ 브리지를 컨테이너 안에서 띄운다 — 호스트의 igris-robot-monitor(cuRobo venv, DDS 없음)가
# 붙을 JSON 엔드포인트를 연다. 상태 PUB 5560(100 Hz), 명령 SUB 5561(→ LowCmd 50 Hz, 워치독 재발행).
# 선행: ./start_stack.sh 로 sim 이 떠 있을 것. 인자는 dds_zmq_bridge.py 로 그대로 전달.
#   CMD_TYPES=simlocal ./start_dds_bridge.sh     # LowCmd 경로 전환(기본 ros — sim 실측으로 유일 수용)
# 끝낼 때:  docker exec igris-sim pkill -f "dds_zmq_bridg[e]"
set -u
cd "$(dirname "$0")"; mkdir -p logs
docker ps --format '{{.Names}}' | grep -q '^igris-sim$' || { echo "컨테이너 없음: IGRIS_SIM_GUI=1 ./up.sh 먼저"; exit 1; }

# SDK ws 도 소스한다 — 기본 경로(--cmd-types ros)가 igris_c_sdk/msg/LowCmd 를 쓴다(네이티브 파이썬
# 타입은 컨트롤러 리더의 DisallowTypeCoercion 에 막힌다, dds_zmq_bridge._ros_cmd_writer 참조).
ROS='source /opt/ros/jazzy/setup.bash; source /igris/ws/install/setup.bash 2>/dev/null; source /igris/sim_local/tools_ws/install/setup.bash 2>/dev/null; export CYCLONEDDS_URI=file:///robros/configs/cyclonedds/cyclonedds_local.xml'
PYP='export PYTHONPATH=/igris/igris-lerobot:${PYTHONPATH:-}'
# 이미 떠 있는 브리지를 먼저 내린다 — 안 그러면 새 프로세스가 5560 바인드에 실패하고, 아래 pgrep 이
# 옛 프로세스를 보고 성공이라 답한다.
docker exec igris-sim pkill -f "dds_zmq_bridg[e]" 2>/dev/null && sleep 1
# 핀 이미지엔 둘 다 없고 컨테이너 재생성 시 소실된다 → 멱등 설치(start_sonic.sh 와 같은 패턴).
docker exec igris-sim bash -lc "python3 -c 'import cyclonedds' 2>/dev/null || pip install --break-system-packages --no-deps 'cyclonedds==11.0.1' >/dev/null 2>&1 && echo '[start_dds_bridge] cyclonedds ready'"
docker exec igris-sim bash -lc "python3 -c 'import zmq' 2>/dev/null || pip install --break-system-packages 'pyzmq' >/dev/null 2>&1 && echo '[start_dds_bridge] pyzmq ready'"

echo "=== [start_dds_bridge] $(date '+%F %T') cmd-types=${CMD_TYPES:-ros} args: $*" >> logs/dds_bridge.log
nohup docker exec igris-sim bash -lc "$ROS; $PYP; exec python3 /igris/igris_curobo/tools/dds_zmq_bridge.py --ns igris_c_001 --cmd-types ${CMD_TYPES:-ros} $*" >> logs/dds_bridge.log 2>&1 &
sleep 2
if docker exec igris-sim pgrep -f "dds_zmq_bridg[e]" >/dev/null; then
  IP=$(docker inspect -f '{{.NetworkSettings.IPAddress}}' igris-sim)
  echo "[start_dds_bridge] OK (로그: logs/dds_bridge.log)"
  echo "[start_dds_bridge] 모니터: igris-robot-monitor --state tcp://$IP:5560 --cmd tcp://$IP:5561"
else
  echo "[start_dds_bridge] 기동 실패 — tail logs/dds_bridge.log"; tail -8 logs/dds_bridge.log
fi
