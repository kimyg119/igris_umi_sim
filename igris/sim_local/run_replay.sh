#!/usr/bin/env bash
# 컨테이너 시뮬 스택에서 pkl 레퍼런스를 정책으로 재생·녹화·지표·렌더까지 한 번에.
#   ./run_replay.sh <pkl(컨테이너 경로 /igris/...)> <params yaml(컨테이너 경로)> <출력 이름> [--host_pkl <호스트 pkl 경로>]
# 예) ./run_replay.sh /igris/retargeting/teleop/box_pickup.clean.pkl /igris/sim_local/params_teleop_wbt.yaml box_pickup_wbt
# 산출: videos/<name>.npz(녹화), videos/<name>_replay.mp4, 지표 stdout (replay_metrics.py)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
PKL="$1"; PARAMS="$2"; NAME="$3"; shift 3
HOST_PKL="${HERE}/..${PKL#/igris}"
while [ $# -gt 0 ]; do case "$1" in --host_pkl) HOST_PKL="$2"; shift 2;; *) echo "unknown $1"; exit 1;; esac; done
DUR="${DUR:-22}"

dx(){ docker exec igris-sim bash -lc "$1"; }
dxd(){ docker exec -d igris-sim bash -lc "$1"; }
ROS='source /opt/ros/jazzy/setup.bash; source /igris/ws/install/setup.bash; export CYCLONEDDS_URI=file:///robros/configs/cyclonedds/cyclonedds_local.xml'

docker ps --filter name=igris-sim --format '{{.Status}}' | grep -q Up || ./up.sh
echo "[1] 잔여 프로세스 정리"
dx 'for pat in "[g]mr_pkl_pub" "[s]tate_recorder" "[g]mt_public_node" "[i]gris_c_mujoco" "[i]gris_c_controller" "[s]imulation.launch"; do pids=$(ps -eo pid,cmd | grep -E "$pat" | awk "{print \$1}"); [ -n "$pids" ] && kill -INT $pids; done; sleep 3; for pat in "[g]mt_public_node" "[i]gris_c_mujoco" "[i]gris_c_controller" "[s]imulation.launch"; do pids=$(ps -eo pid,cmd | grep -E "$pat" | awk "{print \$1}"); [ -n "$pids" ] && kill -9 $pids; done; rm -f /tmp/play.trigger /tmp/sim.log /tmp/policy.log /tmp/pub.log /tmp/rec.log; true'
echo "[2] 시뮬 스택"
dxd "$ROS; cd /tmp; exec ros2 launch igris_c_controller simulation.launch.py config_file:=/robros/configs/robot.yaml > /tmp/sim.log 2>&1"
sleep 6
echo "[3] 정책 노드 ($PARAMS)"
dxd "$ROS; source /igris/gmt_ws/install/setup.bash; N=\$(nproc); exec taskset -c \$((N-4))-\$((N-1)) ros2 run igris_c_gmt_public igris_c_gmt_public_node --config $PARAMS --namespace igris_c_001 > /tmp/policy.log 2>&1"
echo "[4] 퍼블리셔 홀드 ($PKL)"
dxd "$ROS; exec python3 /igris/sim_local/tools/gmr_pkl_pub.py --pkl $PKL --start_file /tmp/play.trigger --no-loop > /tmp/pub.log 2>&1"
sleep 8
dx "grep -v 'Failed to parse' /tmp/pub.log | tail -n 1"
echo "[5] bringup"
./bringup.sh 2>&1 | grep -E "reset #|인계 후|기립 실패|경고"
dx "grep -E 'motion_mode' /tmp/policy.log | tail -n 1 | grep -oE 'motion_mode=\S+|robot_state=\S+|action_hz=\S+' | paste -sd' '"
echo "[6] 녹화 ${DUR}s + 트리거"
dxd "$ROS; exec python3 /igris/sim_local/tools/state_recorder.py --out /igris/sim_local/videos/$NAME.npz --duration $DUR --rate 30 > /tmp/rec.log 2>&1"
sleep 3; dx "touch /tmp/play.trigger"; sleep $((DUR + 4))
T0=$(dx "a=\$(grep -oE '\[[0-9]+\.[0-9]+\].*녹화 시작' /tmp/rec.log | grep -oE '[0-9]+\.[0-9]+' | head -1); b=\$(grep -oE '\[[0-9]+\.[0-9]+\].*트리거 감지' /tmp/pub.log | grep -oE '[0-9]+\.[0-9]+' | head -1); python3 -c \"print(round(\$b-\$a,3))\"")
dx "grep -E '저장' /tmp/rec.log | grep -v 'Failed to parse'"
docker exec igris-sim chown "$(id -u):$(id -g)" "/igris/sim_local/videos/$NAME.npz"
echo "[7] 지표 (t0=$T0)"
~/miniconda3/envs/gmr/bin/python tools/replay_metrics.py --npz "videos/$NAME.npz" --ref_pkl "$HOST_PKL" --t0 "$T0" | tee "videos/$NAME.metrics.txt"
echo "[8] 렌더"
dx "cd /igris/sim_local && MUJOCO_GL=egl python3 tools/render_replay.py --npz videos/$NAME.npz --out videos/${NAME}_replay.mp4 2>&1 | tail -n 1"
docker exec igris-sim chown "$(id -u):$(id -g)" "/igris/sim_local/videos/${NAME}_replay.mp4"
echo "[9] 정리"
dx 'for pat in "[g]mr_pkl_pub" "[g]mt_public_node" "[i]gris_c_mujoco" "[i]gris_c_controller" "[s]imulation.launch"; do pids=$(ps -eo pid,cmd | grep -E "$pat" | awk "{print \$1}"); [ -n "$pids" ] && kill -INT $pids; done; sleep 3; for pat in "[g]mt_public_node" "[i]gris_c_mujoco" "[i]gris_c_controller" "[s]imulation.launch"; do pids=$(ps -eo pid,cmd | grep -E "$pat" | awk "{print \$1}"); [ -n "$pids" ] && kill -9 $pids; done; rm -f /tmp/play.trigger; true'
echo "완료: videos/$NAME.npz videos/${NAME}_replay.mp4 videos/$NAME.metrics.txt"
