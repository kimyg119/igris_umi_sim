#!/usr/bin/env bash
# [터미널 1] 시뮬 스택을 백그라운드로 한 번에: sim + GMT 정책 + Redis→ROS2 브릿지 (로그 logs/{sim,policy,bridge}.log).
# 순서: 터미널 1 ./start_stack.sh → 터미널 2 ./start_planner.sh --keyboard ("[planner] providers=" 뜨면)
#       → 터미널 1 ./bringup.sh → 터미널 2 에서 조종. 끝낼 때 ./stop_stack.sh
# 컨테이너는 떠 있어야 한다: 창을 보려면 IGRIS_SIM_GUI=1 ./up.sh (up.sh 는 기존 컨테이너를 지우고 새로 만든다).
cd "$(dirname "$0")"; mkdir -p logs
docker ps --format '{{.Names}}' | grep -q '^igris-sim$' || { echo "컨테이너 없음: IGRIS_SIM_GUI=1 ./up.sh 먼저"; exit 1; }
# 좀비(<defunct>)는 제외하고 실제 실행 중인 mujoco 만 본다
if docker exec igris-sim bash -lc 'ps -eo stat,cmd | grep -E "igris_c_mujoc[o]" | grep -qv "^Z"'; then echo "시뮬 스택이 이미 떠 있음 (./stop_stack.sh 로 내린 뒤 다시)"; exit 0; fi

ROS='source /opt/ros/jazzy/setup.bash; source /igris/ws/install/setup.bash; export CYCLONEDDS_URI=file:///robros/configs/cyclonedds/cyclonedds_local.xml'
# 코어 고정: 시뮬(MuJoCo/컨트롤러)은 SIM_CORES, 정책은 마지막 4코어. 같은 머신에서 학습이 돌면 학습 쪽을 0-15 로 묶어라
#   (예: taskset -cp 0-15 <train.py pid>). 렌더/학습이 코어를 뺏으면 컨트롤러가 300 Hz → 10 Hz 대로 떨어져 랙이 생긴다.
SIM_CORES=${SIM_CORES:-16-27}
# 정책 설정: 기본 = 손목 홀드(kp 30/kd 3). 홀드 없는 action.yaml 은 0.6 kg 손 진자가 보행 중 낙상을 만든다(README 손목 홀드 실험).
POLICY_CFG=${POLICY_CFG:-/igris/sim_local/params_teleop_wristhold.yaml}
nohup docker exec igris-sim bash -lc "$ROS; cd /tmp; exec taskset -c $SIM_CORES ros2 launch igris_c_controller simulation.launch.py config_file:=/robros/configs/robot.yaml" > logs/sim.log 2>&1 &
sleep 12
nohup docker exec igris-sim bash -lc "$ROS; source /igris/gmt_ws/install/setup.bash; N=\$(nproc); exec taskset -c \$((N-4))-\$((N-1)) ros2 run igris_c_gmt_public igris_c_gmt_public_node --config $POLICY_CFG --namespace igris_c_001" > logs/policy.log 2>&1 &
nohup docker exec igris-sim bash -lc "$ROS; redis-cli ping >/dev/null 2>&1 || redis-server --daemonize yes --bind 0.0.0.0 --protected-mode no --save '' --dir /tmp; redis-cli CONFIG SET save '' >/dev/null 2>&1; cd /igris/retargeting/GMR; exec python3 scripts/redis_to_ros2_bridge.py --robot robros_igris_c_v2 --redis_host 127.0.0.1 --redis_port 6379 --poll_hz 100" > logs/bridge.log 2>&1 &
sleep 6
grep -q "Control:" logs/sim.log && echo "sim OK (cores $SIM_CORES, policy cfg $POLICY_CFG)" || echo "sim 로그에 Control 없음 — tail -f logs/sim.log 확인"
docker exec igris-sim pgrep -f "redis_to_ros2_bridg[e]" >/dev/null && echo "bridge OK" || echo "bridge 실패 — logs/bridge.log"
echo "다음: 터미널 2 → ./start_planner.sh --keyboard  → (providers= 줄 뜨면) 이 터미널 → ./bringup.sh"
