#!/usr/bin/env bash
# SONIC 트래커 노드를 컨테이너 안에서 띄운다 — student_130(igris_c_gmt_public) 대체.
#   50Hz 한 프로세스: 로봇 상태(/igris_c_001/rt/joint_state + /mujoco/imu) → Φ_enc → SONIC
#   enc/dec → 코덱 Φ_dec → SDK 31 lowcmd(손목 홀드). DDS 참여자는 발행 프로세스 안에서 생성(포크 금지).
# 선행: ./start_stack.sh 로 sim/브릿지가 떠 있고, 기립 완료(./bringup.sh)된 상태에서 실행.
#       이 스크립트가 student_130 을 먼저 내리고 트래커를 유일 lowcmd 소스로 띄운다.
# 인자는 sonic_tracker.py 로 그대로 전달(예: --duration 20). 기본은 Ctrl-C/pkill 까지 상시 루프.
#   드라이런:  ./start_sonic.sh --dry-run   (sim 불필요, DDS 미접촉)
# 끝낼 때:  pkill -f "sonic_tracke[r].py"   (또는 student_130 재기동은 start_stack 의 정책 줄 참고)
set -u
cd "$(dirname "$0")"; mkdir -p logs
docker ps --format '{{.Names}}' | grep -q '^igris-sim$' || { echo "컨테이너 없음: IGRIS_SIM_GUI=1 ./up.sh 먼저"; exit 1; }

ROS='source /opt/ros/jazzy/setup.bash; source /igris/ws/install/setup.bash 2>/dev/null; source /igris/sim_local/tools_ws/install/setup.bash 2>/dev/null; export CYCLONEDDS_URI=file:///robros/configs/cyclonedds/cyclonedds_local.xml'
# sonic_engine 이 sys.path 에 자기 디렉토리+코덱을 넣지만, PYTHONPATH 로도 못박아 둔다(SDK DDS 타입은 lowcmd_pub 이 자체 정의 → 별도 SDK 바인딩 불필요).
PYP='export PYTHONPATH=/igris/sim_local/tools/sonic:/igris/retargeting/tools:${PYTHONPATH:-}'
# 벤더 sim2real 게인 스위치: 호스트에서 SONIC_PARAMS_YAML=/igris/sim_local/params_sonic_vendor.yaml 로
# 실행하면 컨테이너 안 sonic_tracker(spike_hold.load_gains)까지 전파된다. 미설정이면 기존 wrist-hold.
# 호스트의 SONIC_* 환경변수 전부를 컨테이너 트래커로 전파한다. (교훈 0901: docker exec 는 호스트
# env 를 상속하지 않는다 — SONIC_PARAMS_YAML 만 넘기던 시절, CODEC_MATRIX/DEPLOY_TUNE/TUNE_LPF/DEC_ONNX
# 플래그가 조용히 기본값으로 무시되어 하룻밤 ablation 이 전부 동일 구성 반복이었다.)
SPY=''
for _v in SONIC_PARAMS_YAML SONIC_CODEC_MATRIX SONIC_DEPLOY_TUNE SONIC_TUNE_LPF SONIC_DEC_ONNX; do
  _val=$(eval echo "\${$_v:-}")
  [ -n "$_val" ] && SPY="$SPY export $_v=$_val;" && echo "[start_sonic] $_v=$_val"
done

# 코덱 Φ_dec(v2 닫힌형)가 scipy 를 쓰는데 핀 이미지엔 없다 → 없으면 numpy 1.26 호환 버전만 설치(numpy 승격 방지).
docker exec igris-sim bash -lc "$ROS; python3 -c 'import scipy' 2>/dev/null || pip install --break-system-packages --no-deps 'scipy==1.13.1' >/dev/null 2>&1 && echo '[start_sonic] scipy ready'"
# cyclonedds 파이썬 바인딩도 핀 이미지에 없다(컨테이너 재생성 시 소실) → 멱등 설치.
docker exec igris-sim bash -lc "python3 -c 'import cyclonedds' 2>/dev/null || pip install --break-system-packages --no-deps 'cyclonedds==11.0.1' >/dev/null 2>&1 && echo '[start_sonic] cyclonedds ready'"

# 드라이런은 sim/DDS 없이 바로 실행하고 끝낸다.
for a in "$@"; do [ "$a" = "--dry-run" ] && { docker exec igris-sim bash -lc "$ROS; $PYP; $SPY python3 /igris/sim_local/tools/sonic/sonic_tracker.py $*"; exit $?; }; done

echo "[start_sonic] stop student_130 (igris_c_gmt_public_node)"
docker exec igris-sim bash -lc 'pkill -f "igris_c_gmt_public_nod[e]"' 2>/dev/null; sleep 0.4

# 정책 코어: student_130 과 동일하게 마지막 4코어에 고정.
nohup docker exec igris-sim bash -lc "$ROS; $PYP; $SPY N=\$(nproc); exec taskset -c \$((N-4))-\$((N-1)) python3 /igris/sim_local/tools/sonic/sonic_tracker.py $*" > logs/sonic.log 2>&1 &
sleep 4
if docker exec igris-sim pgrep -f "sonic_tracke[r].py" >/dev/null; then
  echo "[start_sonic] 트래커 기동 OK (로그: logs/sonic.log, 코어: 마지막 4)"
  grep -m1 "loop @" logs/sonic.log 2>/dev/null || tail -2 logs/sonic.log
else
  echo "[start_sonic] 기동 실패 — tail logs/sonic.log"; tail -8 logs/sonic.log
fi
