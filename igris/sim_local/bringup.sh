#!/usr/bin/env bash
# [터미널 3] 기동 시퀀스. start_sim.sh + start_policy.sh + (텔레옵 소스)가 떠 있어야 한다.
#
# 순서와 이유:
#   1) sim resume - MuJoCo는 일시정지로 시작한다. 안 하면 MotorStatus 0 Hz.
#   2) torque on  - 토크가 꺼져 있으면 컨트롤러가 LOW 요청을 거부하고 TORQUE_OFF_MODE로 떨어진다.
#   3) mode home  - 컨트롤러가 HIGH_LEVEL_MODE에서 HOME 관절 자세를 "붙잡게" 만든다.
#   4) sim reset  - 여기가 핵심. reset은 스폰 자세로 되돌리는데, 컨트롤러가 이미 자세를
#                   붙잡고 있어야 그대로 서 있는다. 아무도 안 잡고 있으면 바로 넘어진다.
#                   (넘어진 상태에서 HOME을 줘도 관절 보간이라 못 일어난다.)
#                   한 번에 안 되면 반복한다.
#   5) mode low   - 정책에 인계. LOW_MODE=lowwalk 이면 WALK_LOW(컨트롤러 내부 기립 + 외부 lowcmd 통과).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAND_MIN="${STAND_MIN:-0.7}"
RESET_TRIES="${RESET_TRIES:-8}"
LOW_MODE="${LOW_MODE:-low}"          # low | lowwalk

run(){ docker exec igris-sim bash -lc "
  source /opt/ros/jazzy/setup.bash
  source /igris/ws/install/setup.bash
  source /igris/sim_local/tools_ws/install/setup.bash
  ros2 run igris_sim_ctl igris_ctl $1 \
    --cyclonedds-xml /robros/configs/cyclonedds/cyclonedds_local.xml" 2>&1 | grep -E "^발행" || true; }

base_z(){ docker exec igris-sim bash -lc '
  source /opt/ros/jazzy/setup.bash; source /igris/ws/install/setup.bash
  timeout 8 ros2 topic echo --once /mujoco/base_odom 2>/dev/null' \
  | awk "/^    position:/{p=1} p&&/^      z:/{print \$2; exit}"; }

stand(){ awk -v v="${1:-0}" -v m="$STAND_MIN" 'BEGIN{exit !(v>m)}'; }

echo "[1/5] MuJoCo 물리 재개"; run "sim resume"; sleep 2
"$HERE/rt_physics.sh" || true   # 물리 스레드 RT 승격 (300 Hz 루프는 이 스레드에 종속)
echo "[2/5] 토크 ON";          run "torque on";  sleep 2
echo "[3/5] HOME 자세 붙잡기"; run "mode home";  sleep 4

echo "[4/5] sim reset 반복 (기립)"
ok=0
for i in $(seq 1 "$RESET_TRIES"); do
  run "sim reset" >/dev/null; sleep 3
  z=$(base_z || echo 0); printf "      reset #%d -> z=%.3f\n" "$i" "${z:-0}"
  if stand "${z:-0}"; then ok=1; break; fi
done
if [ "$ok" -ne 1 ]; then
  echo "      기립 실패. LOW로 넘기지 않습니다."
  echo "      터미널 1에서 SlowStart / errorJoints / 'command time delay' 를 확인하세요."
  exit 1
fi

if [ "${SKIP_LOW:-0}" = "1" ]; then echo "[5/5] 건너뜀 (SKIP_LOW=1): HIGH 스탠바이 기립 상태로 종료"; exit 0; fi
echo "[5/5] LOW_LEVEL($LOW_MODE) - 정책 인계"; run "mode $LOW_MODE"; sleep 5
z=$(base_z); printf "      인계 후 z=%.3f\n" "${z:-0}"
stand "${z:-0}" || echo "      경고: 인계 후 무너짐. 모션 소스가 유효한지 확인하세요(action_hz)."
echo
[ "$LOW_MODE" = lowwalk ] && rs=ROBOT_STATE_WALK_LOW || rs=ROBOT_STATE_LOW
echo "터미널 2 확인: robot_state=$rs, low_command_mode=true, action_hz=50"
echo "  motion_mode=live       -> ROS2 텔레옵 프레임 수신 중"
echo "  motion_mode=onnx_replay-> 녹화 리플레이 모드"
echo "  action_hz=0            -> 모션 프레임이 없거나 무효 (keybody_pos_local 확인)"
