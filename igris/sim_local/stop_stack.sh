#!/usr/bin/env bash
# 시뮬 스택(sim/정책/브릿지)과 호스트 플래너 브릿지를 내린다. 컨테이너는 남긴다(완전히 내리려면 ./down.sh).
# 패턴에 [x] 를 쓰는 이유: pkill -f 가 자기 자신(이 스크립트/셸)의 명령줄에 걸리지 않게.
pkill -f "planner_bridg[e].py" 2>/dev/null
docker exec igris-sim bash -lc 'pkill -f "redis_to_ros2_bridg[e]"; pkill -f "igris_c_gmt_public_nod[e]"; pkill -f "simulation.launc[h].py"; sleep 2; pkill -f "igris_c_mujoc[o]"; pkill -f "igris_c_controlle[r]"; pkill -f "igris_c_param_serve[r]"; pkill -f "igris_c_ca[n]"; pkill -f "igris_c_hand_dumm[y]"; pkill -f "bridge_si[m]"' 2>/dev/null
sleep 1
echo "내림. 남은 프로세스:"; docker exec igris-sim bash -lc 'ps -eo pid,cmd | grep -E "mujoc[o]|igris_[c]|gmt_publi[c]|redis_to_ros[2]" || true'
