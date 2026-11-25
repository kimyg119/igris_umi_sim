#!/usr/bin/env bash
# MuJoCo 물리 스레드를 SCHED_FIFO 50 으로 올린다(컨트롤러 스레드 FIFO 68~70 보다 낮게).
# 300 Hz 제어 루프는 물리 스레드(기본 SCHED_OTHER)에 락스텝으로 종속되어, 이 스레드가 CPU 경쟁에서
# 밀리는 만큼 Control Hz 가 비례해 떨어진다(README "300 Hz 가 떨어지는 진짜 이유"). 물리가 돌고 있을 때
# (sim resume 이후) 실행해야 1 s 동안 가장 바쁜 스레드 = 물리 스레드로 식별된다. bringup.sh 가 자동 호출.
docker exec igris-sim bash -lc '
P=$(ps -eo pid,stat,comm | awk "\$2!~/Z/ && \$3==\"igris_c_mujoco\"{print \$1}" | head -1)
[ -n "$P" ] || { echo "igris_c_mujoco 없음"; exit 1; }
declare -A a; for t in /proc/$P/task/*; do a[$(basename $t)]=$(awk "{print \$14+\$15}" $t/stat); done
sleep 1; best=0; T=
for t in /proc/$P/task/*; do id=$(basename $t); d=$(( $(awk "{print \$14+\$15}" $t/stat) - ${a[$id]:-0} )); [ $d -gt $best ] && best=$d && T=$id; done
[ "$best" -ge 10 ] || { echo "물리 스레드 식별 실패(1 s 최대 CPU ${best}%) — sim resume 후 다시 실행"; exit 1; }
chrt -f -p 50 $T && echo "MuJoCo physics tid $T -> SCHED_FIFO 50 (cpu ${best}%)"'
