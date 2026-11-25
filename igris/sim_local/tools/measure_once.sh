#!/usr/bin/env bash
# usage: measure_once.sh <tag>  (환경변수로 SONIC_* 게이트 전달)
set -u; cd /home/rcv/workspace/Project_Humanoid/igris/sim_local
TAG=$1
docker exec igris-sim pkill -9 -f "sonic_tracke[r].py" 2>/dev/null
./ctl.sh sim reset >/dev/null 2>&1; sleep 2
SKIP_LOW=1 ./bringup.sh >/dev/null 2>&1
./start_sonic.sh >/dev/null 2>&1; sleep 7
N=$(docker exec igris-sim pgrep -fc 'sonic_tracke[r].py'); [ "$N" = "1" ] || { echo "[$TAG] tracker=$N 비정상"; exit 2; }
docker exec -d igris-sim bash -lc "source /opt/ros/jazzy/setup.bash; export CYCLONEDDS_URI=file:///robros/configs/cyclonedds/cyclonedds_local.xml; cd /igris/sim_local && python3 tools/state_recorder.py --out logs/m_${TAG}.npz --duration 60 --rate 50 >/dev/null 2>&1"
sleep 3; ./ctl.sh mode low >/dev/null 2>&1; sleep 62
docker exec igris-sim python3 -c "
import numpy as np
d=np.load('/igris/sim_local/logs/m_${TAG}.npz', allow_pickle=True)
b=d['base']; t=d['t']-d['t'][0]; z=b[:,2]
fell=z<0.5
if fell.any(): print('[${TAG}] 생존 %.2fs' % (t[int(np.argmax(fell))]-3.0))
else: print('[${TAG}] 낙상 없음 (%.0fs, z_min=%.3f)' % (t[-1]-3.0, z.min()))"
