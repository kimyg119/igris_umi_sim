#!/usr/bin/env bash
# One simulator run on the server, wholly inside a private container (network none).
# usage: sim.sh <lane 0..> <container name> <container_run args...>
#   lane s gets three physical cores: CPUs SIM_CPU_BASE+3s .. +3s+2 and their sibling threads.
set -euo pipefail
source "$(dirname "$0")/env.sh"
slot=$1; name=$2; shift 2
a=$((SIM_CPU_BASE + 3 * slot)); cpus="$a,$((a+1)),$((a+2))"
[ "$SIM_CPU_SIBLING" -gt 0 ] && cpus="$cpus,$((a+SIM_CPU_SIBLING)),$((a+SIM_CPU_SIBLING+1)),$((a+SIM_CPU_SIBLING+2))"
server docker run --rm --name "$name" --label owner=$OWNER --network none --ipc private --shm-size 2g \
  --ulimit memlock=-1 --ulimit rtprio=99 --cap-add SYS_NICE --cpuset-cpus "$cpus" -u $RUN_AS -e HOME=/tmp \
  -e PYTHONPATH=$W:$D/vendor_py/mink_cp312:$D/vendor_py/prep_cp312 -e UMI_MINK_PY=$D/vendor_py/mink_cp312 \
  -v $REMOTE/ws:$W:ro -v $REMOTE/ws/igris:/igris:ro \
  -v $REMOTE/ws/igris/configs:/robros/configs:ro -v $REMOTE/DATA:$D -v $REMOTE/DATA/fonts/nanum:/usr/share/fonts/truetype/nanum:ro \
  --tmpfs /robros/logs:mode=1777 -w $W igris-sim:jazzy \
  python3 -m igris.umi_walklow_replay.container_run "$@"
