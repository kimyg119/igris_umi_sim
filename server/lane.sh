#!/usr/bin/env bash
# One job on one lane, tried again (in a new output folder) when the vendor simulation does not come up.
# On the server this work was done on, one bring-up in seven ended with the robot not standing: the physics step
# then takes 12 ms instead of 0.7 ms for the whole run.  Such a run has produced nothing; a second try is a new run.
# usage: lane.sh <lane> <batch name> <tag> <policy checkpoint or -> <container_run args...>
set -uo pipefail
cd "$(dirname "$0")"; source env.sh
lane=$1; name=$2; tag=$3; checkpoint=$4; shift 4
for attempt in 1 2 3; do
  run=$tag; [[ $attempt -gt 1 ]] && run=${tag}_try$attempt
  out=$D/outputs/umi_act/$name/$run
  if [[ $checkpoint != - ]]; then
    ./gpu.sh run "mkdir -p $D/outputs/umi_act/$name/server_logs"
    ./gpu.sh bg outputs/umi_act/$name/server_logs/$run.log "python -m igris.umi_walklow_replay.policy_server --run $out/run --checkpoint $checkpoint --wait-seconds 420"
  fi
  ./sim.sh $lane ${OWNER}_umi_${name}_$run --out $out --dataset $D/datasets/Box_palletizing_UMI_macos_intern_merged "$@" \
    > logs/$name/$run.log 2>&1
  code=$?
  echo "$run exit $code $(grep -o 'RUN_DONE.*' logs/$name/$run.log | cut -c1-200)" >> logs/$name/progress.txt
  grep -q 'WALK_LOW standing failed' logs/$name/$run.log || break
done
