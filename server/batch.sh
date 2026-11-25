#!/usr/bin/env bash
# A batch of simulator runs on the server, <lanes> at a time (each lane: one private container, three cores).
# usage: batch.sh <name> <lanes> <checkpoint or -> <jobs file>      jobs file: one line per run, "<tag> <container_run args...>"
#   checkpoint -: no policy (label-stream replays);  otherwise a policy process per run in the GPU container.
set -uo pipefail
cd "$(dirname "$0")"; source env.sh
name=$1; lanes=$2; checkpoint=$3; jobs=$4
mkdir -p logs/$name
mapfile -t lines < "$jobs"
for lane in $(seq 0 $((lanes - 1))); do
  (
    for index in $(seq $lane $lanes $((${#lines[@]} - 1))); do
      read -r tag args <<< "${lines[$index]}"
      ./lane.sh $((lane + ${LANE_BASE:-0})) $name $tag $checkpoint $args
    done
  ) &
done
wait
echo BATCH_DONE >> logs/$name/progress.txt
