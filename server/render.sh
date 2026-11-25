#!/usr/bin/env bash
# Render the dataset of every finished run of a batch (GPU container, 6 processes; runs already done are skipped).
# usage: render.sh <batch name>
set -euo pipefail
cd "$(dirname "$0")"; source env.sh
./gpu.sh run "cd $D/outputs/umi_act/$1 && ls -d ep* | xargs -P 6 -n 2 python -m igris.umi_walklow_replay.ee_dataset render 2>&1 | grep -E '^\{' > $D/outputs/umi_act/$1/render_last.jsonl; grep -c frames $D/outputs/umi_act/$1/render_last.jsonl; grep -c error $D/outputs/umi_act/$1/render_last.jsonl"
