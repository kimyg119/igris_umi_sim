#!/usr/bin/env bash
# The GPU container on the server (one GPU): rendering, dataset building, training, policy serving.
#   gpu.sh up              start it (idle)
#   gpu.sh run <cmd>       run a shell command in it (bash -lc), with the repository on PYTHONPATH and EGL rendering
#   gpu.sh bg <log> <cmd>  the same, detached; output to <data>/<log>
#   gpu.sh down            remove it
set -euo pipefail
source "$(dirname "$0")/env.sh"
NAME=${OWNER}_umi_act_gpu
ENVS="export PYTHONPATH=$W:$W/third_party/lerobot/src:$D/vendor_py/train_cp312:$D/vendor_py/mink_cp312 MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 UMI_OUTPUTS=$D/outputs/umi_act HOME=$D/home HF_HOME=$D/home/hf TORCH_HOME=$D/home/torch; cd $W;"
case $1 in
  up) server docker run -d --name $NAME --label owner=$OWNER --gpus "\"device=$GPU\"" -e NVIDIA_DRIVER_CAPABILITIES=all \
        -u $RUN_AS --shm-size 32g --cpuset-cpus $GPU_CPUS -v $REMOTE/ws:$W:ro -v $REMOTE/DATA:$D \
        --entrypoint sleep $GPU_IMAGE infinity ;;
  run) shift; server docker exec $NAME bash -lc "$(printf '%q' "$ENVS $*")" ;;
  bg) log=$2; shift 2; server docker exec -d $NAME bash -lc "$(printf '%q' "$ENVS $* > $D/$log 2>&1")" ;;
  down) server docker rm -f $NAME ;;
esac
