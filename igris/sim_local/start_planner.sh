#!/usr/bin/env bash
# [호스트] S0 플래너 브릿지. 컨테이너 쪽은 ./start_sim.sh, ./start_policy.sh, ./start_gmr_bridge.sh, ./bringup.sh 순으로 떠 있어야 한다.
#   ./start_planner.sh --script tools/scripts/s0_walk.json
#   ./start_planner.sh --keyboard
cd "$(dirname "$0")"
MODEL=${MODEL:-$(find ../GR00T-WholeBodyControl -name planner_sonic.onnx -size +100M | head -1)}
PHI=${PHI:-../retargeting/phi/phi_igris.json}
[ -f "$MODEL" ] || { echo "planner_sonic.onnx 없음 — GR00T-WholeBodyControl/download_from_hf.py" >&2; exit 1; }
# onnxruntime-gpu 1.23(CUDA 12) 용 라이브러리: gmr env 는 CUDA 13 pip 패키지라 CUDA EP 가 조용히 CPU 로 떨어진다.
# env_isaaclab_2.3.2 의 CUDA-12 pip 라이브러리 + gmr 의 cudnn 9 를 빌려 쓴다(Task 5 에서 검증). 없으면 CPU 로 동작(57~77 ms).
NV=/home/rcv/miniconda3/envs/env_isaaclab_2.3.2/lib/python3.11/site-packages/nvidia
CUDNN=/home/rcv/miniconda3/envs/gmr/lib/python3.10/site-packages/nvidia/cudnn/lib
if [ -d "$NV/cublas/lib" ]; then
  export LD_LIBRARY_PATH="$CUDNN:$NV/cublas/lib:$NV/cuda_runtime/lib:$NV/cufft/lib:$NV/curand/lib:$NV/cusolver/lib:$NV/cusparse/lib:$NV/nvjitlink/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec conda run --no-capture-output -n gmr python tools/planner_bridge.py --model "$MODEL" --phi "$PHI" "$@"
