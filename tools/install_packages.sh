#!/usr/bin/env bash
# Python packages the pipeline needs beyond Isaac Sim, into vendor_py/ of this repository (not into any environment).
#   mink_cp312   IK packages for the simulator container (Python 3.12) and for the scene preparation in it
#   mink_host    the same without scipy, for the host Python (Isaac Sim's brings its own scipy)
#   train_host   what LeRobot's ACT, the cameras and the dataset writer import on the GPU side
# usage: tools/install_packages.sh [python]      (default: python3; it must be a Python 3.12 with pip)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${1:-python3}
IK="mink==1.3.0 mujoco==3.11.0 qpsolvers==4.13.0 daqp==0.11.2"
$PY -m pip install --quiet --target vendor_py/mink_cp312 $IK scipy==1.13.1
$PY -m pip install --quiet --target vendor_py/mink_host $IK
# numpy, scipy and torch come from the environment that runs the GPU side (they must match its CUDA and its
# compiled packages): whatever pip pulled in of them is taken out again.
$PY -m pip install --quiet --target vendor_py/train_host pyopengl draccus 'huggingface_hub>=1.6,<2' gymnasium termcolor \
    'datasets>=4,<5' 'av>=15,<16' jsonlines einops safetensors 'transformers>=5.4,<5.6'
for name in numpy numpy.libs scipy scipy.libs torch torchvision; do rm -rf vendor_py/train_host/$name vendor_py/train_host/$name-*.dist-info; done
echo "installed into $(pwd)/vendor_py"
