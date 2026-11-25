# Settings of the server scripts.  Copy to env.local.sh (not in git) and fill in.
# Nothing but docker and rsync is run on the server's host; everything else runs in containers.
SERVER_SSH="ssh -i $HOME/.ssh/KEY -o IdentitiesOnly=yes -o LogLevel=ERROR -p PORT"
SERVER=USER@HOST
REMOTE=/data/USER/igris_umi_sim          # work folder on the server: ws/ (this repository), DATA/ (packages, datasets, outputs)
OWNER=USER                                 # prefix of container names and their owner label
RUN_AS=1000:1000                           # the account on the server that owns REMOTE
GPU=0                                      # the one GPU the GPU container may use
GPU_IMAGE=IMAGE                            # an image with Python 3.12, PyTorch and CUDA
GPU_CPUS=16-31                             # CPUs of the GPU container (data loading, rendering)
SIM_CPU_BASE=40                            # first CPU of the simulator lanes; lane s gets SIM_CPU_BASE+3s .. +3s+2
SIM_CPU_SIBLING=64                         # distance to the second thread of the same core (0: no hyper-threading)
