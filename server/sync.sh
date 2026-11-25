#!/usr/bin/env bash
# Send this repository and the analysis scripts to the server (rsync only; nothing is executed on its host).
set -euo pipefail
source "$(dirname "$0")/env.sh"
rsync -a --mkpath --delete -e "$SERVER_SSH" --exclude __pycache__ --exclude '*.pyc' --exclude .git --exclude /vendor_py --exclude /outputs \
  --exclude /datasets --exclude /checkpoints --exclude /server/env.local.sh "$LOCAL_ROOT/" $SERVER:$REMOTE/ws/
rsync -a --mkpath -e "$SERVER_SSH" "$LOCAL_ROOT/tools/analysis/" $SERVER:$REMOTE/DATA/scripts/
echo SYNCED
