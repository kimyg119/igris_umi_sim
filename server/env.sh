# Shared settings of the server scripts (sourced).
HERE_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE_ENV/env.local.sh" ] || { echo "copy server/env.example.sh to server/env.local.sh and fill it in" >&2; exit 1; }
source "$HERE_ENV/env.local.sh"
LOCAL_ROOT="$(cd "$HERE_ENV/.." && pwd)"          # this repository
W=/workspace/igris_umi_sim                        # where the repository is mounted in every container
D=/data                                           # where REMOTE/DATA is mounted in every container
server() { $SERVER_SSH $SERVER "$@" 2>&1 | grep -v -E '^// |^$' ; return ${PIPESTATUS[0]}; }
