#!/usr/bin/env bash
# sim_local 스택을 다른 서버로 옮기기 위한 공유용 zip.
# 컨테이너가 /igris 로 마운트하는 트리를 그대로 담는다(경로가 컨테이너 안에서 동일해야 하므로 구조 유지).
#   ./pack_share.sh [출력경로]     기본: <repo>/igris-sim-local.zip
set -euo pipefail
IG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$IG/igris-sim-local.zip}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
S="$TMP/igris-sim-local"; mkdir -p "$S"

# 1) sim_local — 스크립트/파라미터/모델. 로그·영상·빌드 산출물은 뺀다(재생성 가능).
rsync -a \
  --exclude 'logs/' --exclude 'videos/' --exclude '__pycache__/' --exclude '.pytest_cache/' \
  --exclude 'tools_ws/build/' --exclude 'tools_ws/log/' \
  --exclude 'check_frame*.png' --exclude 'inference_log.csv' --exclude '*.mp4' \
  "$IG/sim_local/" "$S/sim_local/"

# 2) ws — IGRIS-C SDK 바이너리 워크스페이스. install 만 있으면 돌아간다(심볼릭 링크 없음 확인).
#    build/(357M), log/, launch_params_*/ 는 런타임에 안 쓴다.
mkdir -p "$S/ws"
rsync -a "$IG/ws/install" "$IG/ws/setup" "$IG/ws/manager" "$IG/ws/igris_c_generated_mujoco" \
        "$IG/ws/run_manager.sh" "$IG/ws/version.txt" "$S/ws/"

# 3) gmt_ws — GMT 정책 노드. 빌드된 install 을 그대로 실어 바로 실행되게 한다.
mkdir -p "$S/gmt_ws"
rsync -a "$IG/gmt_ws/install" "$S/gmt_ws/"
mkdir -p "$S/gmt_ws/src" && ln -sfn ../../igris_c_gmt_public "$S/gmt_ws/src/igris_c_gmt_public"

# 4) 컨테이너가 /robros/configs 로 마운트하는 설정 + URDF/메시.
rsync -a "$IG/configs" "$S/"
rsync -a --exclude '.git/' "$IG/igris_c_description_public" "$S/"
rsync -a --exclude '.git/' "$IG/igris_c_gmt_public" "$S/"

# 5) Redis→ROS2 브릿지 (start_stack.sh 가 띄운다). GMR 전체(1.9G assets)는 필요 없다.
mkdir -p "$S/retargeting/GMR/scripts"
rsync -a --exclude '__pycache__/' "$IG/retargeting/GMR/general_motion_retargeting" "$S/retargeting/GMR/"
cp "$IG/retargeting/GMR/scripts/redis_to_ros2_bridge.py" "$S/retargeting/GMR/scripts/"

cp "$IG/sim_local/SHARE_README.md" "$S/README.md"

rm -f "$OUT"
( cd "$TMP" && zip -qry "$OUT" igris-sim-local )
echo "생성: $OUT  ($(du -h "$OUT" | cut -f1), 원본 $(du -sh "$S" | cut -f1))"
