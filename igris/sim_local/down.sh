#!/usr/bin/env bash
# 컨테이너(root)가 /igris 마운트에 만든 파일의 소유권을 호스트 사용자로 회수
docker exec igris-sim bash -lc 'find /igris -user root -exec chown 1000:1000 {} + 2>/dev/null' 2>/dev/null || true
docker rm -f igris-sim >/dev/null 2>&1 && echo "igris-sim 내림" || echo "실행 중인 컨테이너 없음"
