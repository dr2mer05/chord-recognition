#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  printf '먼저 bash scripts/setup-dev.sh를 실행해 주세요.\n' >&2
  exit 1
fi
mkdir -p work
# 기존 Codespace에서 포트 공개 설정이 바뀌었어도 개인 검증 서버는 비공개로 시작한다.
if [[ "${CODESPACES:-}" == "true" ]]; then
  gh codespace ports visibility 8000:private -c "$CODESPACE_NAME"
fi
# PID 파일 대신 로컬 health 응답으로 이미 실행 중인 앱을 확인한다.
if .venv/bin/python - <<'PY'
import json
import urllib.request
try:
    # 로컬 접속은 환경의 외부 프록시를 사용하지 않는다.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://127.0.0.1:8000/api/capabilities', timeout=2) as response:
        data = json.load(response)
    raise SystemExit(0 if data.get('audio_source_registration') else 1)
except Exception:
    raise SystemExit(1)
PY
then
  printf 'Chord Analysis Lab이 이미 실행 중입니다.\n'
else
  nohup .venv/bin/python -m chordapp serve --host 0.0.0.0 --port 8000 > work/dev-server.log 2>&1 < /dev/null &
  server_pid=$!
  printf '%s\n' "$server_pid" > work/dev-server.pid
  .venv/bin/python - <<'PY'
import json
import time
import urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for attempt in range(40):
    try:
        with opener.open('http://127.0.0.1:8000/api/capabilities', timeout=1) as response:
            if json.load(response).get('audio_source_registration'):
                break
    except Exception:
        pass
    time.sleep(.25)
else:
    raise SystemExit('서버가 준비되지 않았습니다. work/dev-server.log를 확인해 주세요.')
PY
fi
if [[ "${CODESPACES:-}" == "true" ]]; then
  printf '개인 검증 주소: https://%s-8000.%s\n' "$CODESPACE_NAME" "${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}"
else
  printf '개인 검증 주소: http://127.0.0.1:8000\n'
fi
