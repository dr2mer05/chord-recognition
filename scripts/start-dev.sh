#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${CODESPACES:-}" == "true" && "${1:-}" != "--resumed" ]]; then
  exec bash scripts/resume-dev.sh "$@"
fi
if [[ "${1:-}" == "--resumed" ]]; then shift; fi
if [[ ! -x .venv/bin/python ]]; then
  printf '먼저 bash scripts/setup-dev.sh를 실행해 주세요.\n' >&2
  exit 1
fi
mkdir -p work
# 기존 Codespace에서 포트 공개 설정이 바뀌었어도 개인 검증 서버는 비공개로 시작한다.
if [[ "${CODESPACES:-}" == "true" ]]; then
  gh codespace ports visibility 8000:private -c "$CODESPACE_NAME"
  # 사용자가 승인한 개인 Codespaces 실행 경로. 포트 비공개 확인 후에만 활성화한다.
  export CHORDAPP_PRIVATE_MODE=1
  export CHORDAPP_ENABLE_YTDLP=1
  export CHORDAPP_ENABLE_POT="${CHORDAPP_ENABLE_POT:-1}"
fi
# 공급기 시작 실패가 검색·데모까지 막지 않도록 앱은 실행하고 진단에 상태를 표시한다.
if ! .venv/bin/python -m chordapp.service.pot start; then
  printf '토큰 공급기를 시작하지 못했습니다. 사이트의 실행 상태에서 확인할 수 있습니다.\n' >&2
fi
# --restart는 이 스크립트가 기록한 같은 작업 폴더의 서버만 종료한다.
if [[ "${1:-}" == "--restart" && -f work/dev-server.pid ]]; then
  .venv/bin/python - <<'PY'
import os
from pathlib import Path
import signal
import time
pid = int(Path('work/dev-server.pid').read_text().strip())
proc = Path('/proc') / str(pid)
if proc.exists():
    args = (proc / 'cmdline').read_bytes().split(b'\0')
    if (proc / 'cwd').resolve() != Path.cwd() or b'chordapp' not in args or b'serve' not in args:
        print('이전 PID가 다른 프로세스에 사용 중이어서 종료하지 않았습니다.')
    else:
        os.kill(pid, signal.SIGTERM)
        for _ in range(40):
            try:
                stopped = (proc / 'stat').read_text().split()[2] == 'Z'
            except FileNotFoundError:
                stopped = True
            if stopped:
                break
            time.sleep(.25)
        else:
            raise SystemExit('서버가 종료되지 않았습니다. 터미널에서 상태를 확인해 주세요.')
Path('work/dev-server.pid').unlink(missing_ok=True)
PY
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
