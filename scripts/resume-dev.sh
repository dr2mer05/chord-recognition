#!/usr/bin/env bash
# Codespaces 시작마다 갱신하되 사용자 작업·설정·데이터는 덮어쓰지 않는다.
set -euo pipefail
# 함수 전체를 먼저 읽어 자기 파일이 갱신되어도 현재 실행이 유지되도록 한다.
main() {
cd "$(dirname "$0")/.."
mkdir -p work
if [[ "${CODESPACES:-}" == "true" ]]; then
  branch=$(git branch --show-current)
  if [[ -n "$branch" && -z "$(git status --porcelain --untracked-files=normal)" ]]; then
    if ! GIT_TERMINAL_PROMPT=0 timeout 30 git pull --ff-only origin "$branch" > work/auto-update.log 2>&1; then
      printf '자동 업데이트를 완료하지 못해 현재 버전으로 시작합니다.\n'
    fi
  else
    printf '작업 중인 변경이 있어 자동 업데이트를 건너뜁니다.\n'
  fi
fi
if ! .venv/bin/python - <<'PY'
import hashlib
from pathlib import Path
stamp = Path('work/dev-requirements.sha256')
digest = hashlib.sha256(b''.join(Path(p).read_bytes() for p in ('requirements-cpu.lock', 'requirements-private.txt'))).hexdigest()
raise SystemExit(0 if stamp.exists() and stamp.read_text().strip() == digest else 1)
PY
then
  bash scripts/setup-dev.sh
fi
exec bash scripts/start-dev.sh --resumed --restart
}
main "$@"
