#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.lock
.venv/bin/python -m pip install -r requirements-private.txt
.venv/bin/python -m pip check
.venv/bin/python -m chordapp.service.pot install
mkdir -p work
.venv/bin/python - <<'PY'
import hashlib
from pathlib import Path
digest = hashlib.sha256(b''.join(Path(p).read_bytes() for p in ('requirements-cpu.lock', 'requirements-private.txt'))).hexdigest()
Path('work/dev-requirements.sha256').write_text(digest + '\n')
PY
printf '설치 완료. Codespaces를 시작하면 개인 검증 서버가 자동 실행됩니다.\n'
