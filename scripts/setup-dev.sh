#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.lock
.venv/bin/python -m pip check
mkdir -p work
printf '설치 완료. Codespaces를 시작하면 개인 검증 서버가 자동 실행됩니다.\n'
