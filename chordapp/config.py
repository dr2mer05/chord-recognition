"""로컬 설정 로딩.

윈도우에서 매번 환경변수를 잡는 게 번거로워서, 프로젝트 루트의 `.env`를
읽어 환경변수로 올린다. 이미 설정된 환경변수는 덮어쓰지 않는다
(운영에서는 진짜 환경변수가 항상 이긴다).

`.env`는 .gitignore에 들어 있다. **API 키를 코드나 커밋에 넣지 말 것.**
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

ENV_FILENAME = ".env"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_loaded = False


def parse_env(text: str) -> Dict[str, str]:
    """KEY=VALUE 형식만 읽는다. 주석(#)과 빈 줄은 건너뛴다."""
    values: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def load_env(path: Optional[Path] = None, override: bool = False) -> Dict[str, str]:
    """`.env`를 읽어 os.environ에 채운다. 채운 키 목록을 반환."""
    global _loaded
    target = Path(path) if path else PROJECT_ROOT / ENV_FILENAME
    if not target.is_file():
        _loaded = True
        return {}

    applied: Dict[str, str] = {}
    for key, value in parse_env(target.read_text(encoding="utf-8")).items():
        if override or not os.environ.get(key):
            os.environ[key] = value
            applied[key] = value
    _loaded = True
    return applied


def ensure_loaded() -> None:
    """한 번만 로드. import 시점에 호출된다."""
    if not _loaded:
        load_env()


def status() -> Dict[str, object]:
    """설정이 어떻게 잡혀 있는지. 값은 절대 노출하지 않는다."""
    from .service.youtube import API_KEY_ENV, ENABLE_AUDIO_ENV

    key = os.environ.get(API_KEY_ENV, "")
    return {
        "env_file": str(PROJECT_ROOT / ENV_FILENAME),
        "env_file_exists": (PROJECT_ROOT / ENV_FILENAME).is_file(),
        "youtube_api_key_set": bool(key),
        "youtube_api_key_hint": f"…{key[-4:]}" if len(key) >= 4 else "",
        "ytdlp_enabled": os.environ.get(ENABLE_AUDIO_ENV, "") == "1",
    }
