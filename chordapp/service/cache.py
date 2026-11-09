"""분석 결과 캐시.

파이프라인 1-3의 2번 항목. "같은 입력은 전 세계 누가 요청해도 결과가 같다"는 성질이
비용 구조를 결정하는 핵심이라, 오디오 분석보다 먼저 만들어 둔다.

M1은 로컬 파일 업로드만 받으므로 키를 파일 내용 해시로 잡는다.
M3에서 유튜브를 붙이면 영상 ID를 키로 쓰는 항목이 추가될 뿐 구조는 그대로다.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_DB = Path("data/cache.sqlite3")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_cache (
    key         TEXT PRIMARY KEY,
    namespace   TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  REAL NOT NULL,
    hits        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_cache_namespace ON analysis_cache(namespace);
"""


def file_fingerprint(path: Path, chunk: int = 1 << 20) -> str:
    """파일 내용 해시. 파일명이 달라도 같은 음원이면 캐시가 맞는다."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()[:32]


class AnalysisCache:
    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(_SCHEMA)
            self._expire_metadata(con)

    @staticmethod
    def _expire_metadata(con: sqlite3.Connection) -> None:
        con.execute("DELETE FROM analysis_cache WHERE namespace = 'video_metadata' AND created_at < ?",
                    (time.time() - 24 * 60 * 60,))

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def make_key(self, namespace: str, identifier: str, **options: Any) -> str:
        """옵션이 다르면 다른 캐시. 어휘/전위 검출 설정이 바뀌면 결과가 달라지므로."""
        opts = json.dumps(options, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(f"{identifier}|{opts}".encode("utf-8")).hexdigest()[:32]
        return f"{namespace}:{digest}"

    def get(self, key: str, max_age: Optional[float] = None) -> Optional[Dict[str, Any]]:
        with self._connect() as con:
            self._expire_metadata(con)
            row = con.execute("SELECT payload, created_at FROM analysis_cache WHERE key = ?", (key,)).fetchone()
            if row is None:
                return None
            if max_age is not None and time.time() - row["created_at"] > max_age:
                con.execute("DELETE FROM analysis_cache WHERE key = ?", (key,))
                return None
            con.execute("UPDATE analysis_cache SET hits = hits + 1 WHERE key = ?", (key,))
        return json.loads(row["payload"])

    def put(self, key: str, namespace: str, payload: Dict[str, Any]) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO analysis_cache (key, namespace, payload, created_at, hits)"
                " VALUES (?, ?, ?, ?, COALESCE((SELECT hits FROM analysis_cache WHERE key = ?), 0))",
                (key, namespace, json.dumps(payload, ensure_ascii=False), time.time(), key),
            )

    def stats(self) -> Dict[str, Any]:
        with self._connect() as con:
            row = con.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(hits), 0) AS hits FROM analysis_cache"
            ).fetchone()
        entries, hits = int(row["n"]), int(row["hits"])
        return {
            "entries": entries,
            "hits": hits,
            "saved_ratio": round(hits / (hits + entries), 3) if entries else 0.0,
        }

    def clear(self, namespace: Optional[str] = None) -> int:
        with self._connect() as con:
            if namespace:
                cur = con.execute("DELETE FROM analysis_cache WHERE namespace = ?", (namespace,))
            else:
                cur = con.execute("DELETE FROM analysis_cache")
            return cur.rowcount
