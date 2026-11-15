"""권한을 확인한 별도 음원과 YouTube 영상의 연결. YouTube에서 음원을 받지 않는다."""

from __future__ import annotations

import hashlib
import math
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .youtube import YouTubeError, parse_video_id


class AudioSourceUnavailable(YouTubeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AudioLink:
    video_id: str
    asset_id: str
    revision: int
    path: Path
    label: str
    rights_basis: str
    expires_at: Optional[float]
    same_recording: bool
    video_offset: float
    active: bool

    @property
    def source_key(self) -> str:
        # 연결을 교체했다가 되돌려도 새 검수·교정을 명시적으로 시작한다.
        return f"linked:{self.video_id}:{self.asset_id}:{self.revision}"

    def to_dict(self) -> Dict[str, Any]:
        # 서버 경로와 권한 근거 원문은 목록/분석 응답에 노출하지 않는다.
        return {
            "video_id": self.video_id, "asset_id": self.asset_id,
            "revision": self.revision, "label": self.label,
            "expires_at": self.expires_at, "same_recording": self.same_recording,
            "video_offset": self.video_offset, "active": self.active,
        }


class AudioSourceStore:
    def __init__(self, db_path: Path = Path("data/audio_sources.sqlite3"),
                 audio_dir: Path = Path("data/authorized_audio")):
        self.db_path = Path(db_path)
        self.audio_dir = Path(audio_dir).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS audio_assets (
                    asset_id TEXT PRIMARY KEY, filename TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS video_audio_links (
                    video_id TEXT PRIMARY KEY, asset_id TEXT NOT NULL,
                    revision INTEGER NOT NULL, label TEXT NOT NULL,
                    rights_basis TEXT NOT NULL, expires_at REAL,
                    same_recording INTEGER NOT NULL, video_offset REAL NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1, updated_at REAL NOT NULL
                );
            """)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    @staticmethod
    def fingerprint(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def register(self, video: str, path: Path, *, label: str, rights_basis: str,
                 same_recording: bool, video_offset: float = 0.0,
                 expires_at: Optional[float] = None) -> AudioLink:
        video_id = parse_video_id(video)
        rights_basis = rights_basis.strip()
        if not rights_basis or len(rights_basis) > 2000:
            raise ValueError("음원 분석·보관 권한의 근거를 1~2000자로 입력해 주세요.")
        if not math.isfinite(video_offset) or abs(video_offset) > 900:
            raise ValueError("영상 시작 시차는 -900~900초여야 합니다.")
        if expires_at is not None and (not math.isfinite(expires_at) or expires_at <= time.time()):
            raise ValueError("사용 기한은 현재보다 이후여야 합니다.")
        path = Path(path)
        asset_id = self.fingerprint(path)
        filename = asset_id + path.suffix.lower()
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        destination = self.audio_dir / filename
        if not destination.exists():
            # 완전히 복사하기 전의 파일이 다른 요청에 보이지 않게 한다.
            with tempfile.NamedTemporaryFile(dir=self.audio_dir, delete=False) as stream:
                staged = Path(stream.name)
            try:
                shutil.copyfile(path, staged)
                staged.replace(destination)
            finally:
                staged.unlink(missing_ok=True)
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute("INSERT OR IGNORE INTO audio_assets VALUES (?, ?)", (asset_id, filename))
            row = con.execute("SELECT revision FROM video_audio_links WHERE video_id = ?",
                              (video_id,)).fetchone()
            revision = int(row["revision"]) + 1 if row else 1
            con.execute("""INSERT OR REPLACE INTO video_audio_links
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
                (video_id, asset_id, revision, label[:200], rights_basis, expires_at,
                 int(same_recording), video_offset, time.time()))
        return self.get(video_id)

    def get(self, video: str) -> Optional[AudioLink]:
        video_id = parse_video_id(video)
        with self._connect() as con:
            row = con.execute("""SELECT l.*, a.filename FROM video_audio_links l
                JOIN audio_assets a USING (asset_id) WHERE video_id = ?""", (video_id,)).fetchone()
        if row is None:
            return None
        return AudioLink(
            video_id, row["asset_id"], row["revision"], self.audio_dir / row["filename"],
            row["label"], row["rights_basis"], row["expires_at"], bool(row["same_recording"]),
            row["video_offset"], bool(row["active"]),
        )

    def resolve(self, video: str) -> AudioLink:
        link = self.get(video)
        if link is None:
            raise AudioSourceUnavailable("source_missing", "연결된 음원이 없습니다. 사용 권한이 있는 음원을 한 번 연결해 주세요.")
        if not link.active:
            raise AudioSourceUnavailable("rights_revoked", "이 음원의 사용 권한이 해제되었습니다.")
        if link.expires_at is not None and link.expires_at <= time.time():
            raise AudioSourceUnavailable("rights_expired", "음원 사용 기한이 만료되었습니다. 권한을 확인하고 다시 연결해 주세요.")
        if not link.same_recording:
            raise AudioSourceUnavailable("recording_unconfirmed", "영상과 동일한 녹음·편집본인지 확인이 필요합니다. 다른 버전은 로컬 파일로 분석할 수 있습니다.")
        if not link.path.is_file():
            raise AudioSourceUnavailable("audio_missing", "연결 정보는 있지만 저장된 음원을 찾을 수 없습니다. 다시 연결해 주세요.")
        if self.fingerprint(link.path) != link.asset_id:
            raise AudioSourceUnavailable("audio_changed", "저장된 음원이 변경되었습니다. 원본을 다시 연결해 주세요.")
        return link

    def status(self, video: str) -> Dict[str, Any]:
        video_id = parse_video_id(video)
        link = self.get(video_id)
        try:
            self.resolve(video_id)
            code, message = "ready", "연결된 음원으로 분석할 수 있습니다."
        except AudioSourceUnavailable as exc:
            code, message = exc.code, str(exc)
        return {"video_id": video_id, "status": code, "message": message,
                "link": link.to_dict() if link else None}

    def list_links(self) -> list:
        with self._connect() as con:
            ids = [r["video_id"] for r in con.execute(
                "SELECT video_id FROM video_audio_links ORDER BY updated_at DESC")]
        return [self.status(video_id) for video_id in ids]

    def revoke(self, video: str) -> None:
        with self._connect() as con:
            con.execute("UPDATE video_audio_links SET active = 0, updated_at = ? WHERE video_id = ?",
                        (time.time(), parse_video_id(video)))
