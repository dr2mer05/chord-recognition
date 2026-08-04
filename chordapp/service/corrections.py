"""사용자 교정 저장소 (파이프라인 M3의 "사용자 피드백 루프").

제품 기능이자 개발 도구다. 사용자가 틀린 코드를 고치면
  - 다음에 같은 곡을 열 때 고친 내용이 남아 있고 (제품)
  - 그 곡이 실제 녹음 기준의 정답 데이터가 된다 (개발)

설계에서 중요한 두 가지.

1. **교정은 원본 인식 결과를 덮어쓰지 않는다.** 캐시에는 원본이 그대로 있고
   교정은 별도 레이어로 얹힌다. 덮어쓰면 "교정된 결과 vs 교정"을 비교하게 되어
   정확도 측정이 순환논법이 된다.

2. **곡 단위 '검수 완료' 표시가 따로 있다.** 틀린 코드만 골라 고친 표본은 편향돼
   있어서 정확도를 계산할 수 없다("고친 것 중 몇 %가 틀렸나"는 항상 100%다).
   곡 전체를 확인했다고 표시한 곡만 벤치마크에 쓴다.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..models import ChordSegment
from ..theory.chord import Chord

DEFAULT_DB = Path("data/corrections.sqlite3")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_key  TEXT PRIMARY KEY,
    label       TEXT NOT NULL DEFAULT '',
    audio_path  TEXT NOT NULL DEFAULT '',
    duration    REAL NOT NULL DEFAULT 0,
    verified    INTEGER NOT NULL DEFAULT 0,
    tempo           REAL,
    beats_per_bar   INTEGER,
    downbeat_offset INTEGER NOT NULL DEFAULT 0,
    beat_offset     REAL,
    updated_at      REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS corrections (
    source_key  TEXT NOT NULL,
    start       REAL NOT NULL,
    end         REAL NOT NULL,
    chord       TEXT NOT NULL,
    original    TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL,
    PRIMARY KEY (source_key, start)
);
CREATE INDEX IF NOT EXISTS idx_corr_source ON corrections(source_key);
"""


@dataclass
class Correction:
    start: float
    end: float
    chord: str
    original: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "chord": self.chord,
            "original": self.original,
        }


class CorrectionStore:
    def __init__(self, db_path: Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(_SCHEMA)
            # 이미 만들어진 DB에도 나중에 추가된 열을 붙인다
            existing = {row["name"] for row in con.execute("PRAGMA table_info(sources)")}
            for column, ddl in (
                ("tempo", "REAL"),
                ("beats_per_bar", "INTEGER"),
                ("downbeat_offset", "INTEGER NOT NULL DEFAULT 0"),
                ("beat_offset", "REAL"),
            ):
                if column not in existing:
                    con.execute(f"ALTER TABLE sources ADD COLUMN {column} {ddl}")

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    # -- 곡 -------------------------------------------------------------
    def record_source(
        self, source_key: str, label: str = "", audio_path: str = "", duration: float = 0.0
    ) -> None:
        """교정을 붙이려면 곡이 먼저 등록돼 있어야 한다. 벤치마크에 오디오 경로가 필요."""
        with self._connect() as con:
            con.execute(
                "INSERT INTO sources (source_key, label, audio_path, duration, updated_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(source_key) DO UPDATE SET"
                "   label = excluded.label,"
                "   audio_path = CASE WHEN excluded.audio_path != '' THEN excluded.audio_path"
                "                     ELSE sources.audio_path END,"
                "   duration = excluded.duration,"
                "   updated_at = excluded.updated_at",
                (source_key, label, audio_path, duration, time.time()),
            )

    def set_tempo(
        self, source_key: str, bpm: Optional[float], beat_offset: Optional[float] = None
    ) -> None:
        """곡별 템포 지정. 자동 추정이 배수를 틀릴 때의 확실한 탈출구.

        `beat_offset`(첫 박의 시각, 초)까지 주면 비트 추적을 아예 건너뛰고
        균등 격자를 만든다. 탭 템포로 사용자가 직접 찍은 경우가 이에 해당한다.
        오디오만으로는 4/4의 4분음표와 12/8의 점4분음표를 구별할 수 없어서
        (실측: 온셋 간격이 두 해석에 똑같이 들어맞았다) 사람의 입력이 필요하다.

        bpm이 None이면 지정을 해제하고 자동 추정으로 돌아간다.
        """
        if bpm is not None and not (20.0 <= float(bpm) <= 300.0):
            raise ValueError(f"템포가 범위를 벗어남: {bpm}")
        with self._connect() as con:
            con.execute(
                "UPDATE sources SET tempo = ?, beat_offset = ?, updated_at = ?"
                " WHERE source_key = ?",
                (
                    float(bpm) if bpm is not None else None,
                    float(beat_offset) if (bpm and beat_offset is not None) else None,
                    time.time(), source_key,
                ),
            )

    def beat_offset(self, source_key: str) -> Optional[float]:
        row = self.source(source_key)
        return row.get("beat_offset") if row else None

    def tempo(self, source_key: str) -> Optional[float]:
        row = self.source(source_key)
        return row.get("tempo") if row else None

    def set_meter(
        self,
        source_key: str,
        beats_per_bar: Optional[int] = None,
        downbeat_offset: Optional[int] = None,
    ) -> None:
        """곡별 박자표와 마디 시작 위치.

        박자표 자동 판별은 넣지 않았다. 저역 온셋의 위상 대비로 구별해 보려 했는데
        정답이 4/4인 합성곡에서도 6박이 더 높게 나와서 신뢰할 수 없었다.
        사람은 한 번 들으면 아는 정보라 지정받는 편이 확실하다.
        """
        if beats_per_bar is not None and not (2 <= int(beats_per_bar) <= 12):
            raise ValueError(f"박자가 범위를 벗어남: {beats_per_bar}")

        # 두 값을 항상 함께 쓴다. "안 넘겼으니 유지"와 "None이니 해제"를 구분하려다
        # 조건문이 꼬여서 해제가 안 됐다. 호출부가 늘 둘 다 넘기는 편이 명확하다.
        bpb = int(beats_per_bar) if beats_per_bar else None
        offset = int(downbeat_offset or 0) % (bpb or 4)
        with self._connect() as con:
            con.execute(
                "UPDATE sources SET beats_per_bar = ?, downbeat_offset = ?, updated_at = ?"
                " WHERE source_key = ?",
                (bpb, offset, time.time(), source_key),
            )

    def beats_per_bar(self, source_key: str) -> Optional[int]:
        row = self.source(source_key)
        return row.get("beats_per_bar") if row else None

    def downbeat_offset(self, source_key: str) -> int:
        row = self.source(source_key)
        return int(row.get("downbeat_offset") or 0) if row else 0

    def set_verified(self, source_key: str, verified: bool = True) -> None:
        with self._connect() as con:
            con.execute(
                "UPDATE sources SET verified = ?, updated_at = ? WHERE source_key = ?",
                (1 if verified else 0, time.time(), source_key),
            )

    def source(self, source_key: str) -> Optional[Dict[str, Any]]:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM sources WHERE source_key = ?", (source_key,)
            ).fetchone()
        return dict(row) if row else None

    def sources(self, verified_only: bool = False) -> List[Dict[str, Any]]:
        query = (
            "SELECT s.*, COUNT(c.source_key) AS corrections"
            " FROM sources s LEFT JOIN corrections c ON c.source_key = s.source_key"
            " {where} GROUP BY s.source_key ORDER BY s.updated_at DESC"
        ).format(where="WHERE s.verified = 1" if verified_only else "")
        with self._connect() as con:
            return [dict(row) for row in con.execute(query)]

    # -- 교정 -----------------------------------------------------------
    def add(
        self, source_key: str, start: float, end: float, chord: str, original: str = ""
    ) -> Correction:
        Chord.parse(chord)   # 저장 전에 파싱되는지 확인 (쓰레기 유입 차단)
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO corrections"
                " (source_key, start, end, chord, original, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (source_key, float(start), float(end), chord, original, time.time()),
            )
            con.execute(
                "UPDATE sources SET updated_at = ? WHERE source_key = ?",
                (time.time(), source_key),
            )
        return Correction(float(start), float(end), chord, original)

    def remove(self, source_key: str, start: float, tolerance: float = 1e-3) -> int:
        with self._connect() as con:
            cur = con.execute(
                "DELETE FROM corrections WHERE source_key = ? AND ABS(start - ?) <= ?",
                (source_key, float(start), tolerance),
            )
            return cur.rowcount

    def clear(self, source_key: str) -> int:
        with self._connect() as con:
            return con.execute(
                "DELETE FROM corrections WHERE source_key = ?", (source_key,)
            ).rowcount

    def corrections(self, source_key: str) -> List[Correction]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT start, end, chord, original FROM corrections"
                " WHERE source_key = ? ORDER BY start",
                (source_key,),
            ).fetchall()
        return [Correction(r["start"], r["end"], r["chord"], r["original"]) for r in rows]

    def stats(self) -> Dict[str, Any]:
        with self._connect() as con:
            row = con.execute(
                "SELECT (SELECT COUNT(*) FROM sources) AS songs,"
                "       (SELECT COUNT(*) FROM sources WHERE verified = 1) AS verified,"
                "       (SELECT COUNT(*) FROM corrections) AS corrections"
            ).fetchone()
        return dict(row)


# --------------------------------------------------------------------------
# 교정 적용 / 내보내기
# --------------------------------------------------------------------------

def apply_corrections(
    segments: Sequence[ChordSegment], corrections: Sequence[Correction]
) -> List[ChordSegment]:
    """원본 구간 위에 교정을 레이어로 얹는다. 원본은 건드리지 않는다.

    교정 구간이 원본 구간 경계와 안 맞아도 되도록, 겹치는 구간을 잘라서 덮는다.
    """
    if not corrections:
        return list(segments)

    result: List[ChordSegment] = []
    for seg in segments:
        pieces = [(seg.start, seg.end, seg.chord, False)]
        for fix in corrections:
            next_pieces = []
            for start, end, chord, corrected in pieces:
                if corrected or fix.end <= start + 1e-6 or fix.start >= end - 1e-6:
                    next_pieces.append((start, end, chord, corrected))
                    continue
                if start < fix.start - 1e-6:            # 교정 앞부분은 원본 유지
                    next_pieces.append((start, fix.start, chord, False))
                overlap_end = min(end, fix.end)
                next_pieces.append((max(start, fix.start), overlap_end,
                                    Chord.parse(fix.chord), True))
                if end > fix.end + 1e-6:                # 교정 뒷부분도 원본 유지
                    next_pieces.append((fix.end, end, chord, False))
            pieces = next_pieces

        for start, end, chord, corrected in pieces:
            if end - start <= 1e-6:
                continue
            result.append(ChordSegment(
                start=start, end=end, chord=chord,
                # 사람이 고친 구간은 확신도 만점 — 인식기보다 정확하다
                confidence=1.0 if corrected else seg.confidence,
                agreement=1.0 if corrected else seg.agreement,
                bar=seg.bar, beat=seg.beat,
            ))

    result.sort(key=lambda s: s.start)
    return _merge(result)


def _merge(segments: List[ChordSegment]) -> List[ChordSegment]:
    if not segments:
        return segments
    out = [segments[0]]
    for seg in segments[1:]:
        if seg.chord == out[-1].chord and abs(seg.start - out[-1].end) < 1e-6:
            out[-1].end = seg.end
        else:
            out.append(seg)
    return out


def to_lab(segments: Sequence[ChordSegment], prefer_flat: bool = False) -> str:
    """MIREX .lab 형식으로. 정답 파일과 같은 포맷이라 평가 코드가 그대로 읽는다."""
    lines = [
        f"{s.start:.6f}\t{s.end:.6f}\t{s.chord.name(prefer_flat)}"
        for s in segments
    ]
    return "\n".join(lines) + "\n"
