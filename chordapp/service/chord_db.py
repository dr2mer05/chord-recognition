"""공개 코드 DB 조회 레이어 (파이프라인 1-2의 [B] 분기).

사람이 만든 데이터가 오디오 분석보다 정확하고 연산도 안 드는 게 핵심이라
분석보다 먼저 조회한다. M1에서는 인터페이스와 로컬 JSON 백엔드만 두고,
Chordonomicon / HLSD 적재는 M2에서 이 인터페이스 뒤에 붙인다.

주의: 각 데이터셋의 상업적 이용 라이선스는 적재 전에 개별 확인이 필요하다
(파이프라인 1-8 체크리스트). 이 모듈은 라이선스 판단을 대신하지 않는다.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from ..models import AnalysisResult, ChordSegment
from ..theory.chord import Chord
from ..theory.key import Key


def normalize_query(artist: str, title: str) -> str:
    """'아티스트 - 제목'을 조회 키로 정규화. 괄호 주석과 공백/기호 차이를 흡수."""
    def clean(text: str) -> str:
        text = unicodedata.normalize("NFKC", text).lower()
        text = re.sub(r"\(.*?\)|\[.*?\]", " ", text)
        text = re.sub(r"[^0-9a-z가-힣]+", " ", text)
        return " ".join(text.split())

    return f"{clean(artist)}|{clean(title)}"


@dataclass
class DatabaseEntry:
    artist: str
    title: str
    key: str
    bpm: float
    beats_per_bar: int
    # [코드, 지속 박 수] 목록
    chords: List[List[object]] = field(default_factory=list)
    source: str = "local"

    def to_result(self) -> AnalysisResult:
        beat_dur = 60.0 / self.bpm
        segments: List[ChordSegment] = []
        t = 0.0
        beat_index = 0
        for label, beats in self.chords:
            dur = float(beats) * beat_dur
            segments.append(
                ChordSegment(
                    start=t, end=t + dur, chord=Chord.parse(str(label)),
                    confidence=1.0, agreement=1.0,
                    bar=beat_index // self.beats_per_bar + 1,
                    beat=beat_index % self.beats_per_bar + 1,
                )
            )
            t += dur
            beat_index += int(beats)
        return AnalysisResult(
            source=f"{self.artist} - {self.title}",
            duration=t,
            tempo=self.bpm,
            key=Key.parse(self.key),
            key_confidence=1.0,
            segments=segments,
            beat_times=[i * beat_dur for i in range(beat_index + 1)],
            downbeats=[i * beat_dur for i in range(0, beat_index + 1, self.beats_per_bar)],
            beats_per_bar=self.beats_per_bar,
            recognizers=[f"database:{self.source}"],
            origin="database",
            warnings=[],
        )


class ChordDatabase(Protocol):
    name: str

    def lookup(self, artist: str, title: str) -> Optional[AnalysisResult]: ...


class LocalJsonDatabase:
    """JSON 한 파일짜리 조회 백엔드. 형식은 Chordonomicon 적재 시에도 그대로 쓴다."""

    name = "local-json"

    def __init__(self, path: Path):
        self.path = Path(path)
        self._entries: Dict[str, DatabaseEntry] = {}
        if self.path.exists():
            self.reload()

    def reload(self) -> int:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._entries = {}
        for item in raw:
            entry = DatabaseEntry(**item)
            self._entries[normalize_query(entry.artist, entry.title)] = entry
        return len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def lookup(self, artist: str, title: str) -> Optional[AnalysisResult]:
        entry = self._entries.get(normalize_query(artist, title))
        return entry.to_result() if entry else None


class NullDatabase:
    """DB 미적재 상태. 항상 MISS를 내서 오디오 분석으로 넘긴다."""

    name = "none"

    def lookup(self, artist: str, title: str) -> Optional[AnalysisResult]:
        return None
