"""레이어 간에 오가는 공용 자료형."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .theory.chord import Chord
from .theory.key import Key


# 확신도 등급 경계.
# 합성 음원은 평균 0.83, 실제 상용 녹음은 평균 0.43이 나온다. 이진 표시로는
# 실제 곡이 통째로 "불확실"이 되어 정보가 되지 못하므로 3단계로 나눈다.
# 절대값이 낮은 건 사실이고, 그건 인식기가 1개뿐인 현재 구성의 한계다.
CONFIDENCE_HIGH = 0.60
CONFIDENCE_MID = 0.35


def confidence_grade(value: float) -> str:
    if value >= CONFIDENCE_HIGH:
        return "high"
    return "mid" if value >= CONFIDENCE_MID else "low"


@dataclass
class ChordSegment:
    start: float
    end: float
    chord: Chord
    confidence: float = 0.0     # 인식기 자체 확신도
    agreement: float = 1.0      # 앙상블 일치도 (1.0 = 전원 동의)
    bar: int = 0
    beat: int = 0

    @property
    def duration(self) -> float:
        return max(self.end - self.start, 0.0)

    @property
    def certainty(self) -> float:
        """확신도와 앙상블 일치도 중 낮은 쪽."""
        return min(float(self.confidence), float(self.agreement))

    @property
    def grade(self) -> str:
        return confidence_grade(self.certainty)

    def to_dict(self, prefer_flat: bool = False) -> Dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "chord": self.chord.name(prefer_flat),
            "confidence": round(float(self.confidence), 3),
            "agreement": round(float(self.agreement), 3),
            "grade": self.grade,
            "bar": self.bar,
            "beat": self.beat,
        }


@dataclass
class AnalysisResult:
    source: str                  # 사람이 읽는 이름
    duration: float
    tempo: float
    key: Key
    key_confidence: float
    segments: List[ChordSegment]
    beat_times: List[float] = field(default_factory=list)
    downbeats: List[float] = field(default_factory=list)
    beats_per_bar: int = 4
    recognizers: List[str] = field(default_factory=list)
    origin: str = "audio"        # audio | database | cache
    cached: bool = False
    warnings: List[str] = field(default_factory=list)
    # 교정을 붙이는 안정적인 키 ("youtube:<id>" / "file:<해시>").
    # 표시 이름(source)은 바뀔 수 있으므로 그걸 키로 쓰면 안 된다.
    source_key: str = ""
    audio_path: str = ""         # 벤치마크에서 재분석할 때 필요
    corrected: int = 0           # 사용자가 고친 구간 수

    @property
    def progression(self) -> List[Chord]:
        return [s.chord for s in self.segments if not s.chord.is_no_chord]

    def uncertain_segments(self, threshold: float = 0.6) -> List[int]:
        """LLM 후처리를 붙일 자리. 일치도/확신도가 낮은 구간 인덱스."""
        return [
            i for i, s in enumerate(self.segments)
            if min(s.agreement, s.confidence) < threshold and not s.chord.is_no_chord
        ]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        key = Key.parse(data["key"])
        segments = [
            ChordSegment(
                start=s["start"], end=s["end"], chord=Chord.parse(s["chord"]),
                confidence=s.get("confidence", 0.0), agreement=s.get("agreement", 1.0),
                bar=s.get("bar", 0), beat=s.get("beat", 0),
            )
            for s in data.get("segments", [])
        ]
        return cls(
            source=data["source"],
            duration=data["duration"],
            tempo=data["tempo"],
            key=key,
            key_confidence=data.get("key_confidence", 0.0),
            segments=segments,
            beat_times=data.get("beat_times", []),
            downbeats=data.get("downbeats", []),
            beats_per_bar=data.get("beats_per_bar", 4),
            recognizers=data.get("recognizers", []),
            origin=data.get("origin", "audio"),
            cached=data.get("cached", False),
            warnings=data.get("warnings", []),
            source_key=data.get("source_key", ""),
            audio_path=data.get("audio_path", ""),
            corrected=data.get("corrected", 0),
        )

    def bars(self) -> List[Dict[str, Any]]:
        """구간을 마디 × 박 격자로 펼친다.

        `| C Am |`처럼 코드만 나열하면 Am이 2박인지 3박인지 알 수 없다.
        실제 코드보처럼 박마다 칸을 주고, 앞 박에서 이어지는 칸은 코드 없이
        비워 둔다(화면에서는 `/`). 그러면 `| C / Am / |`로 화성 리듬이 그대로 보인다.
        """
        if not self.segments:
            return []

        boundaries = self._bar_boundaries()
        beats_per_bar = max(int(self.beats_per_bar), 1)
        flat = self.key.prefer_flat
        out: List[Dict[str, Any]] = []

        for index, (bar_start, bar_end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
            beat_len = (bar_end - bar_start) / beats_per_bar
            if beat_len <= 0:
                continue

            slots: List[Dict[str, Any]] = []
            previous: Optional[int] = None
            for beat in range(beats_per_bar):
                # 박의 중앙을 보는 게 경계 오차에 강하다
                moment = bar_start + (beat + 0.5) * beat_len
                seg_index = self._segment_at(moment)
                if seg_index is None:
                    slots.append({"beat": beat + 1, "chord": None, "index": None})
                    continue

                seg = self.segments[seg_index]
                is_new = seg_index != previous
                previous = seg_index
                slots.append({
                    "beat": beat + 1,
                    "index": seg_index,          # segments/analysis 배열과 연결하는 키
                    # chord가 None이면 앞 박에서 이어지는 칸 (화면에서 '/')
                    "chord": seg.chord.name(flat) if is_new else None,
                    "roman": self.key.roman(seg.chord) if is_new else None,
                    "grade": seg.grade,
                    # 이 마디보다 앞에서 시작한 코드인지
                    "held": is_new and seg.start < bar_start - 1e-6,
                })

            if any(slot["index"] is not None for slot in slots):
                out.append({
                    "bar": index,
                    "start": round(bar_start, 3),
                    "end": round(bar_end, 3),
                    "beats_per_bar": beats_per_bar,
                    "slots": slots,
                })
        return out

    def _segment_at(self, moment: float) -> Optional[int]:
        for i, seg in enumerate(self.segments):
            if seg.start - 1e-6 <= moment < seg.end + 1e-6:
                return i
        return None

    def _bar_boundaries(self) -> List[float]:
        """마디 경계 시각. 다운비트가 있으면 그걸 쓰고, 없으면 템포로 만든다."""
        end = max(self.duration, self.segments[-1].end if self.segments else 0.0)
        if len(self.downbeats) >= 2:
            bounds = list(self.downbeats)
            spacing = bounds[1] - bounds[0]
            # 첫 다운비트 앞이 한 마디의 1/4을 넘으면 진짜 못갖춘마디로 보고 한 칸을 준다.
            # 그보다 짧으면 그냥 비트 검출 오차이므로 첫 마디를 0까지 당긴다
            # (안 그러면 23ms짜리 가짜 마디가 생긴다).
            if bounds[0] > spacing * 0.25:
                bounds.insert(0, 0.0)
            else:
                bounds[0] = 0.0
            while bounds[-1] < end - spacing * 0.5:
                bounds.append(bounds[-1] + spacing)
            if bounds[-1] < end:
                bounds.append(end)
            return bounds

        bar_len = (60.0 / max(self.tempo, 1e-6)) * max(self.beats_per_bar, 1)
        count = max(int(np.ceil(end / bar_len)), 1) if bar_len > 0 else 1
        return [i * bar_len for i in range(count + 1)]

    def to_dict(self) -> Dict[str, Any]:
        flat = self.key.prefer_flat
        return {
            "source": self.source,
            "source_key": self.source_key,
            "audio_path": self.audio_path,
            "corrected": self.corrected,
            "duration": round(self.duration, 3),
            "tempo": round(self.tempo, 2),
            "key": self.key.name(),
            "key_confidence": round(self.key_confidence, 3),
            "beats_per_bar": self.beats_per_bar,
            "beat_times": [round(t, 3) for t in self.beat_times],
            "downbeats": [round(t, 3) for t in self.downbeats],
            "segments": [s.to_dict(flat) for s in self.segments],
            "bars": self.bars(),
            "recognizers": self.recognizers,
            "origin": self.origin,
            "cached": self.cached,
            "warnings": self.warnings,
        }
