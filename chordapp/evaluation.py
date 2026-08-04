"""정확도 평가 (MIREX 방식 WCSR).

.lab 정답 파일 하나로 합성 데모와 McGill Billboard / Isophonics를 같은 코드로 잰다.
M2의 "Chordify 대비 정확도 측정" 항목이 여기에 붙는다.

비교 수준
  root    : 근음만 맞으면 정답
  triad   : 근음 + 3화음 품질 (MIREX major/minor 기본 지표)
  seventh : 7음까지 (확장 코드 커버리지 확인용)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .models import AnalysisResult, ChordSegment
from .theory.chord import Chord, simplify_to_triad

Interval = Tuple[float, float, Optional[Chord]]


def load_lab(path: Path) -> List[Interval]:
    intervals: List[Interval] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        start, end, label = float(parts[0]), float(parts[1]), parts[2]
        try:
            chord: Optional[Chord] = Chord.parse(label)
            if chord is not None and chord.is_no_chord:
                chord = None
        except ValueError:
            chord = None
        intervals.append((start, end, chord))
    return intervals


def segments_to_intervals(segments: Sequence[ChordSegment]) -> List[Interval]:
    return [(s.start, s.end, None if s.chord.is_no_chord else s.chord) for s in segments]


def _chord_at(intervals: Sequence[Interval], t: float) -> Optional[Chord]:
    for start, end, chord in intervals:
        if start <= t < end:
            return chord
    return None


def _match(reference: Optional[Chord], estimate: Optional[Chord], level: str) -> bool:
    if reference is None:
        return estimate is None
    if estimate is None:
        return False
    if level == "root":
        return reference.root == estimate.root
    if level == "triad":
        return simplify_to_triad(reference) == simplify_to_triad(estimate)
    if level == "seventh":
        return reference.root == estimate.root and reference.quality == estimate.quality
    raise ValueError(f"알 수 없는 비교 수준: {level}")


@dataclass
class EvalReport:
    name: str
    duration: float
    scores: Dict[str, float]
    confusions: List[Tuple[str, str, float]]  # (정답, 예측, 초)

    def format(self) -> str:
        head = f"{self.name}  ({self.duration:.1f}s)"
        body = "  ".join(f"{k}={v * 100:5.1f}%" for k, v in self.scores.items())
        lines = [f"{head}\n  {body}"]
        if self.confusions:
            lines.append("  주요 오인식:")
            for ref, est, sec in self.confusions[:5]:
                lines.append(f"    {ref:>8s} → {est:<8s} {sec:5.1f}s")
        return "\n".join(lines)


def evaluate(
    reference: Sequence[Interval],
    estimate: Sequence[Interval],
    name: str = "",
    hop: float = 0.02,
    levels: Sequence[str] = ("root", "triad", "seventh"),
) -> EvalReport:
    """프레임 샘플링 기반 WCSR. hop 간격으로 찍어 일치 비율을 잰다."""
    if not reference:
        raise ValueError("정답 구간이 비어 있다")
    duration = max(e for _, e, _ in reference)

    hits = {level: 0.0 for level in levels}
    total = 0
    confusion: Dict[Tuple[str, str], float] = {}

    t = 0.0
    while t < duration:
        ref = _chord_at(reference, t)
        est = _chord_at(estimate, t)
        total += 1
        for level in levels:
            if _match(ref, est, level):
                hits[level] += 1
        if not _match(ref, est, "triad"):
            key = (ref.name() if ref else "N", est.name() if est else "N")
            confusion[key] = confusion.get(key, 0.0) + hop
        t += hop

    scores = {level: hits[level] / total for level in levels} if total else {}
    ranked = sorted(confusion.items(), key=lambda kv: -kv[1])
    return EvalReport(
        name=name,
        duration=duration,
        scores=scores,
        confusions=[(r, e, s) for (r, e), s in ranked],
    )


def evaluate_result(result: AnalysisResult, lab_path: Path, name: str = "") -> EvalReport:
    return evaluate(
        load_lab(lab_path),
        segments_to_intervals(result.segments),
        name=name or Path(lab_path).stem,
    )


def aggregate(reports: Sequence[EvalReport]) -> Dict[str, float]:
    """길이 가중 평균."""
    if not reports:
        return {}
    total = sum(r.duration for r in reports) or 1.0
    keys = reports[0].scores.keys()
    return {k: sum(r.scores.get(k, 0.0) * r.duration for r in reports) / total for k in keys}
