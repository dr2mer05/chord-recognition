"""프레임 샘플 기반 일치율과 코드 경계 평가.

.lab 정답 파일 하나로 합성 데모와 McGill Billboard / Isophonics를 같은 코드로 잰다.
M2의 "Chordify 대비 정확도 측정" 항목이 여기에 붙는다.

비교 수준
  root    : 근음만 맞으면 정답
  triad   : 근음 + 3화음 품질 (MIREX major/minor 기본 지표)
  seventh : 근음 + 내부 화음 종류 전체 (기존 이름 유지, 확장음 포함)

이름 seventh는 표준 MIREX seventh 축약 규칙을 구현했다는 뜻이 아니다.
현재 파서가 해석하지 못하는 정답은 거부하며 무화음으로 대체하지 않는다.
파서 자체의 추가음 축약은 유지하므로 모든 알터드 음을 구분하는 지표는 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .models import AnalysisResult, ChordSegment
from .theory.chord import Chord, simplify_to_triad

Interval = Tuple[float, float, Optional[Chord]]


def load_lab(path: Path) -> List[Interval]:
    intervals: List[Interval] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f'{Path(path).name}:{number}: 정답 구간 형식이 올바르지 않습니다.')
        start, end, label = float(parts[0]), float(parts[1]), parts[2]
        if label.upper() == 'X':
            raise ValueError(f'{Path(path).name}:{number}: 판정 불가 X를 무화음 N으로 채점할 수 없습니다.')
        try:
            chord: Optional[Chord] = Chord.parse(label)
            if chord is not None and chord.is_no_chord:
                chord = None
        except ValueError:
            raise ValueError(f'{Path(path).name}:{number}: 지원하지 않는 정답 코드 {label!r}') from None
        intervals.append((start, end, chord))
    _validate_intervals(intervals)
    return intervals


def segments_to_intervals(segments: Sequence[ChordSegment]) -> List[Interval]:
    return [(s.start, s.end, None if s.chord.is_no_chord else s.chord) for s in segments]


def _chord_at(intervals: Sequence[Interval], t: float) -> Optional[Chord]:
    for start, end, chord in intervals:
        if start <= t < end:
            return chord
    return None


def _validate_intervals(intervals):
    previous_end = 0.0
    for start, end, _ in intervals:
        if not np.isfinite([start, end]).all() or start < 0 or end <= start or start < previous_end - 1e-9:
            raise ValueError('정답·예측 구간은 유한하고 증가하며 서로 겹치지 않아야 합니다.')
        previous_end = end


def _covered(intervals, t):
    return any(start <= t < end for start, end, _ in intervals)


def chord_boundary_f_measure(reference, estimate, tolerance=.1):
    """주석 범위 안의 실제 코드 변화만 비교한다. 양 끝과 단순 구간 분할은 제외한다."""
    def changes(intervals):
        return [current[0] for previous, current in zip(intervals, intervals[1:])
                if abs(previous[1] - current[0]) < 1e-9 and previous[2] != current[2]
                and _covered(reference, current[0] - 1e-7) and _covered(reference, current[0] + 1e-7)]
    result = beat_f_measure(changes(reference), changes(estimate), tolerance)
    result['tolerance_seconds'] = tolerance
    return result


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
    by_quality: Dict[str, dict] = field(default_factory=dict)
    boundaries: Dict[str, float] = field(default_factory=dict)
    exact_chord: float = 0.0  # 근음·화음 종류·베이스를 모두 비교

    def format(self) -> str:
        head = f"{self.name}  ({self.duration:.1f}s)"
        body = "  ".join(f"{k}={v * 100:5.1f}%" for k, v in self.scores.items())
        lines = [f"{head}\n  {body}"]
        if self.boundaries:
            lines.append(f"  전위 포함 {self.exact_chord * 100:.1f}% · 코드 경계 F {self.boundaries['f'] * 100:.1f}% (±{self.boundaries['tolerance_seconds'] * 1000:.0f}ms)")
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
    if not np.isfinite(hop) or hop <= 0:
        raise ValueError('평가 간격은 유한한 양수여야 합니다.')
    _validate_intervals(reference)
    _validate_intervals(estimate)
    end_time = max(e for _, e, _ in reference)
    duration = sum(e - s for s, e, _ in reference)

    hits = {level: 0.0 for level in levels}
    total = 0
    confusion: Dict[Tuple[str, str], float] = {}
    qualities = {}
    exact_hits = 0

    t = 0.0
    while t < end_time:
        if not _covered(reference, t):
            t += hop
            continue  # 주석이 없는 구간을 무화음 정답으로 간주하지 않는다.
        ref = _chord_at(reference, t)
        est = _chord_at(estimate, t)
        predicted = _covered(estimate, t)
        total += 1
        for level in levels:
            if predicted and _match(ref, est, level):
                hits[level] += 1
        full_quality = predicted and _match(ref, est, 'seventh')
        exact_hits += bool(full_quality and (ref is None or (ref.bass if ref.bass is not None else ref.root)
                            == (est.bass if est.bass is not None else est.root)))
        quality = ref.quality if ref else 'N'
        row = qualities.setdefault(quality, {'frames': 0, 'correct': 0})
        row['frames'] += 1
        row['correct'] += int(full_quality)
        if not predicted or not _match(ref, est, "triad"):
            key = (ref.name() if ref else "N", est.name() if est else "N" if predicted else "X")
            confusion[key] = confusion.get(key, 0.0) + hop
        t += hop

    scores = {level: hits[level] / total for level in levels} if total else {}
    ranked = sorted(confusion.items(), key=lambda kv: -kv[1])
    return EvalReport(
        name=name,
        duration=duration,
        scores=scores,
        confusions=[(r, e, s) for (r, e), s in ranked],
        by_quality={quality: {'sampled_seconds': row['frames'] * hop,
                             'accuracy': row['correct'] / row['frames']} for quality, row in qualities.items()},
        boundaries=chord_boundary_f_measure(reference, estimate),
        exact_chord=exact_hits / total if total else 0.0,
    )


def evaluate_result(result: AnalysisResult, lab_path: Path, name: str = "") -> EvalReport:
    return evaluate(
        load_lab(lab_path),
        segments_to_intervals(result.segments),
        name=name or Path(lab_path).stem,
    )


# --------------------------------------------------------------------------
# 비트 추적 평가
# --------------------------------------------------------------------------

BEAT_TOLERANCE = 0.07     # MIREX 표준: ±70ms 안이면 맞은 것으로 본다


def beat_f_measure(
    reference: Sequence[float],
    estimate: Sequence[float],
    tolerance: float = BEAT_TOLERANCE,
) -> Dict[str, float]:
    """비트 위치의 F-measure (MIREX 방식).

    정답 하나에 추정 하나만 짝지어야 한다. 안 그러면 비트를 잔뜩 뿌리기만 해도
    재현율이 올라가서, 가짜 비트가 많은 결과가 좋아 보인다.
    """
    ref = np.sort(np.asarray(reference, dtype=float))
    est = np.sort(np.asarray(estimate, dtype=float))
    if ref.size == 0 or est.size == 0:
        return {"f": 0.0, "precision": 0.0, "recall": 0.0, "n_ref": ref.size, "n_est": est.size}

    used = np.zeros(est.size, dtype=bool)
    hits = 0
    for t in ref:
        window = np.where((np.abs(est - t) <= tolerance) & ~used)[0]
        if window.size:
            best = window[np.argmin(np.abs(est[window] - t))]
            used[best] = True
            hits += 1

    precision = hits / est.size
    recall = hits / ref.size
    f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f": f, "precision": precision, "recall": recall,
            "n_ref": int(ref.size), "n_est": int(est.size)}


def tempo_error(reference_bpm: float, estimate_bpm: float) -> Dict[str, float]:
    """템포 오차와 배수 관계. 옥타브/겹박자 오류를 따로 본다."""
    if reference_bpm <= 0 or estimate_bpm <= 0:
        return {"ratio": 0.0, "percent": 100.0, "octave": False}
    ratio = estimate_bpm / reference_bpm
    percent = abs(ratio - 1.0) * 100
    octave = any(abs(ratio - r) < 0.06 for r in (0.5, 2.0, 1.5, 2 / 3, 3.0, 1 / 3))
    return {"ratio": ratio, "percent": percent, "octave": octave}


def bar_interval_f_measure(reference: Sequence[float], estimate: Sequence[float],
                           tolerance: float = BEAT_TOLERANCE) -> Dict[str, float]:
    """연속한 두 마디 경계가 모두 맞아야 완전한 마디 하나를 맞힌 것으로 센다.

    다운비트를 절반·두 배로 검출해도 경계 일부는 맞을 수 있으므로, 경계 F와
    구간 F를 구분한다. 양 끝 경계가 없는 마지막 불완전 마디는 포함하지 않는다.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("허용 오차는 유한한 0 이상의 값이어야 합니다.")
    values = []
    for boundaries in (reference, estimate):
        times = np.asarray(boundaries, dtype=float)
        if times.ndim != 1 or not np.isfinite(times).all():
            raise ValueError("마디 경계는 유한한 시각의 1차원 목록이어야 합니다.")
        times = np.unique(times)  # 중복 경계로 길이 0인 마디를 만들지 않는다.
        values.append(list(zip(times[:-1], times[1:])))
    ref, est = values
    i = j = hits = 0
    while i < len(ref) and j < len(est):
        start, end = ref[i]
        other_start, other_end = est[j]
        if abs(start - other_start) <= tolerance and abs(end - other_end) <= tolerance:
            hits += 1
            i += 1
            j += 1
        elif other_start < start - tolerance or other_end < end - tolerance:
            j += 1
        else:
            i += 1
    precision = hits / len(est) if est else 0.0
    recall = hits / len(ref) if ref else 0.0
    return {"f": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "precision": precision, "recall": recall, "n_ref": len(ref), "n_est": len(est)}


def aggregate(reports: Sequence[EvalReport]) -> Dict[str, float]:
    """길이 가중 평균."""
    if not reports:
        return {}
    total = sum(r.duration for r in reports) or 1.0
    keys = reports[0].scores.keys()
    return {k: sum(r.scores.get(k, 0.0) * r.duration for r in reports) / total for k in keys}
