"""앙상블 투표 + 조성 문맥 제약 + 비트/마디 스냅.

파이프라인 1-3의 3~4번 항목. 규칙과 투표라서 서버 자원이 거의 안 든다.
정확도 대비 효율이 가장 높은 구간이므로 여기서 최대한 짜낸다.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..models import ChordSegment
from ..theory.chord import Chord
from ..theory.key import Key
from .audio import AudioFeatures
from .recognizers import BeatPrediction


def vote(
    predictions: Sequence[BeatPrediction],
    key: Key,
    key_bonus: float = 0.10,
) -> Tuple[List[Chord], np.ndarray, np.ndarray]:
    """비트별 가중 투표. (코드, 확신도, 일치도)를 반환."""
    usable = [p for p in predictions if p.chords]
    if not usable:
        return [], np.zeros(0), np.zeros(0)

    n_beats = min(len(p.chords) for p in usable)
    chords: List[Chord] = []
    confidence = np.zeros(n_beats, dtype=np.float32)
    agreement = np.zeros(n_beats, dtype=np.float32)

    for i in range(n_beats):
        tally: Dict[Tuple[int, str], float] = defaultdict(float)
        total = 0.0
        for pred in usable:
            chord = pred.chords[i]
            conf = float(pred.confidence[i]) if len(pred.confidence) > i else 0.5
            w = pred.weight * (0.4 + 0.6 * conf)
            tally[(chord.root, chord.quality)] += w
            total += w

        # 조성 문맥 제약: 조성 안에 완전히 들어오는 후보에 가산점
        for candidate_key in list(tally):
            chord = Chord(candidate_key[0], candidate_key[1])
            in_key = sum(1 for pc in chord.pitch_classes() if key.contains_pc(pc))
            ratio = in_key / max(len(chord.pitch_classes()), 1)
            tally[candidate_key] += key_bonus * ratio * total

        best = max(tally.items(), key=lambda kv: kv[1])
        chords.append(Chord(best[0][0], best[0][1]))
        denom = sum(tally.values()) or 1.0
        agreement[i] = best[1] / denom
        confidence[i] = float(
            np.mean([
                float(p.confidence[i]) if len(p.confidence) > i else 0.5
                for p in usable
                if (p.chords[i].root, p.chords[i].quality) == best[0]
            ] or [0.3])
        )

    return chords, confidence, agreement


def apply_bass_to_segments(
    segments: List[ChordSegment],
    bass_chroma: np.ndarray,
    beat_times: np.ndarray,
    margin: float = 1.5,
) -> List[ChordSegment]:
    """저역 크로마로 전위/슬래시 코드를 판정. **구간 단위로** 한다.

    Demucs 베이스 스템 없이 저역 CQT로 근사한다.

    비트 단위로 판정하면 안 된다. 베이스는 한 코드 안에서도 근음↔5도를
    오가기 때문에, 매 박 판정하면 `Dbmaj7 / Dbmaj7/C / Dbmaj7/F / Dbmaj7/C`
    처럼 같은 코드가 잘게 쪼개져 차트가 읽을 수 없게 된다(실제로 그랬다).
    전위는 보통 코드 하나를 통째로 가므로, 구간 전체의 중앙값으로 한 번만 정한다.
    """
    if bass_chroma.ndim != 2 or bass_chroma.shape[1] == 0 or not segments:
        return segments

    n_beats = bass_chroma.shape[1]
    for seg in segments:
        if seg.chord.is_no_chord:
            continue
        start = int(np.searchsorted(beat_times, seg.start, side="right") - 1)
        end = int(np.searchsorted(beat_times, seg.end, side="left"))
        start = max(0, min(start, n_beats - 1))
        end = max(start + 1, min(end, n_beats))

        column = np.median(bass_chroma[:, start:end], axis=1)
        tones = seg.chord.pitch_classes()
        strongest = max(tones, key=lambda pc: column[pc])
        root_energy = column[seg.chord.root] + 1e-6
        if strongest != seg.chord.root and column[strongest] > root_energy * margin:
            seg.chord = seg.chord.with_bass(strongest)
    return segments


def downbeat_phase_from_chords(
    chords: Sequence[Chord],
    beats_per_bar: int,
    current_phase: int,
    margin: float = 1.35,
) -> int:
    """코드 변화 위치로 마디 첫 박을 정한다.

    이게 다운비트의 가장 강력한 단서다. 코드는 마디 첫 박에서 압도적으로 자주
    바뀌기 때문이다. 온셋 세기(킥/스네어)는 편곡에 따라 흔들리지만 화성 리듬은
    곡 구조 그 자체다.

    실측 근거: 윤하 <기다리다> 전주의 정답 코드 전환 위치를 4로 나눈 나머지는
    0번 위상 5회, 2번 3회, 3번 1회였다. beat_this가 고른 위상은 0회였다
    (한 박 늦게 잡았고, 그래서 마디가 통째로 어긋났다).

    확실하지 않으면 원래 위상을 유지한다 — 조용한 구간이나 화성 리듬이 느린
    곡에서는 표본이 부족하다.
    """
    bpb = max(int(beats_per_bar), 1)
    changes = [i for i in range(1, len(chords)) if chords[i] != chords[i - 1]]
    if len(changes) < 4 or bpb < 2:
        return current_phase

    counts = np.bincount([c % bpb for c in changes], minlength=bpb).astype(float)
    best = int(counts.argmax())
    rest = (counts.sum() - counts[best]) / (bpb - 1)
    if rest <= 0 or counts[best] >= rest * margin:
        return best
    return current_phase


def smooth_short_runs(chords: List[Chord], min_beats: int = 2) -> List[Chord]:
    """1비트짜리 튐 제거. 앞뒤가 같으면 그 코드로 흡수한다."""
    if len(chords) < 3:
        return chords
    out = list(chords)
    i = 0
    while i < len(out):
        j = i
        while j + 1 < len(out) and out[j + 1] == out[i]:
            j += 1
        run = j - i + 1
        if run < min_beats and i > 0:
            prev_chord = out[i - 1]
            next_chord = out[j + 1] if j + 1 < len(out) else prev_chord
            out[i : j + 1] = [prev_chord if prev_chord == next_chord else next_chord] * run
        i = j + 1
    return out


def segment_beats(
    chords: Sequence[Chord],
    confidence: np.ndarray,
    agreement: np.ndarray,
    beat_times: np.ndarray,
    duration: float,
    downbeat_indices: np.ndarray,
    beats_per_bar: int = 4,
) -> List[ChordSegment]:
    """같은 코드가 이어지는 비트를 하나의 구간으로 묶는다."""
    if not chords:
        return []
    downbeat_set = set(int(i) for i in np.atleast_1d(downbeat_indices))
    first_downbeat = min(downbeat_set) if downbeat_set else 0

    segments: List[ChordSegment] = []
    start_idx = 0
    for i in range(1, len(chords) + 1):
        if i < len(chords) and chords[i] == chords[start_idx]:
            continue
        # 첫 구간은 0까지 늘린다. 비트 격자가 0보다 뒤에서 시작해도
        # 도입부가 어느 구간에도 안 속하는 일이 없도록.
        start_t = 0.0 if start_idx == 0 else float(beat_times[start_idx])
        end_t = float(beat_times[i]) if i < len(beat_times) else duration
        sl = slice(start_idx, i)
        segments.append(
            ChordSegment(
                start=start_t,
                end=end_t,
                chord=chords[start_idx],
                confidence=float(np.mean(confidence[sl])) if confidence.size else 0.0,
                agreement=float(np.mean(agreement[sl])) if agreement.size else 1.0,
                bar=(start_idx - first_downbeat) // beats_per_bar + 1,
                beat=(start_idx - first_downbeat) % beats_per_bar + 1,
            )
        )
        start_idx = i
    return segments


def snap_to_bars(
    segments: List[ChordSegment],
    beat_times: np.ndarray,
    downbeat_indices: np.ndarray,
    tolerance_beats: float = 1.0,
) -> List[ChordSegment]:
    """구간 경계를 가까운 마디 첫 박으로 당긴다.

    실제 대중음악은 코드가 마디 경계에서 바뀌는 경우가 압도적이라,
    한 비트 어긋난 경계를 마디에 맞추면 오탐이 줄고 읽기도 편해진다.
    """
    if len(segments) < 2 or beat_times.size < 2 or np.size(downbeat_indices) == 0:
        return segments
    beat_period = float(np.median(np.diff(beat_times)))
    tolerance = beat_period * tolerance_beats
    downbeat_times = beat_times[np.clip(np.atleast_1d(downbeat_indices), 0, len(beat_times) - 1)]

    for i in range(1, len(segments)):
        boundary = segments[i].start
        nearest = downbeat_times[int(np.argmin(np.abs(downbeat_times - boundary)))]
        if abs(nearest - boundary) <= tolerance and segments[i - 1].start < nearest < segments[i].end:
            segments[i - 1].end = float(nearest)
            segments[i].start = float(nearest)
    return [s for s in segments if s.duration > 1e-3]


def merge_adjacent(segments: List[ChordSegment]) -> List[ChordSegment]:
    """스냅 이후 같은 코드가 붙어버린 구간을 합친다."""
    if not segments:
        return segments
    out = [segments[0]]
    for seg in segments[1:]:
        last = out[-1]
        if seg.chord == last.chord:
            total = last.duration + seg.duration or 1.0
            last.confidence = (last.confidence * last.duration + seg.confidence * seg.duration) / total
            last.agreement = (last.agreement * last.duration + seg.agreement * seg.duration) / total
            last.end = seg.end
        else:
            out.append(seg)
    return out
