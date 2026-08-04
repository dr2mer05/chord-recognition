"""오디오 → 코드 진행. 파이프라인 1-2의 [C] 분기."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..models import AnalysisResult
from .separation import DemucsSeparator, SeparationError, default_separator
from ..theory.key import Key, rank_keys

# 코드 기반 조성이 오디오 기반 조성을 덮어쓰려면 이만큼 앞서야 한다.
KEY_REFINE_MARGIN = 5.0

# 나란한조(C↔Am)는 구성음도 다이어토닉 코드도 완전히 같아서, 코드 내용만으로는
# 사실상 구분이 안 된다. 실제로 합성 데모에서 C장조 곡이 마진 13.9로 Am에게
# 뒤집혔다. 이 쌍만 문턱을 크게 높여 오디오 쪽 판단을 존중한다.
# (제대로 된 해결은 M2에서 구간 길이 가중 + 프레이즈 시작/끝 분석.)
KEY_REFINE_RELATIVE_MARGIN = 25.0
from .audio import extract_features
from .ensemble import (
    apply_bass_to_segments,
    merge_adjacent,
    segment_beats,
    smooth_short_runs,
    snap_to_bars,
    vote,
)
from .recognizers import Recognizer, default_recognizers


def analyze_audio(
    path: str,
    vocabulary: str = "standard",
    recognizers: Optional[Sequence[Recognizer]] = None,
    beats_per_bar: int = 4,
    max_seconds: Optional[float] = None,
    key_override: Optional[Key] = None,
    detect_inversions: bool = True,
    snap_bars: bool = True,
    separate: Optional[bool] = None,
    separator: Optional["DemucsSeparator"] = None,
) -> AnalysisResult:
    warnings: List[str] = []

    # 소스 분리를 먼저. 성공하면 harmonic/bass 스템으로 크로마를 만든다.
    stems = None
    engine = separator or default_separator()
    want_separation = engine.enabled if separate is None else separate
    if want_separation:
        try:
            stems = engine.separate(Path(path))
        except SeparationError as exc:
            warnings.append(f"소스 분리 실패, 원본 믹스로 진행: {exc}")

    feats = extract_features(
        path, max_seconds=max_seconds, beats_per_bar=beats_per_bar,
        harmonic_paths=[str(p) for p in stems.accompaniment] if stems else None,
        bass_path=str(stems.bass) if stems and stems.bass else None,
    )
    key = key_override or feats.key
    if stems is not None:
        warnings.append(
            f"소스 분리 적용 ({stems.model}, {stems.device}"
            f"{', 캐시' if stems.cached else ''}) — harmonic 스템으로 코드 인식, "
            f"bass 스템으로 전위 판별"
        )
    if key_override is None and feats.key_confidence < 0.45:
        warnings.append(
            f"조성 추정 확신도가 낮음({feats.key_confidence:.2f}). "
            f"{key.name()}로 진행하지만 사용자가 조성을 지정하면 정확도가 올라간다."
        )

    engines = (
        list(recognizers) if recognizers is not None
        else default_recognizers(vocabulary, separated=feats.separated)
    )
    if not engines:
        raise RuntimeError("사용 가능한 인식기가 없다")

    def run(engine_list: Sequence[Recognizer]) -> List[Tuple[Recognizer, "object"]]:
        pairs = []
        for engine in engine_list:
            try:
                pairs.append((engine, engine.recognize(feats)))
            except Exception as exc:  # 외부 모델이 죽어도 내장 인식기로 계속 간다
                warnings.append(f"{engine.name} 인식 실패: {exc}")
        return pairs

    pairs = run(engines)
    if not pairs:
        raise RuntimeError("모든 인식기가 실패했다")

    chords, confidence, agreement = vote([p for _, p in pairs], key)

    # 2패스: 인식된 코드로 조성을 다시 추정한다.
    # 크로마 프로파일은 음 분포만 보지만 코드 진행은 구조(5도 해결, 종지)를
    # 드러내서 더 정확하다. 조성이 바뀌면 조성을 참조하는 인식기만 다시 돌린다.
    if key_override is None:
        ranked = rank_keys([c for c in chords if not c.is_no_chord])
        scores = dict(ranked)
        refined = ranked[0][0] if ranked else key
        # 근소한 차이로는 덮어쓰지 않는다. 나란한조는 구성음이 같아 점수가 늘
        # 붙어 다니는데, 인식 오류 몇 개만으로 뒤집히면 로마숫자 분석이 통째로
        # 어긋난다(C장조 곡이 Am으로 뒤집히는 식).
        margin = scores.get(refined, 0.0) - scores.get(key, float("-inf"))
        threshold = (
            KEY_REFINE_RELATIVE_MARGIN if refined == key.relative() else KEY_REFINE_MARGIN
        )
        if refined != key and margin >= threshold:
            warnings.append(
                f"조성을 코드 진행 기준으로 {key.name()} → {refined.name()}로 정정 (마진 {margin:.1f})"
            )
            key = refined
            feats.key = refined
            rerun = {id(e): p for e, p in run([e for e, _ in pairs if e.uses_key])}
            pairs = [(e, rerun.get(id(e), p)) for e, p in pairs]
            chords, confidence, agreement = vote([p for _, p in pairs], key)

    predictions = [p for _, p in pairs]
    chords = smooth_short_runs(chords, min_beats=2)

    segments = segment_beats(
        chords, confidence, agreement, feats.beat_times, feats.duration,
        feats.downbeat_indices, beats_per_bar,
    )
    if snap_bars:
        segments = snap_to_bars(segments, feats.beat_times, feats.downbeat_indices)
    segments = merge_adjacent(segments)
    # 전위 판정은 구간이 확정된 뒤에. 비트 단위로 하면 같은 코드가 쪼개진다.
    if detect_inversions:
        segments = apply_bass_to_segments(segments, feats.bass_chroma, feats.beat_times)

    if len(predictions) == 1:
        warnings.append(
            "인식기가 1개뿐이라 앙상블 효과가 없다 — 일치도가 항상 1.0이라 신뢰도 지표가 "
            "확신도에만 의존한다. 소스 분리를 켜면 분리본/원본 두 의견으로 교차검증된다."
        )

    downbeat_times = feats.beat_times[
        np.clip(np.atleast_1d(feats.downbeat_indices), 0, len(feats.beat_times) - 1)
    ]

    return AnalysisResult(
        source=path,
        duration=feats.duration,
        tempo=feats.tempo,
        key=key,
        key_confidence=feats.key_confidence,
        segments=segments,
        beat_times=[float(t) for t in feats.beat_times],
        downbeats=[float(t) for t in downbeat_times],
        beats_per_bar=beats_per_bar,
        recognizers=[p.name for p in predictions],
        origin="audio",
        warnings=warnings,
    )
