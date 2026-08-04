"""코드 인식기들.

내장 인식기(크로마 템플릿 + 비터비)는 numpy/librosa만으로 동작하고,
autochord / Chordino는 설치돼 있으면 자동으로 앙상블에 합류한다.
설치 여부와 무관하게 파이프라인이 끝까지 돌아가는 게 M1의 목표.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..theory.chord import Chord
from ..theory.key import Key
from .audio import AudioFeatures
from .templates import build_vocabulary


# 박자 위치별 코드 변화 확률 가중. 마디 첫 박 = 1.0 기준.
WEAK_BEAT_STRENGTH = 0.25    # 마디 안쪽 여린박
HALF_BAR_STRENGTH = 0.6      # 반 마디 지점 (4/4의 3박)
METRICAL_FLOOR = 0.15        # 아무리 약해도 변화 가능성을 완전히 막지는 않는다
METRICAL_GAIN = 1.6          # 마디 경계에서의 변화를 균일 전이보다 적극적으로


def metrical_strength(feats: AudioFeatures, n_beats: int) -> np.ndarray:
    """각 비트의 박자상 강세. 1.0 = 마디 첫 박.

    코드는 아무 데서나 바뀌지 않는다. 마디 첫 박에서 압도적으로 자주 바뀌고,
    그다음이 반 마디 지점이다. 이 정보를 전이 확률에 넣으면 두 방향으로 이득이다.
    - 마디 안쪽에서 흔들리는 오탐이 줄고
    - 마디 경계에서 실제로 바뀐 코드를 놓치지 않는다
    """
    strength = np.full(n_beats, WEAK_BEAT_STRENGTH, dtype=np.float32)
    bpb = max(int(feats.beats_per_bar), 1)
    downbeats = np.atleast_1d(feats.downbeat_indices)
    if downbeats.size == 0:
        return strength

    first = int(downbeats.min())
    offsets = (np.arange(n_beats) - first) % bpb
    strength[offsets == 0] = 1.0
    if bpb % 2 == 0:
        strength[offsets == bpb // 2] = HALF_BAR_STRENGTH
    return strength


def viterbi_path(
    scores: np.ndarray, change_prob: np.ndarray, emission_gain: float = 12.0
) -> np.ndarray:
    """(상태, 시간) 점수 행렬에서 최적 경로. 전이 확률은 시점마다 다를 수 있다."""
    n_states, n_steps = scores.shape
    change = np.clip(change_prob, 1e-6, 0.5)
    log_stay = np.log(1.0 - change)
    log_move = np.log(change / max(n_states - 1, 1))

    emission = scores * emission_gain
    delta = emission[:, 0].copy()
    backptr = np.zeros((n_states, n_steps), dtype=np.int32)

    for t in range(1, n_steps):
        stay = delta + log_stay[t]
        move_best = delta.max() + log_move[t]
        move_arg = int(delta.argmax())
        take_stay = stay > move_best
        backptr[:, t] = np.where(take_stay, np.arange(n_states), move_arg)
        delta = np.where(take_stay, stay, move_best) + emission[:, t]

    path = np.zeros(n_steps, dtype=np.int32)
    path[-1] = int(delta.argmax())
    for t in range(n_steps - 1, 0, -1):
        path[t - 1] = backptr[path[t], t]
    return path


@dataclass
class BeatPrediction:
    """비트 단위 예측. 앙상블은 이 형태로만 대화한다."""

    name: str
    chords: List[Chord]
    confidence: np.ndarray
    weight: float = 1.0
    posterior: Optional[np.ndarray] = None      # (n_beats, V) 있으면 가중투표에 직접 사용
    vocabulary: Optional[List[Chord]] = None    # posterior의 열 순서


class Recognizer:
    name = "base"
    weight = 1.0
    uses_key = False   # True면 조성이 바뀌었을 때 다시 돌릴 가치가 있다

    @property
    def available(self) -> bool:
        return True

    def recognize(self, feats: AudioFeatures) -> BeatPrediction:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------
# 내장: 크로마 템플릿 + 비터비 평활화
# --------------------------------------------------------------------------

class ChromaTemplateRecognizer(Recognizer):
    """비트 동기 크로마와 코드 템플릿의 코사인 유사도를 비터비로 이어붙인다.

    딥러닝 모델보다 정확도는 낮지만 의존성이 numpy뿐이고 CPU에서 실시간의
    수십 배로 돈다. 앙상블의 기준선 역할.
    """

    name = "chroma-template"
    uses_key = True

    def __init__(
        self,
        vocabulary: str = "standard",
        weight: float = 1.0,
        self_transition: float = 0.90,
        key_bonus: float = 0.06,
        source: str = "primary",
        name: Optional[str] = None,
    ):
        self.weight = weight
        self.vocab_name = vocabulary
        self.self_transition = self_transition
        self.key_bonus = key_bonus
        # "primary" = 분리 스템(있으면), "mix" = 원본 믹스.
        # 둘은 서로 다른 오류 편향을 가져서 앙상블의 두 의견이 된다.
        self.source = source
        if name:
            self.name = name
        elif source != "primary":
            self.name = f"chroma-template({source})"
        self.chords, self.templates, self.quality_prior = build_vocabulary(vocabulary)

    def _chroma(self, feats: AudioFeatures) -> Optional[np.ndarray]:
        if self.source == "mix":
            return feats.beat_chroma_alt
        return feats.beat_chroma

    def recognize(self, feats: AudioFeatures) -> BeatPrediction:
        beat_chroma = self._chroma(feats)
        if beat_chroma is None:
            return BeatPrediction(self.name, [], np.zeros(0), self.weight)
        n_beats = beat_chroma.shape[1]
        if n_beats == 0:
            return BeatPrediction(self.name, [], np.zeros(0), self.weight)

        norms = np.linalg.norm(beat_chroma, axis=0, keepdims=True) + 1e-9
        normalized = beat_chroma / norms
        scores = self.templates @ normalized             # (V, n_beats)
        scores = scores + self.quality_prior[:, None]
        scores = scores + self._key_prior(feats.key)[:, None]

        path = self._viterbi(scores, self._metrical_strength(feats, n_beats))
        posterior = _softmax(scores.T * 12.0, axis=1)     # (n_beats, V)
        conf = _margin_confidence(scores)

        return BeatPrediction(
            name=self.name,
            chords=[self.chords[i] for i in path],
            confidence=conf,
            weight=self.weight,
            posterior=posterior,
            vocabulary=self.chords,
        )

    def _key_prior(self, key: Key) -> np.ndarray:
        """조성 문맥 제약. 조성 안 코드에 약한 가산점."""
        bonus = np.zeros(len(self.chords), dtype=np.float32)
        for i, chord in enumerate(self.chords):
            in_key = sum(1 for pc in chord.pitch_classes() if key.contains_pc(pc))
            total = len(chord.pitch_classes())
            bonus[i] = self.key_bonus * (in_key / total)
        return bonus

    def _metrical_strength(self, feats: AudioFeatures, n_beats: int) -> np.ndarray:
        return metrical_strength(feats, n_beats)

    def _viterbi(self, scores: np.ndarray, strength: Optional[np.ndarray] = None) -> np.ndarray:
        """자기 전이에 보너스를 줘서 프레임 단위 튐을 없앤다.

        전이 확률은 위치에 따라 달라진다(`strength`). 균일하게 두면 코드 변화를
        통째로 놓쳐서 여러 코드가 하나로 뭉개진다.
        """
        n_states, n_steps = scores.shape
        base_change = 1.0 - self.self_transition
        if strength is None:
            change = np.full(n_steps, base_change, dtype=np.float64)
        else:
            # 마디 첫 박에서는 코드가 바뀔 확률을 크게, 마디 안쪽에서는 작게
            change = base_change * (METRICAL_FLOOR + (1.0 - METRICAL_FLOOR) * strength)
            change *= METRICAL_GAIN
        change = np.clip(change, 1e-6, 0.5)

        log_stay = np.log(1.0 - change)
        log_move = np.log(change / max(n_states - 1, 1))

        emission = scores * 12.0
        delta = emission[:, 0].copy()
        backptr = np.zeros((n_states, n_steps), dtype=np.int32)

        for t in range(1, n_steps):
            stay = delta + log_stay[t]
            move_best = delta.max() + log_move[t]
            move_arg = int(delta.argmax())
            take_stay = stay > move_best
            backptr[:, t] = np.where(take_stay, np.arange(n_states), move_arg)
            delta = np.where(take_stay, stay, move_best) + emission[:, t]

        path = np.zeros(n_steps, dtype=np.int32)
        path[-1] = int(delta.argmax())
        for t in range(n_steps - 1, 0, -1):
            path[t - 1] = backptr[path[t], t]
        return path


def _margin_confidence(scores: np.ndarray, scale: float = 0.10) -> np.ndarray:
    """1등과 2등 템플릿의 점수 차로 확신도를 잰다.

    softmax 최대값을 쓰면 안 된다. 어휘가 96개라 아무리 명확한 코드도
    확률이 낮게 나와서, 전 구간이 "확신 없음"으로 표시된다(실제로 그랬다).
    "다른 후보를 얼마나 확실히 제쳤나"가 우리가 알고 싶은 값이다.
    """
    if scores.shape[0] < 2:
        return np.ones(scores.shape[1], dtype=np.float32)
    top2 = np.partition(scores, -2, axis=0)[-2:]
    margin = top2[1] - top2[0]
    return np.clip(margin / scale, 0.0, 1.0).astype(np.float32)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


# --------------------------------------------------------------------------
# 외부 모델 어댑터 (설치돼 있을 때만 동작)
# --------------------------------------------------------------------------

@dataclass
class LabeledSegment:
    start: float
    end: float
    label: str


class _SegmentRecognizer(Recognizer):
    """구간 라벨 리스트를 내놓는 외부 도구를 비트 단위로 변환하는 공통 베이스."""

    def _segments(self, audio_path: str) -> List[LabeledSegment]:  # pragma: no cover
        raise NotImplementedError

    def recognize(self, feats: AudioFeatures) -> BeatPrediction:
        segments = self._segments(feats.source_path)  # type: ignore[attr-defined]
        beat_times = feats.beat_times
        chords: List[Chord] = []
        conf = np.zeros(len(beat_times), dtype=np.float32)
        for i, t in enumerate(beat_times):
            center = t + (beat_times[i + 1] - t) / 2 if i + 1 < len(beat_times) else t
            label = next((s.label for s in segments if s.start <= center < s.end), "N")
            try:
                chords.append(Chord.parse(label))
                conf[i] = 0.8
            except ValueError:
                chords.append(Chord(0, "N"))
                conf[i] = 0.1
        return BeatPrediction(self.name, chords, conf, self.weight)


class AutochordRecognizer(_SegmentRecognizer):
    """autochord (사전학습 딥러닝, CPU 가능). 미설치면 앙상블에서 자동 제외."""

    name = "autochord"

    def __init__(self, weight: float = 1.4):
        self.weight = weight

    @property
    def available(self) -> bool:
        try:
            import autochord  # noqa: F401
        except Exception:
            return False
        return True

    def _segments(self, audio_path: str) -> List[LabeledSegment]:
        import autochord

        return [LabeledSegment(float(s), float(e), lab) for s, e, lab in autochord.recognize(audio_path)]


class ChordinoRecognizer(_SegmentRecognizer):
    """chord-extractor(Chordino). DSP 기반이라 연산이 거의 공짜."""

    name = "chordino"

    def __init__(self, weight: float = 1.1):
        self.weight = weight

    @property
    def available(self) -> bool:
        try:
            from chord_extractor.extractors import Chordino  # noqa: F401
        except Exception:
            return False
        return True

    def _segments(self, audio_path: str) -> List[LabeledSegment]:
        from chord_extractor.extractors import Chordino

        result = Chordino().extract(audio_path)
        segments: List[LabeledSegment] = []
        for i, item in enumerate(result):
            end = result[i + 1].timestamp if i + 1 < len(result) else item.timestamp + 2.0
            segments.append(LabeledSegment(float(item.timestamp), float(end), item.chord))
        return segments


def default_recognizers(vocabulary: str = "standard", separated: bool = False) -> List[Recognizer]:
    """설치된 것만 골라 앙상블 구성.

    `separated=True`면 분리 스템용과 원본 믹스용 인식기를 둘 다 넣는다.
    같은 알고리즘이지만 입력이 달라서 오류가 상관되지 않는다 — 윈도우에서는
    autochord/Chordino가 vamp(C++ 컴파일러) 때문에 설치되지 않으므로,
    이게 현실적으로 얻을 수 있는 유일한 교차검증이다.
    분리 스템 쪽을 더 신뢰한다(보컬·드럼이 빠져 있으므로).
    """
    from .hierarchical import HierarchicalRecognizer

    candidates: List[Recognizer] = [
        ChromaTemplateRecognizer(vocabulary=vocabulary),
        # 전혀 다른 방식(근음→3화음→7화음→텐션 단계별 판정)이라 오류가 상관되지 않는다.
        # 합성 데모에서는 템플릿과 동점이지만 실패하는 지점이 서로 다르다.
        HierarchicalRecognizer(),
    ]
    if separated:
        candidates.append(
            ChromaTemplateRecognizer(vocabulary=vocabulary, source="mix", weight=0.7)
        )
    candidates += [AutochordRecognizer(), ChordinoRecognizer()]
    return [r for r in candidates if r.available]
