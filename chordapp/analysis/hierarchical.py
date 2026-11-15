"""단계별 코드 인식: 근음 → 3화음 → 7화음 → 텐션.

96개 템플릿을 한 번에 코사인 유사도로 비교하면, 각 판단이 서로를 방해한다.
maj/min 판별은 사실 **3음 자리 두 칸(root+3 vs root+4)만 보면 되는 문제**인데
템플릿 매칭에서는 나머지 음들의 에너지에 묻힌다. 7음 유무도 마찬가지다.

그래서 결정을 쪼갠다. 각 단계는 그 판단을 실제로 가르는 음만 본다.

  1. 근음  : 베이스 크로마(강한 사전확률) + 화성 크로마 + 조성 문맥, 비터비로 평활화
  2. 3화음 : 근음 기준 3음·5음 자리 비교 (maj/min/dim/aug/sus2/sus4)
  3. 7화음 : b7·M7·6 자리를 화음구성음 평균 대비 상대 세기로 판정
  4. 텐션  : 9·11·13 자리를 같은 방식으로, 더 엄격한 문턱으로 판정

베이스는 근음을 **확정하지 않는다.** 전위·슬래시 코드에서 베이스는 3음이나 5음을
짚기 때문에, 확정해버리면 그 구간이 통째로 틀린다. 강한 가중치를 주되
화성 크로마가 뒤집을 수 있게 둔다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..theory.chord import Chord
from ..theory.key import Key
from .audio import AudioFeatures
from .recognizers import BeatPrediction, Recognizer, metrical_strength, viterbi_path

# --- 근음 판정 -------------------------------------------------------------
BASS_WEIGHT = 0.9        # 베이스 크로마가 근음 판단에 기여하는 정도
FIFTH_WEIGHT = 0.35      # 근음 위 완전5도가 있으면 근음일 가능성이 오른다
THIRD_WEIGHT = 0.30
ROOT_KEY_BONUS = 0.10    # 조성 안 음을 근음으로 선호
ROOT_CHANGE_PROB = 0.22  # 근음이 바뀔 기본 확률 (마디 위치로 조정된다)

# --- 상부구조 판정 ---------------------------------------------------------
# 화음구성음(근음·3음·5음) 평균 에너지 대비 이 비율을 넘으면 "있다"고 본다.
# 0.50이 합성 데모 기준 최적. 스윕해보면 양쪽 끝에서 모두 나빠지는 진짜 최적점이다.
#   0.45 → 3화음 곡에 없는 7음이 붙기 시작 (98.6% → 97.6%)
#   0.55 → 있는 7음을 놓치기 시작 (99.3% → 91.0%)
SEVENTH_THRESHOLD = 0.50
TENSION_THRESHOLD = 0.80
SUS_MARGIN = 1.25        # sus로 판정하려면 3음보다 이만큼 세야 한다
RARE_FIFTH_MARGIN = 1.6  # dim/aug는 드물다. 완전5도를 이만큼 이겨야 인정
AMBIGUOUS_THIRD = 0.15   # 장3도·단3도 차이가 이보다 작으면 조성으로 판단

# 반음 간격
P1, m2, M2, m3, M3, P4, TT, P5, m6, M6, m7, M7 = range(12)


class HierarchicalRecognizer(Recognizer):
    """근음부터 위로 쌓아 올리는 인식기."""

    name = "hierarchical"
    uses_key = True

    def __init__(
        self,
        weight: float = 1.3,
        bass_weight: float = BASS_WEIGHT,
        detect_sevenths: bool = True,
        detect_tensions: bool = True,
        quality_mode: str = "local",
    ):
        if quality_mode not in {"local", "legacy"}:
            raise ValueError("지원하지 않는 화음 종류 판정 모드입니다.")
        self.weight = weight
        self.bass_weight = bass_weight
        self.detect_sevenths = detect_sevenths
        self.detect_tensions = detect_tensions
        self.quality_mode = quality_mode

    # -- 1단계: 근음 ---------------------------------------------------
    def _root_scores(self, feats: AudioFeatures) -> np.ndarray:
        """(12, n_beats) 근음 점수."""
        chroma = _normalize(feats.beat_chroma)
        n_beats = chroma.shape[1]
        scores = np.zeros((12, n_beats), dtype=np.float32)

        bass = feats.bass_chroma
        use_bass = bass is not None and bass.shape[1] == n_beats
        bass_norm = _normalize(bass) if use_bass else None

        for pc in range(12):
            third = np.maximum(chroma[(pc + m3) % 12], chroma[(pc + M3) % 12])
            support = (
                chroma[pc]
                + FIFTH_WEIGHT * chroma[(pc + P5) % 12]
                + THIRD_WEIGHT * third
            )
            scores[pc] = support
            if bass_norm is not None:
                scores[pc] += self.bass_weight * bass_norm[pc]

        key = feats.key
        for pc in range(12):
            if key.contains_pc(pc):
                scores[pc] += ROOT_KEY_BONUS
        return scores / (scores.max(axis=0, keepdims=True) + 1e-9)

    # -- 2~4단계: 근음 위에 쌓기 ---------------------------------------
    def _quality_for(self, column: np.ndarray, root: int, key: Key) -> Tuple[str, float]:
        """근음이 정해진 뒤 품질을 정한다. 판단을 가르는 음만 본다."""
        def at(interval: int) -> float:
            return float(column[(root + interval) % 12])

        # --- 3화음: 3음 자리와 5음 자리 ---
        minor_third, major_third = at(m3), at(M3)
        fifth, dim_fifth, aug_fifth = at(P5), at(TT), at(m6)
        sus4, sus2 = at(P4), at(M2)

        third_energy = max(minor_third, major_third)
        third_margin = abs(major_third - minor_third) / (third_energy + 1e-6)

        # 3음이 애매하면(두 후보가 붙어 있으면) 크로마로 우기지 말고 조성으로 푼다.
        # 실제 녹음에서는 보컬·리버브 때문에 3음 자리가 자주 흐려지는데, 그때
        # 억지로 고르면 DbmMaj7 같은 실존하지 않는 코드가 나온다(실제로 나왔다).
        # 조성 안에서 그 근음이 원래 무슨 3화음인지가 훨씬 믿을 만한 단서다.
        if third_margin < AMBIGUOUS_THIRD:
            is_minor = _diatonic_is_minor(key, root)
        else:
            is_minor = minor_third > major_third
        quality_third = "min" if is_minor else "maj"

        # sus는 3음이 약하고 4음(또는 2음)이 뚜렷할 때만
        sus_kind = None
        if sus4 > third_energy * SUS_MARGIN and sus4 > sus2:
            sus_kind = "sus4"
        elif sus2 > third_energy * SUS_MARGIN:
            sus_kind = "sus2"

        core = [at(P1), third_energy, fifth]
        positive_core = [c for c in core if c > 0]
        base = float(np.mean(positive_core)) if positive_core else 1e-6

        # --- 5도 변화 ---
        # dim/aug는 대중음악에서 드물다. 5도 자리만 보고 쉽게 넘어가면
        # 실제 녹음에서 남발된다(실측에서 Dbmaj7 구간을 Dbdim으로 읽었다).
        # 완전5도를 뚜렷하게 이겨야 하고, 3음 판정도 같은 방향이어야 한다.
        third_is_clear = third_margin >= AMBIGUOUS_THIRD
        if (sus_kind is None and is_minor and third_is_clear
                and dim_fifth > fifth * RARE_FIFTH_MARGIN):
            triad = "dim"
        elif (sus_kind is None and not is_minor and third_is_clear
                and aug_fifth > fifth * RARE_FIFTH_MARGIN):
            triad = "aug"
        elif sus_kind is not None:
            triad = sus_kind
        else:
            triad = quality_third

        # --- 3단계: 7화음 ---
        seventh = None
        if self.detect_sevenths and triad not in ("dim", "aug"):
            b7, maj7, sixth = at(m7), at(M7), at(M6)
            best, kind = 0.0, None
            for value, name in ((b7, "b7"), (maj7, "maj7"), (sixth, "6")):
                if value > base * SEVENTH_THRESHOLD and value > best:
                    best, kind = value, name
            seventh = kind
        elif self.detect_sevenths and triad == "dim":
            b7, dim7 = at(m7), at(M6)
            if dim7 > base * SEVENTH_THRESHOLD and dim7 >= b7:
                seventh = "dim7"
            elif b7 > base * SEVENTH_THRESHOLD:
                seventh = "b7"

        quality = _assemble(triad, seventh)

        # --- 4단계: 텐션 ---
        if self.detect_tensions and quality in _TENSION_UPGRADE:
            ninth = at(M2)
            if ninth > base * TENSION_THRESHOLD and key.contains_pc((root + M2) % 12):
                quality = _TENSION_UPGRADE[quality]

        # 확신도: 3음 판정이 얼마나 확실했는지
        margin = abs(major_third - minor_third) / (third_energy + 1e-6)
        return quality, float(np.clip(margin, 0.0, 1.0))

    # -- 실행 -----------------------------------------------------------
    def recognize(self, feats: AudioFeatures) -> BeatPrediction:
        chroma = _normalize(feats.beat_chroma)
        n_beats = chroma.shape[1]
        if n_beats == 0:
            return BeatPrediction(self.name, [], np.zeros(0), self.weight)

        root_scores = self._root_scores(feats)
        strength = metrical_strength(feats, n_beats)
        change = ROOT_CHANGE_PROB * (0.15 + 0.85 * strength)
        roots = viterbi_path(root_scores, change)

        chords: List[Chord] = [Chord(0, "maj")] * n_beats
        confidence = np.zeros(n_beats, dtype=np.float32)

        # 근음이 유지돼도 C→C7→Cmaj7처럼 화음 종류는 바뀔 수 있다.
        # 각 박의 품질 판정으로 경계를 찾은 뒤 같은 근음·품질 구간만 묶어 집계한다.
        # 과거의 근음 구간 전체 중앙값 방식은 비교용 legacy 옵션으로 보존한다.
        for start, end in _runs(roots):
            root = int(roots[start])
            if self.quality_mode == "legacy":
                spans = [(0, end - start)]
            else:
                qualities = np.asarray([self._quality_for(chroma[:, i], root, feats.key)[0]
                                        for i in range(start, end)])
                spans = _runs(qualities)
            for lo, hi in spans:
                left, right = start + lo, start + hi
                column = np.median(chroma[:, left:right], axis=1)
                quality, quality_conf = self._quality_for(column, root, feats.key)
                chord = Chord(root, quality)
                for i in range(left, right):
                    chords[i] = chord
                    confidence[i] = 0.65 * _margin(root_scores[:, i]) + 0.35 * quality_conf

        return BeatPrediction(
            name=self.name, chords=chords, confidence=confidence, weight=self.weight
        )


# 7음이 있는 코드에만 9음 확장을 허용한다 (add9는 오탐이 많아 제외).
_TENSION_UPGRADE: Dict[str, str] = {
    "7": "9", "maj7": "maj9", "min7": "min9",
}


def _assemble(triad: str, seventh: Optional[str]) -> str:
    """3화음 + 7음 조합을 내부 품질 이름으로."""
    if seventh is None:
        return triad
    if triad == "maj":
        return {"b7": "7", "maj7": "maj7", "6": "6"}[seventh]
    if triad == "min":
        return {"b7": "min7", "maj7": "minmaj7", "6": "min6"}[seventh]
    if triad == "dim":
        return {"dim7": "dim7", "b7": "hdim7"}.get(seventh, "dim")
    if triad in ("sus4", "sus2"):
        return "7sus4" if seventh == "b7" and triad == "sus4" else triad
    return triad


def _diatonic_is_minor(key: Key, root: int) -> bool:
    """조성 안에서 이 근음의 3화음이 단3화음인지. 조성 밖이면 장3화음으로 본다
    (차용화음·세컨더리 도미넌트는 장3화음인 경우가 압도적)."""
    for triad in key.diatonic_triads():
        if triad.root == root % 12:
            return triad.quality in ("min", "dim")
    return False


def _runs(values: np.ndarray) -> List[Tuple[int, int]]:
    """같은 값이 이어지는 구간 [시작, 끝) 목록."""
    spans: List[Tuple[int, int]] = []
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            spans.append((start, i))
            start = i
    return spans


def _normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix is None or matrix.size == 0:
        return matrix
    return matrix / (matrix.max(axis=0, keepdims=True) + 1e-9)


def _margin(column: np.ndarray) -> float:
    """1등과 2등의 차이를 확신도로."""
    if column.size < 2:
        return 1.0
    top2 = np.partition(column, -2)[-2:]
    return float(np.clip((top2[1] - top2[0]) / 0.25, 0.0, 1.0))
