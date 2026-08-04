"""조성(Key): 다이어토닉 코드, 로마숫자 분석, 관계조/나란한조.

대체 코드 엔진의 모든 규칙이 "이 코드가 조성 안에서 무슨 기능인가"를 먼저 묻기 때문에
로마숫자 분석이 사실상 엔진의 입구다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .chord import Chord
from .notes import note_name, parse_note, prefers_flat

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
NATURAL_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)
HARMONIC_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 11)

MAJOR_TRIADS = ("maj", "min", "min", "maj", "maj", "min", "dim")
MAJOR_SEVENTHS = ("maj7", "min7", "min7", "maj7", "7", "min7", "hdim7")
MINOR_TRIADS = ("min", "dim", "maj", "min", "min", "maj", "maj")
MINOR_SEVENTHS = ("min7", "hdim7", "maj7", "min7", "min7", "maj7", "7")

MAJOR_NUMERALS = ("I", "ii", "iii", "IV", "V", "vi", "vii")
MINOR_NUMERALS = ("i", "ii", "III", "iv", "v", "VI", "VII")

_ROMAN_BY_DEGREE = ("I", "II", "III", "IV", "V", "VI", "VII")

# 조성 밖 근음은 올림/내림 어느 쪽으로도 이름 붙일 수 있어서 관례를 따른다.
# 장조: bII bIII bVI bVII은 차용화음으로 흔하고, 트라이톤만 #IV로 부른다.
# 단조: bII(나폴리)만 내림이고 #III #IV #VI #VII은 올림으로 부른다.
_SHARP_DEGREES = {"major": frozenset({6}), "minor": frozenset({4, 6, 9, 11})}

# 오디오 크로마용 Krumhansl-Kessler 프로파일 (연속적인 에너지 분포에 맞는 쪽)
KK_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
KK_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)

# 코드 구성음 히스토그램용 프로파일. 으뜸음 기준 반음 간격별 가중치.
# 조성 밖 음에 음수를 주는 게 핵심 (KK에는 그 성질이 없다).
# bVII/#vi 처럼 차용화음으로 흔한 자리는 감점을 약하게 준다.
CHORD_MAJOR_PROFILE = (2.6, -2.0, 1.2, -1.6, 1.6, 1.6, -2.0, 2.2, -1.6, 1.4, -1.4, 1.0)
CHORD_MINOR_PROFILE = (2.6, -1.6, 1.2, 1.6, -1.2, 1.6, -2.0, 2.2, 1.4, -1.0, 1.4, 0.6)


@dataclass(frozen=True)
class Key:
    tonic: int
    mode: str = "major"  # "major" | "minor"

    def __post_init__(self) -> None:
        if self.mode not in ("major", "minor"):
            raise ValueError(f"모드는 major/minor만: {self.mode!r}")
        object.__setattr__(self, "tonic", self.tonic % 12)

    @classmethod
    def parse(cls, text: str) -> "Key":
        """'C', 'Am', 'F# minor', 'Bb major' 형태를 파싱."""
        t = text.strip().replace("maj", " major").replace("min", " minor")
        parts = t.split()
        head = parts[0]
        mode = "major"
        if len(parts) > 1 and parts[1].lower().startswith("min"):
            mode = "minor"
        elif head.endswith("m") and len(head) > 1 and head[-2] not in ("#", "b"):
            head, mode = head[:-1], "minor"
        elif len(head) > 1 and head[-1] == "m":
            head, mode = head[:-1], "minor"
        return cls(parse_note(head), mode)

    # --- 스케일 -------------------------------------------------------
    @property
    def scale(self) -> Tuple[int, ...]:
        base = MAJOR_SCALE if self.mode == "major" else NATURAL_MINOR_SCALE
        return tuple((self.tonic + i) % 12 for i in base)

    @property
    def prefer_flat(self) -> bool:
        return prefers_flat(self.tonic, self.mode)

    def contains_pc(self, pc: int) -> bool:
        return pc % 12 in self.scale

    # --- 다이어토닉 코드 ----------------------------------------------
    def diatonic_triads(self) -> List[Chord]:
        quals = MAJOR_TRIADS if self.mode == "major" else MINOR_TRIADS
        return [Chord(pc, q) for pc, q in zip(self.scale, quals)]

    def diatonic_sevenths(self) -> List[Chord]:
        quals = MAJOR_SEVENTHS if self.mode == "major" else MINOR_SEVENTHS
        return [Chord(pc, q) for pc, q in zip(self.scale, quals)]

    def degree_of(self, pc: int) -> Optional[int]:
        """근음이 스케일 몇 번째 음인지 (0-based). 조성 밖이면 None."""
        pc %= 12
        for i, s in enumerate(self.scale):
            if s == pc:
                return i
        return None

    def dominant_of(self, target_pc: int) -> Chord:
        """해당 음을 임시 으뜸음으로 보는 V7."""
        return Chord((target_pc + 7) % 12, "7")

    # --- 관계조 -------------------------------------------------------
    def relative(self) -> "Key":
        if self.mode == "major":
            return Key((self.tonic + 9) % 12, "minor")
        return Key((self.tonic + 3) % 12, "major")

    def parallel(self) -> "Key":
        return Key(self.tonic, "minor" if self.mode == "major" else "major")

    # --- 분석 ---------------------------------------------------------
    def roman(self, chord: Chord) -> str:
        """코드의 로마숫자 표기. 조성 밖 근음은 b/#를 붙여 표기한다."""
        if chord.is_no_chord:
            return "N.C."
        interval = (chord.root - self.tonic) % 12
        base = MAJOR_SCALE if self.mode == "major" else NATURAL_MINOR_SCALE
        accidental = ""
        if interval in base:
            degree = base.index(interval)
        elif interval in _SHARP_DEGREES[self.mode]:
            degree = base.index((interval - 1) % 12)
            accidental = "#"
        else:
            degree = base.index((interval + 1) % 12)
            accidental = "b"

        numeral = _ROMAN_BY_DEGREE[degree]
        tq = chord.triad_quality()
        if tq in ("min", "dim", "hdim"):
            numeral = numeral.lower()

        suffix = ""
        if tq == "dim":
            suffix = "°"
        elif tq == "hdim":
            suffix = "ø"
        elif tq == "aug":
            suffix = "+"
        elif tq == "dom":
            suffix = "7"
        elif chord.quality in ("maj7", "maj9", "maj13"):
            suffix = "maj7"
        elif chord.quality in ("min7", "min9", "min11"):
            suffix = "7"
        elif chord.quality in ("sus2", "sus4"):
            suffix = chord.quality

        text = accidental + numeral + suffix
        if chord.bass is not None:
            text += "/" + note_name(chord.bass, self.prefer_flat)
        return text

    def is_diatonic(self, chord: Chord) -> bool:
        return all(self.contains_pc(pc) for pc in chord.pitch_classes())

    def function_of(self, chord: Chord) -> str:
        """토닉 / 서브도미넌트 / 도미넌트 기능 분류."""
        deg = self.degree_of(chord.root)
        if deg is None:
            if chord.is_dominant:
                return "dominant"
            return "chromatic"
        if self.mode == "major":
            table = {0: "tonic", 1: "subdominant", 2: "tonic", 3: "subdominant",
                     4: "dominant", 5: "tonic", 6: "dominant"}
        else:
            table = {0: "tonic", 1: "subdominant", 2: "tonic", 3: "subdominant",
                     4: "dominant", 5: "subdominant", 6: "dominant"}
        return table[deg]

    def name(self) -> str:
        return note_name(self.tonic, self.prefer_flat) + ("" if self.mode == "major" else "m")

    def __str__(self) -> str:  # pragma: no cover
        return self.name()


def scale_pcs(tonic: int, mode: str) -> Tuple[int, ...]:
    base = MAJOR_SCALE if mode == "major" else NATURAL_MINOR_SCALE
    return tuple((tonic + i) % 12 for i in base)


def key_from_chord_counts(counts: Dict[int, float]) -> Key:
    """코드 구성음 히스토그램만으로 조성을 추정.

    여기서 Krumhansl-Kessler를 쓰면 안 된다. KK는 선율의 음 지속시간을 전제로
    만들어진 프로파일이라 이끔음(7음) 가중치가 평균보다 낮다. 그래서 코드
    구성음에 적용하면 B가 있다는 사실이 오히려 C장조에 불리하게 작용해서
    Dm7-G7-Cmaj7을 G장조로 판정한다(실제로 그렇게 틀렸다).

    코드는 스케일 소속 여부가 훨씬 강한 단서이므로, 조성 밖 음에 감점을 주는
    전용 프로파일을 쓴다. 조성 안 음은 기능 중요도(I > V > IV·vi > …) 순으로 가중.
    """
    vec = [float(counts.get(i, 0.0)) for i in range(12)]
    if sum(vec) <= 0:
        return Key(0, "major")

    best, best_score = Key(0, "major"), float("-inf")
    for tonic in range(12):
        for mode, profile in (("major", CHORD_MAJOR_PROFILE), ("minor", CHORD_MINOR_PROFILE)):
            score = sum(vec[(tonic + i) % 12] * profile[i] for i in range(12))
            if score > best_score:
                best, best_score = Key(tonic, mode), score
    return best


def key_from_chords(chords: Sequence[Chord]) -> Key:
    """코드 진행에서 조성을 추정. 음 분포 + 진행의 구조를 함께 본다.

    음 분포만 보면 짧은 진행에서 자주 틀린다. Dm7-G7-Cmaj7-A7은 C장조의
    ii-V-I + V7/ii인데, A7의 C# 하나 때문에 음 분포상으로는 A단조가 이긴다.
    사람은 "G7이 C로 5도 해결한다"는 구조를 보고 판단하므로 그 단서를 넣는다.
    """
    ranked = rank_keys(chords)
    return ranked[0][0] if ranked else Key(0, "major")


def rank_keys(chords: Sequence[Chord]) -> List[Tuple[Key, float]]:
    """24개 조성 전체를 점수 순으로. 점수 차이를 봐야 하는 쪽에서 쓴다.

    오디오 기반 추정을 코드 기반 추정으로 덮어쓸지 판단할 때, 1등만 보면
    나란한조끼리 근소한 차이로 뒤집혀서 오히려 나빠진다.
    """
    playable = [c for c in chords if not c.is_no_chord]
    if not playable:
        return []

    counts: Dict[int, float] = {}
    for chord in playable:
        for pc in chord.pitch_classes():
            counts[pc] = counts.get(pc, 0.0) + 1.0
        counts[chord.root] = counts.get(chord.root, 0.0) + 1.0  # 근음에 가중

    scored: List[Tuple[Key, float]] = []
    for tonic in range(12):
        for mode, profile in (("major", CHORD_MAJOR_PROFILE), ("minor", CHORD_MINOR_PROFILE)):
            key = Key(tonic, mode)
            score = sum(counts.get(pc, 0.0) * profile[(pc - tonic) % 12] for pc in counts)
            score += _structural_score(key, playable)
            scored.append((key, score))

    scored.sort(key=lambda kv: -kv[1])
    return scored


def _structural_score(key: Key, chords: Sequence[Chord]) -> float:
    """진행이 이 조성에 얼마나 "말이 되는가"."""
    diatonic = {(c.root, c.triad_quality()) for c in key.diatonic_triads()}
    diatonic |= {(c.root, c.triad_quality()) for c in key.diatonic_sevenths()}
    scale = set(key.scale)
    tonic_quality = "maj" if key.mode == "major" else "min"

    score = 0.0
    for chord in chords:
        if (chord.root, chord.triad_quality()) in diatonic:
            score += 1.5
        elif chord.root in scale:
            score += 0.3      # 근음은 맞고 품질만 다름 (차용/세컨더리 도미넌트)
        else:
            score -= 1.0

    # 종지: 마지막(그리고 첫) 코드가 으뜸화음이면 강한 단서
    if chords[-1].root == key.tonic and chords[-1].triad_quality() == tonic_quality:
        score += 3.0
    if chords[0].root == key.tonic and chords[0].triad_quality() == tonic_quality:
        score += 1.5

    # 도미넌트의 5도 하행 해결. 해결 지점이 으뜸음이면 결정적이다.
    for a, b in zip(chords, chords[1:]):
        if a.is_dominant and (a.root - b.root) % 12 == 7:
            score += 2.0 if b.root == key.tonic else (0.7 if b.root in scale else 0.0)
    return score
