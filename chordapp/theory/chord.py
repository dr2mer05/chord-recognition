"""코드(화음) 표현: 파싱 / 표기 / 구성음.

코드 인식기와 화성학 엔진이 주고받는 공통 자료형.
품질(quality)은 정규화된 내부 이름을 쓰고, 표기용 심볼은 따로 둔다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

from .notes import note_name, parse_note

# 내부 품질 이름 -> 근음 기준 반음 간격
QUALITY_INTERVALS: Dict[str, Tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "5": (0, 7),
    "6": (0, 4, 7, 9),
    "min6": (0, 3, 7, 9),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "minmaj7": (0, 3, 7, 11),
    "dim7": (0, 3, 6, 9),
    "hdim7": (0, 3, 6, 10),
    "aug7": (0, 4, 8, 10),
    "7sus4": (0, 5, 7, 10),
    "add9": (0, 4, 7, 2),
    "minadd9": (0, 3, 7, 2),
    "9": (0, 4, 7, 10, 2),
    "maj9": (0, 4, 7, 11, 2),
    "min9": (0, 3, 7, 10, 2),
    "69": (0, 4, 7, 9, 2),
    "7b9": (0, 4, 7, 10, 1),
    "7#9": (0, 4, 7, 10, 3),
    "7#5": (0, 4, 8, 10),
    "7b5": (0, 4, 6, 10),
    "7#11": (0, 4, 7, 10, 6),
    "13": (0, 4, 7, 10, 2, 9),
    "min11": (0, 3, 7, 10, 2, 5),
    "maj13": (0, 4, 7, 11, 2, 9),
    "N": (),  # no chord (무음/비화성 구간)
}

# 내부 이름 -> 화면 표기 심볼
QUALITY_SYMBOL: Dict[str, str] = {
    "maj": "", "min": "m", "dim": "dim", "aug": "aug",
    "sus2": "sus2", "sus4": "sus4", "5": "5",
    "6": "6", "min6": "m6",
    "7": "7", "maj7": "maj7", "min7": "m7", "minmaj7": "mMaj7",
    "dim7": "dim7", "hdim7": "m7b5", "aug7": "7#5", "7sus4": "7sus4",
    "add9": "add9", "minadd9": "madd9",
    "9": "9", "maj9": "maj9", "min9": "m9", "69": "6/9",
    "7b9": "7b9", "7#9": "7#9", "7#5": "7#5", "7b5": "7b5", "7#11": "7#11",
    "13": "13", "min11": "m11", "maj13": "maj13",
    "N": "N.C.",
}

# 입력 문자열 별칭 -> 내부 이름 (긴 것부터 매칭해야 하므로 정렬해서 사용)
QUALITY_ALIASES: Dict[str, str] = {
    "": "maj", "maj": "maj", "M": "maj", "major": "maj",
    "m": "min", "min": "min", "-": "min", "minor": "min",
    "dim": "dim", "o": "dim", "°": "dim",
    "aug": "aug", "+": "aug",
    "sus2": "sus2", "sus4": "sus4", "sus": "sus4",
    "5": "5", "no3": "5",
    "6": "6", "m6": "min6", "min6": "min6", "-6": "min6",
    "7": "7", "dom7": "7",
    "maj7": "maj7", "M7": "maj7", "ma7": "maj7", "△7": "maj7", "△": "maj7",
    "m7": "min7", "min7": "min7", "-7": "min7",
    "mMaj7": "minmaj7", "mM7": "minmaj7", "minmaj7": "minmaj7", "m△7": "minmaj7",
    "dim7": "dim7", "o7": "dim7", "°7": "dim7",
    "m7b5": "hdim7", "hdim7": "hdim7", "ø7": "hdim7", "ø": "hdim7", "min7b5": "hdim7",
    "7#5": "7#5", "7+5": "7#5", "aug7": "7#5", "7alt": "7#9",
    "7b5": "7b5",
    "7sus4": "7sus4", "7sus": "7sus4",
    "add9": "add9", "madd9": "minadd9", "m(add9)": "minadd9", "add2": "add9",
    "9": "9", "maj9": "maj9", "M9": "maj9", "m9": "min9", "min9": "min9",
    "6/9": "69", "69": "69",
    "7b9": "7b9", "7#9": "7#9", "7#11": "7#11",
    "13": "13", "m11": "min11", "min11": "min11", "11": "13", "maj13": "maj13",
    "N": "N", "N.C.": "N", "NC": "N",
}

_CHORD_RE = re.compile(
    r"^\s*(?P<root>[A-Ga-g](?:#|b|♯|♭)*|N\.?C\.?|N)"
    r"(?P<quality>[^/]*)"
    r"(?:/(?P<bass>[A-Ga-g](?:#|b|♯|♭)*))?\s*$"
)

# MIREX 형식: 'C:maj', 'A:min7/b3', 'F:maj(9)', 'N', 'X'
# autochord/Chordino 출력과 McGill Billboard·Isophonics의 .lab이 모두 이 형식이다.
_MIREX_RE = re.compile(
    r"^\s*(?P<root>[A-Ga-g](?:#|b|♯|♭)*|N|X)"
    r"(?::(?P<quality>[^/]*))?"
    r"(?:/(?P<bass>[^/\s]+))?\s*$"
)

# MIREX 음정 표기 -> 근음 기준 반음
_MIREX_DEGREES: Dict[str, int] = {
    "1": 0, "b2": 1, "2": 2, "#2": 3, "b3": 3, "3": 4, "b4": 4, "4": 5,
    "#4": 6, "b5": 6, "5": 7, "#5": 8, "b6": 8, "6": 9, "bb7": 9,
    "b7": 10, "7": 11, "b9": 1, "9": 2, "#9": 3, "11": 5, "#11": 6, "13": 9,
}

# MIREX 축약형 -> 내부 품질 이름 (별칭에 없는 것만)
_MIREX_SHORTHAND: Dict[str, str] = {
    "maj6": "6", "min6": "min6", "maj9": "maj9", "min9": "min9",
    "hdim7": "hdim7", "minmaj7": "minmaj7", "sus4": "sus4", "sus2": "sus2",
    "": "maj",
}

# 재즈/팝에서 실제로 기능이 같은 품질 묶음. 대체 코드 규칙이 이 분류를 참조한다.
DOMINANT_QUALITIES = {"7", "9", "13", "7b9", "7#9", "7#5", "7b5", "7#11", "7sus4", "aug7"}
MAJORISH_QUALITIES = {"maj", "maj7", "maj9", "6", "69", "add9", "maj13"}
MINORISH_QUALITIES = {"min", "min7", "min9", "min6", "minadd9", "min11", "minmaj7"}
DIMINISHED_QUALITIES = {"dim", "dim7", "hdim7"}
SUS_QUALITIES = {"sus2", "sus4", "7sus4"}


@dataclass(frozen=True)
class Chord:
    """근음 피치 클래스 + 품질 + (선택) 베이스 피치 클래스."""

    root: int
    quality: str = "maj"
    bass: Optional[int] = None

    def __post_init__(self) -> None:
        if self.quality not in QUALITY_INTERVALS:
            raise ValueError(f"알 수 없는 코드 품질: {self.quality!r}")
        object.__setattr__(self, "root", self.root % 12)
        if self.bass is not None:
            object.__setattr__(self, "bass", self.bass % 12)

    # --- 생성 ---------------------------------------------------------
    @classmethod
    def parse(cls, text: str) -> "Chord":
        """'Am7', 'C/E', 'F#m7b5', 'N.C.', 'C:maj', 'A:min7/b3' 등을 파싱."""
        # N = 무화음, X = 판정 불가. .lab 정답 파일에 둘 다 나온다.
        if text.strip().upper().replace(".", "") in ("N", "NC", "X"):
            return cls(0, "N")
        if ":" in text:
            return cls.parse_mirex(text)
        m = _CHORD_RE.match(text)
        if not m:
            raise ValueError(f"코드 문자열을 해석할 수 없음: {text!r}")
        root_txt = m.group("root")
        if root_txt.upper().replace(".", "") in ("N", "NC"):
            return cls(0, "N")
        root = parse_note(root_txt)
        qual_txt = (m.group("quality") or "").strip()
        qual_txt = qual_txt.replace("(", "").replace(")", "")
        if qual_txt not in QUALITY_ALIASES:
            # 'Cmaj7(9)' 처럼 꼬리가 붙은 경우 가장 긴 접두 별칭으로 근사
            best = None
            for alias in sorted(QUALITY_ALIASES, key=len, reverse=True):
                if alias and qual_txt.startswith(alias):
                    best = alias
                    break
            if best is None:
                raise ValueError(f"알 수 없는 코드 품질: {qual_txt!r} in {text!r}")
            qual_txt = best
        bass_txt = m.group("bass")
        bass = parse_note(bass_txt) if bass_txt else None
        return cls(root, QUALITY_ALIASES[qual_txt], bass)

    @classmethod
    def parse_mirex(cls, text: str) -> "Chord":
        """MIREX 축약 표기를 파싱.

        `근음:축약형(추가음)/베이스` 형태로, 베이스가 음이름이 아니라 근음 기준
        음정(`/b3`, `/5`)으로 온다. autochord·Chordino 출력과 McGill Billboard·
        Isophonics의 정답 .lab이 전부 이 형식이라, 앙상블과 벤치마크 양쪽에 필요하다.
        """
        m = _MIREX_RE.match(text)
        if not m:
            raise ValueError(f"MIREX 코드 라벨이 아님: {text!r}")

        root_txt = m.group("root")
        if root_txt in ("N", "X"):   # 무화음 / 판정 불가
            return cls(0, "N")
        root = parse_note(root_txt)

        quality_txt = (m.group("quality") or "").strip()
        # 'maj(9)' 처럼 붙는 추가음 표기는 축약형으로 근사한다.
        # 정확히 반영하려면 어휘를 크게 늘려야 하는데 인식 정확도 이득이 없다.
        quality_txt = re.sub(r"\(.*?\)", "", quality_txt).strip()
        if quality_txt in _MIREX_SHORTHAND:
            quality = _MIREX_SHORTHAND[quality_txt]
        elif quality_txt in QUALITY_ALIASES:
            quality = QUALITY_ALIASES[quality_txt]
        else:
            raise ValueError(f"알 수 없는 MIREX 축약형: {quality_txt!r} in {text!r}")

        bass_txt = m.group("bass")
        bass: Optional[int] = None
        if bass_txt:
            if bass_txt in _MIREX_DEGREES:
                bass = (root + _MIREX_DEGREES[bass_txt]) % 12
            else:
                bass = parse_note(bass_txt)   # 음이름으로 온 경우도 허용
        return cls(root, quality, bass)

    def with_bass(self, bass: Optional[int]) -> "Chord":
        if bass is not None and bass % 12 == self.root:
            bass = None
        return replace(self, bass=bass)

    def with_quality(self, quality: str) -> "Chord":
        return replace(self, quality=quality)

    # --- 조회 ---------------------------------------------------------
    @property
    def is_no_chord(self) -> bool:
        return self.quality == "N"

    def pitch_classes(self) -> Tuple[int, ...]:
        pcs = tuple(sorted({(self.root + i) % 12 for i in QUALITY_INTERVALS[self.quality]}))
        if self.bass is not None:
            pcs = tuple(sorted(set(pcs) | {self.bass}))
        return pcs

    def triad_quality(self) -> str:
        """3화음 골격만 뽑는다. 대체 규칙에서 기능 판정에 쓴다."""
        q = self.quality
        if q in MAJORISH_QUALITIES:
            return "maj"
        if q in MINORISH_QUALITIES:
            return "min"
        if q in DOMINANT_QUALITIES:
            return "dom"
        if q == "dim7" or q == "dim":
            return "dim"
        if q == "hdim7":
            return "hdim"
        if q in SUS_QUALITIES:
            return "sus"
        if q == "aug":
            return "aug"
        return "maj"

    @property
    def is_dominant(self) -> bool:
        return self.quality in DOMINANT_QUALITIES

    def common_tones(self, other: "Chord") -> int:
        return len(set(self.pitch_classes()) & set(other.pitch_classes()))

    # --- 표기 ---------------------------------------------------------
    def name(self, prefer_flat: bool = False) -> str:
        if self.is_no_chord:
            return "N.C."
        text = note_name(self.root, prefer_flat) + QUALITY_SYMBOL[self.quality]
        if self.bass is not None:
            text += "/" + note_name(self.bass, prefer_flat)
        return text

    def __str__(self) -> str:  # pragma: no cover - 디버깅 편의용
        return self.name()


def simplify_to_triad(chord: Chord) -> Chord:
    """텐션을 걷어낸 3화음. 인식 결과 비교(정확도 평가)의 기준."""
    tq = chord.triad_quality()
    mapping = {"maj": "maj", "min": "min", "dom": "maj", "dim": "dim",
               "hdim": "dim", "sus": "maj", "aug": "aug"}
    return Chord(chord.root, mapping.get(tq, "maj"))
