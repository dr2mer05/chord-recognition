"""피치 클래스와 음이름 표기.

화성학 엔진 전체가 이 모듈의 정수 피치 클래스(0=C .. 11=B)를 쓴다.
표기(스펠링)는 조성에 따라 달라지므로 계산과 분리해 둔다.
"""

from __future__ import annotations

from typing import Optional

SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# 조표에 플랫이 붙는 조성들. 이 조성에서는 Bb/Eb 계열로 표기한다.
_FLAT_MAJOR_TONICS = {5, 10, 3, 8, 1, 6}       # F, Bb, Eb, Ab, Db, Gb
_FLAT_MINOR_TONICS = {2, 7, 0, 5, 10, 3}       # d, g, c, f, bb, eb


def parse_note(name: str) -> int:
    """'C', 'F#', 'Bb', 'Ebb' 같은 음이름을 피치 클래스로."""
    name = name.strip()
    if not name:
        raise ValueError("빈 음이름")
    letter = name[0].upper()
    if letter not in _LETTER_PC:
        raise ValueError(f"음이름이 아님: {name!r}")
    pc = _LETTER_PC[letter]
    for ch in name[1:]:
        if ch in ("#", "♯"):
            pc += 1
        elif ch in ("b", "♭"):
            pc -= 1
        else:
            raise ValueError(f"임시표가 아님: {ch!r} in {name!r}")
    return pc % 12


def note_name(pc: int, prefer_flat: bool = False) -> str:
    pc %= 12
    return FLAT_NAMES[pc] if prefer_flat else SHARP_NAMES[pc]


def prefers_flat(tonic_pc: Optional[int], mode: str = "major") -> bool:
    """해당 조성이 플랫 표기를 쓰는지. 조성을 모르면 샤프를 기본으로."""
    if tonic_pc is None:
        return False
    tonic_pc %= 12
    if mode == "minor":
        return tonic_pc in _FLAT_MINOR_TONICS
    return tonic_pc in _FLAT_MAJOR_TONICS


def interval_name(semitones: int) -> str:
    names = {
        0: "1도", 1: "단2도", 2: "장2도", 3: "단3도", 4: "장3도", 5: "완전4도",
        6: "트라이톤", 7: "완전5도", 8: "단6도", 9: "장6도", 10: "단7도", 11: "장7도",
    }
    return names[semitones % 12]
