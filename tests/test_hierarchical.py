"""단계별 인식기의 각 판정 단계 테스트.

오디오 없이 크로마 벡터를 직접 만들어 각 단계를 검증한다.
"""

import numpy as np
import pytest

from chordapp.analysis.hierarchical import HierarchicalRecognizer, _runs
from chordapp.theory.chord import Chord
from chordapp.theory.key import Key


def chroma_for(notes, weights=None):
    """음이름 목록으로 크로마 한 칸을 만든다."""
    column = np.zeros(12, dtype=np.float32)
    for i, name in enumerate(notes):
        pc = Chord.parse(name).root if len(name) <= 2 else None
        column[pc] = weights[i] if weights else 1.0
    return column


def quality_of(notes, weights=None, key="C", root="C"):
    rec = HierarchicalRecognizer()
    column = chroma_for(notes, weights)
    quality, _ = rec._quality_for(column, Chord.parse(root).root, Key.parse(key))
    return quality


# --- 2단계: 3화음 -----------------------------------------------------

def test_major_and_minor_triads():
    assert quality_of(["C", "E", "G"]) == "maj"
    assert quality_of(["C", "D#", "G"]) == "min"


def test_sus_needs_a_weak_third():
    # 4음이 3음보다 뚜렷하게 세야 sus로 본다
    assert quality_of(["C", "F", "G"], [1.0, 1.0, 1.0]) == "sus4"
    assert quality_of(["C", "E", "F", "G"], [1.0, 1.0, 0.5, 1.0]) == "maj"


def test_diminished_needs_a_clear_flat_five():
    """dim은 드물다. 완전5도를 확실히 이겨야 인정한다."""
    assert quality_of(["C", "D#", "F#"], [1.0, 1.0, 1.0]) == "dim"
    # 감5도가 완전5도보다 살짝 셀 뿐이면 그냥 단3화음
    assert quality_of(["C", "D#", "F#", "G"], [1.0, 1.0, 0.6, 0.5]) == "min"


def test_ambiguous_third_is_resolved_by_key():
    """3음이 붙어 있으면 크로마로 우기지 말고 조성을 따른다.

    이걸 안 하면 실제 녹음에서 DbmMaj7 같은 실존하지 않는 코드가 나온다.
    """
    ambiguous = ["C", "D#", "E", "G"]
    weights = [1.0, 0.80, 0.82, 1.0]      # 3음 두 후보가 거의 동률
    # C장조에서 D의 다이어토닉은 단3화음
    assert quality_of(ambiguous, weights, key="C", root="C") == "maj"
    assert quality_of(["D", "F", "F#", "A"], [1.0, 0.80, 0.82, 1.0],
                      key="C", root="D") == "min"


# --- 3단계: 7화음 -----------------------------------------------------

@pytest.mark.parametrize("notes,weights,expected", [
    (["C", "E", "G", "A#"], [1.0, 1.0, 1.0, 0.9], "7"),
    (["C", "E", "G", "B"], [1.0, 1.0, 1.0, 0.9], "maj7"),
    (["C", "D#", "G", "A#"], [1.0, 1.0, 1.0, 0.9], "min7"),
    (["C", "E", "G", "A"], [1.0, 1.0, 1.0, 0.9], "6"),
    (["C", "E", "G"], None, "maj"),                       # 7음 없음
])
def test_seventh_detection(notes, weights, expected):
    assert quality_of(notes, weights) == expected


def test_weak_seventh_is_ignored():
    """희미하게 새어 들어온 7음으로 7화음을 만들면 안 된다."""
    assert quality_of(["C", "E", "G", "B"], [1.0, 1.0, 1.0, 0.2]) == "maj"


# --- 4단계: 텐션 ------------------------------------------------------

def test_ninth_upgrades_only_on_seventh_chords():
    assert quality_of(["C", "D", "E", "G", "B"], [1.0, 0.95, 1.0, 1.0, 0.9]) == "maj9"
    # 7음이 없으면 9음이 있어도 텐션으로 올리지 않는다 (add9는 오탐이 많다)
    assert quality_of(["C", "D", "E", "G"], [1.0, 0.95, 1.0, 1.0]) == "maj"


def test_ninth_must_be_in_key():
    """조성 밖 음을 텐션으로 올리지 않는다. 대개 옆 코드가 새어 들어온 것이다."""
    voicing = (["C", "D", "E", "G", "A#"], [1.0, 0.95, 1.0, 1.0, 0.9])
    assert quality_of(*voicing, key="C") == "9"     # C장조에 D 있음
    assert quality_of(*voicing, key="Db") == "7"    # Db장조에 D 없음


# --- 보조 ------------------------------------------------------------

def test_runs_groups_consecutive_values():
    assert _runs(np.array([0, 0, 3, 3, 3, 7])) == [(0, 2), (2, 5), (5, 6)]
    assert _runs(np.array([5])) == [(0, 1)]
