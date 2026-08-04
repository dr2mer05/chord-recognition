import pytest

from chordapp.models import ChordSegment
from chordapp.service.corrections import (
    Correction,
    CorrectionStore,
    apply_corrections,
    to_lab,
)
from chordapp.theory.chord import Chord

KEY = "youtube:abc12345678"


def seg(start, end, chord, conf=0.4):
    return ChordSegment(start, end, Chord.parse(chord), confidence=conf, agreement=conf)


@pytest.fixture
def store(tmp_path):
    return CorrectionStore(tmp_path / "c.sqlite3")


# --- 저장소 -----------------------------------------------------------

def test_add_and_list(store):
    store.record_source(KEY, "테스트 곡", "/x.wav", 30.0)
    store.add(KEY, 0.0, 2.0, "Am7", original="Am")
    store.add(KEY, 4.0, 6.0, "F")

    fixes = store.corrections(KEY)
    assert [f.chord for f in fixes] == ["Am7", "F"]
    assert fixes[0].original == "Am"


def test_rejects_unparseable_chord(store):
    store.record_source(KEY)
    with pytest.raises(ValueError):
        store.add(KEY, 0.0, 1.0, "Zx9")
    assert store.corrections(KEY) == []


def test_same_span_is_replaced_not_duplicated(store):
    store.record_source(KEY)
    store.add(KEY, 0.0, 2.0, "Am")
    store.add(KEY, 0.0, 2.0, "Am7")
    assert [f.chord for f in store.corrections(KEY)] == ["Am7"]


def test_remove_and_clear(store):
    store.record_source(KEY)
    store.add(KEY, 0.0, 2.0, "Am")
    store.add(KEY, 2.0, 4.0, "F")
    assert store.remove(KEY, 0.0) == 1
    assert len(store.corrections(KEY)) == 1
    assert store.clear(KEY) == 1
    assert store.corrections(KEY) == []


def test_verified_flag_gates_the_benchmark_set(store):
    """틀린 것만 고친 표본은 편향돼 있다. 검수 완료한 곡만 벤치마크에 쓴다."""
    store.record_source(KEY, "곡 A")
    store.record_source("file:zzz", "곡 B")
    assert store.sources(verified_only=True) == []

    store.set_verified(KEY)
    verified = store.sources(verified_only=True)
    assert [s["source_key"] for s in verified] == [KEY]
    assert store.stats()["verified"] == 1


# --- 템포 / 박자 지정 --------------------------------------------------

def test_tempo_override_roundtrip(store):
    store.record_source(KEY)
    assert store.tempo(KEY) is None
    store.set_tempo(KEY, 69.0)
    assert store.tempo(KEY) == 69.0
    store.set_tempo(KEY, None)          # 자동 추정으로 복귀
    assert store.tempo(KEY) is None


@pytest.mark.parametrize("bpm", [0, 19, 301, 1000])
def test_tempo_override_rejects_absurd_values(store, bpm):
    store.record_source(KEY)
    with pytest.raises(ValueError):
        store.set_tempo(KEY, bpm)


def test_tap_tempo_stores_the_grid_offset(store):
    """탭 템포는 템포와 첫 박 시각을 함께 남긴다.

    오디오만으로는 4/4의 4분음표와 12/8의 점4분음표를 구별할 수 없어서
    (온셋 간격이 두 해석에 똑같이 들어맞는다) 사람이 찍은 격자가 필요하다.
    """
    store.record_source(KEY)
    store.set_tempo(KEY, 46.5, beat_offset=2.0)
    assert store.tempo(KEY) == 46.5
    assert store.beat_offset(KEY) == 2.0

    # 템포만 지정하면 격자는 다시 추적기가 잡는다
    store.set_tempo(KEY, 69.0)
    assert store.beat_offset(KEY) is None

    store.set_tempo(KEY, None, beat_offset=2.0)   # 템포 없이 오프셋만은 무의미
    assert store.beat_offset(KEY) is None


def test_meter_override_roundtrip(store):
    """박자표 자동 판별은 신뢰할 수 없어서 지정받는다."""
    store.record_source(KEY)
    assert store.beats_per_bar(KEY) is None
    assert store.downbeat_offset(KEY) == 0

    store.set_meter(KEY, beats_per_bar=6)
    assert store.beats_per_bar(KEY) == 6

    store.set_meter(KEY, beats_per_bar=6, downbeat_offset=2)
    assert store.downbeat_offset(KEY) == 2

    store.set_meter(KEY, beats_per_bar=None, downbeat_offset=0)   # 자동으로 복귀
    assert store.beats_per_bar(KEY) is None
    assert store.downbeat_offset(KEY) == 0


def test_downbeat_offset_wraps_within_the_bar(store):
    store.record_source(KEY)
    store.set_meter(KEY, beats_per_bar=4, downbeat_offset=5)
    assert store.downbeat_offset(KEY) == 1        # 5 % 4
    store.set_meter(KEY, beats_per_bar=4, downbeat_offset=-1)
    assert store.downbeat_offset(KEY) == 3        # 파이썬 모듈로


@pytest.mark.parametrize("bpb", [1, 0, 13, 20])
def test_meter_rejects_implausible_values(store, bpb):
    store.record_source(KEY)
    with pytest.raises(ValueError):
        store.set_meter(KEY, beats_per_bar=bpb)


# --- 교정 적용 --------------------------------------------------------

def test_apply_replaces_matching_segment():
    segments = [seg(0, 2, "C"), seg(2, 4, "G"), seg(4, 6, "Am")]
    fixed = apply_corrections(segments, [Correction(2.0, 4.0, "G7")])
    assert [s.chord.name() for s in fixed] == ["C", "G7", "Am"]


def test_correction_marks_full_confidence():
    """사람이 고친 구간은 인식기보다 정확하다."""
    fixed = apply_corrections([seg(0, 2, "C", conf=0.3)], [Correction(0.0, 2.0, "Cmaj7")])
    assert fixed[0].confidence == 1.0
    assert fixed[0].grade == "high"


def test_partial_overlap_splits_the_segment():
    """교정 구간이 원본 경계와 안 맞아도 잘라서 덮는다."""
    fixed = apply_corrections([seg(0, 4, "C")], [Correction(1.0, 3.0, "F")])
    assert [(s.start, s.end, s.chord.name()) for s in fixed] == [
        (0.0, 1.0, "C"), (1.0, 3.0, "F"), (3.0, 4.0, "C"),
    ]


def test_correction_spanning_many_segments():
    segments = [seg(0, 2, "C"), seg(2, 4, "G"), seg(4, 6, "Am")]
    fixed = apply_corrections(segments, [Correction(0.0, 6.0, "F")])
    assert [s.chord.name() for s in fixed] == ["F"]      # 인접 동일 코드는 병합
    assert (fixed[0].start, fixed[0].end) == (0.0, 6.0)


def test_no_corrections_returns_original():
    segments = [seg(0, 2, "C")]
    assert [s.chord.name() for s in apply_corrections(segments, [])] == ["C"]


def test_original_segments_are_not_mutated():
    """원본은 캐시에 남아야 한다. 덮어쓰면 정확도 측정이 순환논법이 된다."""
    segments = [seg(0, 2, "C")]
    apply_corrections(segments, [Correction(0.0, 2.0, "Cmaj7")])
    assert segments[0].chord.name() == "C"


# --- 내보내기 ---------------------------------------------------------

def test_to_lab_roundtrips_through_the_evaluator():
    from chordapp.evaluation import load_lab

    segments = [seg(0, 2, "C"), seg(2, 4, "Am7")]
    text = to_lab(segments)
    assert text.splitlines()[0].split("\t")[2] == "C"

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.lab"
        path.write_text(text, encoding="utf-8")
        assert [c.name() for _, _, c in load_lab(path)] == ["C", "Am7"]
