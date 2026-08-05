"""오디오 분석 통합 테스트. librosa가 있어야 돌아간다."""

import numpy as np
import pytest

from chordapp.evaluation import evaluate, evaluate_result, load_lab, segments_to_intervals
from chordapp.models import ChordSegment
from chordapp.theory.chord import Chord

librosa = pytest.importorskip("librosa")

from chordapp.analysis import analyze_audio  # noqa: E402
from chordapp.tools.synth import DEMO_TRACKS, render_track, write_lab, write_wav  # noqa: E402


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    """가장 짧은 데모 하나만 렌더링해서 전 구간을 통과시킨다."""
    track = next(t for t in DEMO_TRACKS if t.name == "minor_loop")
    audio, labels = render_track(track, seed=7)
    out = tmp_path_factory.mktemp("demo")
    wav = out / "minor_loop.wav"
    write_wav(wav, audio)
    write_lab(wav.with_suffix(".lab"), labels)
    return wav


def test_analyze_recovers_the_progression(demo):
    result = analyze_audio(str(demo), separate=False)
    assert result.origin == "audio"
    assert result.tempo == pytest.approx(112, abs=6)
    assert result.key.name() in ("Am", "C")  # 나란한조끼리는 구분이 어렵다
    assert len(result.segments) >= 8

    report = evaluate_result(result, demo.with_suffix(".lab"))
    assert report.scores["triad"] > 0.85

    # 코드 순서 자체는 정확히 나와야 한다
    names = [s.chord.name() for s in result.segments]
    assert names[:4] == ["Am", "F", "C", "G"]


def test_segments_are_contiguous_and_ordered(demo):
    result = analyze_audio(str(demo), separate=False)
    for a, b in zip(result.segments, result.segments[1:]):
        assert a.end == pytest.approx(b.start, abs=1e-6)
        assert a.chord != b.chord
    assert result.segments[0].start == pytest.approx(0.0, abs=1e-6)


def test_beat_grid_covers_the_start_without_a_fake_beat(demo):
    """도입부가 빠지면 안 되지만, 격자에 0.0을 억지로 끼워 넣어도 안 된다.

    0.0을 끼워 넣으면 그 한 박이 격자 위상을 밀어서 다운비트 추정이 통째로
    어긋난다. 대신 첫 코드 구간을 0까지 늘려서 도입부를 덮는다.
    """
    result = analyze_audio(str(demo), separate=False)
    period = float(np.median(np.diff(result.beat_times)))

    assert result.beat_times[0] < period          # 첫 박이 한 박 안에 있다
    assert result.segments[0].start == pytest.approx(0.0, abs=1e-6)
    assert all(b < a for a, b in zip(result.beat_times[1:], result.beat_times))

    gaps = np.diff(result.beat_times)
    assert gaps.min() > period * 0.5, "격자에 비정상적으로 짧은 박이 있다"


def test_key_override_is_respected(demo):
    from chordapp.theory.key import Key

    result = analyze_audio(str(demo), key_override=Key.parse("Eb"), separate=False)
    assert result.key.name() == "Eb"


def test_tapped_grid_is_uniform_and_anchored(demo):
    """탭 템포로 찍으면 그 격자를 그대로 쓴다.

    오디오만으로는 4/4의 4분음표와 12/8의 점4분음표를 구별할 수 없다
    (실측: 온셋 간격이 두 해석에 똑같이 들어맞았다). 사람이 확정한 격자를
    추적기가 다시 흔들면 안 된다.
    """
    from chordapp.analysis.audio import HOP, SR, extract_features

    feats = extract_features(str(demo), tempo_override=46.5, beat_offset=2.0)
    assert feats.tempo == pytest.approx(46.5)

    # 프레임 양자화(HOP/SR ≈ 23ms) 말고는 완전 균등해야 한다
    tolerance = 1.5 * HOP / SR
    gaps = np.diff(feats.beat_times)
    assert np.abs(gaps - 60.0 / 46.5).max() < tolerance

    downbeat = feats.beat_times[int(np.atleast_1d(feats.downbeat_indices).min())]
    assert downbeat == pytest.approx(2.0, abs=tolerance)


def test_downbeat_phase_is_correct(demo):
    """마디 첫 박을 맞혀야 한다.

    마디 번호·마디 스냅·마디별 표기·박자 기반 전이확률이 전부 여기에 의존한다.
    한때 5곡 전부 틀렸는데(스네어 백비트를 마디 첫 박으로 오인) 티가 안 났다.
    """
    import numpy as np

    from chordapp.analysis.audio import extract_features
    from chordapp.tools.synth import DEMO_TRACKS

    track = next(t for t in DEMO_TRACKS if t.name == "minor_loop")
    feats = extract_features(str(demo))

    beat_dur = 60.0 / track.bpm
    # 격자가 몇 번째 박에서 시작하는지에 따라 정답 위상이 달라진다
    grid_offset = int(round(feats.beat_times[0] / beat_dur)) % track.beats_per_bar
    expected = (track.beats_per_bar - grid_offset) % track.beats_per_bar

    assert int(np.atleast_1d(feats.downbeat_indices).min()) == expected


def test_bars_are_a_beat_grid(demo):
    """마디마다 박 수만큼 칸이 있어야 코드가 몇 박째인지 읽을 수 있다."""
    result = analyze_audio(str(demo), separate=False)
    bars = result.bars()
    assert bars, "마디가 하나도 안 나오면 안 된다"

    for bar in bars:
        assert len(bar["slots"]) == result.beats_per_bar
        assert [s["beat"] for s in bar["slots"]] == list(range(1, result.beats_per_bar + 1))
        assert bar["end"] > bar["start"]

    numbers = [b["bar"] for b in bars]
    assert numbers == sorted(numbers) and len(set(numbers)) == len(numbers)

    # minor_loop은 마디마다 코드가 하나씩 바뀐다 → 1박에 코드, 나머지는 이어짐
    assert [b["slots"][0]["chord"] for b in bars[:4]] == ["Am", "F", "C", "G"]
    for bar in bars[:4]:
        assert [s["chord"] for s in bar["slots"][1:]] == [None] * (result.beats_per_bar - 1)


def test_bar_slots_show_where_a_chord_lands():
    """반 마디마다 바뀌면 1박과 3박에 코드가 찍혀야 한다."""
    from chordapp.models import AnalysisResult
    from chordapp.theory.key import Key

    result = AnalysisResult(
        source="t", duration=4.0, tempo=120.0, key=Key.parse("C"), key_confidence=1.0,
        segments=[
            ChordSegment(0.0, 2.0, Chord.parse("C"), 1.0, 1.0),
            ChordSegment(2.0, 4.0, Chord.parse("Am"), 1.0, 1.0),
        ],
        downbeats=[0.0, 4.0], beats_per_bar=4,
    )
    slots = result.bars()[0]["slots"]
    assert [s["chord"] for s in slots] == ["C", None, "Am", None]

    # 칸마다 자기 시간 범위를 들고 있어야 그 박에서 코드를 갈라낼 수 있다
    assert [s["start"] for s in slots] == [0.0, 1.0, 2.0, 3.0]
    assert [s["end"] for s in slots] == [1.0, 2.0, 3.0, 4.0]
    # 새 코드를 넣으면 원래 코드가 끝나던 데까지 적용된다
    assert [s["segment_end"] for s in slots] == [2.0, 2.0, 4.0, 4.0]


def test_editing_a_middle_slot_splits_the_segment():
    """한 코드로 묶인 구간을 중간 박에서 갈라낼 수 있어야 한다."""
    from chordapp.service.corrections import Correction, apply_corrections

    segments = [ChordSegment(0.0, 4.0, Chord.parse("Am"), 1.0, 1.0)]
    # 3박(2.0초)째 칸에 F를 넣으면 거기서부터 구간 끝까지 F
    split = apply_corrections(segments, [Correction(2.0, 4.0, "F")])
    assert [(s.start, s.end, s.chord.name()) for s in split] == [
        (0.0, 2.0, "Am"), (2.0, 4.0, "F"),
    ]


# --- 평가 코드 자체 ---------------------------------------------------

def test_evaluate_perfect_match():
    reference = [(0.0, 2.0, Chord.parse("C")), (2.0, 4.0, Chord.parse("G"))]
    report = evaluate(reference, reference, name="t")
    assert report.scores["triad"] == pytest.approx(1.0)
    assert report.confusions == []


def test_evaluate_penalises_wrong_quality_only_at_seventh_level():
    reference = [(0.0, 2.0, Chord.parse("Dm7"))]
    estimate = [(0.0, 2.0, Chord.parse("Dm"))]
    report = evaluate(reference, estimate)
    assert report.scores["root"] == pytest.approx(1.0)
    assert report.scores["triad"] == pytest.approx(1.0)
    assert report.scores["seventh"] == pytest.approx(0.0)


def test_lab_roundtrip(tmp_path):
    path = tmp_path / "x.lab"
    write_lab(path, [(0.0, 1.5, "C"), (1.5, 3.0, "Am7"), (3.0, 4.0, "N")])
    intervals = load_lab(path)
    assert [c.name() if c else None for _, _, c in intervals] == ["C", "Am7", None]
