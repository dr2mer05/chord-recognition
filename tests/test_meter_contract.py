"""추적기의 박자 정보가 코드 구간·API 자료·화면 격자까지 보존되는지 검증한다.

합성 음원과 통제된 추적 결과를 사용한다. 자동 박자 인식 성능 평가는 아니다.
"""

import numpy as np
import pytest
import soundfile as sf

from chordapp.analysis.beat_tracker import BeatGrid
from chordapp.analysis.pipeline import analyze_audio
from chordapp.analysis.recognizers import BeatPrediction, Recognizer
from chordapp.models import AnalysisResult
from chordapp.theory.chord import Chord
from chordapp.theory.key import Key
from chordapp.tools.synth import DEMO_TRACKS, render_track, write_wav


@pytest.fixture(scope="module")
def audio(tmp_path_factory):
    wav = tmp_path_factory.mktemp("meter-contract") / "synth.wav"
    signal, _ = render_track(DEMO_TRACKS[2], seed=7)
    write_wav(wav, signal)
    return wav


class ControlledTracker:
    name = "controlled-meter"
    available = True
    device = "cpu"

    def __init__(self, meter, phase=0):
        self.meter = meter
        self.phase = phase
        self.output = None

    def track(self, path):
        beats = np.arange(0, sf.info(path).duration, .5)
        self.output = BeatGrid(beats, 120.0, beats[self.phase::self.meter], self.meter, self.name)
        return self.output


class BarChanges(Recognizer):
    """구간의 마디·박 번호를 확인하기 위해 알려진 위치에서 코드를 바꾼다."""

    name = "controlled-chords"

    def recognize(self, feats):
        n = len(feats.beat_times)
        chords = [Chord.parse("C" if (i // feats.beats_per_bar) % 2 == 0 else "G")
                  for i in range(n)]
        return BeatPrediction(self.name, chords, np.ones(n))


@pytest.mark.parametrize("meter", [3, 4, 6])
def test_tracker_meter_reaches_segments_and_serialized_chart(audio, meter):
    tracker = ControlledTracker(meter)
    result = analyze_audio(str(audio), tracker=tracker, recognizers=[BarChanges()],
                           separate=False, detect_inversions=False, key_override=Key.parse("C"))

    # 추적기의 위치를 수정하지 않고 그 해석을 끝까지 전달해야 한다.
    np.testing.assert_allclose(result.beat_times, tracker.output.beat_times)
    np.testing.assert_allclose(result.downbeats, tracker.output.downbeat_times)
    assert result.beats_per_bar == meter
    assert all(seg.beat == 1 for seg in result.segments)
    assert [seg.bar for seg in result.segments] == list(range(1, len(result.segments) + 1))

    # 서버 응답·캐시 왕복 후에도 마디당 칸 수와 코드 전환 위치가 일치해야 한다.
    restored = AnalysisResult.from_dict(result.to_dict())
    payload = restored.to_dict()
    assert payload["beats_per_bar"] == meter
    for bar in payload["bars"]:
        assert bar["beats_per_bar"] == meter
        assert len(bar["slots"]) == meter
        assert [slot["beat"] for slot in bar["slots"]] == list(range(1, meter + 1))
    assert [b["slots"][0]["chord"] for b in payload["bars"][:4]] == ["C", "G", "C", "G"]


@pytest.mark.parametrize("meter", [3, 6])
def test_manual_meter_without_tracker_is_preserved(audio, meter):
    result = analyze_audio(str(audio), tracker=None, beats_per_bar=meter,
                           tempo_override=120, beat_offset=0, separate=False,
                           detect_inversions=False, recognizers=[BarChanges()],
                           key_override=Key.parse("C"))
    assert result.beats_per_bar == meter
    assert all(len(bar["slots"]) == meter for bar in result.bars())
    assert all(seg.beat == 1 for seg in result.segments)


@pytest.mark.parametrize("manual,automatic", [(3, 4), (4, 3), (6, 4)])
def test_explicit_meter_takes_precedence_over_tracker_meter(audio, manual, automatic):
    tracker = ControlledTracker(automatic)
    result = analyze_audio(str(audio), tracker=tracker, beats_per_bar=manual,
                           separate=False, detect_inversions=False, recognizers=[BarChanges()],
                           key_override=Key.parse("C"))
    assert result.beats_per_bar == manual
    np.testing.assert_allclose(result.beat_times, tracker.output.beat_times)
    np.testing.assert_allclose(np.diff(result.downbeats), manual * .5)
    assert all(seg.beat == 1 for seg in result.segments)
    assert all(len(bar["slots"]) == manual for bar in result.bars())


@pytest.mark.parametrize("manual,phase,shift", [(3, 1, 1), (4, 2, -1), (6, 1, 2)])
def test_manual_meter_uses_tracker_anchor_and_user_phase(audio, manual, phase, shift):
    tracker = ControlledTracker(4, phase=phase)
    result = analyze_audio(str(audio), tracker=tracker, beats_per_bar=manual,
                           downbeat_offset=shift, separate=False, detect_inversions=False,
                           recognizers=[BarChanges()], key_override=Key.parse("C"))
    assert result.downbeats[0] == pytest.approx(((phase + shift) % manual) * .5)
    np.testing.assert_allclose(np.diff(result.downbeats), manual * .5)
    np.testing.assert_allclose(result.beat_times, tracker.output.beat_times)


def test_service_roundtrip_distinguishes_auto_and_explicit_four_beats(audio, tmp_path, monkeypatch):
    from chordapp.service.audio_sources import AudioSourceStore
    from chordapp.service.cache import AnalysisCache
    from chordapp.service.corrections import CorrectionStore
    from chordapp.service.pipeline import ChordService

    tracker = ControlledTracker(3)
    monkeypatch.setattr("chordapp.analysis.audio.default_tracker", lambda: tracker)
    monkeypatch.setattr("chordapp.analysis.beat_tracker.default_tracker", lambda: tracker)
    service = ChordService(cache=AnalysisCache(tmp_path / "cache.sqlite3"),
                           corrections=CorrectionStore(tmp_path / "corrections.sqlite3"),
                           audio_sources=AudioSourceStore(tmp_path / "sources.sqlite3", tmp_path / "assets"))
    automatic = service.analyze_file(audio, separate=False, detect_inversions=False, key_override="C")
    assert automatic.beats_per_bar == 3
    service.corrections.set_meter(automatic.source_key, beats_per_bar=4)
    manual = service.analyze_file(audio, separate=False, detect_inversions=False, key_override="C")
    assert manual.beats_per_bar == 4 and not manual.cached
    np.testing.assert_allclose(np.diff(manual.downbeats), 2.0)
    service.corrections.set_meter(automatic.source_key, beats_per_bar=None)
    restored = service.analyze_file(audio, separate=False, detect_inversions=False, key_override="C")
    assert restored.beats_per_bar == 3 and restored.cached
