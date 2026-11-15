"""화음 종류 변화와 짧은 화성 리듬의 구조적 회귀를 검증한다.

정답을 직접 지정한 특징/인식기 시험이며 실제 음원 정확도 평가는 아니다.
"""

from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from chordapp.analysis.hierarchical import HierarchicalRecognizer
from chordapp.analysis.pipeline import analyze_audio
from chordapp.analysis.recognizers import BeatPrediction, Recognizer
from chordapp.theory.chord import Chord
from chordapp.theory.key import Key


@pytest.mark.parametrize("root", range(12))
@pytest.mark.parametrize("qualities", [
    ("maj", "7", "maj7", "6"),
    ("min", "min7", "minmaj7", "min6"),
    ("maj", "sus4", "sus2", "7"),
])
def test_same_root_quality_changes_survive_in_all_keys(root, qualities):
    expected = [Chord(root, quality) for quality in qualities for _ in range(2)]
    chroma = np.zeros((12, len(expected)), dtype=np.float32)
    bass = np.zeros_like(chroma)
    for i, chord in enumerate(expected):
        chroma[chord.pitch_classes(), i] = 1
        bass[root, i] = 1
    features = SimpleNamespace(beat_chroma=chroma, bass_chroma=bass,
        key=Key(root, 'major'), beats_per_bar=4, downbeat_indices=np.array([0, 4]))
    current = HierarchicalRecognizer().recognize(features)
    assert current.chords == expected
    legacy = HierarchicalRecognizer(quality_mode='legacy').recognize(features)
    assert len(set(legacy.chords)) == 1
    assert legacy.chords != expected


def test_zero_energy_quality_has_finite_confidence_without_runtime_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        _, confidence = HierarchicalRecognizer()._quality_for(np.zeros(12), 0, Key.parse('C'))
    assert np.isfinite(confidence)


class KnownChords(Recognizer):
    name = 'independent-known-chords'

    def __init__(self, names):
        self.chords = [Chord.parse(name) for name in names]

    def recognize(self, features):
        return BeatPrediction(self.name, self.chords, np.ones(len(self.chords)))


def run_known(monkeypatch, names, **options):
    n = len(names)
    features = SimpleNamespace(beat_times=np.arange(n) * .5, duration=n * .5,
        tempo=120., beats_per_bar=4, downbeat_indices=np.arange(0, n, 4),
        key=Key.parse('C'), key_confidence=1., separated=False, bass_chroma=None,
        beat_source='controlled', beat_tracker_requested='controlled',
        beat_tracker_fallback=None, meter_source='manual')
    monkeypatch.setattr('chordapp.analysis.pipeline.extract_features', lambda *a, **k: features)
    return analyze_audio('unused.wav', recognizers=[KnownChords(names)], separate=False,
        detect_inversions=False, key_override=features.key, **options)


def test_default_preserves_one_beat_chords(monkeypatch):
    names = ['C', 'G', 'Am', 'F'] * 2
    result = run_known(monkeypatch, names)
    assert [s.chord.name() for s in result.segments] == names
    assert [s.start for s in result.segments] == list(np.arange(8) * .5)
    legacy = run_known(monkeypatch, names, smooth_chords=True)
    assert [s.chord.name() for s in legacy.segments] != names


def test_default_preserves_change_before_bar_boundary(monkeypatch):
    names = ['C'] * 3 + ['G'] * 5
    result = run_known(monkeypatch, names)
    assert result.segments[1].start == 1.5
    assert result.segments[0].end == 1.5
    legacy = run_known(monkeypatch, names, snap_bars=True)
    assert legacy.segments[1].start == 2.
