"""모델 출력 보존·오류 폴백·분석 범위 계약. 모의 모델이며 실제 모델 성능 시험은 아니다."""
import numpy as np
import pytest

from chordapp.analysis.beat_tracker import BeatGrid, BeatThisTracker
from chordapp.analysis.pipeline import analyze_audio
from test_meter_contract import audio, ControlledTracker, BarChanges


def adapter(beats, downs, mode='model'):
    tracker = BeatThisTracker(device='cpu', postprocess=mode)
    tracker._model = lambda path: (beats, downs)
    return tracker


def test_model_mode_preserves_genuine_tempo_change():
    beats = np.r_[np.arange(24)*.5, 12+np.arange(8)*.25]
    downs = beats[::4]
    raw = adapter(beats, downs).track('not-read')
    legacy = adapter(beats, downs, 'legacy').track('not-read')
    np.testing.assert_array_equal(raw.beat_times, beats)
    np.testing.assert_array_equal(raw.downbeat_times, downs)
    assert raw.beats_per_bar == 4
    assert len(legacy.beat_times) < len(beats)


def test_model_mode_preserves_meter_change_without_claiming_one_meter():
    beats = np.arange(28)*.5
    downs = beats[[0,4,8,12,15,18,21,24,27]]
    grid = adapter(beats, downs).track('not-read')
    np.testing.assert_array_equal(grid.downbeat_times, downs)
    assert grid.beats_per_bar is None


@pytest.mark.parametrize('meter',[2,3,4,6,7])
def test_consistent_model_meter_has_no_four_beat_prior(meter):
    beats=np.arange(42)*.5
    assert adapter(beats,beats[::meter]).track('not-read').beats_per_bar==meter


@pytest.mark.parametrize('beats,tempo,downs',[
    ([0,.5,.5],120,None),([0,float('nan'),1],120,None),([1,0],120,None),
    ([0,.5],0,None),([0,.5],float('inf'),None),([0,.5],120,[.5,0]),
])
def test_invalid_tracker_output_rejected(beats,tempo,downs):
    with pytest.raises(ValueError):BeatGrid(beats,tempo,downs)


def test_tracker_failure_is_visible_without_exposing_original_error(audio):
    class Broken:
        name='broken-model'
        def track(self,path):raise RuntimeError('secret-value https://private.invalid/model')
    result=analyze_audio(str(audio),tracker=Broken(),max_seconds=3.5,separate=False,
                         recognizers=[BarChanges()],detect_inversions=False)
    assert result.execution['beat_tracker']=='librosa'
    assert result.execution['beat_tracker_requested']=='broken-model'
    assert result.execution['beat_tracker_fallback']=='RuntimeError'
    assert any('실행 실패' in text for text in result.warnings)
    assert 'secret-value' not in str(result.to_dict())


def test_model_grid_is_cropped_to_requested_audio_range(audio):
    result=analyze_audio(str(audio),tracker=ControlledTracker(4),max_seconds=3.5,
                         separate=False,recognizers=[BarChanges()],detect_inversions=False)
    assert max(result.beat_times)<result.duration
    assert max(result.downbeats)<result.duration
    assert max(segment.end for segment in result.segments)<=result.duration
    assert result.execution['beat_tracker_fallback'] is None


def test_shifted_downbeat_outside_range_is_dropped_not_clamped(audio):
    result=analyze_audio(str(audio),tracker=ControlledTracker(4),max_seconds=3.5,
                         downbeat_offset=3,separate=False,recognizers=[BarChanges()],detect_inversions=False)
    assert result.downbeats==[1.5]
