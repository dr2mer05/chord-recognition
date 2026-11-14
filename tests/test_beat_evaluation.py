"""정답 박자 입력을 분리하고 동일 음원에서 평가 조건을 비교하는지 검증한다."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from chordapp import beat_evaluation
from chordapp.cli import build_parser, cmd_beateval
from chordapp.evaluation import bar_interval_f_measure, beat_f_measure
from chordapp.tools.synth import DemoTrack


@pytest.fixture
def fixture_run(tmp_path, monkeypatch):
    track = DemoTrack('three', 'C', 120, 3, ['C', 'F'], repeats=1)
    calls = []
    def extract(path, **kwargs):
        calls.append(kwargs)
        meter = kwargs['beats_per_bar'] or 4
        return SimpleNamespace(tempo=120., beat_times=np.arange(7)*.5,
            downbeat_indices=np.arange(0,7,meter), beats_per_bar=meter, beat_source='fixture')
    monkeypatch.setattr(beat_evaluation.audio, 'extract_features', extract)
    def run(**kwargs):
        return beat_evaluation.evaluate_tracks(tmp_path, tracks=[track], **kwargs)
    return run, calls, tmp_path, track


def test_auto_evaluation_does_not_supply_ground_truth_meter(fixture_run):
    run, calls, _, _ = fixture_run
    report = run()
    assert calls == [{'beats_per_bar':None}]
    assert report['settings']['reference_meter_supplied'] is False
    assert report['summary']['beats_per_bar_accuracy'] == 0
    assert report['tracks'][0]['estimate']['beats_per_bar'] == 4
    json.dumps(report, allow_nan=False)


def test_reference_mode_is_explicit_and_reuses_identical_audio(fixture_run):
    run, calls, _, _ = fixture_run
    automatic, reference = run(), run(meter_mode='reference')
    assert calls[-1] == {'beats_per_bar':3}
    assert reference['settings']['reference_meter_supplied'] is True
    assert reference['summary']['beats_per_bar_accuracy'] is None
    assert automatic['tracks'][0]['audio_sha256'] == reference['tracks'][0]['audio_sha256']
    assert reference['summary']['bar_f_macro'] == 1


def test_explicit_librosa_baseline_bypasses_optional_model(fixture_run):
    run, calls, _, _ = fixture_run
    run(tracker='librosa')
    assert calls[0]['tracker'] is None


def test_requested_candidate_is_not_scored_when_it_falls_back(fixture_run):
    run, _, _, _ = fixture_run
    with pytest.raises(ValueError, match='폴백을 후보 성능으로 계산하지 않습니다'):
        run(tracker='plp')


def test_changed_audio_cannot_be_scored_against_stale_truth(fixture_run):
    run, calls, directory, _ = fixture_run
    run()
    with (directory/'three.wav').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='생성 조건'):
        run()
    assert len(calls) == 1
    run(regenerate=True)
    assert len(calls) == 2


def test_changed_reference_spec_requires_regeneration(fixture_run):
    run, calls, _, track = fixture_run
    run()
    track.bpm = 90
    with pytest.raises(ValueError, match='생성 조건'):
        run()
    assert len(calls) == 1


def test_old_fixture_without_manifest_is_not_silently_accepted(fixture_run):
    run, calls, directory, _ = fixture_run
    (directory/'three.wav').write_bytes(b'not a generated wave')
    with pytest.raises(ValueError, match='regenerate'):
        run()
    assert calls == []


@pytest.mark.parametrize('contents',['[]','not json'])
def test_invalid_generation_manifest_is_reported(fixture_run, contents):
    run, calls, directory, _ = fixture_run
    (directory/'.beat-fixtures.json').write_text(contents)
    with pytest.raises(ValueError, match='생성 기록'):
        run()
    assert calls == []
    run(regenerate=True)
    assert len(calls) == 1


def test_cli_defaults_and_json_report(fixture_run,monkeypatch,capsys):
    run, _, directory, _ = fixture_run
    actual = run()
    seen = []
    def evaluate(directory, **kwargs):
        seen.append(kwargs)
        return actual
    monkeypatch.setattr(beat_evaluation,'evaluate_tracks',evaluate)
    output=directory/'report.json'
    args=build_parser().parse_args(['beateval','--dir',str(directory),'--json-out',str(output)])
    assert cmd_beateval(args)==0
    assert seen[0]['meter_mode']=='auto'
    assert json.loads(output.read_text())['dataset_kind']=='synthetic'
    assert '정답 박자 미입력' in capsys.readouterr().out


def test_bar_metric_requires_both_ends_and_penalizes_missing_boundaries():
    ref, est=[0,1,2],[0,2]
    assert beat_f_measure(ref,est)['f']>.5
    assert bar_interval_f_measure(ref,est)=={'f':0.,'precision':0.,'recall':0.,'n_ref':2,'n_est':1}


def test_bar_metric_penalizes_extra_boundaries_and_allows_small_jitter():
    assert bar_interval_f_measure([0,2,4],[0,1,2,3,4])['f']==0
    assert bar_interval_f_measure([0,2,4],[.03,2.02,4.01])['f']==1


@pytest.mark.parametrize('ref,est,tolerance',[([0,float('nan')],[0,1],.07),([0,1],[0,1],-.1)])
def test_invalid_bar_metric_input_rejected(ref,est,tolerance):
    with pytest.raises(ValueError):
        bar_interval_f_measure(ref,est,tolerance)
