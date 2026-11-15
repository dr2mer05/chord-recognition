"""검증 세트가 기존 데모·사용자 음원을 섞지 않고 열리는지 확인한다."""
import pytest
from fastapi.testclient import TestClient

from chordapp.api import server
from chordapp.models import AnalysisResult, ChordSegment
from chordapp.theory.chord import Chord
from chordapp.theory.key import Key


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'DEMO_DIR', tmp_path / 'demo')
    monkeypatch.setattr(server, 'STRESS_DIR', tmp_path / 'stress')
    return TestClient(server.app)


def test_stress_tracks_truth_and_audio_are_separate_from_basic_demo(client):
    stress = client.get('/api/demo?suite=stress').json()['tracks']
    basic = client.get('/api/demo').json()['tracks']
    assert len(stress) == 7 and len(basic) == 5
    assert not ({t['name'] for t in stress} & {t['name'] for t in basic})
    major = next(t for t in stress if t['name'] == 'same_root_major')
    assert major['truth'][:4] == ['C', 'C7', 'Cmaj7', 'C6']
    response = client.get(major['audio_url'])
    assert response.status_code == 200 and response.content[:4] == b'RIFF'
    assert client.get('/api/demo/audio/same_root_major').status_code == 404
    assert client.post('/api/demo/same_root_major').status_code == 404


@pytest.mark.parametrize('method,url', [
    ('get', '/api/demo?suite=../'),
    ('post', '/api/demo/any?suite=../'),
    ('get', '/api/demo/audio/any?suite=../'),
])
def test_suite_is_bounded_not_a_filesystem_path(client, method, url):
    assert getattr(client, method)(url).status_code == 422


def test_stress_response_exposes_independent_truth_and_distinct_metrics(client, monkeypatch):
    client.get('/api/demo?suite=stress')
    result = AnalysisResult(source='test', duration=20, tempo=96, key=Key.parse('C'),
        key_confidence=.8, segments=[ChordSegment(0, 20, Chord.parse('C'))],
        beat_times=[0, 1], downbeats=[0])
    monkeypatch.setattr(server.service, 'analyze_file', lambda *a, **k: result)
    payload = client.post('/api/demo/same_root_major?suite=stress').json()
    assert payload['reference_chords'][:4] == ['C', 'C7', 'Cmaj7', 'C6']
    assert payload['accuracy']['root'] == 100
    assert payload['accuracy']['seventh'] < 30
    assert payload['accuracy']['boundary_f'] == 0
    assert payload['audio_url'].endswith('?suite=stress')
