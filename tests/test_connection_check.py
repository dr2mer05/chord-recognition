"""접속 비교가 캐시·교정을 건드리지 않고 두 설정의 실제 다운로드 경로를 호출하는지 검증."""
from pathlib import Path
import subprocess

import pytest
from fastapi.testclient import TestClient

from chordapp.service import connection_check, pot
from chordapp.service.private_youtube import PrivateYouTubeProvider
from test_audio_sources import setup, VID
from test_private_youtube import private


@pytest.fixture
def check_provider(private, monkeypatch, tmp_path):
    monkeypatch.setenv('CHORDAPP_ENABLE_POT', '1')
    monkeypatch.setattr(pot, 'ensure_started', lambda: None)
    monkeypatch.setattr(pot, 'status', lambda: {'state':'ready'})
    created = []
    original_download = private[5]._download

    class Provider(PrivateYouTubeProvider):
        def __init__(self, **kw):
            super().__init__(temp_dir=tmp_path/'compare', **kw)
            created.append(self)
        def _download(self, command):
            return original_download(command)

    monkeypatch.setattr(connection_check, 'PrivateYouTubeProvider', Provider)
    return private, created


def test_both_profiles_download_and_cleanup_without_analysis(check_provider):
    private, providers = check_provider
    updates=[]
    report=connection_check.compare(VID, updates.append)
    assert report['state']=='complete'
    assert [p.profile for p in providers]==['legacy_clients','current']
    assert [r['ok'] for r in report['rows']]==[True,True]
    assert [len(v['rows']) for v in updates]==[0,1,2,2]
    assert len(private[6])==2 and all(not p.exists() for p in private[6])
    assert private[3]==[]  # DSP is never invoked.
    legacy, current=[p.command(VID,Path('/tmp')) for p in providers]
    assert 'youtube:player_client=android,ios,tv,web' in legacy
    assert '--no-plugin-dirs' in legacy and not any('bgutil' in x for x in legacy)
    assert 'youtube:player_client=mweb' in current
    assert current[-1]==legacy[-1]=='https://www.youtube.com/watch?v='+VID


def test_current_token_failure_does_not_prevent_legacy_test(check_provider, monkeypatch):
    def fail():raise pot.PotError('토큰 공급 서버가 시작되지 않았습니다.')
    monkeypatch.setattr(pot,'ensure_started',fail)
    report=connection_check.compare(VID,lambda report:None)
    assert report['rows'][0]['ok'] is True
    assert report['rows'][1]['code']=='pot_unavailable'
    assert len(check_provider[0][6])==1


def test_two_failures_keep_safe_stage_evidence_and_release_lock(check_provider, monkeypatch):
    def fail(self,command):
        return subprocess.CompletedProcess(command,1,'[download] Destination: temp',
            'HTTP Error 403: Forbidden https://user:secret@example.invalid/?token=private')
    monkeypatch.setattr(connection_check.PrivateYouTubeProvider,'_download',fail)
    report=connection_check.compare(VID,lambda report:None)
    assert [r['code'] for r in report['rows']]==['youtube_access_denied']*2
    assert all(r['details']['http_status']==403 for r in report['rows'])
    assert all(r['details']['stage']=='음원 다운로드' for r in report['rows'])
    assert 'secret' not in str(report) and 'example.invalid' not in str(report)
    assert connection_check._lock.acquire(blocking=False)
    connection_check._lock.release()


def test_busy_comparison_is_not_queued(check_provider):
    connection_check._lock.acquire()
    try:
        with pytest.raises(connection_check.CheckBusy):
            connection_check.compare(VID,lambda report:None)
        assert check_provider[1]==[]
    finally:connection_check._lock.release()


@pytest.mark.parametrize('headers,enabled', [({},True),({'X-Chordapp-Settings':'1','Origin':'https://other.invalid'},True),({'X-Chordapp-Settings':'1'},False)])
def test_api_rejects_nonprivate_or_cross_origin_checks(check_provider, monkeypatch, headers, enabled):
    from chordapp.api import server
    monkeypatch.setattr(server,'service',check_provider[0][0])
    if not enabled:monkeypatch.delenv('CHORDAPP_PRIVATE_MODE')
    with TestClient(server.app) as client:
        response=client.post('/api/diagnostics/youtube-connection',json={'url':VID},headers=headers)
    assert response.status_code==403 and check_provider[1]==[]


def test_api_uses_download_path_and_keeps_report_for_reload(check_provider,monkeypatch):
    from chordapp.api import server
    monkeypatch.setattr(server,'service',check_provider[0][0])
    monkeypatch.setattr(server.app.state,'connection_comparison',None,raising=False)
    with TestClient(server.app) as client:
        response=client.post('/api/diagnostics/youtube-connection',json={'url':VID},
            headers={'X-Chordapp-Settings':'1','Origin':'http://testserver'})
        assert response.status_code==200
        assert client.get('/api/diagnostics').json()['connection_comparison']==response.json()
        invalid=client.post('/api/diagnostics/youtube-connection',json={'url':'https://example.invalid/audio'},
            headers={'X-Chordapp-Settings':'1'})
        assert invalid.status_code==400 and len(check_provider[1])==2
