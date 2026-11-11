"""키 설정부터 검색 결과까지의 계약, 비밀 비노출, Google 오류 및 할당량 절약 검증."""
import io
import json
import os
from pathlib import Path
import urllib.error
import urllib.parse

import pytest
from fastapi.testclient import TestClient

from chordapp import config
from chordapp.service.youtube import DataApiSearch, SearchUnavailable
from chordapp.service.pipeline import ChordService
from chordapp.service.cache import AnalysisCache
from chordapp.service.corrections import CorrectionStore
from chordapp.service.audio_sources import AudioSourceStore

KEY = "AIza" + "x" * 35
HEADERS = {"X-Chordapp-Settings": "1", "Origin": "http://testserver"}


@pytest.fixture
def settings(tmp_path, monkeypatch):
    from chordapp.api import server
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("CHORDAPP_PRIVATE_MODE", "1")
    monkeypatch.setenv("YOUTUBE_API_KEY", "")
    service = ChordService(cache=AnalysisCache(tmp_path / "cache.db"),
        corrections=CorrectionStore(tmp_path / "fix.db"),
        audio_sources=AudioSourceStore(tmp_path / "source.db", tmp_path / "audio"))
    monkeypatch.setattr(server, "service", service)
    monkeypatch.setattr(server.app.state, "last_error", None, raising=False)
    calls = []
    def google(url, **kwargs):
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        calls.append((parsed.path, query))
        if parsed.path.endswith("/search"):
            data = {"items": [{"id": {"videoId": "TestVideo01"}}]}
        elif query["part"] == ["id"]:
            data = {"items": []}
        else:
            data = {"items": [{"id": "TestVideo01", "snippet": {"title": "검증 곡", "channelTitle": "테스트", "thumbnails": {}},
                               "contentDetails": {"duration": "PT3M10S"}}]}
        return io.BytesIO(json.dumps(data).encode())
    monkeypatch.setattr("urllib.request.urlopen", google)
    with TestClient(server.app) as client:
        yield client, service, calls, tmp_path


def save(client, key=KEY, headers=HEADERS):
    return client.post("/api/settings/youtube", json={"api_key": key}, headers=headers)


def test_save_enables_search_without_restart_and_preserves_env(settings):
    client, service, calls, root = settings
    path = root / ".env"
    path.write_text('# keep\nOTHER=value\nYOUTUBE_API_KEY=old\nYOUTUBE_API_KEY=duplicate\n')
    assert not client.get("/api/capabilities").json()["youtube_search"]
    response = save(client)
    assert response.status_code == 200 and KEY not in response.text
    assert path.read_text() == '# keep\nOTHER=value\nYOUTUBE_API_KEY=' + KEY + '\n'
    assert path.stat().st_mode & 0o777 == 0o600
    assert client.get("/api/capabilities").json()["youtube_search"]
    assert len(calls) == 1 and calls[0][1]["part"] == ["id"]
    assert client.get("/api/search", params={"q": "검증 곡"}).json()["results"][0]["duration"] == 190
    assert len(calls) == 3
    assert calls[1][1]["videoCategoryId"] == ["10"]
    assert client.get("/api/search", params={"q": "검증 곡"}).status_code == 200
    assert len(calls) == 3  # 같은 검색은 추가 할당량을 쓰지 않는다.
    for endpoint in ("/api/settings/youtube", "/api/diagnostics", "/api/capabilities"):
        assert KEY not in client.get(endpoint).text


def test_key_survives_server_restart(settings, monkeypatch):
    client, _, _, root = settings
    assert save(client).status_code == 200
    monkeypatch.delenv("YOUTUBE_API_KEY")
    config.load_env(root / ".env")
    assert DataApiSearch().available and DataApiSearch().api_key == KEY


@pytest.mark.parametrize("headers", [{}, {"X-Chordapp-Settings":"1", "Origin":"https://other.example"}])
def test_reject_cross_site_settings(settings, headers):
    client, _, calls, root = settings
    assert save(client, headers=headers).status_code == 403
    assert not (root / ".env").exists() and calls == []


def test_private_mode_required(settings, monkeypatch):
    client, _, calls, root = settings
    monkeypatch.delenv("CHORDAPP_PRIVATE_MODE")
    assert not client.get("/api/settings/youtube").json()["can_edit"]
    assert save(client).status_code == 403 and calls == []


@pytest.mark.parametrize("key", ["invalid", KEY + '\nOTHER=oops', KEY + 'extra'])
def test_invalid_format_never_echoes_input_or_calls_google(settings, key):
    client, _, calls, root = settings
    response = save(client, key)
    assert response.status_code == 400 and key not in response.text
    assert calls == [] and not (root / '.env').exists()


@pytest.mark.parametrize("status,reason,expected", [(400,"API_KEY_INVALID","invalid_api_key"),
    (403,"quotaExceeded","quota_exceeded"),(403,"SERVICE_DISABLED","api_not_enabled"),
    (403,"API_KEY_HTTP_REFERRER_BLOCKED","key_restricted"),(500,"unknown","api_error")])
def test_google_failure_preserves_working_key_and_hides_secret(settings, monkeypatch, status, reason, expected):
    client, _, _, root = settings
    old_key = 'AIza' + 'y'*35
    (root / '.env').write_text('YOUTUBE_API_KEY=' + old_key + '\n')
    monkeypatch.setenv('YOUTUBE_API_KEY', old_key)
    def error(url, **kwargs):
        body = {"error": {"message": KEY, "details": [{"reason":reason}], "errors":[]}}
        raise urllib.error.HTTPError(url, status, 'failed '+KEY, {}, io.BytesIO(json.dumps(body).encode()))
    monkeypatch.setattr('urllib.request.urlopen', error)
    response = save(client)
    assert response.status_code == 503 and response.json()['detail']['code'] == expected
    assert KEY not in response.text and old_key not in response.text
    assert os.environ['YOUTUBE_API_KEY'] == old_key and old_key in (root / '.env').read_text()
    diagnostics = client.get('/api/diagnostics')
    assert KEY not in diagnostics.text and old_key not in diagnostics.text
    assert diagnostics.json()['last_error']['code'] == expected


def test_network_error_does_not_expose_request_url(settings, monkeypatch):
    client = settings[0]
    def error(*args, **kwargs): raise urllib.error.URLError('sensitive ' + KEY)
    monkeypatch.setattr('urllib.request.urlopen', error)
    response = save(client)
    assert response.status_code == 503 and KEY not in response.text
    assert response.json()['detail']['code'] == 'network_error'


def test_failed_atomic_write_keeps_old_key_and_cleans_staging(settings, monkeypatch):
    _, _, _, root = settings
    target = root / '.env'
    target.write_text('OTHER=keep\n')
    def error(*args): raise OSError('disk failure')
    monkeypatch.setattr(Path, 'replace', error)
    with pytest.raises(OSError): config.save_youtube_key(KEY)
    assert target.read_text() == 'OTHER=keep\n' and not os.environ.get('YOUTUBE_API_KEY')
    assert not list(root.glob('.env-key-*.tmp'))


def test_search_cache_expires_after_hour(settings, monkeypatch):
    import time
    client, _, calls, _ = settings
    save(client)
    client.get('/api/search', params={'q':'test'})
    now = time.time()
    monkeypatch.setattr('chordapp.service.cache.time.time', lambda: now+3601)
    client.get('/api/search', params={'q':'test'})
    assert len(calls) == 5
