"""개인 공급 경로·임시 음원 수명·캐시·교정 회귀. YouTube 네트워크를 모사한다."""

from pathlib import Path
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from chordapp.service.pipeline import ChordService
from chordapp.service.private_youtube import PrivateYouTubeError, PrivateYouTubeProvider
from chordapp.service.audio_sources import AudioSourceUnavailable
from test_audio_sources import setup, register, VID, audio_bytes


@pytest.fixture
def private(setup, monkeypatch, tmp_path):
    service, store, wav, calls, identity = setup
    monkeypatch.setenv("CHORDAPP_ENABLE_YTDLP", "1")
    monkeypatch.setenv("CHORDAPP_PRIVATE_MODE", "1")
    monkeypatch.setattr("chordapp.service.private_youtube.find_spec", lambda name: object())
    monkeypatch.setattr("chordapp.service.private_youtube.find_ffmpeg", lambda: "/test/ffmpeg")
    monkeypatch.setattr("chordapp.service.private_youtube.find_deno", lambda: "/test/deno")
    provider = PrivateYouTubeProvider(tmp_path / "private-temp")
    service.private_youtube = provider
    downloads = []
    def download(command):
        path = Path(command[command.index("-o") + 1].replace("%(ext)s", "wav"))
        path.write_bytes(wav.read_bytes())
        downloads.append(path)
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(provider, "_download", download)
    return service, store, wav, calls, identity, provider, downloads


@pytest.mark.parametrize("private_flag,enabled_flag,expected", [(None,None,False),("1",None,False),(None,"1",False),("1","1",True)])
def test_both_private_flags_required(monkeypatch, private_flag, enabled_flag, expected):
    for name, value in [("CHORDAPP_PRIVATE_MODE",private_flag),("CHORDAPP_ENABLE_YTDLP",enabled_flag)]:
        if value: monkeypatch.setenv(name, value)
        else: monkeypatch.delenv(name, raising=False)
    assert PrivateYouTubeProvider().enabled is expected


def test_cleanup_and_cache_after_restart_without_audio(private):
    service, store, _, calls, _, provider, downloads = private
    first, _ = service.analyze_youtube(VID)
    assert not downloads[0].exists() and list(provider.temp_dir.iterdir()) == []
    assert first.audio_path == ""
    assert service.corrections.source(first.source_key)["audio_path"] == ""
    restored = ChordService(cache=service.cache, corrections=service.corrections,
        search=service.search_backend, audio_sources=store, private_youtube=provider)
    reopened, _ = restored.analyze_youtube(f"https://youtu.be/{VID}?t=10")
    assert len(downloads) == len(calls) == 1
    assert reopened.cached and reopened.origin == "audio"
    assert reopened.source_key == first.source_key
    assert reopened.audio_link["provider"] == "yt-dlp"


def test_corrections_preserve_original_and_meter_invalidates_cache(private):
    service, _, _, calls, _, _, downloads = private
    first, _ = service.analyze_youtube(VID)
    service.corrections.add(first.source_key, 0, 2, "Am7", original="C")
    again, _ = service.analyze_youtube(VID)
    assert again.segments[0].chord.name() == "C"
    assert service.full_report(again)["segments"][0]["chord"] == "Am7"
    assert len(downloads) == 1
    service.corrections.set_meter(first.source_key, 3, 0)
    changed, _ = service.analyze_youtube(VID)
    assert len(downloads) == len(calls) == 2 and calls[-1][1]["beats_per_bar"] == 3
    service.corrections.set_meter(first.source_key, None, 0)
    auto, _ = service.analyze_youtube(VID)
    assert auto.cached and len(downloads) == 2


def test_private_mode_disabled_denies_cached_results(private, monkeypatch):
    service = private[0]
    service.analyze_youtube(VID)
    monkeypatch.delenv("CHORDAPP_PRIVATE_MODE")
    with pytest.raises(AudioSourceUnavailable, match="연결된 음원"):
        service.analyze_youtube(VID)


def test_explicit_link_priority_and_revocation(private):
    service, store, wav, _, _, _, downloads = private
    link = register(store, wav)
    result, _ = service.analyze_youtube(VID)
    assert result.source_key == link.source_key and downloads == []
    store.revoke(VID)
    with pytest.raises(AudioSourceUnavailable) as error:
        service.analyze_youtube(VID)
    assert error.value.code == "rights_revoked" and downloads == []
    assert service.youtube_source_status(VID)["status"] == "rights_revoked"


def test_expired_cache_and_changed_audio_do_not_inherit_corrections(private, monkeypatch):
    import time
    service, _, wav, _, _, _, downloads = private
    first, _ = service.analyze_youtube(VID)
    service.corrections.add(first.source_key, 0, 2, "Am7")
    now = time.time()
    monkeypatch.setattr("chordapp.service.cache.time.time", lambda: now + 86401)
    wav.write_bytes(audio_bytes(.2))
    changed, _ = service.analyze_youtube(VID)
    assert len(downloads) == 2 and changed.source_key != first.source_key
    assert service.full_report(changed)["corrected"] == 0


def test_options_environment_and_fallback_do_not_share_cache(private):
    service, _, _, calls, identity, _, downloads = private
    service.analyze_youtube(VID)
    service.analyze_youtube(VID, max_seconds=2)
    assert len(downloads) == 2
    identity["packages"]["test-engine"] = "2"
    service.analyze_youtube(VID)
    assert len(downloads) == 3
    identity["beat_tracker"] = "preferred-unavailable"
    service.analyze_youtube(VID)
    service.analyze_youtube(VID)
    assert len(downloads) == len(calls) == 5


def test_analysis_failure_cleans_audio_and_never_caches(private, monkeypatch):
    service, _, _, _, _, provider, _ = private
    def fail(*args, **kwargs): raise RuntimeError("DSP failure")
    monkeypatch.setattr("chordapp.analysis.pipeline.analyze_audio", fail)
    with pytest.raises(RuntimeError, match="DSP failure"):
        service.analyze_youtube(VID)
    assert list(provider.temp_dir.iterdir()) == []
    assert service.cache.stats()["entries"] == 0


@pytest.mark.parametrize("failure,expected", [("exit","download_failed"),("timeout","download_timeout"),("missing","unsupported_video"),("corrupt","audio_invalid"),("spawn","download_failed")])
def test_downloader_failures_remove_partial_files(private, monkeypatch, failure, expected):
    provider = private[5]
    def fail(command):
        path = Path(command[command.index("-o") + 1].replace("%(ext)s", "wav"))
        path.write_text("invalid")
        if failure == "exit": return subprocess.CompletedProcess(command, 1, "", "blocked")
        if failure == "timeout": raise subprocess.TimeoutExpired(command, 1)
        if failure == "spawn": raise OSError("missing")
        if failure == "missing": path.unlink()
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(provider, "_download", fail)
    with pytest.raises(PrivateYouTubeError) as error:
        with provider.audio(VID): pass
    assert error.value.code == expected
    assert list(provider.temp_dir.iterdir()) == []


def test_timeout_terminates_child_group(tmp_path):
    provider = PrivateYouTubeProvider(timeout=.2)
    command = [sys.executable, "-c", "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);time.sleep(30)"]
    import time
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired): provider._download(command)
    assert time.monotonic() - start < 5


def test_api_ready_without_upload_and_errors(private, monkeypatch):
    from chordapp.api import server
    service, _, _, _, _, provider, _ = private
    monkeypatch.setattr(server, "service", service)
    with TestClient(server.app) as client:
        assert client.get("/api/capabilities").json()["youtube_audio"] is True
        assert client.get("/api/youtube/source", params={"url":VID}).json()["status"] == "ready"
        response = client.post("/api/youtube", json={"url":VID})
        assert response.status_code == 200 and response.json()["audio_link"]["provider"] == "yt-dlp"
        service.cache.clear()
        def fail(*args): raise PrivateYouTubeError("download_timeout", "시간 초과")
        monkeypatch.setattr(provider, "_download", fail)
        response = client.post("/api/youtube", json={"url":VID})
        assert response.status_code == 504 and response.json()["detail"]["code"] == "download_timeout"


def test_missing_dependency_is_reported(private, monkeypatch):
    provider = private[5]
    monkeypatch.setattr("chordapp.service.private_youtube.find_spec", lambda name: None)
    assert not provider.available
    with pytest.raises(PrivateYouTubeError) as error: provider.ensure_available()
    assert error.value.code == "dependency_missing"


@pytest.mark.parametrize("stderr,expected,status", [
    ("ERROR: Sign in to confirm you’re not a bot. Use --cookies-from-browser ...", "youtube_bot_check", None),
    ("Unable to connect to proxy: Tunnel connection failed: 403 Forbidden", "network_proxy_blocked", 403),
    ("ERROR: HTTP Error 429: Too Many Requests", "youtube_rate_limited", 429),
    ("Video not available in your country", "video_region_restricted", None),
    ("Sign in to confirm your age", "video_login_required", None),
    ("Private video", "video_login_required", None),
    ("Video unavailable", "video_unavailable", None),
    ("CERTIFICATE_VERIFY_FAILED", "network_tls_error", None),
    ("WARNING: No supported JavaScript runtime could be found. Requested format is not available", "youtube_runtime_error", None),
    ("n challenge solving failed: Requested format is not available", "youtube_runtime_error", None),
    ("Requested format is not available", "audio_format_unavailable", None),
    ("ERROR: Postprocessing: ffmpeg not found", "audio_conversion_failed", None),
    ("HTTP Error 403: Forbidden", "youtube_access_denied", 403),
    ("Unrecognized failure", "download_failed", None),
])
def test_classifies_evidence_without_exposing_raw_error(stderr, expected, status):
    from chordapp.service.private_youtube import download_failure
    secret = 'AIza' + 's' * 35
    raw = stderr + ' https://user:password@example.invalid/?key=' + secret + ' /home/private/user/file'
    error = download_failure(subprocess.CompletedProcess([], 1, '[youtube] Downloading webpage', raw))
    assert error.code == expected and error.diagnostics['http_status'] == status
    shared = str(error) + str(error.diagnostics)
    assert secret not in shared and 'password' not in shared and '/home/private' not in shared
    assert '--cookies' not in shared


def test_conversion_stage_is_distinct_from_metadata_failure():
    from chordapp.service.private_youtube import download_failure
    result = subprocess.CompletedProcess([], 1, '[download] Destination: x\n[ExtractAudio] Destination: y', 'unknown')
    assert download_failure(result).diagnostics['stage'] == '음원 변환'


def test_classified_failure_reaches_api_and_diagnostic_screen(private, monkeypatch):
    from chordapp.api import server
    service, _, _, _, _, provider, _ = private
    def blocked(command):
        return subprocess.CompletedProcess(command, 1, '[youtube] Downloading webpage', 'Sign in to confirm you are not a bot')
    monkeypatch.setattr(provider, '_download', blocked)
    monkeypatch.setattr(server, 'service', service)
    with TestClient(server.app) as client:
        response = client.post('/api/youtube', json={'url':VID})
        assert response.status_code == 502 and response.json()['detail']['code'] == 'youtube_bot_check'
        diagnostic = client.get('/api/diagnostics').json()['last_error']
        assert diagnostic['code'] == 'youtube_bot_check'
        assert diagnostic['details']['stage'] == '영상 정보 조회'
        assert list(provider.temp_dir.iterdir()) == []


def test_runtime_probe_detects_execution_failure_and_never_exposes_output(monkeypatch):
    from chordapp.service import private_youtube as module
    module.runtime_versions.cache_clear()
    try:
        monkeypatch.setattr(module, 'find_deno', lambda: '/test/deno')
        monkeypatch.setattr(module, 'find_ffmpeg', lambda: '/test/ffmpeg')
        def run(command, **kwargs):
            if 'deno' in command[0]: raise OSError('private machine path')
            return subprocess.CompletedProcess(command, 0, 'ffmpeg version 7.0.2 privatepath', '')
        monkeypatch.setattr(module.subprocess, 'run', run)
        versions = module.runtime_versions()
        assert versions['deno'] is None and versions['ffmpeg'] == '7.0.2'
        assert 'private' not in str(versions)
    finally:
        module.runtime_versions.cache_clear()
