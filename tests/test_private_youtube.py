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
