"""허용 음원 연결의 수명주기, 원본·교정 분리, API 경계 검증. 외부 네트워크 없음."""

import copy
import io
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from chordapp.models import AnalysisResult, ChordSegment
from chordapp.service.audio_sources import AudioSourceStore, AudioSourceUnavailable
from chordapp.service.cache import AnalysisCache
from chordapp.service.corrections import CorrectionStore
from chordapp.service.pipeline import ChordService
from chordapp.service.youtube import VideoInfo
from chordapp.theory.chord import Chord
from chordapp.theory.key import Key

VID = "dQw4w9WgXcQ"
IDENTITY = {"beat_tracker": "librosa", "separated": False,
            "recognizers": ["fixture"], "packages": {"test-engine": "1"}}


def audio_bytes(value=0.0, duration=4.0, sr=8000):
    stream = io.BytesIO()
    sf.write(stream, np.full(int(duration * sr), value), sr, format="WAV")
    return stream.getvalue()


class Search:
    available = False

    def __init__(self):
        self.calls = 0

    def metadata(self, video_id):
        self.calls += 1
        return VideoInfo(video_id, title="직접 녹음한 테스트", channel="테스트")


@pytest.fixture
def setup(tmp_path, monkeypatch):
    store = AudioSourceStore(tmp_path / "sources.sqlite3", tmp_path / "assets")
    service = ChordService(cache=AnalysisCache(tmp_path / "cache.sqlite3"),
                           corrections=CorrectionStore(tmp_path / "fix.sqlite3"),
                           search=Search(), audio_sources=store)
    calls = []
    identity = copy.deepcopy(IDENTITY)
    monkeypatch.setattr("chordapp.analysis.runtime.runtime_identity", lambda *a: copy.deepcopy(identity))

    def analyze(path, **options):
        calls.append((path, options))
        return AnalysisResult(
            source=path, duration=4, tempo=120, key=Key.parse("C"), key_confidence=.8,
            segments=[ChordSegment(0, 2, Chord.parse("C")), ChordSegment(2, 4, Chord.parse("F"))],
            beat_times=list(np.arange(0, 4, .5)), downbeats=[0, 2], recognizers=["fixture"],
            execution={"beat_tracker": "librosa", "separated": False, "recognizers": ["fixture"]})

    monkeypatch.setattr("chordapp.analysis.pipeline.analyze_audio", analyze)
    wav = tmp_path / "original.wav"
    wav.write_bytes(audio_bytes())
    return service, store, wav, calls, identity


def register(store, wav, **kwargs):
    return store.register(VID, wav, label="원본.wav", rights_basis="직접 녹음, 분석·보관 허용",
                          same_recording=kwargs.pop("same_recording", True), **kwargs)


def test_link_reopens_after_restart_without_upload(setup):
    service, store, wav, calls, _ = setup
    link = register(store, wav, video_offset=5)
    restored = AudioSourceStore(store.db_path, store.audio_dir)
    assert restored.resolve(VID) == link
    assert link.path.read_bytes() == wav.read_bytes()
    first, _ = service.analyze_youtube(f"https://youtu.be/{VID}")
    again, _ = service.analyze_youtube(f"https://www.youtube.com/watch?v={VID}&t=30")
    assert len(calls) == 1
    assert first.source_key == again.source_key == link.source_key
    assert first.origin == again.origin == "audio"
    assert not first.cached and again.cached
    assert again.audio_link["video_offset"] == 5
    assert service.search_backend.calls == 1


def test_original_cache_survives_correction_and_reopen(setup):
    service, store, wav, calls, _ = setup
    link = register(store, wav)
    raw, _ = service.analyze_youtube(VID)
    assert service.corrections.source(link.source_key) is not None
    service.corrections.add(link.source_key, 0, 2, "Am", original="C")
    reopened, _ = service.analyze_youtube(VID)
    report = service.full_report(reopened)
    assert report["segments"][0]["chord"] == "Am"
    assert reopened.segments[0].chord.name() == raw.segments[0].chord.name() == "C"
    assert len(calls) == 1


def test_replacement_separates_cache_corrections_and_verified_state(setup):
    service, store, wav, calls, _ = setup
    old = register(store, wav)
    service.analyze_youtube(VID)
    service.corrections.add(old.source_key, 0, 2, "Am")
    service.corrections.set_verified(old.source_key)
    wav.write_bytes(audio_bytes(.1))
    new = register(store, wav)
    raw, _ = service.analyze_youtube(VID)
    assert new.asset_id != old.asset_id and new.source_key != old.source_key
    assert len(calls) == 2 and not raw.cached
    assert service.full_report(raw)["corrected"] == 0
    assert service.full_report(raw)["verified"] is False
    assert len(service.corrections.corrections(old.source_key)) == 1


def test_rebinding_same_audio_starts_new_link_revision(setup):
    service, store, wav, _, _ = setup
    old = register(store, wav)
    new = register(store, wav, video_offset=10)
    assert old.asset_id == new.asset_id
    assert new.revision == old.revision + 1
    assert new.source_key != old.source_key


@pytest.mark.parametrize("failure", ["rights_revoked", "rights_expired", "audio_missing", "audio_changed"])
def test_source_checked_even_on_cache_hit(setup, monkeypatch, failure):
    service, store, wav, calls, _ = setup
    link = register(store, wav, expires_at=time.time() + 100)
    service.analyze_youtube(VID)
    if failure == "rights_revoked":
        store.revoke(VID)
    elif failure == "rights_expired":
        monkeypatch.setattr("chordapp.service.audio_sources.time.time", lambda: link.expires_at + 1)
    elif failure == "audio_missing":
        link.path.unlink()
    else:
        link.path.write_bytes(audio_bytes(.2))
    with pytest.raises(AudioSourceUnavailable) as exc:
        service.analyze_youtube(VID)
    assert exc.value.code == failure and len(calls) == 1


def test_database_title_cannot_supply_synchronized_audio(setup):
    service, store, wav, calls, _ = setup
    class Database:
        def lookup(self, *a):
            raise AssertionError("제목 기반 코드 DB를 영상 분석의 정답으로 사용하면 안 된다")
    service.database = Database()
    with pytest.raises(AudioSourceUnavailable) as exc:
        service.analyze_youtube(VID)
    assert exc.value.code == "source_missing" and not calls


def test_unconfirmed_recording_cannot_start_analysis(setup):
    service, store, wav, calls, _ = setup
    register(store, wav, same_recording=False)
    assert store.status(VID)["status"] == "recording_unconfirmed"
    with pytest.raises(AudioSourceUnavailable):
        service.analyze_youtube(VID)
    assert not calls


def test_engine_version_change_invalidates_cache(setup):
    service, store, wav, calls, identity = setup
    register(store, wav)
    service.analyze_youtube(VID)
    identity["packages"]["test-engine"] = "2"
    service.analyze_youtube(VID)
    assert len(calls) == 2


def test_fallback_is_not_cached_as_preferred_engine(setup):
    service, store, wav, calls, identity = setup
    register(store, wav)
    identity["beat_tracker"] = "beat_this"
    service.analyze_youtube(VID)
    service.analyze_youtube(VID)
    assert len(calls) == 2


def test_metadata_expiration_does_not_invalidate_audio_analysis(setup):
    service, store, wav, calls, _ = setup
    register(store, wav)
    service.analyze_youtube(VID)
    with service.cache._connect() as con:
        con.execute("UPDATE analysis_cache SET created_at = 0 WHERE namespace = 'video_metadata'")
    service.analyze_youtube(VID)
    assert service.search_backend.calls == 2 and len(calls) == 1


@pytest.mark.parametrize("options", [{"video_offset": float("nan")}, {"video_offset": 901},
                                      {"expires_at": float("inf")}, {"expires_at": 1}])
def test_invalid_link_metadata_is_not_stored(setup, options):
    _, store, wav, _, _ = setup
    with pytest.raises(ValueError):
        register(store, wav, **options)
    assert not store.list_links()


@pytest.fixture
def client(setup, monkeypatch, tmp_path):
    from chordapp.api import server
    monkeypatch.setattr(server, "service", setup[0])
    monkeypatch.setattr(server, "UPLOAD_DIR", tmp_path / "uploads")
    with TestClient(server.app) as client:
        yield client


def upload(client, **overrides):
    data = {"url": VID, "rights_basis": "직접 녹음", "rights_confirmed": "true",
            "same_recording": "true", "video_offset": "5"}
    data.update(overrides)
    return client.post("/api/audio-sources", data=data,
                       files={"file": ("원본.wav", audio_bytes(), "audio/wav")})


def test_api_register_analyze_correct_reopen_and_revoke(client, setup):
    response = upload(client)
    assert response.status_code == 201
    assert response.json()["status"] == "ready"
    assert "path" not in str(response.json()) and "rights_basis" not in str(response.json())
    report = client.post("/api/youtube", json={"url": VID}).json()
    source_key = report["source_key"]
    assert report["audio_link"]["video_offset"] == 5
    assert client.post("/api/corrections", json={"source_key": source_key,
        "start": 0, "end": 2, "chord": "Am", "original": "C"}).status_code == 200
    report = client.post("/api/youtube", json={"url": VID}).json()
    assert report["cached"] and report["origin"] == "audio"
    assert report["segments"][0]["chord"] == "Am"
    assert len(client.get("/api/audio-sources").json()["sources"]) == 1
    assert client.delete(f"/api/audio-sources/{VID}").status_code == 200
    denied = client.post("/api/youtube", json={"url": VID})
    assert denied.status_code == 409 and denied.json()["detail"]["code"] == "rights_revoked"


def test_api_missing_source_is_actionable(client):
    response = client.post("/api/youtube", json={"url": VID})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "source_missing"
    assert client.get("/api/youtube/source", params={"url": VID}).json()["video"]["title"]


def test_api_rebinding_rejects_edits_from_stale_browser(client):
    upload(client)
    old_key = client.post("/api/youtube", json={"url": VID}).json()["source_key"]
    upload(client, video_offset="10")
    for route, fields in [("", {"start": 0, "end": 2, "chord": "Am"}),
                          ("/tempo", {"bpm": 120}), ("/meter", {"beats_per_bar": 3}),
                          ("/verify", {"verified": True})]:
        response = client.post("/api/corrections" + route, json={"source_key": old_key, **fields})
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "source_replaced"


@pytest.mark.parametrize("start,end,status", [(2, 1, 400), (0, 5, 400), (-1, 2, 422)])
def test_api_correction_time_bounds(client, start, end, status):
    upload(client)
    key = client.post("/api/youtube", json={"url": VID}).json()["source_key"]
    response = client.post("/api/corrections", json={"source_key": key,
        "start": start, "end": end, "chord": "Am"})
    assert response.status_code == status


@pytest.mark.parametrize("options", [{"vocabulary": "unknown"}, {"max_seconds": -1}, {"limit": 1000}])
def test_api_analysis_options_are_bounded(client, options):
    assert client.post("/api/youtube", json={"url": VID, **options}).status_code == 422


@pytest.mark.parametrize("data", [{"rights_confirmed": "false"}, {"url": "https://example.com"},
                                  {"rights_basis": " "}, {"video_offset": "nan"}, {"expires_at": "1"}])
def test_api_rejects_invalid_registration(client, setup, data):
    assert upload(client, **data).status_code == 400
    assert not setup[1].list_links()


def test_api_checks_audio_contents_and_cleans_up(client, tmp_path):
    response = client.post("/api/audio-sources", data={"url": VID, "rights_basis": "자작",
        "rights_confirmed": "true", "same_recording": "true"},
        files={"file": ("fake.wav", b"not an audio file", "audio/wav")})
    assert response.status_code == 400
    assert list((tmp_path / "uploads").iterdir()) == []


def test_api_upload_size_limit_cleans_up(client, monkeypatch, tmp_path):
    monkeypatch.setattr("chordapp.api.server.MAX_UPLOAD_BYTES", 100)
    assert upload(client).status_code == 413
    assert list((tmp_path / "uploads").iterdir()) == []


def test_api_rejects_overlong_audio(client):
    response = client.post("/api/audio-sources", data={"url": VID, "rights_basis": "자작",
        "rights_confirmed": "true", "same_recording": "true"},
        files={"file": ("long.wav", audio_bytes(duration=901, sr=10), "audio/wav")})
    assert response.status_code == 400


def test_old_extraction_flag_cannot_enable_audio_download(client, monkeypatch):
    monkeypatch.setenv("CHORDAPP_ENABLE_YTDLP", "1")
    assert client.get("/api/capabilities").json()["youtube_audio"] is False
    assert client.post("/api/youtube", json={"url": VID}).json()["detail"]["code"] == "source_missing"
