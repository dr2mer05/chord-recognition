import pytest

from chordapp.service.youtube import (
    AudioFetchDisabled,
    DisabledAudioProvider,
    DisabledSearch,
    InvalidVideoUrl,
    SearchUnavailable,
    VideoInfo,
    YtDlpProvider,
    _parse_iso8601_duration,
    parse_video_id,
    split_artist_title,
)

VID = "dQw4w9WgXcQ"


@pytest.mark.parametrize("text", [
    VID,
    f"https://www.youtube.com/watch?v={VID}",
    f"http://youtube.com/watch?v={VID}",
    f"https://m.youtube.com/watch?v={VID}&t=42s",
    f"https://music.youtube.com/watch?v={VID}&list=RDAMVM",
    f"https://youtu.be/{VID}",
    f"https://youtu.be/{VID}?t=90",
    f"https://www.youtube.com/shorts/{VID}",
    f"https://www.youtube.com/embed/{VID}",
    f"https://www.youtube.com/live/{VID}",
    f"www.youtube.com/watch?v={VID}",
    f"  https://www.youtube.com/watch?v={VID}  ",
])
def test_parse_video_id(text):
    assert parse_video_id(text) == VID


@pytest.mark.parametrize("text", [
    "", "   ", "not a url",
    "https://vimeo.com/12345",
    "https://www.youtube.com/watch?v=tooshort",
    "https://www.youtube.com/results?search_query=abc",
    "https://evil.example.com/watch?v=" + VID,
])
def test_parse_video_id_rejects(text):
    with pytest.raises(InvalidVideoUrl):
        parse_video_id(text)


@pytest.mark.parametrize("title,channel,expected", [
    ("아이유(IU) - 밤편지", "1theK", ("아이유", "밤편지")),
    ("Radiohead - Creep (Official Music Video)", "Radiohead", ("Radiohead", "Creep")),
    ("[MV] BOL4 - Some", "채널", ("BOL4", "Some")),
    ("Song Title Only", "The Band", ("The Band", "Song Title Only")),
    ("Artist – Title [Official Audio]", "ch", ("Artist", "Title")),
    # 1theK 등 국내 뮤직비디오 채널 형식
    ("[MV] IU(아이유) _ Through the Night(밤편지)", "1theK", ("IU", "Through the Night")),
    ("[MV] BOL4(볼빨간사춘기) _ Some(썸 탈래)", "1theK", ("BOL4", "Some")),
    # 방송/무대 정보가 꼬리로 붙는 경우
    ("아이유 - 밤편지 [유희열의 스케치북] | KBS 200918 방송", "KBS Kpop", ("아이유", "밤편지")),
    # 유튜브 자동 생성 아티스트 채널 (' - Topic')
    ("Through the Night (밤편지)", "IU - Topic", ("IU", "Through the Night")),
    ("Some Song", "ArtistVEVO", ("Artist", "Some Song")),
])
def test_split_artist_title(title, channel, expected):
    assert split_artist_title(title, channel) == expected


@pytest.mark.parametrize("text,seconds", [
    ("PT4M13S", 253.0), ("PT1H2M3S", 3723.0), ("PT45S", 45.0), ("", None),
])
def test_iso_duration(text, seconds):
    assert _parse_iso8601_duration(text) == seconds


def test_video_info_dict_has_playable_url_and_guesses():
    info = VideoInfo(VID, title="Artist - Title (Official)", channel="ch", duration=210.0)
    data = info.to_dict()
    assert data["url"] == f"https://www.youtube.com/watch?v={VID}"
    assert data["guessed_artist"] == "Artist"
    assert data["guessed_title"] == "Title"
    assert VID in data["thumbnail"]


def test_search_disabled_explains_how_to_enable():
    with pytest.raises(SearchUnavailable, match="YOUTUBE_API_KEY"):
        DisabledSearch().search("아무거나")


def test_metadata_falls_back_to_oembed_without_api_key(monkeypatch):
    """키가 없어도 제목은 공식 oEmbed로 받아 코드 DB 조회까지 갈 수 있어야 한다."""
    calls = []

    def fake(video_id, timeout=8.0):
        calls.append(video_id)
        return VideoInfo(video_id, title="A - B", channel="ch")

    monkeypatch.setattr("chordapp.service.youtube.fetch_oembed", fake)
    info = DisabledSearch().metadata(VID)
    assert calls == [VID]
    assert info is not None and info.title == "A - B"


def test_oembed_returns_none_when_offline(monkeypatch):
    """네트워크가 막혀도 예외를 던지지 말고 조용히 None이어야 한다."""
    import urllib.error

    from chordapp.service.youtube import fetch_oembed

    def boom(*args, **kwargs):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert fetch_oembed(VID) is None


def test_audio_fetch_is_off_by_default(tmp_path):
    """약관 검토 전까지 오디오 추출은 꺼져 있어야 한다."""
    with pytest.raises(AudioFetchDisabled, match="CHORDAPP_ENABLE_YTDLP"):
        DisabledAudioProvider().fetch(VID, tmp_path)


def test_ytdlp_provider_refuses_without_env_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("CHORDAPP_ENABLE_YTDLP", raising=False)
    provider = YtDlpProvider()
    assert provider.enabled is False
    with pytest.raises(AudioFetchDisabled):
        provider.fetch(VID, tmp_path)


def test_ytdlp_provider_needs_the_binary_even_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("CHORDAPP_ENABLE_YTDLP", "1")
    monkeypatch.setattr("shutil.which", lambda name: None)
    provider = YtDlpProvider()
    assert provider.enabled is False
    with pytest.raises(AudioFetchDisabled, match="yt-dlp"):
        provider.fetch(VID, tmp_path)


def test_service_caches_by_video_id(tmp_path):
    """캐시 키가 영상 ID 기준이어야 URL 형태가 달라도 같은 캐시를 맞힌다."""
    from chordapp.service.cache import AnalysisCache
    from chordapp.service.pipeline import NAMESPACE_YOUTUBE, ChordService

    service = ChordService(cache=AnalysisCache(tmp_path / "c.sqlite3"))
    a = service.cache.make_key(NAMESPACE_YOUTUBE, parse_video_id(f"https://youtu.be/{VID}"),
                               vocabulary="standard")
    b = service.cache.make_key(NAMESPACE_YOUTUBE, parse_video_id(f"https://m.youtube.com/watch?v={VID}&t=9"),
                               vocabulary="standard")
    assert a == b


class _StubSearch:
    """API 키 없이 oEmbed로 제목만 얻은 상황을 흉내낸다."""

    available = False

    def __init__(self, info):
        self.info = info

    def search(self, query, limit=8):
        raise SearchUnavailable("stub")

    def metadata(self, video_id):
        return self.info


class _ExplodingAudio:
    enabled = True

    def fetch(self, video_id, dest_dir):
        raise AssertionError("코드 DB가 HIT면 오디오 분석까지 가면 안 된다")


def _db(tmp_path):
    import json

    from chordapp.service.chord_db import LocalJsonDatabase

    path = tmp_path / "db.json"
    path.write_text(json.dumps([{
        "artist": "Demo Artist", "title": "Demo Song", "key": "C", "bpm": 120,
        "beats_per_bar": 4, "chords": [["C", 4], ["G", 4], ["Am", 4], ["F", 4]],
    }]), encoding="utf-8")
    return LocalJsonDatabase(path)


def test_youtube_link_hits_chord_db_without_any_credentials(tmp_path):
    """API 키도 yt-dlp도 없이, 링크 → 제목 → 코드 DB HIT 경로가 살아 있어야 한다."""
    from chordapp.service.cache import AnalysisCache
    from chordapp.service.pipeline import ChordService

    info = VideoInfo(VID, title="Demo Artist - Demo Song (Official Video)", channel="Demo Artist")
    service = ChordService(
        cache=AnalysisCache(tmp_path / "c.sqlite3"), database=_db(tmp_path),
        search=_StubSearch(info), audio_provider=_ExplodingAudio(),
    )

    result, video = service.analyze_youtube(f"https://youtu.be/{VID}")
    assert result.origin == "database"
    assert [s.chord.name() for s in result.segments] == ["C", "G", "Am", "F"]
    assert video is not None and video.video_id == VID

    cached, _ = service.analyze_youtube(f"https://www.youtube.com/watch?v={VID}&t=30")
    assert cached.origin == "cache" and cached.cached is True


def test_service_reports_audio_disabled(tmp_path, monkeypatch):
    """플래그가 없으면 꺼져 있어야 한다.

    개발자 로컬 .env가 켜 놨더라도 이 계약은 유지돼야 하므로 환경을 비우고 본다.
    """
    monkeypatch.delenv("CHORDAPP_ENABLE_YTDLP", raising=False)

    from chordapp.service.cache import AnalysisCache
    from chordapp.service.pipeline import ChordService
    from chordapp.service.youtube import default_audio_provider

    service = ChordService(cache=AnalysisCache(tmp_path / "c.sqlite3"),
                           audio_provider=default_audio_provider())
    assert service.audio_provider.enabled is False
    with pytest.raises(AudioFetchDisabled):
        service.analyze_youtube(f"https://youtu.be/{VID}")
