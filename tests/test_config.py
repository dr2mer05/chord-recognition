import pytest

from chordapp.config import load_env, parse_env, status


def test_parse_env_ignores_comments_and_blanks():
    text = """
    # 주석
    YOUTUBE_API_KEY=abc123

    CHORDAPP_ENABLE_YTDLP=1
    잘못된줄
    QUOTED="with spaces"
    SINGLE='x'
    """
    assert parse_env(text) == {
        "YOUTUBE_API_KEY": "abc123",
        "CHORDAPP_ENABLE_YTDLP": "1",
        "QUOTED": "with spaces",
        "SINGLE": "x",
    }


def test_load_env_does_not_clobber_real_env_vars(tmp_path, monkeypatch):
    """진짜 환경변수가 항상 이겨야 한다 (운영에서 .env가 배포 설정을 덮으면 안 된다)."""
    env = tmp_path / ".env"
    env.write_text("YOUTUBE_API_KEY=from_file\nOTHER=from_file\n", encoding="utf-8")
    monkeypatch.setenv("YOUTUBE_API_KEY", "from_environment")
    monkeypatch.delenv("OTHER", raising=False)

    applied = load_env(env)
    assert "YOUTUBE_API_KEY" not in applied
    assert applied["OTHER"] == "from_file"

    import os

    assert os.environ["YOUTUBE_API_KEY"] == "from_environment"


def test_load_env_missing_file_is_fine(tmp_path):
    assert load_env(tmp_path / "nope.env") == {}


def test_status_never_leaks_the_key(monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "SUPERSECRETVALUE")
    info = status()
    assert info["youtube_api_key_set"] is True
    assert "SUPERSECRETVALUE" not in repr(info)
    assert info["youtube_api_key_hint"] == "…ALUE"


def test_search_backend_picks_up_key_without_reconstruction(monkeypatch):
    """.env를 고치고 서버만 재시작하면 되도록, 키는 매번 환경변수에서 읽는다."""
    from chordapp.service.youtube import DataApiSearch

    backend = DataApiSearch()
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    assert backend.available is False
    monkeypatch.setenv("YOUTUBE_API_KEY", "k")
    assert backend.available is True
