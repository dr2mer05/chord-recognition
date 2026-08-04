import json

import pytest

from chordapp.models import AnalysisResult, ChordSegment
from chordapp.service.cache import AnalysisCache, file_fingerprint
from chordapp.service.chord_db import LocalJsonDatabase, normalize_query
from chordapp.theory.chord import Chord
from chordapp.theory.key import Key


def _result() -> AnalysisResult:
    return AnalysisResult(
        source="t.wav", duration=8.0, tempo=120.0, key=Key.parse("C"), key_confidence=0.8,
        segments=[
            ChordSegment(0.0, 4.0, Chord.parse("C"), 0.9, 1.0, 1, 1),
            ChordSegment(4.0, 8.0, Chord.parse("G7"), 0.7, 0.8, 2, 1),
        ],
        beat_times=[0.0, 2.0, 4.0, 6.0], downbeats=[0.0, 4.0], recognizers=["x"],
    )


def test_result_dict_roundtrip():
    original = _result()
    restored = AnalysisResult.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()


def test_uncertain_segments_flags_low_agreement():
    result = _result()
    result.segments[1].agreement = 0.3
    assert result.uncertain_segments() == [1]


def test_cache_roundtrip(tmp_path):
    cache = AnalysisCache(tmp_path / "c.sqlite3")
    key = cache.make_key("audio", "fingerprint", vocabulary="standard")
    assert cache.get(key) is None
    cache.put(key, "audio", _result().to_dict())
    assert cache.get(key)["tempo"] == 120.0
    assert cache.stats()["entries"] == 1
    assert cache.stats()["hits"] == 1


def test_cache_key_depends_on_options(tmp_path):
    cache = AnalysisCache(tmp_path / "c.sqlite3")
    a = cache.make_key("audio", "fp", vocabulary="standard")
    b = cache.make_key("audio", "fp", vocabulary="extended")
    assert a != b


def test_fingerprint_ignores_filename(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"same"), b.write_bytes(b"same")
    assert file_fingerprint(a) == file_fingerprint(b)


@pytest.mark.parametrize("artist,title", [
    ("The Beatles", "Let It Be"),
    ("the  beatles", "let it be (remastered)"),
    ("The Beatles!", "Let It Be [2009 Remaster]"),
])
def test_query_normalization(artist, title):
    assert normalize_query(artist, title) == "the beatles|let it be"


def test_local_database_lookup(tmp_path):
    path = tmp_path / "db.json"
    path.write_text(json.dumps([{
        "artist": "Demo", "title": "Loop", "key": "C", "bpm": 120, "beats_per_bar": 4,
        "chords": [["C", 4], ["Am", 4], ["F", 4], ["G", 4]],
    }]), encoding="utf-8")

    db = LocalJsonDatabase(path)
    assert len(db) == 1
    assert db.lookup("nobody", "nothing") is None

    result = db.lookup("DEMO", "loop")
    assert result is not None
    assert result.origin == "database"
    assert [s.chord.name() for s in result.segments] == ["C", "Am", "F", "G"]
    assert result.duration == pytest.approx(8.0)
    assert result.segments[1].bar == 2
