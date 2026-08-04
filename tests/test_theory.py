import pytest

from chordapp.theory.chord import Chord, simplify_to_triad
from chordapp.theory.key import Key
from chordapp.theory.substitution import Context, rule_tritone_substitution, suggest


# --- 파싱 / 표기 ------------------------------------------------------

@pytest.mark.parametrize("text,root,quality,bass", [
    ("C", 0, "maj", None),
    ("Am", 9, "min", None),
    ("F#m7b5", 6, "hdim7", None),
    ("Bb7", 10, "7", None),
    ("C/E", 0, "maj", 4),
    ("Dm7/G", 2, "min7", 7),
    ("Cmaj7", 0, "maj7", None),
    ("G7sus4", 7, "7sus4", None),
    ("N.C.", 0, "N", None),
])
def test_parse(text, root, quality, bass):
    chord = Chord.parse(text)
    assert (chord.root, chord.quality, chord.bass) == (root, quality, bass)


@pytest.mark.parametrize("text", ["C", "Am7", "F#dim7", "Bbmaj9", "C/E", "G7"])
def test_name_roundtrip(text):
    assert Chord.parse(Chord.parse(text).name()) == Chord.parse(text)


@pytest.mark.parametrize("text,expected", [
    ("C:maj", "C"),
    ("A:min", "Am"),
    ("G:7", "G7"),
    ("D:min7", "Dm7"),
    ("F:maj7", "Fmaj7"),
    ("B:hdim7", "Bm7b5"),
    ("C:maj6", "C6"),
    ("E:sus4", "Esus4"),
    ("C:maj(9)", "C"),          # 괄호 추가음은 축약형으로 근사
    ("C:maj/3", "C/E"),         # 베이스가 근음 기준 음정으로 온다
    ("C:maj/5", "C/G"),
    ("A:min7/b7", "Am7/G"),
    ("F:maj/E", "F/E"),         # 음이름 베이스도 허용
    ("N", "N.C."),
    ("X", "N.C."),
])
def test_parse_mirex_labels(text, expected):
    """autochord/Chordino 출력과 McGill Billboard .lab이 모두 이 형식이다."""
    assert Chord.parse(text).name() == expected


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        Chord.parse("Hm")
    with pytest.raises(ValueError):
        Chord.parse("")


def test_pitch_classes():
    assert Chord.parse("C").pitch_classes() == (0, 4, 7)
    assert Chord.parse("Cmaj7").pitch_classes() == (0, 4, 7, 11)
    assert Chord.parse("C/E").pitch_classes() == (0, 4, 7)
    assert Chord.parse("C/D").pitch_classes() == (0, 2, 4, 7)


def test_simplify_to_triad():
    assert simplify_to_triad(Chord.parse("Cmaj9")) == Chord.parse("C")
    assert simplify_to_triad(Chord.parse("Am11")) == Chord.parse("Am")
    assert simplify_to_triad(Chord.parse("G13")) == Chord.parse("G")


# --- 조성 -------------------------------------------------------------

def test_diatonic_triads_c_major():
    names = [c.name() for c in Key.parse("C").diatonic_triads()]
    assert names == ["C", "Dm", "Em", "F", "G", "Am", "Bdim"]


def test_diatonic_sevenths_a_minor():
    names = [c.name() for c in Key.parse("Am").diatonic_sevenths()]
    assert names == ["Am7", "Bm7b5", "Cmaj7", "Dm7", "Em7", "Fmaj7", "G7"]


@pytest.mark.parametrize("key,chord,roman", [
    ("C", "C", "I"),
    ("C", "Dm", "ii"),
    ("C", "G7", "V7"),
    ("C", "Bdim", "vii°"),
    ("C", "Bb", "bVII"),
    ("C", "Ab", "bVI"),
    ("C", "F#dim", "#iv°"),
    ("Am", "Am", "i"),
    ("Am", "F", "VI"),
    ("Am", "E7", "V7"),
])
def test_roman(key, chord, roman):
    assert Key.parse(key).roman(Chord.parse(chord)) == roman


def test_relative_and_parallel():
    assert Key.parse("C").relative() == Key.parse("Am")
    assert Key.parse("Am").relative() == Key.parse("C")
    assert Key.parse("C").parallel() == Key.parse("Cm")


@pytest.mark.parametrize("progression,expected", [
    (["Dm7", "G7", "Cmaj7"], "C"),
    (["Dm7", "G7", "Cmaj7", "A7"], "C"),   # 끝에 V7/ii가 붙어도 흔들리면 안 된다
    (["C", "G", "Am", "F"], "C"),
    (["G", "D", "Em", "C"], "G"),
    (["F", "Bb", "C", "F"], "F"),
    (["Cm", "Ab", "Eb", "Bb"], "Cm"),
])
def test_key_estimation_from_chords(progression, expected):
    from chordapp.theory.key import key_from_chords

    estimated = key_from_chords([Chord.parse(t) for t in progression])
    assert estimated.name() == expected


@pytest.mark.parametrize("progression", [
    ["Am", "F", "C", "G"],      # 나란한조 진행은 C / Am 어느 쪽으로 읽어도 맞다
    ["Am", "Dm", "E7", "Am"],
])
def test_relative_key_ambiguity_is_resolved_to_one_of_the_pair(progression):
    from chordapp.theory.key import key_from_chords

    estimated = key_from_chords([Chord.parse(t) for t in progression])
    assert estimated.name() in ("C", "Am")


def test_flat_spelling_follows_key():
    assert Chord.parse("Bb").name(Key.parse("F").prefer_flat) == "Bb"
    assert Chord.parse("Bb").name(Key.parse("C").prefer_flat) == "A#"


# --- 대체 코드 엔진 ---------------------------------------------------

def test_suggest_returns_ranked_unique_candidates():
    subs = suggest(Chord.parse("Am"), Key.parse("C"),
                   prev=Chord.parse("F"), next=Chord.parse("F"), limit=8)
    assert subs
    assert all(0.0 <= s.score <= 100.0 for s in subs)
    assert subs == sorted(subs, key=lambda s: (-s.score, s.chord.root))
    keys = [(s.chord.root, s.chord.quality, s.chord.bass, s.kind) for s in subs]
    assert len(keys) == len(set(keys))
    assert all(s.reason and s.label for s in subs)


def test_suggest_never_returns_the_same_chord():
    chord = Chord.parse("G7")
    subs = suggest(chord, Key.parse("C"), next=Chord.parse("C"), limit=12)
    assert all(not (s.chord == chord and s.kind == "replace") for s in subs)


def test_no_chord_has_no_substitutions():
    assert suggest(Chord.parse("N.C."), Key.parse("C")) == []


def test_tritone_sub_shares_the_tritone():
    ctx = Context(Chord.parse("G7"), Key.parse("C"), next=Chord.parse("C"))
    subs = rule_tritone_substitution(ctx)
    assert [s.chord.name() for s in subs] == ["C#7"]
    original = set(Chord.parse("G7").pitch_classes())
    # 3음(B)과 7음(F)이 그대로 유지되는 게 트라이톤 대체의 정의
    assert {11, 5} <= original & set(subs[0].chord.pitch_classes())


def test_secondary_dominant_targets_the_next_chord():
    subs = suggest(Chord.parse("Am"), Key.parse("C"), next=Chord.parse("Dm"), limit=14)
    labels = {s.label: s.chord.name() for s in subs}
    assert labels.get("V7/ii") == "A7"


def test_diatonic_substitution_stays_in_key():
    key = Key.parse("C")
    subs = [s for s in suggest(Chord.parse("C"), key, next=Chord.parse("F"), limit=14)
            if s.category == "diatonic-sub"]
    assert subs
    assert all(key.is_diatonic(s.chord) for s in subs)


def test_tension_extensions_stay_in_key_for_non_dominants():
    key = Key.parse("C")
    subs = [s for s in suggest(Chord.parse("F"), key, prev=Chord.parse("C"), limit=14)
            if s.category == "tension"]
    assert {s.chord.name() for s in subs} >= {"Fmaj7", "F6", "Fadd9"}
    assert all(key.is_diatonic(s.chord) for s in subs)


def test_dominant_gets_seventh_when_it_resolves_down_a_fifth():
    subs = suggest(Chord.parse("G"), Key.parse("C"), next=Chord.parse("C"), limit=14)
    assert "G7" in {s.chord.name() for s in subs}


def test_passing_diminished_is_an_insertion():
    subs = [s for s in suggest(Chord.parse("C"), Key.parse("C"), next=Chord.parse("Dm"), limit=14)
            if s.category == "passing-dim"]
    assert len(subs) == 1
    assert subs[0].kind == "insert_after"
    assert subs[0].chord.name() == "C#dim7"


def test_modal_interchange_borrows_from_parallel_key():
    subs = [s for s in suggest(Chord.parse("F"), Key.parse("C"), next=Chord.parse("C"), limit=20)
            if s.category == "modal-interchange"]
    borrowed = {s.chord.name() for s in subs}
    assert "Fm" in borrowed  # iv 차용은 IV 자리에서 가장 흔한 수법
    assert all(not Key.parse("C").is_diatonic(s.chord) for s in subs)


def test_inversion_notes_bass_line_motion():
    subs = suggest(Chord.parse("G"), Key.parse("C"), next=Chord.parse("C"), limit=14)
    inv = next(s for s in subs if s.chord.name() == "G/B")
    assert "베이스 라인" in inv.tags


def test_diversify_mixes_categories():
    subs = suggest(Chord.parse("C"), Key.parse("C"), next=Chord.parse("F"), limit=6)
    assert len({s.category for s in subs}) >= 3


def test_progression_context_changes_suggestions():
    chord, key = Chord.parse("C"), Key.parse("C")
    to_f = {s.chord.name() for s in suggest(chord, key, next=Chord.parse("F"), limit=10)}
    to_am = {s.chord.name() for s in suggest(chord, key, next=Chord.parse("Am"), limit=10)}
    assert to_f != to_am
