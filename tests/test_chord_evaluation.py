"""독립 정답의 범위·화음 종류·전위·코드 경계를 구분한다."""

import pytest

from chordapp.evaluation import evaluate, load_lab
from chordapp.theory.chord import Chord


def interval(start, end, label):
    return start, end, None if label == 'N' else Chord.parse(label)


@pytest.mark.parametrize('line', ['0 1 X', '0 1 unknown', '0 1', '-1 1 C',
    '0 nan C', '2 1 C', '0 2 C\n1 3 G'])
def test_invalid_truth_is_rejected_instead_of_becoming_no_chord(tmp_path, line):
    lab = tmp_path / 'truth.lab'
    lab.write_text(line)
    with pytest.raises(ValueError):
        load_lab(lab)


@pytest.mark.parametrize('hop', [0, -1, float('nan'), float('inf')])
def test_invalid_sampling_interval_rejected(hop):
    with pytest.raises(ValueError):
        evaluate([interval(0, 1, 'C')], [], hop=hop)


def test_unannotated_gap_is_excluded_and_missing_prediction_is_not_no_chord():
    reference = [interval(0, 1, 'C'), interval(3, 4, 'G')]
    report = evaluate(reference, reference)
    assert report.duration == 2
    assert report.scores['seventh'] == 1
    assert report.boundaries['n_ref'] == 0
    missing = evaluate([interval(0, 1, 'N')], [])
    assert missing.scores['root'] == 0
    assert missing.confusions[0][1] == 'X'


def test_inversion_quality_and_boundary_scores_are_distinct():
    reference = [interval(0, 1, 'C/E'), interval(1, 2, 'Cmaj7')]
    estimate = [interval(0, .4, 'C'), interval(.4, 1.04, 'C'), interval(1.04, 2, 'Cmaj7')]
    report = evaluate(reference, estimate)
    assert report.scores['root'] == 1
    assert report.scores['seventh'] > .95
    assert report.exact_chord < .5
    assert report.by_quality['maj']['accuracy'] == 1
    assert report.boundaries['n_ref'] == report.boundaries['n_est'] == 1
    assert report.boundaries['f'] == 1


def test_late_boundary_fails_even_when_labels_match_most_frames():
    reference = [interval(0, 1, 'C'), interval(1, 2, 'G')]
    estimate = [interval(0, 1.25, 'C'), interval(1.25, 2, 'G')]
    report = evaluate(reference, estimate)
    assert report.scores['seventh'] > .8
    assert report.boundaries['f'] == 0
