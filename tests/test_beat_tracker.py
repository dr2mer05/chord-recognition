"""박자 정규화 테스트.

모델이 낸 다운비트는 프레임 단위 독립 판정이라 들쭉날쭉하다. 이를 일정한
주기로 펴는 로직인데, 여기서 틀리면 마디가 통째로 어긋난다.
"""

import numpy as np
import pytest

from chordapp.analysis.beat_tracker import (
    DEFAULT_METER,
    BeatGrid,
    regularize_downbeats,
    sanitize_beats,
)


# --- 비트 목록 정리 ---------------------------------------------------
#
# 가짜 비트 하나가 이후 박 번호를 통째로 밀어서 마디가 어긋난다.
# 실측: 기다리다 342비트 중 69개가 중앙 간격의 60% 미만이었고, 그중 하나 때문에
# 마디 첫 박이 한 박 늦게 잡히고 없는 못갖춘마디가 생겼다.

def test_clean_list_is_untouched():
    b = np.arange(40) * 0.86
    assert np.allclose(sanitize_beats(b), b)


def test_spurious_beat_is_dropped():
    b = np.arange(40) * 0.86
    dirty = np.sort(np.append(b, b[18] + 0.12))     # 0.12초짜리 가짜 비트
    clean = sanitize_beats(dirty)
    assert len(clean) == len(b)
    assert np.allclose(clean, b)


def test_missing_beat_is_filled():
    b = np.arange(40) * 0.86
    holed = np.delete(b, 20)
    clean = sanitize_beats(holed)
    assert len(clean) == len(b)
    assert np.allclose(clean, b, atol=1e-6)


def test_double_time_run_is_thinned():
    """배속으로 튄 구간은 한 박씩 걸러진다.

    실측: 기다리다 208초 이후에 간격이 0.42~0.46초(정상 0.86초의 절반)로 튀는
    구간이 있었다.
    """
    period = 0.86
    normal = np.arange(40) * period
    fast = normal[-1] + np.arange(1, 13) * (period / 2)
    clean = sanitize_beats(np.concatenate([normal, fast]))
    gaps = np.diff(clean)
    assert gaps.min() > period * 0.6


def test_thinning_needs_the_normal_tempo_to_be_dominant():
    """알려진 한계: 배속 구간이 곡의 절반을 넘으면 중앙 간격 자체가 뒤집힌다.

    그 경우 무엇이 '정상'인지 정의할 수 없으므로 손대지 않는다.
    곡 전체가 그 템포라면 그게 맞는 해석일 수도 있다.
    """
    period = 0.86
    normal = np.arange(8) * period
    fast = normal[-1] + np.arange(1, 25) * (period / 2)   # 배속이 다수
    clean = sanitize_beats(np.concatenate([normal, fast]))
    assert np.median(np.diff(clean)) == pytest.approx(period / 2, abs=0.01)


def test_sanitized_gaps_are_all_plausible():
    rng = np.random.default_rng(0)
    b = np.cumsum(rng.normal(0.86, 0.02, 200))
    dirty = np.sort(np.concatenate([b, b[::17] + 0.1]))
    gaps = np.diff(sanitize_beats(dirty))
    med = np.median(gaps)
    assert gaps.min() > med * 0.6 and gaps.max() < med * 1.6


def test_too_short_list_is_returned_as_is():
    b = np.array([0.0, 0.9, 1.8])
    assert np.allclose(sanitize_beats(b), b)


def beats(n=64, period=0.5):
    return np.arange(n) * period


def test_clean_four_four_is_recovered():
    b = beats()
    downs = b[::4]
    regular, meter = regularize_downbeats(b, downs)
    assert meter == 4
    assert np.allclose(regular, b[::4])


def test_offset_phase_is_recovered():
    b = beats()
    downs = b[1::4]          # 마디가 1번 비트에서 시작
    regular, meter = regularize_downbeats(b, downs)
    assert meter == 4
    assert np.allclose(regular, b[1::4])


def test_noisy_downbeats_still_give_four():
    """일부가 2박 간격으로 튀어도 4/4로 봐야 한다.

    2박은 4의 약수라서 4/4 곡의 다운비트가 정의상 2박 격자에도 전부 들어맞는다.
    잡음이 조금만 껴도 2박이 이기는 구조라, 실제로 4/4 발라드를 2박으로 잡았다.
    """
    b = beats()
    downs = np.concatenate([b[:16:2], b[17::4]])   # 앞부분만 2박 간격
    regular, meter = regularize_downbeats(b, downs)
    assert meter == 4


def test_two_four_is_never_chosen():
    """2/4는 후보에서 뺐다 — 대중음악에 거의 없고 구조적으로 유리하다."""
    b = beats()
    regular, meter = regularize_downbeats(b, b[::2])
    assert meter == DEFAULT_METER == 4


def test_clear_three_four_is_accepted():
    """왈츠처럼 증거가 뚜렷하면 3박으로 간다."""
    b = beats(n=90)
    regular, meter = regularize_downbeats(b, b[::3])
    assert meter == 3
    assert np.allclose(regular, b[::3])


def test_output_is_perfectly_regular():
    b = beats()
    downs = np.array([b[0], b[2], b[4], b[9], b[13], b[17], b[21]])
    regular, meter = regularize_downbeats(b, downs)
    gaps = np.diff(regular)
    assert np.allclose(gaps, gaps[0]), "정규화 결과는 등간격이어야 한다"


def test_too_little_evidence_returns_nothing():
    assert regularize_downbeats(beats(4), beats(4)[::2]) == (None, None)
    assert regularize_downbeats(beats(64), np.array([0.0])) == (None, None)


# --- BeatGrid ---------------------------------------------------------

def test_grid_maps_downbeat_times_to_indices():
    b = beats()
    grid = BeatGrid(beat_times=b, tempo=120.0, downbeat_times=b[1::4], beats_per_bar=4)
    assert np.array_equal(grid.downbeat_indices(), np.arange(1, len(b), 4))


def test_grid_falls_back_to_phase_when_model_gives_nothing():
    b = beats()
    grid = BeatGrid(beat_times=b, tempo=120.0)
    idx = grid.downbeat_indices(fallback_phase=2, beats_per_bar=4)
    assert np.array_equal(idx, np.arange(2, len(b), 4))
