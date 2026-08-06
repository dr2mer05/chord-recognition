"""오디오 로딩, 비트/다운비트, 조성 추정.

librosa만 쓴다. GPU도 외부 모델도 필요 없다.
소스 분리(Demucs)는 1단계 보류 항목이라, 대신 HPSS로 타악 성분만 걷어내
화성 악기 쪽 스펙트럼을 조금이라도 깨끗하게 만드는 선에서 처리한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..theory.key import KK_MAJOR, KK_MINOR, Key
from .beat_tracker import BeatTracker, default_tracker

# extract_features의 tracker 인자 기본값. None은 "librosa만 써라"라는 뜻이라
# "설정된 걸 알아서 골라라"와 구분해야 해서 별도 센티널을 둔다.
_USE_DEFAULT_TRACKER = object()

SR = 22050
HOP = 512

# 화성 판정용 크로마의 음역: C3~C7.
# C2부터 잡으면 베이스 기본음과 그 배음이 크로마를 지배해서 3화음만 남고
# 7음·텐션이 묻힌다. 합성 데모 기준으로 C2/6옥타브 → C3/4옥타브로 바꾸자
# 7th 인식률이 8% → 99%로 올랐다. 화성 악기(기타/피아노/패드)가 실제로
# 자리하는 대역이기도 하다. 베이스는 bass_chroma로 따로 본다.
# 다만 이 수치는 합성 음원에 맞춰 고른 것이므로, M2에서 McGill Billboard로
# 다시 스윕해야 한다.
CHROMA_FMIN = "C3"
CHROMA_OCTAVES = 4

# 다운비트 판정에서 broadband 온셋(=스네어 백비트)에 주는 감점.
# 대중음악은 2·4박에 스네어가 있으므로 broadband가 센 자리는 마디 첫 박이 아니다.
BACKBEAT_PENALTY = 0.5


@dataclass
class AudioFeatures:
    source_path: str         # 외부 인식기 어댑터가 원본 파일을 다시 읽을 때 필요
    y: np.ndarray            # 원본 모노 신호
    y_harmonic: np.ndarray   # 타악 성분 제거본
    sr: int
    duration: float
    tempo: float
    beat_times: np.ndarray   # 초 단위 비트 위치
    downbeat_indices: np.ndarray  # beat_times 안에서 마디 첫 박의 인덱스
    beats_per_bar: int
    chroma: np.ndarray       # (12, n_frames) 프레임 단위 크로마
    beat_chroma: np.ndarray  # (12, n_beats) 비트 동기 크로마 (주 입력)
    bass_chroma: np.ndarray  # (12, n_beats) 저역 크로마 (전위/슬래시 판별용)
    key: Key
    key_confidence: float
    beat_source: str = "librosa"   # 어느 추적기가 격자를 만들었는지
    # 소스 분리를 썼을 때만 채워진다. 원본 믹스 기준 크로마로, 분리본과
    # 서로 다른 오류 편향을 갖기 때문에 앙상블의 두 번째 의견으로 쓴다.
    beat_chroma_alt: Optional[np.ndarray] = None
    separated: bool = False


def load_audio(path: str, sr: int = SR, max_seconds: Optional[float] = None) -> Tuple[np.ndarray, int]:
    import librosa

    y, sr = librosa.load(path, sr=sr, mono=True, duration=max_seconds)
    if y.size == 0:
        raise ValueError(f"오디오가 비어 있음: {path}")
    return y, sr


def extract_features(
    path: str,
    sr: int = SR,
    max_seconds: Optional[float] = None,
    beats_per_bar: int = 4,
    chroma_fmin: str = CHROMA_FMIN,
    chroma_octaves: int = CHROMA_OCTAVES,
    harmonic_paths: Optional[Sequence[str]] = None,
    bass_path: Optional[str] = None,
    tempo_override: Optional[float] = None,
    downbeat_offset: int = 0,
    beat_offset: Optional[float] = None,
    tracker: Optional["BeatTracker"] = _USE_DEFAULT_TRACKER,
) -> AudioFeatures:
    """`harmonic_paths`/`bass_path`를 주면 그 스템으로 크로마를 만든다.

    `harmonic_paths`는 여러 스템을 받아 합친다. `other`(기타·건반)만 쓰면
    베이스가 빠져서 코드 근음 정보를 잃는다 — 실측에서 `other` 단독은 원본 믹스보다
    확신도가 오히려 낮았다(0.418 vs 0.433). `other + bass`(= 믹스에서 보컬·드럼만
    뺀 반주)가 맞는 조합이다.

    비트 추적과 다운비트는 **항상 원본 믹스**로 한다. 드럼이 빠지면 온셋이
    약해져서 비트 추적이 오히려 나빠지기 때문이다.
    """
    import librosa

    y, sr = load_audio(path, sr=sr, max_seconds=max_seconds)

    if tracker is _USE_DEFAULT_TRACKER:
        tracker = default_tracker()

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    manual_grid = bool(tempo_override and beat_offset is not None)

    # 더 나은 추적기(beat_this)가 있으면 그걸 쓴다. 사용자가 템포를 지정했거나
    # 설치돼 있지 않으면 librosa로 간다.
    grid = None
    if not tempo_override and tracker is not None:
        try:
            grid = tracker.track(path)
        except Exception:
            grid = None

    if grid is not None:
        beat_times = np.asarray(grid.beat_times, dtype=float)
        tempo = grid.tempo
        if grid.beats_per_bar:
            beats_per_bar = grid.beats_per_bar
    else:
        tempo, beat_frames = _beat_track(
            onset_env, sr, tempo_override, beat_offset, duration=len(y) / sr
        )
        tempo = float(np.atleast_1d(tempo)[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
        if not manual_grid:
            beat_times = _ensure_beat_grid(beat_times, tempo, len(y) / sr)
    beat_frames_grid = librosa.time_to_frames(beat_times, sr=sr, hop_length=HOP)

    def harmonic_chroma(signal: np.ndarray) -> np.ndarray:
        return _denoise_chroma(librosa.feature.chroma_cqt(
            y=signal, sr=sr, hop_length=HOP, bins_per_octave=36,
            n_octaves=chroma_octaves, fmin=librosa.note_to_hz(chroma_fmin),
        ))

    # 화성 크로마: 분리된 스템이 있으면 그걸 쓰고, 없으면 HPSS로 근사한다
    mix_harmonic, _ = librosa.effects.hpss(y, margin=(1.0, 3.0))
    separated = bool(harmonic_paths)
    if separated:
        y_harm = np.zeros(len(y), dtype=np.float32)
        for stem_path in harmonic_paths:      # type: ignore[union-attr]
            stem, _ = load_audio(stem_path, sr=sr, max_seconds=max_seconds)
            y_harm = y_harm + _match_length(stem, len(y))
    else:
        y_harm = mix_harmonic

    chroma = harmonic_chroma(y_harm)
    beat_chroma = _beat_sync(chroma, beat_frames_grid, chroma.shape[1])

    # 분리를 썼다면 원본 믹스 크로마도 남긴다 (앙상블의 두 번째 의견)
    beat_chroma_alt = None
    if separated:
        alt = harmonic_chroma(mix_harmonic)
        beat_chroma_alt = _beat_sync(alt, beat_frames_grid, alt.shape[1])

    # 베이스: 분리된 bass 스템이 있으면 그게 정확하고, 없으면 저역 CQT로 근사한다
    if bass_path is not None:
        y_bass, _ = load_audio(bass_path, sr=sr, max_seconds=max_seconds)
        y_bass = _match_length(y_bass, len(y))
        bass_cqt = librosa.feature.chroma_cqt(
            y=y_bass, sr=sr, hop_length=HOP, bins_per_octave=36, n_octaves=3,
            fmin=librosa.note_to_hz("C1"),
        )
    else:
        bass_cqt = librosa.feature.chroma_cqt(
            y=mix_harmonic, sr=sr, hop_length=HOP, bins_per_octave=36, n_octaves=2,
            fmin=librosa.note_to_hz("C1"),
        )
    bass_chroma = _beat_sync(_denoise_chroma(bass_cqt), beat_frames_grid, bass_cqt.shape[1])

    key, conf = estimate_key(chroma)
    if manual_grid:
        # 사용자가 찍은 첫 박이 곧 마디 첫 박이다. 여기서 추정을 다시 돌리면
        # 사람이 확정한 걸 기계가 뒤집는 셈이 된다.
        period = 60.0 / max(tempo, 1e-6)
        first = int(round((float(beat_offset) - float(beat_times[0])) / period))
        phase = (first + downbeat_offset) % max(beats_per_bar, 1)
        downbeats = np.arange(phase, len(beat_times), max(beats_per_bar, 1), dtype=int)
    elif grid is not None and grid.downbeat_times is not None:
        # 추적기가 다운비트를 직접 준다. 온셋 세기로 위상을 추측하던 자체
        # 휴리스틱보다 이쪽이 훨씬 낫다.
        downbeats = grid.downbeat_indices(beats_per_bar=beats_per_bar)
        if downbeat_offset:
            downbeats = np.clip(downbeats + downbeat_offset, 0, len(beat_times) - 1)
    else:
        downbeats = estimate_downbeats(
            y, sr, beat_times, beats_per_bar, onset_env=onset_env,
            phase_offset=downbeat_offset,
        )

    return AudioFeatures(
        source_path=path,
        y=y,
        y_harmonic=y_harm,
        sr=sr,
        duration=len(y) / sr,
        tempo=tempo,
        beat_times=beat_times,
        downbeat_indices=downbeats,
        beats_per_bar=beats_per_bar,
        chroma=chroma,
        beat_chroma=beat_chroma,
        bass_chroma=bass_chroma,
        key=key,
        key_confidence=conf,
        beat_chroma_alt=beat_chroma_alt,
        separated=separated,
        beat_source=grid.source if grid is not None else "librosa",
    )


def _match_length(signal: np.ndarray, target: int) -> np.ndarray:
    """스템 길이를 믹스에 맞춘다. demucs 출력이 몇 샘플 어긋나는 경우가 있다."""
    if len(signal) == target:
        return signal
    if len(signal) > target:
        return signal[:target]
    return np.pad(signal, (0, target - len(signal)))


def _denoise_chroma(chroma: np.ndarray) -> np.ndarray:
    """중앙값 필터 + 정규화. 배음 잔재와 순간적인 튐을 줄인다."""
    import librosa
    from scipy.ndimage import median_filter

    chroma = np.maximum(chroma, 0.0)
    chroma = median_filter(chroma, size=(1, 9))
    return librosa.util.normalize(chroma, norm=2, axis=0)


def _beat_sync(feature: np.ndarray, beat_frames: np.ndarray, n_frames: int) -> np.ndarray:
    """비트 구간별 중앙값. 평균보다 순간 오염에 강하다.

    출력 열 i는 정확히 beat_times[i]부터 다음 비트까지를 담는다.
    (librosa.util.sync(pad=True)를 쓰면 첫 비트 이전 구간이 0번 열로 들어와
     전체가 한 박씩 밀린다. 실제로 그 버그를 한 번 만들었다.)
    """
    import librosa

    if n_frames <= 0 or np.size(beat_frames) == 0:
        return librosa.util.normalize(feature.mean(axis=1, keepdims=True), norm=1, axis=0)

    bounds = np.clip(np.asarray(beat_frames, dtype=int), 0, n_frames)
    bounds = np.append(bounds, n_frames)
    columns = []
    for i in range(len(bounds) - 1):
        start = int(bounds[i])
        end = max(int(bounds[i + 1]), start + 1)
        columns.append(np.median(feature[:, start:end], axis=1))
    synced = np.maximum(np.column_stack(columns), 0.0)
    return librosa.util.normalize(synced, norm=1, axis=0)


def _ensure_beat_grid(beat_times: np.ndarray, tempo: float, duration: float) -> np.ndarray:
    """비트 검출이 실패했을 때 템포 기반 균등 격자로 대체.

    도입부를 채울 때 0.0을 그냥 끼워 넣으면 안 된다. 그 한 박이 격자의 위상을
    통째로 밀어서 다운비트 추정이 어긋난다(합성 데모 5곡 전부 마디 첫 박을
    3번째 박으로 잡고 있었다). 박 간격을 유지한 채 뒤로 외삽해야 한다.
    남는 한 박 미만 구간은 첫 코드 구간을 0까지 늘려서 처리한다.
    """
    if beat_times.size < 4:
        period = 60.0 / max(tempo, 40.0)
        return np.arange(0.0, duration, period)

    period = float(np.median(np.diff(beat_times)))
    if period > 0 and beat_times[0] > period:
        n_back = int(np.floor(beat_times[0] / period))
        pre = beat_times[0] - period * np.arange(n_back, 0, -1)
        beat_times = np.concatenate([pre[pre >= 0.0], beat_times])
    return beat_times


def _beat_track(
    onset_env: np.ndarray,
    sr: int,
    bpm: Optional[float] = None,
    beat_offset: Optional[float] = None,
    duration: Optional[float] = None,
):
    """비트 추적.

    `bpm`과 `beat_offset`이 둘 다 오면 추적을 아예 하지 않고 균등 격자를 만든다.
    사용자가 탭 템포로 직접 찍은 경우인데, 그때는 사람의 판단이 정답이므로
    추적기가 다시 위상을 고르게 두면 안 된다.
    """
    import librosa

    if bpm and beat_offset is not None and duration:
        period = 60.0 / float(bpm)
        start = float(beat_offset) % period          # 첫 박 이전도 격자로 덮는다
        times = np.arange(start, duration, period)
        return float(bpm), librosa.time_to_frames(times, sr=sr, hop_length=HOP)

    if bpm:
        return librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=HOP, bpm=float(bpm), trim=False
        )
    return librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sr, hop_length=HOP,
        start_bpm=TEMPO_PRIOR_BPM, trim=False,
    )


# 템포 옥타브 오류에 대한 기록.
#
# 짝수/홀수 박의 온셋 세기를 비교해 판별하는 방법을 시도했다가 되돌렸다.
# 4/4 대중음악은 킥과 스네어가 번갈아 나와서 인접 박의 세기가 원래 다르고,
# 그걸 "템포 2배"로 오판해 격자의 절반(스네어 박)만 남긴다.
# 합성 데모 한 곡이 98.8% → 31.3%로 무너졌다. 그 신호는 백비트와 템포 배수를
# 구별하지 못한다 (실측: 밤편지는 2배 오류인데도 짝/홀 비가 1.03이었다).
#
# 진짜 원인은 librosa의 템포 사전확률이 120 BPM 중심이라는 것이었다.
# 69 BPM 발라드는 138 쪽이 사전확률상 유리해서 배음으로 끌려간다.
# 사전확률을 90으로 낮추면 느린 곡은 고쳐지는데, 빠른 곡이 대신 깨진다.
# beat_this를 기준으로 실측한 결과(5곡):
#
#   곡                 정답    prior 120   prior 90
#   윤하 기다리다        69.8    136 ✗       69.8 ✓
#   아이유 밤편지        78.9    152 ✗       80.7 ✓
#   다시 여기 바닷가     130.4    129 ✓       89.1 ✗
#   실제곡4            136.4    136 ✓       89.1 ✗
#   실제곡5            111.1    112 ✓       112  ✓
#                            ---------   --------
#                              3/5         3/5
#
# **고정 사전확률로는 해결되지 않는다.** 어느 쪽으로 옮겨도 손익이 상쇄된다.
#
# 한때 90으로 낮췄다가 librosa 기본값으로 되돌렸다. 발라드를 고치는 대신 빠른
# 곡을 깨뜨리는 맞바꿈이었을 뿐이고(3/5 대 3/5), 두 곡을 보고 정한 값이라
# 일반화 근거가 없었다. 진짜 해결은 beat_this(beat_tracker.py)이고 —
# 16곡 평가에서 템포 배수 오류 0건 — 이 값은 그게 없을 때의 폴백일 뿐이다.
TEMPO_PRIOR_BPM = 120.0


def estimate_key(chroma: np.ndarray) -> Tuple[Key, float]:
    """Krumhansl-Kessler 프로파일 상관으로 조성 추정."""
    profile = chroma.mean(axis=1)
    if profile.sum() <= 0:
        return Key(0, "major"), 0.0
    profile = profile / profile.sum()

    scores: List[Tuple[float, Key]] = []
    for tonic in range(12):
        rotated = np.roll(profile, -tonic)
        for mode, template in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            t = np.asarray(template)
            t = (t - t.mean()) / (t.std() + 1e-9)
            r = (rotated - rotated.mean()) / (rotated.std() + 1e-9)
            scores.append((float(np.dot(t, r) / 12.0), Key(tonic, mode)))

    scores.sort(key=lambda s: -s[0])
    best_score, best_key = scores[0]
    runner_up = scores[1][0]
    confidence = float(np.clip((best_score - runner_up) * 3.0 + 0.35, 0.0, 1.0))
    return best_key, confidence


def estimate_downbeats(
    y: np.ndarray,
    sr: int,
    beat_times: np.ndarray,
    beats_per_bar: int = 4,
    onset_env: Optional[np.ndarray] = None,
    phase_offset: int = 0,
) -> np.ndarray:
    """마디 첫 박 추정.

    전용 다운비트 모델(madmom 등)은 설치 부담이 커서 위상(phase) 선택으로 근사한다.

    판정 기준: **저역이 세고(킥) 전대역이 세지 않은(스네어가 아닌)** 자리.

    전대역 온셋을 더하면 안 된다. 스네어가 broadband 플럭스를 가장 크게 만드는데
    스네어는 여린박(2·4박)에 있기 때문이다. 실제로 전대역을 가산했을 때
    5곡 전부 마디 첫 박 대신 4박을 골랐다. 빼야 맞는 항이었다.

    온셋 곡선은 실제 어택보다 1~2 프레임 뒤에서 정점을 찍으므로 비트 시각을
    정확히 샘플링하면 안 되고, 주변 창의 최댓값을 봐야 한다.
    """
    import librosa

    if beat_times.size < beats_per_bar * 2:
        return np.array([0], dtype=int)

    if onset_env is None:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    # 저역만 본 온셋 (킥/베이스). 마디 첫 박 단서로 가장 강력하다.
    # mel 필터로 자르면 안 된다 — 200Hz 이하에 mel 밴드를 여러 개 넣으면 대부분이
    # 빈 필터가 되어 온셋이 쓰레기값이 된다(Empty filters 경고). STFT를 직접 자른다.
    spec = np.abs(librosa.stft(y, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr)
    low_band = spec[freqs <= 250.0]
    low_env = librosa.onset.onset_strength(
        S=librosa.amplitude_to_db(low_band, ref=np.max), sr=sr, hop_length=HOP
    )

    def phase_scores(env: np.ndarray) -> np.ndarray:
        peaks = _window_peak(env, beat_times, sr)
        scores = np.array([peaks[p::beats_per_bar].mean() for p in range(beats_per_bar)])
        span = scores.max() - scores.min()
        return (scores - scores.min()) / span if span > 0 else np.zeros_like(scores)

    combined = phase_scores(low_env) - BACKBEAT_PENALTY * phase_scores(onset_env)
    best_phase = (int(np.argmax(combined)) + int(phase_offset)) % beats_per_bar
    return np.arange(best_phase, len(beat_times), beats_per_bar, dtype=int)


def _window_peak(env: np.ndarray, beat_times: np.ndarray, sr: int, ahead: int = 3) -> np.ndarray:
    """각 비트 주변 창의 온셋 최댓값. 온셋 곡선의 지연을 흡수한다."""
    import librosa

    frames = librosa.time_to_frames(beat_times, sr=sr, hop_length=HOP)
    out = np.zeros(len(frames), dtype=np.float32)
    for i, f in enumerate(frames):
        lo = max(int(f) - 1, 0)
        hi = min(int(f) + ahead + 1, len(env))
        if hi > lo:
            out[i] = float(env[lo:hi].max())
    return out
