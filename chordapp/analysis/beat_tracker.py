"""비트·다운비트 추적기 어댑터와 비교용 기준선.

librosa DP/PLP는 박자표·다운비트를 제공하지 않는다. beat_this는 두 이벤트의
시각을 제공하며 박자표는 별도 해석이다. 특정 모델의 우열은 동일 데이터에서
검증해야 한다. 기존 전역 보정은 기준선으로 보존하고 원출력 보존 모드를 제공한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

import numpy as np


@dataclass
class BeatGrid:
    """비트 위치와 (알 수 있으면) 다운비트·박자표."""

    beat_times: np.ndarray
    tempo: float
    downbeat_times: Optional[np.ndarray] = None
    beats_per_bar: Optional[int] = None
    source: str = "librosa"

    def __post_init__(self):
        self.beat_times = np.asarray(self.beat_times, dtype=float)
        if (self.beat_times.ndim != 1 or self.beat_times.size < 2
                or not np.isfinite(self.beat_times).all() or self.beat_times[0] < 0
                or np.any(np.diff(self.beat_times) <= 0)):
            raise ValueError("비트 시각은 증가하는 유한한 비음수 목록이어야 합니다.")
        if not np.isfinite(self.tempo) or self.tempo <= 0:
            raise ValueError("템포는 유한한 양수여야 합니다.")
        if self.beats_per_bar is not None and (self.beats_per_bar < 1 or int(self.beats_per_bar) != self.beats_per_bar):
            raise ValueError("마디당 박 수는 양의 정수여야 합니다.")
        if self.downbeat_times is not None:
            self.downbeat_times = np.asarray(self.downbeat_times, dtype=float)
            if (self.downbeat_times.ndim != 1 or not np.isfinite(self.downbeat_times).all()
                    or np.any(self.downbeat_times < 0) or np.any(np.diff(self.downbeat_times) <= 0)):
                raise ValueError("다운비트 시각은 증가하는 유한한 비음수 목록이어야 합니다.")

    def downbeat_indices(self, fallback_phase: int = 0, beats_per_bar: int = 4) -> np.ndarray:
        """다운비트를 beat_times 안의 인덱스로. 추적기가 안 주면 위상만 적용."""
        bpb = self.beats_per_bar or beats_per_bar
        if self.downbeat_times is None or len(self.downbeat_times) == 0:
            return np.arange(fallback_phase, len(self.beat_times), bpb, dtype=int)
        # 가장 가까운 비트로 스냅
        idx = [int(np.argmin(np.abs(self.beat_times - t))) for t in self.downbeat_times]
        return np.array(sorted(set(idx)), dtype=int)


class BeatTracker(Protocol):
    name: str

    @property
    def available(self) -> bool: ...

    def track(self, audio_path: str) -> BeatGrid: ...


class BeatThisTracker:
    """beat_this 비트·다운비트 연결. legacy 보정과 모델 출력 보존을 비교한다."""

    name = "beat_this"

    def __init__(self, device: Optional[str] = None, dbn: bool = False,
                 postprocess: str = "legacy", checkpoint_path: str = "final0"):
        if postprocess not in {"legacy", "model"}:
            raise ValueError("지원하지 않는 비트 후처리 모드입니다.")
        self._device = device
        self.dbn = dbn
        self.postprocess = postprocess
        self.checkpoint_path = checkpoint_path
        self.name = "beat_this_model" if postprocess == "model" else "beat_this"
        self._model = None

    @property
    def device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @property
    def available(self) -> bool:
        try:
            from beat_this.inference import File2Beats  # noqa: F401
        except Exception:
            # Python 3.9에서는 문법 오류(TypeError)로 죽는다. 조용히 폴백.
            return False
        return True

    def _load(self):
        if self._model is None:
            from beat_this.inference import File2Beats

            self._model = File2Beats(checkpoint_path=self.checkpoint_path, device=self.device, dbn=self.dbn)
        return self._model

    def track(self, audio_path: str) -> BeatGrid:
        beats, downbeats = self._load()(audio_path)
        beats = np.asarray(beats, dtype=float)
        downbeats = np.asarray(downbeats, dtype=float)
        if beats.size < 2:
            raise RuntimeError("beat_this가 비트를 찾지 못했다")

        if self.postprocess == "model":
            # 전체 곡의 중앙 간격으로 빠른 구간을 삭제하거나 느린 구간에 비트를
            # 채우지 않는다. 변박·누락으로 마디 길이가 다르면 전역 박자표는 미정이다.
            period = float(np.median(np.diff(beats)))
            if period <= 0:
                raise ValueError("모델의 비트 간격이 올바르지 않습니다.")
            grid = BeatGrid(beats, 60.0 / period, downbeats if downbeats.size else None,
                            source=self.name)
            if downbeats.size >= 3:
                gaps = np.diff(grid.downbeat_indices())
                if gaps.size and np.all(gaps == gaps[0]) and 2 <= gaps[0] <= 12:
                    grid.beats_per_bar = int(gaps[0])
            return grid

        # 가짜 비트 하나가 이후 박 번호를 통째로 밀어버리므로 먼저 정리한다
        beats = sanitize_beats(beats)
        tempo = float(60.0 / np.median(np.diff(beats)))
        regular, meter = regularize_downbeats(beats, downbeats)
        return BeatGrid(
            beat_times=beats,
            tempo=tempo,
            downbeat_times=regular if regular is not None
            else (downbeats if downbeats.size else None),
            beats_per_bar=meter,
            source=self.name,
        )


# 박자 후보. 2박은 뺐다 — 대중음악에서 2/4는 거의 없는 데다,
# **4의 약수라서 구조적으로 유리하다.** 4/4 곡의 다운비트는 정의상 2박 격자에도
# 전부 들어맞기 때문에, 잡음이 조금만 끼면 항상 2박이 이긴다.
# (실측: 4/4 발라드의 2박 점수 0.478 vs 4박 0.232 — 도입부의 불규칙한
#  다운비트 몇 개 때문에 뒤집혔다.)
DEFAULT_METER = 4
_ALT_METERS = (3, 6)

# 4/4가 아니라고 판단하려면 이만큼 확실해야 한다. 대중음악은 4/4가 압도적이라
# 어중간한 증거로 3박·6박으로 넘어가면 손해가 더 크다.
_ALT_METER_MARGIN = 1.6


def sanitize_beats(
    beats: np.ndarray, drop_below: float = 0.6, fill_above: float = 1.6
) -> np.ndarray:
    """비트 목록에서 가짜 비트를 버리고 빠진 비트를 채운다.

    beat_this는 프레임 단위로 비트를 독립 판정해서 목록이 깨끗하지 않다. 실측:

      윤하 기다리다   342비트 중 69개(20%)가 중앙 간격의 60% 미만
      다시 여기 바닷가 521비트 중 12개
      아이유 밤편지   짧은 것 3개 + 긴 것 6개(비트를 놓친 구간)

    이게 치명적인 이유는 **개수가 밀리기 때문**이다. 기다리다에서는 0.12초짜리
    가짜 비트 하나 때문에 이후 모든 박 번호가 하나씩 밀렸고, 그래서 마디 첫 박이
    한 박 늦게 잡히고 없는 못갖춘마디가 생겼다.

    두 종류를 다룬다.
      - 간격이 너무 짧으면(중앙값의 60% 미만) 뒤 비트를 버린다. 배속으로 튄
        구간(간격이 절반)도 이 규칙으로 한 박씩 걸러진다.
      - 간격이 너무 길면(160% 초과) 그 사이를 균등하게 채운다.
    """
    beats = np.asarray(beats, dtype=float)
    if beats.size < 4:
        return beats

    period = float(np.median(np.diff(beats)))
    if period <= 0:
        return beats

    out: List[float] = [float(beats[0])]
    for t in beats[1:]:
        gap = float(t) - out[-1]
        if gap < period * drop_below:
            continue                                  # 가짜 비트
        if gap > period * fill_above:                 # 놓친 비트를 채운다
            missing = max(int(round(gap / period)) - 1, 0)
            base, step = out[-1], gap / (missing + 1)
            out.extend(base + step * k for k in range(1, missing + 1))
        out.append(float(t))
    return np.asarray(out)


def regularize_downbeats(
    beats: np.ndarray, downbeats: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """모델이 낸 다운비트를 일정한 주기로 정리한다.

    beat_this는 다운비트를 프레임 단위로 독립 판정해서 결과가 들쭉날쭉하다.
    실측에서 느린 발라드의 마디 길이 변동이 27.7%, 4박 일관성이 85%였다
    (빠른 곡은 8~9%, 94~96%로 괜찮다). 원래는 DBN 후처리가 이걸 잡아주는데
    그건 madmom을 요구해서 윈도우에서 쓸 수 없다.

    그래서 직접 편다: (박자, 위상) 조합 중 모델의 다운비트를 가장 많이 설명하는
    것을 고르고, 그 주기로 균등하게 다시 깐다. 모델의 판단을 버리는 게 아니라
    "몇 박마다"라는 제약을 씌우는 것이다.
    """
    if beats.size < 8 or downbeats.size < 3:
        return None, None

    # 각 다운비트가 몇 번째 비트인지
    idx = np.array([int(np.argmin(np.abs(beats - t))) for t in downbeats])
    idx = np.unique(idx)

    def evaluate(meter: int) -> Tuple[float, int]:
        """이 박자로 봤을 때 최선의 위상과 점수(우연 초과분)."""
        chance = 1.0 / meter
        best_phase, best_rate = 0, -1.0
        for phase in range(meter):
            rate = float(np.sum((idx % meter) == phase)) / len(idx)
            if rate > best_rate:
                best_phase, best_rate = phase, rate
        return (best_rate - chance) / (1.0 - chance), best_phase

    base_score, base_phase = evaluate(DEFAULT_METER)
    meter, phase, score = DEFAULT_METER, base_phase, base_score

    # 4/4에서 벗어나려면 뚜렷하게 나아야 한다
    for candidate in _ALT_METERS:
        alt_score, alt_phase = evaluate(candidate)
        if alt_score > max(base_score, 0.05) * _ALT_METER_MARGIN and alt_score > score:
            meter, phase, score = candidate, alt_phase, alt_score

    regular = np.arange(phase, len(beats), meter, dtype=int)
    return beats[regular], meter


def default_tracker() -> Optional[BeatTracker]:
    """설치된 선택 추적기를 준다. 우열을 의미하지 않으며 미설치 시 librosa를 쓴다."""
    tracker = BeatThisTracker()
    return tracker if tracker.available else None


class LibrosaPLPTracker:
    """기본 매개변수의 국소 펄스 후보. 다운비트·박자표는 추정하지 않는다."""

    name = "librosa_plp"
    available = True
    device = "cpu"

    def track(self, audio_path: str) -> BeatGrid:
        import librosa
        from .audio import HOP, SR, load_audio

        signal, sr = load_audio(audio_path, sr=SR)
        onset = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=HOP)
        pulse = librosa.beat.plp(onset_envelope=onset, sr=sr, hop_length=HOP)
        beats = librosa.frames_to_time(np.flatnonzero(librosa.util.localmax(pulse)), sr=sr, hop_length=HOP)
        if len(beats) < 2:
            raise RuntimeError("PLP가 충분한 비트를 찾지 못했습니다.")
        return BeatGrid(beats, float(60.0 / np.median(np.diff(beats))), source=self.name)
