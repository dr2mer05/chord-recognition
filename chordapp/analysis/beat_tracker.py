"""비트·다운비트 추적기.

librosa 기본 추적기는 이 용도에 한계가 뚜렷하다.

  1. **템포 배수를 고정 사전확률로 정한다.** `start_bpm`이 기본 120이라 느린
     발라드는 배음으로 끌려간다(69 BPM 곡을 136으로). 그렇다고 90으로 낮추면
     이번엔 빠른 곡이 절반이 된다(130 BPM 곡을 89로). 실측 5곡에서 어느 쪽도
     3/5를 못 넘겼다 — **고정 사전확률로는 원리적으로 해결되지 않는다.**
  2. **다운비트를 아예 안 준다.** 온셋 세기로 위상을 고르는 자체 휴리스틱을
     붙였는데, 합성음에서는 맞아도 실제 녹음에서는 자주 어긋난다.
  3. **박자표를 안 준다.**

`beat_this`(ISMIR 2024, 트랜스포머)는 셋을 한 번에 해결한다. 실측에서 위 5곡의
템포를 모두 맞혔고 다운비트도 함께 낸다. GPU에서 4분 곡이 1~3초.

**설치 조건**: `beat_this`는 Python 3.10+ 문법을 쓴다. 3.9에서는 import 자체가
실패하므로 자동으로 librosa로 폴백한다. 파이썬을 올리면 그 순간부터 켜진다.
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
    """beat_this (ISMIR 2024). 비트·다운비트·박자표를 한 번에 준다."""

    name = "beat_this"

    def __init__(self, device: Optional[str] = None, dbn: bool = False):
        self._device = device
        self.dbn = dbn
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

            self._model = File2Beats(device=self.device, dbn=self.dbn)
        return self._model

    def track(self, audio_path: str) -> BeatGrid:
        beats, downbeats = self._load()(audio_path)
        beats = np.asarray(beats, dtype=float)
        downbeats = np.asarray(downbeats, dtype=float)
        if beats.size < 2:
            raise RuntimeError("beat_this가 비트를 찾지 못했다")

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
    """쓸 수 있으면 더 나은 추적기를 준다. 없으면 None(=librosa 사용)."""
    tracker = BeatThisTracker()
    return tracker if tracker.available else None
