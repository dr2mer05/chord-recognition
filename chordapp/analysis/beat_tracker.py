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

        tempo = float(60.0 / np.median(np.diff(beats)))
        return BeatGrid(
            beat_times=beats,
            tempo=tempo,
            downbeat_times=downbeats if downbeats.size else None,
            beats_per_bar=_infer_meter(beats, downbeats),
            source=self.name,
        )


def _infer_meter(beats: np.ndarray, downbeats: np.ndarray) -> Optional[int]:
    """다운비트 사이에 비트가 몇 개인지 세서 박자표를 얻는다."""
    if downbeats.size < 3:
        return None
    counts = [
        int(np.sum((beats >= downbeats[i]) & (beats < downbeats[i + 1])))
        for i in range(len(downbeats) - 1)
    ]
    counts = [c for c in counts if 2 <= c <= 12]
    if len(counts) < 2:
        return None
    return int(np.bincount(counts).argmax())


def default_tracker() -> Optional[BeatTracker]:
    """쓸 수 있으면 더 나은 추적기를 준다. 없으면 None(=librosa 사용)."""
    tracker = BeatThisTracker()
    return tracker if tracker.available else None
