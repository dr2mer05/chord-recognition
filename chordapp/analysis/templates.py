"""코드 템플릿 (크로마 매칭용 어휘 사전)."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..theory.chord import QUALITY_INTERVALS, Chord

# 화음 안에서 각 음이 실제 스펙트럼에 기여하는 정도의 근사치.
# 근음·3음이 강하고 텐션으로 갈수록 약하다.
_DEGREE_WEIGHT: Dict[int, float] = {0: 1.00, 1: 0.90, 2: 0.75, 3: 0.70, 4: 0.45, 5: 0.40}

# 어휘 프리셋. 넓을수록 확장 코드를 잡을 수 있지만 오탐도 같이 는다.
VOCABULARIES: Dict[str, Sequence[str]] = {
    "triads": ("maj", "min"),
    "standard": ("maj", "min", "7", "min7", "maj7", "dim", "hdim7", "sus4"),
    "extended": (
        "maj", "min", "7", "min7", "maj7", "dim", "dim7", "hdim7",
        "sus4", "sus2", "6", "min6", "9", "min9", "maj9", "add9", "aug",
    ),
}

# 흔하지 않은 품질에 붙는 감점. 증거가 확실할 때만 이기게 한다.
_QUALITY_PRIOR: Dict[str, float] = {
    "maj": 0.0, "min": 0.0,
    "7": -0.015, "min7": -0.015, "maj7": -0.020,
    "sus4": -0.035, "sus2": -0.045,
    "dim": -0.050, "hdim7": -0.050, "dim7": -0.055, "aug": -0.070,
    "6": -0.045, "min6": -0.055,
    "9": -0.055, "min9": -0.055, "maj9": -0.060, "add9": -0.050,
}


def build_vocabulary(name: str = "standard") -> Tuple[List[Chord], np.ndarray, np.ndarray]:
    """(코드 목록, 정규화된 템플릿 행렬 (V,12), 품질 사전점수 (V,))를 반환."""
    if name not in VOCABULARIES:
        raise ValueError(f"알 수 없는 어휘: {name!r} (가능: {list(VOCABULARIES)})")

    chords: List[Chord] = []
    rows: List[np.ndarray] = []
    priors: List[float] = []

    for quality in VOCABULARIES[name]:
        intervals = QUALITY_INTERVALS[quality]
        for root in range(12):
            vec = np.zeros(12, dtype=np.float32)
            for i, semitone in enumerate(intervals):
                vec[(root + semitone) % 12] += _DEGREE_WEIGHT.get(i, 0.35)
            norm = np.linalg.norm(vec)
            if norm == 0:
                continue
            chords.append(Chord(root, quality))
            rows.append(vec / norm)
            priors.append(_QUALITY_PRIOR.get(quality, -0.05))

    return chords, np.vstack(rows), np.asarray(priors, dtype=np.float32)
