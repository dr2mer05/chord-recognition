"""코드 전이 통계 (대체안 랭킹 보정용).

파이프라인 1-4의 "통계 보정" 자리. 지금은 팝/록에서 흔한 진행을 반영한
하드코딩 사전 확률이고, `TransitionPrior` 인터페이스만 유지하면
Hooktheory 공식 API 응답으로 그대로 갈아끼울 수 있다.

이론적으로 가능하지만 실제로 잘 안 쓰이는 대체안의 순위를 낮추는 게 목적이므로
정밀한 확률값보다 상대적 순서가 중요하다.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Protocol

# 로마숫자 -> 다음 로마숫자 전이 가중치 (장조 기준, 합이 1일 필요 없음)
_MAJOR_TRANSITIONS: Dict[str, Dict[str, float]] = {
    "I":    {"V": 0.24, "IV": 0.22, "vi": 0.18, "ii": 0.11, "iii": 0.05, "bVII": 0.05, "V/ii": 0.04, "V/IV": 0.04, "vii": 0.02},
    "ii":   {"V": 0.45, "I": 0.12, "IV": 0.10, "iii": 0.06, "vi": 0.05, "bII": 0.04, "V/V": 0.04},
    "iii":  {"vi": 0.30, "IV": 0.24, "ii": 0.12, "I": 0.10, "V/vi": 0.06},
    "IV":   {"I": 0.30, "V": 0.26, "vi": 0.12, "ii": 0.08, "iv": 0.07, "bVII": 0.05, "iii": 0.04},
    "V":    {"I": 0.52, "vi": 0.16, "IV": 0.10, "ii": 0.04, "iii": 0.03, "bVI": 0.03},
    "vi":   {"IV": 0.30, "V": 0.20, "ii": 0.13, "I": 0.12, "iii": 0.07, "V/V": 0.04, "bVI": 0.03},
    "vii":  {"I": 0.55, "iii": 0.12, "vi": 0.10},
    "bVII": {"I": 0.42, "IV": 0.20, "bVI": 0.12, "V": 0.08},
    "bVI":  {"bVII": 0.30, "IV": 0.18, "I": 0.16, "V": 0.14},
    "bIII": {"IV": 0.28, "bVII": 0.22, "I": 0.16, "bVI": 0.12},
    "iv":   {"I": 0.42, "V": 0.16, "bVII": 0.10, "vi": 0.08},
    "bII":  {"I": 0.40, "V": 0.30},
    "V/V":  {"V": 0.72, "I": 0.08},
    "V/vi": {"vi": 0.74, "IV": 0.06},
    "V/ii": {"ii": 0.72, "V": 0.06},
    "V/IV": {"IV": 0.70, "I": 0.06},
    "V/iii": {"iii": 0.70},
}

# 단조는 로마숫자 표기가 달라 별도 표를 둔다.
_MINOR_TRANSITIONS: Dict[str, Dict[str, float]] = {
    "i":    {"VI": 0.22, "VII": 0.20, "iv": 0.18, "V": 0.14, "III": 0.10, "ii": 0.05},
    "ii":   {"V": 0.44, "i": 0.14, "VII": 0.08},
    "III":  {"VI": 0.26, "VII": 0.22, "iv": 0.16, "i": 0.12},
    "iv":   {"i": 0.30, "V": 0.26, "VII": 0.14, "VI": 0.10},
    "v":    {"i": 0.44, "VI": 0.14, "iv": 0.10},
    "V":    {"i": 0.62, "VI": 0.16},
    "VI":   {"VII": 0.30, "V": 0.20, "iv": 0.14, "III": 0.12, "i": 0.10},
    "VII":  {"i": 0.34, "III": 0.24, "VI": 0.16, "iv": 0.08},
    "bII":  {"V": 0.36, "i": 0.34},
}

_DEFAULT_WEIGHT = 0.015  # 표에 없는 전이의 기본값 (0으로 두면 신선한 대체안이 다 죽는다)

_SUFFIX_RE = re.compile(r"(maj7|7|9|11|13|sus[24]|°|ø|\+|/.*$)")


def normalize_numeral(roman: str) -> str:
    """'V7', 'imaj7', 'IV/A' 등을 전이표 키 형태로 정규화."""
    return _SUFFIX_RE.sub("", roman).strip() or roman


class TransitionPrior(Protocol):
    def weight(self, from_numeral: str, to_numeral: str, mode: str) -> float: ...


class BuiltinPrior:
    """의존성 없는 기본 사전 확률."""

    def weight(self, from_numeral: str, to_numeral: str, mode: str = "major") -> float:
        table = _MAJOR_TRANSITIONS if mode == "major" else _MINOR_TRANSITIONS
        row = table.get(normalize_numeral(from_numeral))
        if not row:
            return _DEFAULT_WEIGHT
        return row.get(normalize_numeral(to_numeral), _DEFAULT_WEIGHT)


class HooktheoryPrior:
    """Hooktheory API 연동 자리 (M2). 키가 없으면 기본 사전 확률로 폴백."""

    def __init__(self, api_key: Optional[str] = None, fallback: Optional[TransitionPrior] = None):
        self.api_key = api_key
        self.fallback = fallback or BuiltinPrior()
        self._cache: Dict[str, Dict[str, float]] = {}

    def weight(self, from_numeral: str, to_numeral: str, mode: str = "major") -> float:
        if not self.api_key:
            return self.fallback.weight(from_numeral, to_numeral, mode)
        raise NotImplementedError("M2에서 Hooktheory API 연동")


DEFAULT_PRIOR: TransitionPrior = BuiltinPrior()
