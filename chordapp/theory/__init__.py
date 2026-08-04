"""화성학 엔진 (순수 로직, 외부 의존성 없음)."""

from .chord import Chord, simplify_to_triad
from .key import Key
from .stats import DEFAULT_PRIOR, BuiltinPrior, HooktheoryPrior
from .substitution import Substitution, suggest, suggest_for_progression

__all__ = [
    "Chord",
    "Key",
    "Substitution",
    "simplify_to_triad",
    "suggest",
    "suggest_for_progression",
    "DEFAULT_PRIOR",
    "BuiltinPrior",
    "HooktheoryPrior",
]
