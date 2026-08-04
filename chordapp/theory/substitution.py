"""대체 코드 생성 + 랭킹 엔진.

파이프라인 1-4의 구현체. 순수 로직이라 서버 자원이 거의 들지 않고
기존 앱과의 차별화 지점이므로 규칙을 촘촘하게 넣는다.

설계
  - 규칙 하나 = 함수 하나(`Rule`). 후보(`Substitution`)를 0개 이상 낸다.
  - 각 후보는 "왜 대체 가능한가"를 기능 라벨 + 설명 문장으로 항상 들고 다닌다(UI 원칙).
  - 점수 = 규칙 기본점수 + 전이 통계(앞뒤 문맥) + 보이싱 매끄러움.
  - LLM은 여기 관여하지 않는다. 설명 문구를 더 자연스럽게 다듬는 용도로만 선택 적용.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .chord import (
    DOMINANT_QUALITIES,
    QUALITY_INTERVALS,
    Chord,
    simplify_to_triad,
)
from .key import Key
from .notes import note_name
from .stats import DEFAULT_PRIOR, TransitionPrior

REPLACE = "replace"
INSERT_BEFORE = "insert_before"
INSERT_AFTER = "insert_after"


@dataclass
class Substitution:
    chord: Chord
    category: str
    label: str          # UI에 뱃지로 붙는 짧은 기능 라벨
    reason: str         # 한 줄 설명
    score: float = 0.0
    kind: str = REPLACE
    tags: Tuple[str, ...] = ()

    def to_dict(self, prefer_flat: bool = False) -> Dict[str, object]:
        return {
            "chord": self.chord.name(prefer_flat),
            "category": self.category,
            "label": self.label,
            "reason": self.reason,
            "score": round(self.score, 1),
            "kind": self.kind,
            "tags": list(self.tags),
        }


@dataclass
class Context:
    chord: Chord
    key: Key
    prev: Optional[Chord] = None
    next: Optional[Chord] = None

    @property
    def roman(self) -> str:
        return self.key.roman(self.chord)

    @property
    def function(self) -> str:
        return self.key.function_of(self.chord)

    def name(self, chord: Chord) -> str:
        return chord.name(self.key.prefer_flat)


Rule = Callable[[Context], List[Substitution]]

# 규칙별 기본 점수. 화성학적 안전도(원곡을 크게 안 해치는 정도) 기준.
BASE_SCORES: Dict[str, float] = {
    "tension": 0.82,
    "relative": 0.78,
    "secondary-dominant": 0.74,
    "diatonic-sub": 0.72,
    "inversion": 0.70,
    "modal-interchange": 0.66,
    "sus": 0.58,
    "tritone-sub": 0.60,
    "passing-dim": 0.55,
    "backdoor": 0.52,
    "deceptive": 0.50,
    "neapolitan": 0.45,
    "chromatic-mediant": 0.34,
}


# --------------------------------------------------------------------------
# 헬퍼
# --------------------------------------------------------------------------

def _chord_tone(chord: Chord, degree_index: int) -> Optional[int]:
    intervals = QUALITY_INTERVALS[chord.quality]
    if degree_index >= len(intervals):
        return None
    return (chord.root + intervals[degree_index]) % 12


def _all_in_key(chord: Chord, key: Key) -> bool:
    return all(key.contains_pc(pc) for pc in chord.pitch_classes())


def _semitone_distance(a: int, b: int) -> int:
    d = (a - b) % 12
    return min(d, 12 - d)


def _resolves_down_fifth(a: Chord, b: Optional[Chord]) -> bool:
    return b is not None and (a.root - b.root) % 12 == 7


# --------------------------------------------------------------------------
# 규칙들
# --------------------------------------------------------------------------

def rule_diatonic_substitution(ctx: Context) -> List[Substitution]:
    """3도 관계 대리코드. 공통음 2개 이상을 공유하는 같은 조성의 코드."""
    out: List[Substitution] = []
    base = simplify_to_triad(ctx.chord)
    sevenths = ctx.key.diatonic_sevenths()
    for i, cand7 in enumerate(sevenths):
        if cand7.root == ctx.chord.root:
            continue
        triad = ctx.key.diatonic_triads()[i]
        shared = base.common_tones(simplify_to_triad(triad))
        if shared < 2:
            continue
        if _semitone_distance(triad.root, ctx.chord.root) not in (3, 4):
            continue  # 3도 관계만 (2도/5도 관계는 대리코드가 아니라 진행)
        direction = "위" if (triad.root - ctx.chord.root) % 12 in (3, 4) else "아래"
        out.append(
            Substitution(
                chord=triad,
                category="diatonic-sub",
                label=f"{ctx.key.roman(triad)} 대리",
                reason=(
                    f"{ctx.roman}와 구성음 {shared}개를 공유하는 3도 {direction} 다이어토닉 코드. "
                    f"기능({_fn_ko(ctx.key.function_of(triad))})이 유지돼 진행을 해치지 않는다."
                ),
                tags=("안전", "다이어토닉"),
            )
        )
    return out


def rule_relative_key(ctx: Context) -> List[Substitution]:
    """나란한조 토닉 교체 (C ↔ Am)."""
    if ctx.key.degree_of(ctx.chord.root) is None:
        return []
    is_tonic = ctx.chord.root == ctx.key.tonic
    if not is_tonic:
        return []
    rel = ctx.key.relative()
    cand = Chord(rel.tonic, "min" if rel.mode == "minor" else "maj")
    return [
        Substitution(
            chord=cand,
            category="relative",
            label="나란한조 토닉",
            reason=(
                f"{ctx.key.name()}의 나란한조 {rel.name()}의 으뜸화음. "
                f"구성음 2개를 공유해 토닉 기능을 그대로 이어받으면서 밝기만 바뀐다."
            ),
            tags=("안전", "분위기 전환"),
        )
    ]


def rule_tension(ctx: Context) -> List[Substitution]:
    """텐션 확장. 조성 안에 머무는 확장을 우선하고, 도미넌트만 변화음을 허용한다."""
    c = ctx.chord
    tq = c.triad_quality()
    out: List[Substitution] = []

    def add(quality: str, label: str, reason: str, allow_out_of_key: bool = False) -> None:
        if quality == c.quality:
            return
        cand = Chord(c.root, quality, c.bass)
        if not allow_out_of_key and not _all_in_key(cand, ctx.key):
            return
        out.append(Substitution(cand, "tension", label, reason, tags=("텐션",)))

    if tq == "maj":
        fn = ctx.function
        if fn == "dominant" or _resolves_down_fifth(c, ctx.next):
            add("7", "도미넌트 7th", "다음 코드로 5도 하행 해결하는 자리라 7음을 넣으면 끌어당기는 힘이 생긴다.", True)
            add("9", "9th 확장", "도미넌트에 9음을 얹어 색을 더한다. 해결 방향은 그대로.", True)
            add("13", "13th 확장", "도미넌트 13음. 팝/소울에서 V 자리에 가장 흔한 확장.", True)
        else:
            add("maj7", "메이저 7th", f"{ctx.roman}의 장7도를 추가. 조성 안 음이라 위화감 없이 세련된 색이 된다.")
            add("6", "6th", "장6도를 넣어 7th보다 가볍고 복고적인 색.")
            add("add9", "add9", "9음만 얹고 7음은 빼서 밝고 열린 소리. 어쿠스틱 기타에서 특히 잘 맞는다.")
            add("maj9", "maj9", "7음과 9음을 함께. 발라드/시티팝 톤.")
    elif tq == "min":
        add("min7", "마이너 7th", f"{ctx.roman}의 단7도 추가. 다이어토닉 안이라 가장 안전한 확장.")
        add("min9", "m9", "9음까지 확장. 네오소울/알앤비 색.")
        add("min11", "m11", "11음까지 쌓은 넓은 보이싱. 근음을 베이스에 맡기고 쓰면 좋다.")
        add("min6", "m6", "도리안 색의 장6도. 단조 토닉에서 쓰면 어두움이 덜어진다.")
    elif tq == "dom":
        add("9", "9th 확장", "도미넌트 9음. 무난하게 두꺼워진다.", True)
        add("13", "13th 확장", "도미넌트 13음. V 자리 기본값처럼 쓰인다.", True)
        if ctx.next is not None and ctx.next.triad_quality() == "min":
            add("7b9", "b9 변화음", "단조로 해결하는 도미넌트. b9가 해결음을 반음 위에서 눌러준다.", True)
        add("7#9", "#9 변화음", "블루스/펑크의 긴장. 장3도와 단3도가 부딪히는 소리를 의도적으로 쓴다.", True)
        add("7#5", "#5 변화음", "5음을 올려 해결 직전 긴장을 최대로.", True)
    elif tq == "dim":
        add("dim7", "dim7", "감7화음으로 확장. 경과 화음으로 쓸 때 더 매끄럽다.", True)
        add("hdim7", "m7b5", "반감7화음. ii-V의 ii 자리라면 이쪽이 자연스럽다.", True)
    return out


def rule_inversion(ctx: Context) -> List[Substitution]:
    """전위 / 슬래시 코드. 베이스 순차진행을 만들면 점수를 올린다."""
    c = ctx.chord
    if c.is_no_chord:
        return []
    out: List[Substitution] = []
    labels = [(1, "1전위", "3음"), (2, "2전위", "5음"), (3, "7음 베이스", "7음")]
    for idx, label, tone_ko in labels:
        pc = _chord_tone(c, idx)
        if pc is None or pc == c.root:
            continue
        cand = c.with_bass(pc)
        if cand.bass is None:
            continue
        reason = f"{tone_ko}을 베이스로 내린 {label}. 화성은 그대로고 베이스 라인만 바뀐다."
        tags: Tuple[str, ...] = ("전위",)
        if ctx.next is not None and not ctx.next.is_no_chord:
            target = ctx.next.bass if ctx.next.bass is not None else ctx.next.root
            if _semitone_distance(pc, target) in (1, 2):
                reason += f" 다음 {ctx.name(ctx.next)}의 베이스로 순차 이동해서 라인이 이어진다."
                tags = tags + ("베이스 라인",)
        if ctx.prev is not None and not ctx.prev.is_no_chord:
            src = ctx.prev.bass if ctx.prev.bass is not None else ctx.prev.root
            if _semitone_distance(pc, src) in (1, 2):
                tags = tags + ("라인 클리셰",)
        out.append(Substitution(cand, "inversion", label, reason, tags=tags))
    return out


def rule_secondary_dominant(ctx: Context) -> List[Substitution]:
    """세컨더리 도미넌트. 다음 코드를 임시 으뜸음으로 보고 그 V7을 쓴다."""
    out: List[Substitution] = []
    nxt = ctx.next
    if nxt is None or nxt.is_no_chord:
        return out
    if nxt.root == ctx.chord.root:
        return out
    cand = Chord((nxt.root + 7) % 12, "7")
    target_roman = ctx.key.roman(nxt)
    # 근음이 같아도 품질이 다르면 유효한 대체다 (Am → A7 → Dm 이 대표적).
    if cand != ctx.chord:
        out.append(
            Substitution(
                chord=cand,
                category="secondary-dominant",
                label=f"V7/{target_roman}",
                reason=(
                    f"다음 코드 {ctx.name(nxt)}({target_roman})를 잠깐 으뜸음 취급해서 그 도미넌트를 놓는다. "
                    f"5도 하행으로 강하게 끌어당긴다."
                ),
                tags=("추진력", "세컨더리 도미넌트"),
            )
        )
    # 현재 코드를 그대로 두고 그 앞에 V7/현재코드를 끼워넣는 변형
    if ctx.key.degree_of(ctx.chord.root) is not None and ctx.chord.root != ctx.key.tonic:
        pre = Chord((ctx.chord.root + 7) % 12, "7")
        out.append(
            Substitution(
                chord=pre,
                category="secondary-dominant",
                label=f"V7/{ctx.roman} 삽입",
                reason=(
                    f"{ctx.name(ctx.chord)}를 그대로 두고 그 앞 반 마디에 도미넌트를 끼워 넣는 방식. "
                    f"코드를 바꾸지 않고 추진력만 추가한다."
                ),
                kind=INSERT_BEFORE,
                tags=("추진력", "삽입"),
            )
        )
    return out


def rule_tritone_substitution(ctx: Context) -> List[Substitution]:
    """트라이톤 대체. 도미넌트의 3음·7음(트라이톤)을 공유하는 감5도 위 도미넌트."""
    c = ctx.chord
    out: List[Substitution] = []
    is_functional_dominant = c.is_dominant or (
        c.root == (ctx.key.tonic + 7) % 12 and c.triad_quality() == "maj"
    )
    if not is_functional_dominant:
        return out
    cand = Chord((c.root + 6) % 12, "7")
    out.append(
        Substitution(
            chord=cand,
            category="tritone-sub",
            label="트라이톤 대체",
            reason=(
                f"{ctx.name(c)}의 3음·7음을 그대로 가진 도미넌트. "
                f"베이스가 해결음으로 반음 하행해서 5도 하행보다 매끄럽게 들어간다."
            ),
            tags=("재즈", "반음 하행"),
        )
    )
    return out


def rule_modal_interchange(ctx: Context) -> List[Substitution]:
    """모달 인터체인지. 같은 으뜸음의 반대 모드에서 코드를 빌려온다."""
    c = ctx.chord
    par = ctx.key.parallel()
    out: List[Substitution] = []
    popular = _borrowed_popularity(ctx.key.mode)
    for triad, seventh in zip(par.diatonic_triads(), par.diatonic_sevenths()):
        if ctx.key.is_diatonic(triad):
            continue  # 이미 원래 조성에 있는 코드는 차용이 아님
        if triad.root == c.root and triad.quality == c.quality:
            continue
        shared = simplify_to_triad(c).common_tones(triad)
        if shared < 1 and triad.root != c.root:
            continue
        roman = ctx.key.roman(triad)
        weight = popular.get(_strip_roman(roman), 0.5)
        sub = Substitution(
            chord=triad,
            category="modal-interchange",
            label=f"{roman} 차용",
            reason=(
                f"같은 으뜸음 {par.name()}에서 빌려온 코드. "
                f"{'같은 근음이라 자리만 그대로 두고 색이 어두워진다.' if triad.root == c.root else f'원래 코드와 {shared}음 공유.'}"
            ),
            tags=("모달 인터체인지", "색채"),
        )
        sub.score = weight  # 뒤에서 기본점수에 곱해 쓰기 위한 임시 저장
        out.append(sub)
    return out


def _strip_roman(roman: str) -> str:
    for suffix in ("maj7", "7", "°", "ø", "+"):
        if roman.endswith(suffix):
            return roman[: -len(suffix)]
    return roman


def _borrowed_popularity(mode: str) -> Dict[str, float]:
    if mode == "major":
        return {"iv": 1.0, "bVII": 1.0, "bVI": 0.95, "bIII": 0.8, "i": 0.55, "ii": 0.6, "v": 0.6}
    return {"IV": 1.0, "V": 1.0, "I": 0.7, "ii": 0.7, "vi": 0.6, "iii": 0.5, "vii": 0.5}


def rule_sus(ctx: Context) -> List[Substitution]:
    """해결 지연용 sus."""
    c = ctx.chord
    tq = c.triad_quality()
    if tq not in ("maj", "min", "dom"):
        return []
    out: List[Substitution] = []
    if tq == "dom":
        out.append(
            Substitution(
                Chord(c.root, "7sus4", c.bass), "sus", "7sus4",
                "3음을 4음으로 미뤄 해결을 한 박 늦춘다. 원래 코드로 되돌아가면 고전적인 sus-해결.",
                tags=("해결 지연",),
            )
        )
    else:
        out.append(
            Substitution(
                Chord(c.root, "sus4", c.bass), "sus", "sus4",
                "3음 대신 4음. 장/단 색을 지워 다음 코드로 넘어가는 긴장을 만든다.",
                tags=("해결 지연",),
            )
        )
        if ctx.key.contains_pc((c.root + 2) % 12):
            out.append(
                Substitution(
                    Chord(c.root, "sus2", c.bass), "sus", "sus2",
                    "3음 대신 2음. sus4보다 열려 있고 가볍다.",
                    tags=("해결 지연",),
                )
            )
    return out


def rule_passing_diminished(ctx: Context) -> List[Substitution]:
    """경과 감7화음 삽입. 근음이 온음 이동할 때 사이를 반음으로 메운다."""
    nxt = ctx.next
    if nxt is None or nxt.is_no_chord or ctx.chord.is_no_chord:
        return []
    up = (nxt.root - ctx.chord.root) % 12
    if up == 2:
        pc = (ctx.chord.root + 1) % 12
    elif up == 10:
        pc = (ctx.chord.root - 1) % 12
    else:
        return []
    cand = Chord(pc, "dim7")
    direction = "상행" if up == 2 else "하행"
    return [
        Substitution(
            chord=cand,
            category="passing-dim",
            label="경과 dim7",
            reason=(
                f"{ctx.name(ctx.chord)} → {ctx.name(nxt)}가 온음 {direction}이라 "
                f"사이에 감7화음을 넣으면 베이스가 반음씩 움직인다."
            ),
            kind=INSERT_AFTER,
            tags=("경과화음", "베이스 라인"),
        )
    ]


def rule_backdoor(ctx: Context) -> List[Substitution]:
    """백도어 도미넌트 bVII7 → I."""
    nxt = ctx.next
    if nxt is None or nxt.root != ctx.key.tonic:
        return []
    if ctx.function not in ("subdominant", "dominant"):
        return []
    cand = Chord((ctx.key.tonic + 10) % 12, "7")
    if cand.root == ctx.chord.root:
        return []
    return [
        Substitution(
            chord=cand,
            category="backdoor",
            label="백도어 도미넌트",
            reason=(
                "다음이 토닉이라 bVII7으로 뒤에서 들어갈 수 있다. "
                "V7보다 부드럽고 소울/네오소울에서 흔한 진입 방식."
            ),
            tags=("소울", "차용"),
        )
    ]


def rule_neapolitan(ctx: Context) -> List[Substitution]:
    """나폴리 6화음 bII6."""
    if ctx.function != "subdominant":
        return []
    root = (ctx.key.tonic + 1) % 12
    cand = Chord(root, "maj", (root + 4) % 12)
    return [
        Substitution(
            chord=cand,
            category="neapolitan",
            label="나폴리 6화음",
            reason=(
                "서브도미넌트 자리를 bII의 1전위로 바꾼다. 베이스는 4도음 그대로 두고 "
                "윗성부만 반음 올려 극적으로 만드는 고전적 수법."
            ),
            tags=("클래식", "극적"),
        )
    ]


def rule_deceptive(ctx: Context) -> List[Substitution]:
    """위장종지. 도미넌트 다음 토닉 자리를 vi/bVI로."""
    if ctx.prev is None or ctx.chord.root != ctx.key.tonic:
        return []
    if ctx.key.function_of(ctx.prev) != "dominant":
        return []
    out: List[Substitution] = []
    if ctx.key.mode == "major":
        out.append(
            Substitution(
                Chord((ctx.key.tonic + 9) % 12, "min"), "deceptive", "위장종지 vi",
                "도미넌트 다음에 토닉 대신 vi를 놓아 종지를 회피한다. 곡을 더 끌고 갈 때.",
                tags=("종지 회피",),
            )
        )
        out.append(
            Substitution(
                Chord((ctx.key.tonic + 8) % 12, "maj"), "deceptive", "위장종지 bVI",
                "차용화음 bVI로 받는 위장종지. vi보다 이탈감이 크다.",
                tags=("종지 회피", "차용"),
            )
        )
    else:
        out.append(
            Substitution(
                Chord((ctx.key.tonic + 8) % 12, "maj"), "deceptive", "위장종지 VI",
                "단조 도미넌트 다음 VI로 받는 위장종지.",
                tags=("종지 회피",),
            )
        )
    return out


def rule_chromatic_mediant(ctx: Context) -> List[Substitution]:
    """반음계적 중음관계. 공통음 1개만 공유하는 3도 관계 코드."""
    c = ctx.chord
    if c.triad_quality() not in ("maj", "min"):
        return []
    base = simplify_to_triad(c)
    out: List[Substitution] = []
    for interval in (3, 4, 8, 9):
        for quality in ("maj", "min"):
            cand = Chord((c.root + interval) % 12, quality)
            if ctx.key.is_diatonic(cand):
                continue
            if base.common_tones(cand) != 1:
                continue
            out.append(
                Substitution(
                    cand, "chromatic-mediant", "반음계적 중음",
                    f"{ctx.name(c)}와 공통음 1개만 공유하는 3도 관계 코드. "
                    f"조성 밖이라 이탈감이 크지만 후렴 직전 색 전환에 쓸 만하다.",
                    tags=("색채", "이탈"),
                )
            )
    return out


ALL_RULES: Sequence[Rule] = (
    rule_diatonic_substitution,
    rule_relative_key,
    rule_tension,
    rule_inversion,
    rule_secondary_dominant,
    rule_tritone_substitution,
    rule_modal_interchange,
    rule_sus,
    rule_passing_diminished,
    rule_backdoor,
    rule_neapolitan,
    rule_deceptive,
    rule_chromatic_mediant,
)


# --------------------------------------------------------------------------
# 랭킹
# --------------------------------------------------------------------------

def _fn_ko(function: str) -> str:
    return {"tonic": "토닉", "subdominant": "서브도미넌트",
            "dominant": "도미넌트", "chromatic": "반음계"}.get(function, function)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _score(sub: Substitution, ctx: Context, prior: TransitionPrior) -> float:
    base = BASE_SCORES.get(sub.category, 0.5)
    if sub.category == "modal-interchange":
        base *= sub.score if sub.score else 1.0  # 규칙이 넘긴 인기도 가중치

    mode = ctx.key.mode
    cand_roman = ctx.key.roman(sub.chord)

    # 앞뒤 문맥 전이 확률 (Hooktheory 자리)
    fwd = 0.5
    if sub.kind == REPLACE and ctx.next is not None and not ctx.next.is_no_chord:
        fwd = _clamp01(prior.weight(cand_roman, ctx.key.roman(ctx.next), mode) / 0.30)
    elif sub.kind == INSERT_BEFORE:
        fwd = _clamp01(prior.weight(cand_roman, ctx.roman, mode) / 0.30)
    elif sub.kind == INSERT_AFTER and ctx.next is not None:
        fwd = _clamp01(prior.weight(cand_roman, ctx.key.roman(ctx.next), mode) / 0.30)

    bwd = 0.5
    if ctx.prev is not None and not ctx.prev.is_no_chord and sub.kind == REPLACE:
        bwd = _clamp01(prior.weight(ctx.key.roman(ctx.prev), cand_roman, mode) / 0.30)

    # 원래 코드와 얼마나 가까운가 (전위/텐션이 위로 오게 하는 항)
    a, b = set(ctx.chord.pitch_classes()), set(sub.chord.pitch_classes())
    smooth = len(a & b) / max(len(a | b), 1)

    raw = base + 0.30 * fwd + 0.15 * bwd + 0.18 * smooth
    return _clamp01(raw / 1.45) * 100.0


def suggest(
    chord: Chord,
    key: Key,
    prev: Optional[Chord] = None,
    next: Optional[Chord] = None,
    limit: int = 8,
    prior: TransitionPrior = DEFAULT_PRIOR,
    min_score: float = 0.0,
    rules: Sequence[Rule] = ALL_RULES,
    diversify: bool = True,
) -> List[Substitution]:
    """한 코드 자리에 대한 대체안을 점수 순으로 반환."""
    if chord.is_no_chord:
        return []
    ctx = Context(chord=chord, key=key, prev=prev, next=next)

    raw: List[Substitution] = []
    for rule in rules:
        try:
            raw.extend(rule(ctx))
        except Exception:  # 규칙 하나가 깨져도 나머지는 살린다
            continue

    # 세컨더리 도미넌트에는 트라이톤 대체를 파생 적용 (subV/x)
    derived: List[Substitution] = []
    for sub in raw:
        if sub.category == "secondary-dominant" and sub.kind == REPLACE:
            alt = Chord((sub.chord.root + 6) % 12, "7")
            derived.append(
                Substitution(
                    alt, "tritone-sub", f"sub{sub.label}",
                    f"{sub.label}를 트라이톤 대체한 형태. 베이스가 반음으로 내려간다.",
                    tags=("재즈", "반음 하행"),
                )
            )
    raw.extend(derived)

    scored: Dict[Tuple[int, str, Optional[int], str], Substitution] = {}
    for sub in raw:
        if sub.chord == chord and sub.kind == REPLACE:
            continue
        sub.score = _score(sub, ctx, prior)
        dedup_key = (sub.chord.root, sub.chord.quality, sub.chord.bass, sub.kind)
        best = scored.get(dedup_key)
        if best is None or sub.score > best.score:
            if best is not None:
                sub.tags = tuple(dict.fromkeys(sub.tags + best.tags))
            scored[dedup_key] = sub
        else:
            best.tags = tuple(dict.fromkeys(best.tags + sub.tags))

    result = [s for s in scored.values() if s.score >= min_score]
    result.sort(key=lambda s: (-s.score, s.chord.root))
    if diversify:
        result = _diversify(result, limit)
    return result[:limit]


def _diversify(subs: List[Substitution], limit: int, decay: float = 0.80) -> List[Substitution]:
    """같은 카테고리가 상위를 독식하지 않게 그리디로 재선택.

    텐션 확장은 점수가 구조적으로 높아서 보정 없이 두면 목록이 전부 텐션이 된다.
    사용자가 실제로 원하는 건 "무슨 수를 쓸 수 있는지"의 목록이므로 종류를 섞는다.
    원래 점수는 그대로 두고 선택 순서에만 감쇠를 적용한다.
    """
    remaining = list(subs)
    picked: List[Substitution] = []
    used: Dict[str, int] = {}
    while remaining and len(picked) < limit:
        best_i, best_val = 0, float("-inf")
        for i, sub in enumerate(remaining):
            val = sub.score * (decay ** used.get(sub.category, 0))
            if val > best_val:
                best_i, best_val = i, val
        chosen = remaining.pop(best_i)
        used[chosen.category] = used.get(chosen.category, 0) + 1
        picked.append(chosen)
    # 감쇠는 "무엇을 담을지"에만 쓰고, 담은 뒤 순서는 원래 점수대로 되돌린다.
    picked.sort(key=lambda s: (-s.score, s.chord.root))
    return picked


def suggest_for_progression(
    chords: Sequence[Chord],
    key: Key,
    limit_per_chord: int = 6,
    prior: TransitionPrior = DEFAULT_PRIOR,
) -> List[List[Substitution]]:
    """진행 전체에 대해 자리별 대체안. 앞뒤 문맥을 넣어 호출한다."""
    out = []
    for i, chord in enumerate(chords):
        prev = chords[i - 1] if i > 0 else None
        nxt = chords[i + 1] if i + 1 < len(chords) else None
        out.append(suggest(chord, key, prev=prev, next=nxt, limit=limit_per_chord, prior=prior))
    return out
