"""파이프라인 1-2 오케스트레이션: [A] 캐시 → [B] 코드 DB → [C] 오디오 분석.

순서 자체가 비용 구조다. 오디오 분석은 마지막 수단.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..analysis.constants import ANALYSIS_VERSION
from ..models import AnalysisResult
from ..theory.key import Key
from ..theory.stats import DEFAULT_PRIOR, TransitionPrior
from ..theory.substitution import suggest
from .cache import AnalysisCache, file_fingerprint
from .chord_db import ChordDatabase, NullDatabase
from .corrections import CorrectionStore, apply_corrections
from .audio_sources import AudioSourceStore
from .youtube import (
    VideoInfo,
    VideoSearch,
    default_search,
    parse_video_id,
)

NAMESPACE_AUDIO = "audio"
NAMESPACE_YOUTUBE = "youtube"
NAMESPACE_SUBS = "subs"

AUDIO_DIR = Path("data/youtube")


class ChordService:
    def __init__(
        self,
        cache: Optional[AnalysisCache] = None,
        database: Optional[ChordDatabase] = None,
        vocabulary: str = "standard",
        prior: TransitionPrior = DEFAULT_PRIOR,
        search: Optional[VideoSearch] = None,
        audio_sources: Optional[AudioSourceStore] = None,
        corrections: Optional[CorrectionStore] = None,
    ):
        self.cache = cache or AnalysisCache()
        self.database = database or NullDatabase()
        self.vocabulary = vocabulary
        self.prior = prior
        self.search_backend = search or default_search()
        self.audio_sources = audio_sources or AudioSourceStore()
        self.corrections = corrections or CorrectionStore()

    # -- 사용자 교정 ----------------------------------------------------
    def corrected_result(self, result: AnalysisResult) -> AnalysisResult:
        """원본 인식 결과 위에 사용자 교정을 얹은 사본.

        원본은 캐시에 그대로 남는다. 덮어쓰면 정확도 측정이 순환논법이 된다.
        """
        if not result.source_key:
            return result
        fixes = self.corrections.corrections(result.source_key)
        if not fixes:
            return result

        merged = copy.copy(result)
        merged.segments = apply_corrections(result.segments, fixes)
        merged.corrected = len(fixes)
        return merged

    # -- [B] 조회 ------------------------------------------------------
    def lookup(self, artist: str, title: str) -> Optional[AnalysisResult]:
        return self.database.lookup(artist, title)

    @staticmethod
    def _separation_setting(separate: Optional[bool]) -> bool:
        """None이면 환경설정을 따른다. 캐시 키에 넣으려면 값이 확정돼야 한다."""
        if separate is not None:
            return separate
        from ..analysis.separation import default_separator

        return default_separator().enabled

    # -- 유튜브 --------------------------------------------------------
    def search_videos(self, query: str, limit: int = 8) -> List[VideoInfo]:
        return self.search_backend.search(query, limit=limit)

    def analyze_youtube(
        self,
        url_or_id: str,
        key_override: Optional[str] = None,
        vocabulary: Optional[str] = None,
        detect_inversions: bool = True,
        separate: Optional[bool] = None,
        max_seconds: Optional[float] = None,
        use_cache: bool = True,
    ) -> Tuple[AnalysisResult, Optional[VideoInfo]]:
        """허용된 별도 음원 연결 → 원본 분석 캐시 → 분석. 권한은 HIT 전에도 확인한다."""
        video_id = parse_video_id(url_or_id)
        link = self.audio_sources.resolve(video_id)
        result = self.analyze_file(
            link.path, key_override=key_override, vocabulary=vocabulary,
            detect_inversions=detect_inversions, separate=separate,
            max_seconds=max_seconds, use_cache=use_cache, source_key=link.source_key,
        )
        # 영상 메타데이터는 분석 결과와 별도 캐시·만료 주기를 가진다.
        info = self.video_metadata(video_id)
        result.source = info.title if info and info.title else link.label
        result.audio_link = link.to_dict()
        self._register(result)
        return result, info

    def video_metadata(self, video_id: str) -> Optional[VideoInfo]:
        cache_key = self.cache.make_key("video_metadata", video_id)
        hit = self.cache.get(cache_key, max_age=24 * 60 * 60)
        if hit is not None:
            return VideoInfo(**hit)
        info = self.search_backend.metadata(video_id)
        if info is not None:
            from dataclasses import asdict
            self.cache.put(cache_key, "video_metadata", asdict(info))
        return info

    # -- [A] + [C] -----------------------------------------------------
    def analyze_file(
        self,
        path: Path,
        key_override: Optional[str] = None,
        vocabulary: Optional[str] = None,
        detect_inversions: bool = True,
        separate: Optional[bool] = None,
        max_seconds: Optional[float] = None,
        use_cache: bool = True,
        source_key: Optional[str] = None,
    ) -> AnalysisResult:
        path = Path(path)
        vocab = vocabulary or self.vocabulary
        fingerprint = file_fingerprint(path)
        key_ = source_key or f"file:{fingerprint}"
        tempo_override = self.corrections.tempo(key_)
        beats_per_bar = self.corrections.beats_per_bar(key_)
        downbeat_offset = self.corrections.downbeat_offset(key_)
        beat_offset = self.corrections.beat_offset(key_)
        options: Dict[str, Any] = {
            "vocabulary": vocab,
            "key": key_override,
            "inversions": detect_inversions,
            "max_seconds": max_seconds,
            # 분리 여부로 결과가 달라지므로 캐시를 분리해야 한다
            "separate": self._separation_setting(separate),
            # 템포·박자를 지정하면 격자가 통째로 달라진다
            "tempo": tempo_override,
            "meter": (beats_per_bar, downbeat_offset, beat_offset),
            # 분석 로직이 바뀌면 옛 결과를 그대로 쓰면 안 된다
            "algo": ANALYSIS_VERSION,
        }
        from ..analysis.runtime import runtime_identity, cacheable_execution
        identity = runtime_identity(vocab, options["separate"], tempo_override)
        options["runtime"] = identity
        options["source_key"] = key_
        cache_key = self.cache.make_key(NAMESPACE_AUDIO, fingerprint, **options)

        if use_cache:
            hit = self.cache.get(cache_key)
            if hit is not None:
                result = AnalysisResult.from_dict(hit)
                result.cached = True
                # 캐시 payload를 믿지 않고 다시 세운다. 예전에 저장된 항목에는
                # 이 필드들이 없어서, 그대로 두면 교정을 붙일 수 없다.
                result.source_key = key_
                result.audio_path = str(path.resolve())
                self._register(result)
                return result

        # 무거운 의존성(librosa)은 실제로 분석할 때만 로드한다
        from ..analysis.pipeline import analyze_audio

        result = analyze_audio(
            str(path),
            vocabulary=vocab,
            key_override=Key.parse(key_override) if key_override else None,
            detect_inversions=detect_inversions,
            separate=options["separate"],
            tempo_override=tempo_override,
            beats_per_bar=beats_per_bar,
            downbeat_offset=downbeat_offset,
            beat_offset=beat_offset,
            max_seconds=max_seconds,
        )
        result.source = path.name
        result.source_key = key_
        result.audio_path = str(path.resolve())
        result.execution["environment"] = identity
        if use_cache and cacheable_execution(identity, result.execution):
            self.cache.put(cache_key, NAMESPACE_AUDIO, result.to_dict())
        self._register(result)
        return result

    def _register(self, result: AnalysisResult) -> None:
        """교정을 붙일 수 있도록 곡을 등록한다. 오디오 경로는 벤치마크 재분석에 쓴다."""
        if not result.source_key:
            return
        self.corrections.record_source(
            result.source_key, label=result.source,
            audio_path=result.audio_path, duration=result.duration,
        )

    # -- 화성학 엔진 연결 ------------------------------------------------
    def substitutions(
        self,
        result: AnalysisResult,
        limit: int = 6,
        key_override: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """구간별 대체 코드. 앞뒤 코드를 문맥으로 넣어 호출한다."""
        key = Key.parse(key_override) if key_override else result.key
        flat = key.prefer_flat
        segments = result.segments
        out: List[Dict[str, Any]] = []
        for i, seg in enumerate(segments):
            prev = segments[i - 1].chord if i > 0 else None
            nxt = segments[i + 1].chord if i + 1 < len(segments) else None
            subs = suggest(seg.chord, key, prev=prev, next=nxt, limit=limit, prior=self.prior)
            out.append(
                {
                    "index": i,
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "chord": seg.chord.name(flat),
                    "roman": key.roman(seg.chord),
                    "function": key.function_of(seg.chord),
                    "bar": seg.bar,
                    "agreement": round(float(seg.agreement), 3),
                    "substitutions": [s.to_dict(flat) for s in subs],
                }
            )
        return out

    def full_report(
        self,
        result: AnalysisResult,
        limit: int = 6,
        key_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        # 사용자에게는 교정이 반영된 결과를 보여준다 (원본은 캐시에 그대로)
        shown = self.corrected_result(result)
        data = shown.to_dict()
        data["analysis"] = self.substitutions(shown, limit=limit, key_override=key_override)
        data["uncertain"] = shown.uncertain_segments()
        data["cache_stats"] = self.cache.stats()
        source = self.corrections.source(shown.source_key) if shown.source_key else None
        data["verified"] = bool(source and source.get("verified"))
        data["tempo_override"] = source.get("tempo") if source else None
        data["meter_override"] = source.get("beats_per_bar") if source else None
        data["downbeat_offset"] = int(source.get("downbeat_offset") or 0) if source else 0
        data["corrections"] = [c.to_dict() for c in
                               self.corrections.corrections(shown.source_key)] if shown.source_key else []
        return data
