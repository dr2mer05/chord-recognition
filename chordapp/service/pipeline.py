"""파이프라인 1-2 오케스트레이션: [A] 캐시 → [B] 코드 DB → [C] 오디오 분석.

순서 자체가 비용 구조다. 오디오 분석은 마지막 수단.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..models import AnalysisResult
from ..theory.key import Key
from ..theory.stats import DEFAULT_PRIOR, TransitionPrior
from ..theory.substitution import suggest
from .cache import AnalysisCache, file_fingerprint
from .chord_db import ChordDatabase, NullDatabase
from .corrections import CorrectionStore, apply_corrections
from .youtube import (
    AudioProvider,
    VideoInfo,
    VideoSearch,
    default_audio_provider,
    default_search,
    parse_video_id,
    split_artist_title,
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
        audio_provider: Optional[AudioProvider] = None,
        corrections: Optional[CorrectionStore] = None,
    ):
        self.cache = cache or AnalysisCache()
        self.database = database or NullDatabase()
        self.vocabulary = vocabulary
        self.prior = prior
        self.search_backend = search or default_search()
        self.audio_provider = audio_provider or default_audio_provider()
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
        """[A] 캐시 → [B] 코드 DB → [C] 오디오 분석 순서로 처리한다.

        캐시 키가 영상 ID라 같은 영상은 누가 요청해도 연산이 한 번뿐이다.
        이 순서가 곧 비용 구조라서, 오디오 분석은 마지막 수단이다.
        """
        video_id = parse_video_id(url_or_id)
        vocab = vocabulary or self.vocabulary
        options: Dict[str, Any] = {
            "vocabulary": vocab,
            "key": key_override,
            "inversions": detect_inversions,
            "max_seconds": max_seconds,
            # 분리 여부로 결과가 달라지므로 캐시를 분리해야 한다
            "separate": self._separation_setting(separate),
        }
        cache_key = self.cache.make_key(NAMESPACE_YOUTUBE, video_id, **options)

        # [A] 캐시
        if use_cache:
            hit = self.cache.get(cache_key)
            if hit is not None:
                result = AnalysisResult.from_dict(hit)
                result.cached = True
                result.origin = "cache"
                result.source_key = f"youtube:{video_id}"
                if not result.audio_path:
                    candidate = AUDIO_DIR / f"{video_id}.wav"
                    result.audio_path = str(candidate) if candidate.is_file() else ""
                self._register(result)
                return result, self._video_from_cache(hit)

        info = self.search_backend.metadata(video_id)

        # [B] 공개 코드 DB — 사람이 만든 데이터가 더 정확하고 연산도 0
        if info is not None and info.title:
            artist, title = split_artist_title(info.title, info.channel)
            if artist and title:
                found = self.database.lookup(artist, title)
                if found is not None:
                    found.source = info.title
                    if use_cache:
                        self.cache.put(cache_key, NAMESPACE_YOUTUBE,
                                       self._with_video(found.to_dict(), info))
                    return found, info

        # [C] 오디오 분석
        audio_path = self.audio_provider.fetch(video_id, AUDIO_DIR)
        from ..analysis.pipeline import analyze_audio

        result = analyze_audio(
            str(audio_path),
            vocabulary=vocab,
            key_override=Key.parse(key_override) if key_override else None,
            detect_inversions=detect_inversions,
            separate=separate,
            max_seconds=max_seconds,
        )
        result.source = info.title if info and info.title else video_id
        result.source_key = f"youtube:{video_id}"
        result.audio_path = str(audio_path)
        if use_cache:
            self.cache.put(cache_key, NAMESPACE_YOUTUBE, self._with_video(result.to_dict(), info))
        self._register(result)
        return result, info

    @staticmethod
    def _with_video(payload: Dict[str, Any], info: Optional[VideoInfo]) -> Dict[str, Any]:
        if info is not None:
            payload = dict(payload, video=info.to_dict())
        return payload

    @staticmethod
    def _video_from_cache(payload: Dict[str, Any]) -> Optional[VideoInfo]:
        raw = payload.get("video")
        if not raw:
            return None
        return VideoInfo(
            video_id=raw["video_id"], title=raw.get("title", ""),
            channel=raw.get("channel", ""), duration=raw.get("duration"),
            thumbnail=raw.get("thumbnail", ""),
        )

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
    ) -> AnalysisResult:
        path = Path(path)
        vocab = vocabulary or self.vocabulary
        options: Dict[str, Any] = {
            "vocabulary": vocab,
            "key": key_override,
            "inversions": detect_inversions,
            "max_seconds": max_seconds,
            # 분리 여부로 결과가 달라지므로 캐시를 분리해야 한다
            "separate": self._separation_setting(separate),
        }
        fingerprint = file_fingerprint(path)
        cache_key = self.cache.make_key(NAMESPACE_AUDIO, fingerprint, **options)

        if use_cache:
            hit = self.cache.get(cache_key)
            if hit is not None:
                result = AnalysisResult.from_dict(hit)
                result.cached = True
                result.origin = "cache"
                # 캐시 payload를 믿지 않고 다시 세운다. 예전에 저장된 항목에는
                # 이 필드들이 없어서, 그대로 두면 교정을 붙일 수 없다.
                result.source_key = f"file:{fingerprint}"
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
            separate=separate,
            max_seconds=max_seconds,
        )
        result.source = path.name
        result.source_key = f"file:{fingerprint}"
        result.audio_path = str(path.resolve())
        if use_cache:
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
        data["corrections"] = [c.to_dict() for c in
                               self.corrections.corrections(shown.source_key)] if shown.source_key else []
        return data
