"""공식 YouTube 검색·메타데이터와 URL 처리. 음원 공급 모듈과 분리한다."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

API_KEY_ENV = "YOUTUBE_API_KEY"

_API_BASE = "https://www.googleapis.com/youtube/v3"

# 11자 영상 ID. URL 형태가 여럿이라 정규식 하나로 뭉개지 않고 파싱해서 뽑는다.
_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com",
    "music.youtube.com", "youtu.be", "www.youtu.be",
}


class YouTubeError(Exception):
    """유튜브 레이어 공통 예외."""


class InvalidVideoUrl(YouTubeError):
    pass


class SearchUnavailable(YouTubeError):
    def __init__(self, message: str, code: str = "search_unavailable"):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------
# 링크 파싱
# --------------------------------------------------------------------------

def parse_video_id(text: str) -> str:
    """유튜브 URL 또는 영상 ID에서 11자 ID를 뽑는다."""
    text = (text or "").strip()
    if not text:
        raise InvalidVideoUrl("입력이 비어 있다")
    if _VIDEO_ID_RE.match(text):
        return text

    candidate = text if "://" in text else "https://" + text
    try:
        parsed = urllib.parse.urlparse(candidate)
    except ValueError as exc:
        raise InvalidVideoUrl(f"URL을 해석할 수 없다: {text!r}") from exc

    host = (parsed.hostname or "").lower()
    if host not in _HOSTS:
        raise InvalidVideoUrl(f"유튜브 주소가 아니다: {text!r}")

    if host.endswith("youtu.be"):
        vid = parsed.path.lstrip("/").split("/")[0]
    else:
        query = urllib.parse.parse_qs(parsed.query)
        if "v" in query and query["v"]:
            vid = query["v"][0]
        else:
            # /shorts/<id>, /embed/<id>, /live/<id>, /v/<id>
            parts = [p for p in parsed.path.split("/") if p]
            vid = ""
            if len(parts) >= 2 and parts[0] in ("shorts", "embed", "live", "v"):
                vid = parts[1]

    if not _VIDEO_ID_RE.match(vid):
        raise InvalidVideoUrl(f"영상 ID를 찾을 수 없다: {text!r}")
    return vid


def watch_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


# 영상 제목에서 걷어낼 관용적 꼬리표. 코드 DB 조회 정확도를 위해 정리한다.
_NOISE_PATTERNS = [
    r"\[[^\]]*\]", r"\([^)]*\)", r"【[^】]*】", r"「[^」]*」",
    r"(?i)\bofficial\b.*$", r"(?i)\bm/?v\b.*$", r"(?i)\blyrics?\b.*$",
    r"(?i)\baudio\b.*$", r"(?i)\bfull\s+album\b.*$", r"(?i)\b4k\b", r"(?i)\bhd\b",
]


def split_artist_title(video_title: str, channel: str = "") -> "tuple[str, str]":
    """'아티스트 - 곡명 (Official MV)' 형태를 (아티스트, 곡명)으로.

    구분자가 없으면 채널명을 아티스트로 본다. 완벽할 수 없는 휴리스틱이고,
    실패해도 코드 DB 조회가 MISS 날 뿐이라 오디오 분석으로 넘어간다.
    """
    text = video_title or ""
    for pattern in _NOISE_PATTERNS:
        text = re.sub(pattern, " ", text)
    text = " ".join(text.split())

    # " _ "는 1theK 등 국내 뮤직비디오 채널이 쓰는 형식
    # (예: "[MV] IU(아이유) _ Through the Night(밤편지)")
    for sep in (" - ", " – ", " — ", " _ ", " ~ ", " | "):
        if sep in text:
            left, right = text.split(sep, 1)
            return left.strip(), _trim_tail(right)
    return _clean_channel(channel), _trim_tail(text)


def _trim_tail(title: str) -> str:
    """곡명 뒤에 붙은 방송/무대 정보를 잘라낸다.

    "밤편지 | KBS 200918 방송" 같은 꼬리가 남으면 코드 DB 조회가 MISS 난다.
    '|'는 곡명에 거의 안 쓰이므로 첫 '|' 이후를 버린다.
    """
    return title.split(" | ")[0].strip(" -–—_~").strip()


# 유튜브가 자동 생성한 아티스트 채널에 붙는 꼬리표
_CHANNEL_NOISE_RE = re.compile(r"\s*[-–]\s*Topic$|VEVO$|\s*Official$", re.IGNORECASE)


def _clean_channel(channel: str) -> str:
    """'IU - Topic' → 'IU'. 채널명을 아티스트로 쓸 때만 의미가 있다."""
    text = (channel or "").strip()
    previous = None
    while text != previous:
        previous = text
        text = _CHANNEL_NOISE_RE.sub("", text).strip()
    return text


# --------------------------------------------------------------------------
# 검색 / 메타데이터
# --------------------------------------------------------------------------

@dataclass
class VideoInfo:
    video_id: str
    title: str = ""
    channel: str = ""
    duration: Optional[float] = None      # 초
    thumbnail: str = ""

    @property
    def url(self) -> str:
        return watch_url(self.video_id)

    def to_dict(self) -> Dict[str, Any]:
        artist, title = split_artist_title(self.title, self.channel)
        return {
            "video_id": self.video_id,
            "title": self.title,
            "channel": self.channel,
            "duration": self.duration,
            "thumbnail": self.thumbnail or f"https://i.ytimg.com/vi/{self.video_id}/mqdefault.jpg",
            "url": self.url,
            "guessed_artist": artist,
            "guessed_title": title,
        }


class VideoSearch(Protocol):
    available: bool

    def search(self, query: str, limit: int = 8) -> List[VideoInfo]: ...

    def metadata(self, video_id: str) -> Optional[VideoInfo]: ...


class DataApiSearch:
    """공식 YouTube Data API v3. 검색과 메타데이터 조회만 사용한다."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0):
        self._explicit_key = api_key
        self.timeout = timeout

    @property
    def api_key(self) -> str:
        # 매번 환경변수를 다시 읽는다. .env를 고친 뒤 서버만 재시작하면 되도록.
        return self._explicit_key or os.environ.get(API_KEY_ENV, "")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _get(self, endpoint: str, params: Dict[str, str]) -> Dict[str, Any]:
        params = dict(params, key=self.api_key)
        url = f"{_API_BASE}/{endpoint}?{urllib.parse.urlencode(params)}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            # Google의 원문 오류나 요청 URL에 키가 포함될 수 있어 반환하지 않는다.
            try:
                detail = json.loads(exc.read().decode("utf-8", "replace"))
                reasons = {entry.get("reason", "") for entry in detail.get("error", {}).get("errors", [])}
                reasons.update(entry.get("reason", "") for entry in detail.get("error", {}).get("details", []))
            except (ValueError, AttributeError, TypeError):
                reasons = set()
            if reasons & {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"} or exc.code == 429:
                code, message = "quota_exceeded", "YouTube 검색 할당량을 초과했습니다. Google Cloud의 할당량을 확인하거나 나중에 다시 시도해 주세요."
            elif reasons & {"accessNotConfigured", "SERVICE_DISABLED"}:
                code, message = "api_not_enabled", "이 키의 프로젝트에서 YouTube Data API v3를 사용 설정해 주세요."
            elif exc.code == 400:
                code, message = "invalid_api_key", "API 키를 확인해 주세요. 키 전체를 공백 없이 입력해야 합니다."
            elif exc.code == 403:
                code, message = "key_restricted", "API 키의 제한 설정을 확인해 주세요. 서버 요청에 YouTube Data API v3 사용이 허용되어야 합니다."
            else:
                code, message = "api_error", f"YouTube 검색 서버가 오류를 반환했습니다 ({exc.code}). 잠시 후 다시 시도해 주세요."
            raise SearchUnavailable(message, code) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise SearchUnavailable("서버에서 YouTube Data API에 연결하지 못했습니다. 네트워크 상태를 확인해 주세요.", "network_error") from None

    def validate(self) -> None:
        """검색 할당량을 쓰지 않고 videos.list 요청으로 키·API 사용 설정을 확인한다."""
        self._get("videos", {"part": "id", "id": "BaW_jenozKc"})

    def search(self, query: str, limit: int = 8) -> List[VideoInfo]:
        if not self.available:
            raise SearchUnavailable(
                f"{API_KEY_ENV} 환경변수가 없다. 유튜브 검색을 쓰려면 "
                f"YouTube Data API v3 키를 발급받아 설정해야 한다."
            )
        data = self._get("search", {
            "part": "snippet", "q": query, "type": "video",
            "maxResults": str(max(1, min(limit, 25))),
            "videoCategoryId": "10",   # Music
        })
        ids = [item["id"]["videoId"] for item in data.get("items", []) if item.get("id", {}).get("videoId")]
        return self._details(ids)

    def metadata(self, video_id: str) -> Optional[VideoInfo]:
        if not self.available:
            return fetch_oembed(video_id)
        try:
            found = self._details([video_id])
        except SearchUnavailable:
            return fetch_oembed(video_id)   # 할당량 초과 등에서도 제목은 얻는다
        return found[0] if found else fetch_oembed(video_id)

    def _details(self, video_ids: List[str]) -> List[VideoInfo]:
        if not video_ids:
            return []
        data = self._get("videos", {"part": "snippet,contentDetails", "id": ",".join(video_ids)})
        out: List[VideoInfo] = []
        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            thumbs = snippet.get("thumbnails", {})
            best = thumbs.get("medium") or thumbs.get("default") or {}
            out.append(VideoInfo(
                video_id=item["id"],
                title=snippet.get("title", ""),
                channel=snippet.get("channelTitle", ""),
                duration=_parse_iso8601_duration(item.get("contentDetails", {}).get("duration", "")),
                thumbnail=best.get("url", ""),
            ))
        return out


_OEMBED_URL = "https://www.youtube.com/oembed"


def fetch_oembed(video_id: str, timeout: float = 8.0) -> Optional[VideoInfo]:
    """공식 oEmbed 엔드포인트로 제목/채널만 가져온다. API 키가 필요 없다.

    검색은 못 하지만, 링크를 이미 아는 상황에서는 이것만으로 제목을 얻어
    [B] 공개 코드 DB 조회까지 갈 수 있다. 즉 자격증명 0으로도
    "링크 붙여넣기 → 코드 DB HIT" 경로가 살아 있다.
    """
    params = urllib.parse.urlencode({"url": watch_url(video_id), "format": "json"})
    try:
        with urllib.request.urlopen(f"{_OEMBED_URL}?{params}", timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError):
        return None
    return VideoInfo(
        video_id=video_id,
        title=data.get("title", ""),
        channel=data.get("author_name", ""),
        thumbnail=data.get("thumbnail_url", ""),
    )


class DisabledSearch:
    """API 키가 없을 때. 검색은 막히지만 링크 직접 입력은 계속 동작한다."""

    available = False

    def search(self, query: str, limit: int = 8) -> List[VideoInfo]:
        raise SearchUnavailable(
            f"유튜브 검색에는 {API_KEY_ENV}가 필요하다. 키 없이도 영상 링크를 "
            f"직접 붙여넣으면 제목 조회와 코드 DB 검색은 동작한다."
        )

    def metadata(self, video_id: str) -> Optional[VideoInfo]:
        return fetch_oembed(video_id)


_ISO_DURATION_RE = re.compile(
    r"P(?:(?P<d>\d+)D)?T?(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?"
)


def _parse_iso8601_duration(text: str) -> Optional[float]:
    """'PT4M13S' -> 253.0"""
    if not text:
        return None
    m = _ISO_DURATION_RE.match(text)
    if not m:
        return None
    parts = {k: int(v) for k, v in m.groupdict(default="0").items()}
    return float(parts["d"] * 86400 + parts["h"] * 3600 + parts["m"] * 60 + parts["s"])


def default_search() -> VideoSearch:
    # 웹에서 키를 저장하면 서버 재시작 없이 available이 바뀐다.
    return DataApiSearch()
