"""유튜브 입력 레이어: 검색 · 링크 파싱 · 오디오 확보.

파이프라인 M3의 "유튜브 연동" 항목. 다만 1-8 체크리스트에
"유튜브 오디오 추출 방식의 이용약관 적합성 검토"가 미해결로 남아 있으므로,
법적 판단이 필요한 부분과 그렇지 않은 부분을 코드에서 분리해 둔다.

  - 검색 / 메타데이터: 공식 YouTube Data API v3. API 키만 있으면 문제 없음.
  - 링크 파싱 / 캐시 키: 순수 로직. 아무 리스크 없음.
  - 오디오 추출: 검토 대상. **기본 비활성**이며 환경변수로 명시적으로 켜야 한다.

이렇게 두면 약관 검토가 끝나기 전에도 나머지 파이프라인을 전부 돌려볼 수 있고,
검토가 끝나면 설정 한 줄로 켜진다.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

ENABLE_AUDIO_ENV = "CHORDAPP_ENABLE_YTDLP"
API_KEY_ENV = "YOUTUBE_API_KEY"

# yt-dlp가 시도할 유튜브 player client 목록. 유튜브가 추출을 막는 방식이
# 계속 바뀌므로 여러 개를 넘겨 둔다. 추출이 깨지면 여기부터 손보면 된다.
PLAYER_CLIENTS = "android,ios,tv,web"

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
    pass


class AudioFetchDisabled(YouTubeError):
    pass


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
            detail = exc.read().decode("utf-8", "replace")[:200]
            raise SearchUnavailable(f"YouTube API 오류 {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise SearchUnavailable(f"YouTube API에 접속할 수 없다: {exc.reason}") from exc

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


# --------------------------------------------------------------------------
# 오디오 확보 (약관 검토 대상 — 기본 비활성)
# --------------------------------------------------------------------------

class AudioProvider(Protocol):
    enabled: bool

    def fetch(self, video_id: str, dest_dir: Path) -> Path: ...


class DisabledAudioProvider:
    """기본값. 무엇을 어떻게 켜야 하는지 명시적으로 알려주고 멈춘다."""

    enabled = False
    reason = (
        "유튜브 오디오 추출이 꺼져 있다. 파이프라인 1-8의 '유튜브 오디오 추출 방식의 "
        "이용약관 적합성 검토'가 끝난 뒤, yt-dlp를 설치하고 "
        f"{ENABLE_AUDIO_ENV}=1 을 설정하면 켜진다. "
        "그 전까지는 코드 DB 조회(HIT) 또는 로컬 파일 업로드로 분석할 수 있다."
    )

    def fetch(self, video_id: str, dest_dir: Path) -> Path:
        raise AudioFetchDisabled(self.reason)


def find_ffmpeg() -> Optional[str]:
    """ffmpeg 실행파일 위치. 시스템 설치본을 우선하고, 없으면 pip으로 받은 번들을 쓴다.

    시스템 전역 설치를 강요하지 않으려고 imageio-ffmpeg 폴백을 둔다
    (`pip install imageio-ffmpeg`이면 끝난다).
    """
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg

        path = imageio_ffmpeg.get_ffmpeg_exe()
        return path if os.path.isfile(path) else None
    except Exception:
        return None


class YtDlpProvider:
    """yt-dlp로 오디오만 받아온다. 명시적으로 켠 경우에만 동작한다.

    분석에 필요한 형태(22.05kHz 모노 wav)로 바로 받는다. 원본 품질을 보존할
    이유가 없고 — 어차피 크로마 계산 전에 다운샘플된다 — wav로 받으면
    디코딩에 ffmpeg가 다시 필요 없어서 soundfile이 바로 읽는다.
    4분 곡이 약 5MB.
    """

    def __init__(self, max_duration: float = 900.0, timeout: float = 300.0):
        self.max_duration = max_duration
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return (
            os.environ.get(ENABLE_AUDIO_ENV, "") == "1"
            and self.binary is not None
            and find_ffmpeg() is not None
        )

    @property
    def binary(self) -> Optional[str]:
        return shutil.which("yt-dlp")

    def _command(self, video_id: str, dest_dir: Path) -> List[str]:
        template = str(dest_dir / f"{video_id}.%(ext)s")
        command = [
            self.binary or "yt-dlp", "--no-playlist", "--quiet", "--no-warnings",
            # 여러 player client를 함께 넘긴다. 유튜브가 추출 경로를 자주 막는데
            # 클라이언트마다 막히는 시점이 달라서, 하나가 죽어도 다른 게 산다.
            # (기본 web 클라이언트는 "The page needs to be reloaded"로 실패했다.)
            "--extractor-args", f"youtube:player_client={PLAYER_CLIENTS}",
            # audio-only DASH 포맷은 PO 토큰으로 막혀 있는 경우가 많다.
            # 그때는 오디오가 들어 있는 통짜 mp4(포맷 18)로 폴백한다 — 어차피
            # 오디오만 뽑아 쓰므로 영상 트랙이 붙어 오는 건 대역폭 낭비일 뿐이다.
            "-f", "bestaudio/best",
            "-x", "--audio-format", "wav",
            # 분석기가 쓰는 포맷으로 바로 변환한다
            "--postprocessor-args", "ffmpeg:-ar 22050 -ac 1",
            "--match-filter", f"duration < {int(self.max_duration)}",
            "-o", template,
        ]
        ffmpeg = find_ffmpeg()
        if ffmpeg:
            command += ["--ffmpeg-location", ffmpeg]
        return command + [watch_url(video_id)]

    def fetch(self, video_id: str, dest_dir: Path) -> Path:
        if os.environ.get(ENABLE_AUDIO_ENV, "") != "1":
            raise AudioFetchDisabled(DisabledAudioProvider.reason)
        if self.binary is None:
            raise AudioFetchDisabled("yt-dlp가 설치돼 있지 않다: pip install yt-dlp")
        if find_ffmpeg() is None:
            raise AudioFetchDisabled(
                "오디오 추출에 ffmpeg가 필요하다. 시스템에 설치하거나 "
                "`pip install imageio-ffmpeg`로 번들을 받으면 된다."
            )

        dest_dir.mkdir(parents=True, exist_ok=True)
        existing = dest_dir / f"{video_id}.wav"
        if existing.is_file():
            return existing   # 같은 영상을 두 번 받을 이유가 없다

        try:
            # 윈도우 로케일 코덱(cp949)으로 읽으면 yt-dlp의 UTF-8 출력에서 터진다
            proc = subprocess.run(
                self._command(video_id, dest_dir),
                capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise YouTubeError(f"오디오 추출 시간 초과 ({self.timeout:.0f}초)") from exc
        if proc.returncode != 0:
            raise YouTubeError(f"yt-dlp 실패: {(proc.stderr or proc.stdout or '').strip()[:300]}")

        for candidate in sorted(dest_dir.glob(f"{video_id}.*")):
            if candidate.suffix.lower() in (".wav", ".m4a", ".mp3", ".opus", ".webm"):
                return candidate
        raise YouTubeError(
            f"{self.max_duration / 60:.0f}분을 넘는 영상이거나 오디오를 받지 못했다"
        )


def default_search() -> VideoSearch:
    api = DataApiSearch()
    return api if api.available else DisabledSearch()


def default_audio_provider() -> AudioProvider:
    provider = YtDlpProvider()
    return provider if provider.enabled else DisabledAudioProvider()
