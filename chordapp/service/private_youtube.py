"""개인 전용 yt-dlp 공급 경로. 분석 동안만 파일을 보유한다."""

from __future__ import annotations

from contextlib import contextmanager
from importlib.util import find_spec
import os
import re
import signal
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache

from .youtube import YouTubeError, parse_video_id, watch_url

ENABLE_ENV = "CHORDAPP_ENABLE_YTDLP"
PRIVATE_ENV = "CHORDAPP_PRIVATE_MODE"


class PrivateYouTubeError(YouTubeError):
    def __init__(self, code: str, message: str, diagnostics=None):
        super().__init__(message)
        self.code = code
        self.diagnostics = diagnostics


def download_failure(proc):
    """원문을 노출하지 않고 알려진 실패 증거와 처리 단계만 추출한다."""
    text = (proc.stderr or "").lower()
    output = (proc.stdout or "").lower()
    code, message = "download_failed", "음원 연결에 실패했습니다. 실행 상태에서 실패 단계와 응답 상태를 확인해 주세요."
    if "unable to connect to proxy" in text or "tunnel connection failed" in text:
        code, message = "network_proxy_blocked", "서버의 네트워크 프록시가 YouTube 연결을 거부했습니다. 현재 서버의 접속 환경을 확인해야 합니다."
    elif ("not a bot" in text and "sign in" in text) or "captcha" in text:
        code, message = "youtube_bot_check", "YouTube가 이 서버의 접속에 자동화 확인을 요구했습니다. 코드 분석 전에 음원 연결이 차단된 상태입니다."
    elif "http error 429" in text or "too many requests" in text:
        code, message = "youtube_rate_limited", "YouTube가 서버의 요청 빈도를 제한했습니다. 잠시 후 다시 확인해 주세요."
    elif "not available in your country" in text or "geo-restricted" in text:
        code, message = "video_region_restricted", "이 영상은 서버가 위치한 지역에서 이용할 수 없습니다."
    elif "confirm your age" in text or "age-restricted" in text or "private video" in text or "members-only" in text:
        code, message = "video_login_required", "이 영상은 연령·로그인·멤버십 등의 접근 권한을 요구합니다."
    elif "video unavailable" in text or "has been removed" in text:
        code, message = "video_unavailable", "선택한 영상을 현재 서버에서 이용할 수 없습니다. 영상의 공개 상태를 확인해 주세요."
    elif "certificate verify failed" in text or "certificate_verify_failed" in text:
        code, message = "network_tls_error", "서버가 YouTube 연결의 인증서를 검증하지 못했습니다. 서버의 인증서 설정을 확인해야 합니다."
    elif ("po token" in text or "pot server" in text or "bgutil" in text) and any(word in text for word in ("failed", "error", "unavailable", "not available", "not found")):
        code, message = "token_generation_failed", "YouTube 요청 증명 토큰을 생성하거나 전달하지 못했습니다. 실행 상태에서 토큰 공급기 연결을 확인해 주세요."
    elif "no supported javascript runtime" in text or "challenge solving failed" in text:
        code, message = "youtube_runtime_error", "YouTube 연결에 필요한 JavaScript 처리에 실패했습니다. 실행 상태에서 Deno와 EJS 설치 상태를 확인해 주세요."
    elif "requested format is not available" in text:
        code, message = "audio_format_unavailable", "YouTube에서 다운로드 가능한 음원 형식을 받지 못했습니다. 영상 또는 연결 도구의 호환성 확인이 필요합니다."
    elif "ffmpeg" in text and ("not found" in text or "postprocessing" in text or "error" in text):
        code, message = "audio_conversion_failed", "음원 변환 도구 FFmpeg 처리에 실패했습니다. 실행 상태에서 도구 실행 여부를 확인해 주세요."
    elif "http error 403" in text or "http error 451" in text:
        code, message = "youtube_access_denied", "YouTube가 서버의 음원 요청을 거부했습니다. 응답 코드만으로는 영상 제한과 서버 접속 제한을 구분할 수 없습니다."
    stage = "영상 정보 조회"
    if "[download]" in output:
        stage = "음원 다운로드"
    if "[extractaudio]" in output or code == "audio_conversion_failed":
        stage = "음원 변환"
    status = re.search(r'(?:http error|tunnel connection failed:)\s*(\d{3})', text)
    return PrivateYouTubeError(code, message, {"stage": stage, "exit_code": proc.returncode,
        "http_status": int(status[1]) if status else None})


def find_ffmpeg():
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def find_deno():
    local = Path(sys.executable).parent / ("deno.exe" if os.name == "nt" else "deno")
    return str(local) if local.is_file() else shutil.which("deno")


@lru_cache(maxsize=1)
def runtime_versions():
    """파일 존재와 별도로 실행 가능 여부를 점검한다. 경로·원문은 반환하지 않는다."""
    from importlib.metadata import PackageNotFoundError, version
    versions = {}
    for name, binary, args, pattern in (
        ("deno", find_deno(), ["--version"], r"deno (\d+\.\d+\.\d+)"),
        ("ffmpeg", find_ffmpeg(), ["-version"], r"ffmpeg version ([A-Za-z0-9._-]+)"),
    ):
        try:
            result = subprocess.run([binary, *args], capture_output=True, text=True, timeout=3) if binary else None
            match = re.search(pattern, result.stdout) if result and result.returncode == 0 else None
            versions[name] = match[1] if match else None
        except (OSError, subprocess.TimeoutExpired):
            versions[name] = None
    try:
        versions["ejs"] = version("yt-dlp-ejs")
    except PackageNotFoundError:
        versions["ejs"] = None
    return versions


class PrivateYouTubeProvider:
    def __init__(self, temp_dir=Path("work/private-youtube"), timeout=180):
        self.temp_dir = Path(temp_dir)
        self.timeout = timeout

    @property
    def enabled(self):
        return os.environ.get(ENABLE_ENV) == "1" and os.environ.get(PRIVATE_ENV) == "1"

    @property
    def reason(self):
        if not self.enabled:
            return "개인 전용 링크 분석이 꺼져 있습니다. 별도 음원을 연결하거나 서버의 개인 모드를 설정해 주세요."
        if find_spec("yt_dlp") is None or not find_ffmpeg() or not find_deno():
            return "개인 링크 분석에 필요한 도구가 없습니다. 서버에서 bash scripts/setup-dev.sh를 실행해 주세요."
        return "개인 전용 링크 분석을 사용할 수 있습니다. 음원은 분석 후 삭제됩니다."

    @property
    def available(self):
        return self.enabled and find_spec("yt_dlp") is not None and bool(find_ffmpeg()) and bool(find_deno())

    def ensure_available(self):
        if not self.available:
            raise PrivateYouTubeError("private_mode_disabled" if not self.enabled else "dependency_missing", self.reason)

    def command(self, video_id, directory):
        # 고정 파일명·검증된 ID만 전달하고, 사용자의 전역 yt-dlp 설정은 읽지 않는다.
        args = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-cache-dir",
                "--no-playlist", "--no-progress",
                "--socket-timeout", "15", "--retries", "1", "--extractor-retries", "1",
                "--fragment-retries", "1", "--abort-on-unavailable-fragments",
                "--match-filter", "!is_live & duration > 0 & duration <= 900",
                "--max-filesize", "40M", "-f", "bestaudio/best",
                "-x", "--audio-format", "wav",
                "--postprocessor-args", "ffmpeg:-ar 22050 -ac 1",
                "--ffmpeg-location", find_ffmpeg(),
                "-o", str(directory / "audio.%(ext)s")]
        # EJS는 설치된 패키지만 사용한다. 외부 스크립트 자동 다운로드는 하지 않는다.
        args += ["--js-runtimes", "deno:" + find_deno()]
        from . import pot
        if pot.enabled():
            args += ["--extractor-args", "youtube:player_client=mweb",
                     "--extractor-args", "youtubepot-bgutilhttp:base_url=" + pot.URL]
        else:
            # 비교 기준선은 설치된 외부 공급 플러그인을 자동 사용하지 않는다.
            args += ["--no-plugin-dirs"]
        return args + [watch_url(video_id)]

    def _download(self, command):
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", errors="replace",
                                start_new_session=os.name == "posix")
        try:
            stdout, stderr = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            # yt-dlp가 띄운 ffmpeg도 함께 종료해야 임시 파일 정리가 끝난다.
            if os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                proc.kill()
            proc.communicate()
            raise
        return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)

    @contextmanager
    def audio(self, video):
        video_id = parse_video_id(video)
        self.ensure_available()
        from . import pot
        if pot.enabled():
            try:
                pot.ensure_started()
            except pot.PotError as exc:
                raise PrivateYouTubeError("pot_unavailable", str(exc)) from None
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        # 동일 영상의 동시 요청도 서로의 파일을 덮어쓰거나 삭제하지 않는다.
        with tempfile.TemporaryDirectory(prefix=video_id + "-", dir=self.temp_dir) as scratch:
            directory = Path(scratch)
            try:
                proc = self._download(self.command(video_id, directory))
            except subprocess.TimeoutExpired as exc:
                raise PrivateYouTubeError("download_timeout", "영상 음원 연결 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.") from exc
            except OSError as exc:
                raise PrivateYouTubeError("download_failed", "음원 연결 도구를 실행하지 못했습니다. 서버 설치 상태를 확인해 주세요.") from exc
            if proc.returncode:
                error = download_failure(proc)
                if pot.enabled():
                    error.diagnostics["token_provider"] = pot.status()["state"]
                raise error
            path = directory / "audio.wav"
            if not path.is_file():
                raise PrivateYouTubeError("unsupported_video", "분석할 음원이 없습니다. 라이브 영상과 15분 초과 영상은 지원하지 않습니다.")
            import soundfile as sf
            try:
                info = sf.info(str(path))
                valid = 0 < info.duration <= 900 and path.stat().st_size <= 40 * 1024 * 1024
            except (OSError, RuntimeError, ValueError):
                valid = False
            if not valid:
                raise PrivateYouTubeError("audio_invalid", "연결된 음원을 읽을 수 없거나 15분·40MB 제한을 초과했습니다.")
            yield path
