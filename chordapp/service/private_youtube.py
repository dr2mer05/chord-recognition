"""개인 전용 yt-dlp 공급 경로. 분석 동안만 파일을 보유한다."""

from __future__ import annotations

from contextlib import contextmanager
from importlib.util import find_spec
import os
import signal
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .youtube import YouTubeError, parse_video_id, watch_url

ENABLE_ENV = "CHORDAPP_ENABLE_YTDLP"
PRIVATE_ENV = "CHORDAPP_PRIVATE_MODE"


class PrivateYouTubeError(YouTubeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


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
                "--no-playlist", "--no-progress", "--no-warnings",
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
                raise PrivateYouTubeError("download_failed", "YouTube 음원에 접근하지 못했습니다. 영상 공개 여부·접근 제한 또는 연결 도구의 호환성을 확인해 주세요.")
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
