"""동일 서버·동일 설치에서 클라이언트 설정만 바꿔 실제 음원 확보를 비교한다."""
from datetime import datetime, timezone
import threading
import time

from .private_youtube import PrivateYouTubeError, PrivateYouTubeProvider
from .youtube import parse_video_id

_lock = threading.Lock()


class CheckBusy(Exception):
    pass


def compare(video, publish):
    video_id = parse_video_id(video)
    if not _lock.acquire(blocking=False):
        raise CheckBusy("다른 연결 비교가 진행 중입니다. 잠시 후 다시 확인해 주세요.")
    report = {"state": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "rows": [], "scope": "same_server_same_dependencies_client_settings_only"}
    try:
        publish({**report, "rows": []})
        for profile, label in (("legacy_clients", "기존 클라이언트 설정"), ("current", "현재 설정")):
            started = time.monotonic()
            provider = PrivateYouTubeProvider(timeout=90, profile=profile)
            row = {"profile": profile, "label": label}
            try:
                with provider.audio(video_id) as path:
                    # 실제 WAV 검증은 공급 경로가 수행한다. 분석·캐시·교정에는 쓰지 않는다.
                    row.update(ok=True, code="audio_received", message="음원 확보·변환 성공",
                               audio_bytes=path.stat().st_size)
            except PrivateYouTubeError as exc:
                row.update(ok=False, code=exc.code, message=str(exc), details=exc.diagnostics)
            except OSError:
                row.update(ok=False, code="local_io_error", message="서버의 임시 파일 또는 도구 실행에 실패했습니다.")
            row["seconds"] = round(time.monotonic() - started, 1)
            report["rows"].append(row)
            publish({**report, "rows": list(report["rows"])})
        report["state"] = "complete"
        publish(report)
        return report
    finally:
        _lock.release()
