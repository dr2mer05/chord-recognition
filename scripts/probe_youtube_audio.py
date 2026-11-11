"""음원 확보만 시험하는 독립 프로그램. 다운로드 원문·음원은 외부에 제공하지 않는다."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import wave

DEFAULT_VIDEO = "BzYnNdJhZQw"
MAX_BYTES = 40 * 1024 * 1024
PACKAGES = ("yt-dlp", "yt-dlp-ejs", "imageio-ffmpeg", "deno")


def validate_id(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
        raise ValueError("SOURCE_PROBE_VIDEO_ID에는 11자 YouTube 영상 ID만 지정할 수 있습니다.")
    return value


def tools():
    deno = Path(sys.executable).parent / ("deno.exe" if os.name == "nt" else "deno")
    deno = str(deno) if deno.is_file() else shutil.which("deno")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            ffmpeg = None
    return deno, ffmpeg


def environment():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    deno, ffmpeg = tools()
    runnable, runtime_versions = {}, {}
    for name, binary, arguments in (("deno", deno, ["--version"]), ("ffmpeg", ffmpeg, ["-version"])):
        try:
            result = subprocess.run([binary, *arguments], capture_output=True, text=True, timeout=5) if binary else None
            runnable[name] = bool(result and result.returncode == 0)
            pattern = r"deno ([0-9.]+)" if name == "deno" else r"ffmpeg version ([A-Za-z0-9._-]+)"
            match = re.search(pattern, result.stdout) if runnable[name] else None
            runtime_versions[name] = match[1] if match else None
        except (OSError, subprocess.TimeoutExpired):
            runnable[name] = False
            runtime_versions[name] = None
    return {"python": sys.version.split()[0], "packages": versions,
            "runnable": runnable, "runtime_versions": runtime_versions}


def command(video_id, directory, *, ipv4, binaries):
    video_id = validate_id(video_id)
    deno, ffmpeg = binaries
    args = [sys.executable, "-m", "yt_dlp", "--ignore-config", "--no-plugin-dirs",
        "--no-cache-dir", "--no-playlist", "--no-progress", "--verbose",
        "--socket-timeout", "15", "--retries", "0", "--extractor-retries", "0",
        "--fragment-retries", "0", "--abort-on-unavailable-fragments",
        "--match-filter", "!is_live & duration > 0 & duration <= 900",
        "--max-filesize", "40M", "-f", "bestaudio/best", "-x", "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ar 22050 -ac 1", "--ffmpeg-location", ffmpeg,
        "--js-runtimes", "deno:" + deno, "-o", str(directory / "audio.%(ext)s")]
    if ipv4:
        args.append("--force-ipv4")
    return args + ["https://www.youtube.com/watch?v=" + video_id]


def download(args):
    child_env = dict(os.environ)
    child_env.pop("YOUTUBE_API_KEY", None)
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", start_new_session=os.name == "posix", env=child_env)
    try:
        stdout, stderr = proc.communicate(timeout=90)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            pass
        proc.communicate()
        raise
    return subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)


def failure(result):
    # 원문이나 외부 URL을 반환하지 않고 고정값과 숫자만 추출한다.
    text = ((result.stdout or "") + "\n" + (result.stderr or "")).lower()
    code = "download_failed"
    if "tunnel connection failed" in text or "unable to connect to proxy" in text:
        code = "network_proxy_blocked"
    elif "not a bot" in text and "sign in" in text:
        code = "youtube_bot_check"
    elif "certificate_verify_failed" in text or "certificate verify failed" in text:
        code = "network_tls_error"
    elif "http error 403" in text:
        code = "youtube_access_denied"
    elif "http error 429" in text:
        code = "youtube_rate_limited"
    elif "requested format is not available" in text:
        code = "audio_format_unavailable"
    stage = "conversion" if "[extractaudio]" in text else "download" if "[download]" in text else "metadata"
    match = re.search(r"(?:http error|tunnel connection failed:)\s*(\d{3})", text)
    return {"ok": False, "code": code, "stage": stage,
            "http_status": int(match[1]) if match else None, "exit_code": result.returncode}


def verify_audio(path):
    size = path.stat().st_size
    with wave.open(str(path), "rb") as audio:
        duration = audio.getnframes() / audio.getframerate()
        valid = (audio.getnchannels() == 1 and audio.getframerate() == 22050
                 and 0 < duration <= 900 and 0 < size <= MAX_BYTES and bool(audio.readframes(1)))
    if not valid:
        raise ValueError("invalid audio")
    return {"ok": True, "code": "audio_received", "stage": "wav_verified",
            "duration_seconds": round(duration, 2), "audio_bytes": size}


def probe(video_id, *, ipv4):
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="chord-source-probe-") as folder:
            directory = Path(folder)
            result = download(command(video_id, directory, ipv4=ipv4, binaries=tools()))
            if result.returncode:
                row = failure(result)
            else:
                try:
                    row = verify_audio(directory / "audio.wav")
                except (OSError, ValueError, wave.Error, EOFError):
                    row = {"ok": False, "code": "audio_invalid", "stage": "wav_validation"}
    except subprocess.TimeoutExpired:
        row = {"ok": False, "code": "download_timeout", "stage": "unknown"}
    except OSError:
        row = {"ok": False, "code": "local_io_error", "stage": "local_execution"}
    row.update(profile="default_ipv4" if ipv4 else "default", seconds=round(time.monotonic() - started, 1))
    return row


def run(video_id, publish):
    validate_id(video_id)
    report = {"state": "running", "video_id": video_id,
        "started_at": datetime.now(timezone.utc).isoformat(), "environment": environment(), "rows": []}
    publish(report)
    if not all(report["environment"]["packages"].values()) or not all(report["environment"]["runnable"].values()):
        report.update(state="failed", code="dependency_missing")
    else:
        for ipv4 in (False, True):
            report = {**report, "rows": [*report["rows"], probe(video_id, ipv4=ipv4)]}
            publish(report)
            # 성공하거나 환경 프록시가 명시적으로 거부하면 추가 요청을 하지 않는다.
            if report["rows"][-1]["ok"] or report["rows"][-1]["code"] == "network_proxy_blocked":
                break
        report = {**report, "state": "complete"}
    publish(report)
    return report


def page(report):
    labels = {"default": "표준 설정", "default_ipv4": "IPv4 지정"}
    stages = {"metadata": "영상 정보 조회", "download": "음원 다운로드", "conversion": "음원 변환", "wav_verified": "음원 확보·WAV 검증 완료"}
    rows = []
    for row in report.get("rows", []):
        rows.append(f'<li><strong>{escape(labels.get(row["profile"], row["profile"]))}: '
            f'{"성공" if row["ok"] else "실패"}</strong><p>{escape(row["code"])} · '
            f'{escape(stages.get(row["stage"], row["stage"]))} · HTTP {row.get("http_status") or "확인되지 않음"}'
            f' · {row["seconds"]}초</p></li>')
    refresh = '<meta http-equiv="refresh" content="5">' if report["state"] in {"running", "pending"} else ""
    return ('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        + refresh + '<title>Chord 음원 확보 검증</title><style>body{font:17px/1.6 system-ui;max-width:800px;margin:30px auto;padding:16px;background:#101217;color:#eee}pre{white-space:pre-wrap;overflow-wrap:anywhere}li{margin-bottom:22px}</style>'
        + '<h1>음원 확보만 검증</h1><p>영상: ' + escape(report.get("video_id", "")) + '</p><p>상태: '
        + escape({"pending":"검증 시작 대기", "running":"진행 중", "complete":"완료", "failed":"실행 준비 실패", "disabled":"실행하지 않음"}.get(report["state"], report["state"]))
        + '</p><ul>' + ''.join(rows) + '</ul><p>실제 음원을 받아 검증한 경우만 성공입니다. 음원은 삭제하며 코드 분석은 수행하지 않습니다.</p>'
        + '<details><summary>검증 기록</summary><pre>' + escape(json.dumps(report, ensure_ascii=False, indent=2)) + '</pre></details></html>')


def serve(port, video_id, enabled):
    report = {"state": "pending" if enabled else "disabled", "video_id": video_id, "rows": []}
    lock = threading.Lock()

    def publish(value):
        nonlocal report
        with lock:
            report = value

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            with lock:
                snapshot = report
            if self.path == "/healthz":
                body, content_type = b"ok", "text/plain"
            elif self.path == "/report.json":
                body, content_type = json.dumps(snapshot, ensure_ascii=False).encode(), "application/json"
            elif self.path == "/":
                body, content_type = page(snapshot).encode(), "text/html"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    if enabled:
        def worker():
            try:
                run(video_id, publish)
            except Exception:
                # 알 수 없는 예외의 원문·환경 값도 공개 페이지에 쓰지 않는다.
                publish({"state": "failed", "code": "internal_error", "video_id": video_id, "rows": []})
        threading.Thread(target=worker, daemon=True).start()
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("selfcheck", "run", "serve"))
    args = parser.parse_args()
    video_id = validate_id(os.environ.get("SOURCE_PROBE_VIDEO_ID", DEFAULT_VIDEO))
    if args.action == "selfcheck":
        print(json.dumps({"state": "not_run", "environment": environment()}, ensure_ascii=False))
    elif args.action == "run":
        print(json.dumps(run(video_id, lambda report: None), ensure_ascii=False))
    else:
        serve(int(os.environ.get("PORT", "8080")), video_id, os.environ.get("SOURCE_PROBE_RUN") == "1")


if __name__ == "__main__":
    main()
