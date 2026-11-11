"""개인용 PO Token 공급기 설치·시작·상태. 토큰 원문을 앱에 반환하지 않는다."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request
from importlib.metadata import PackageNotFoundError, version

VERSION = "2.0.2"
COMMIT = "26475e9d3665b972c4aeb83b4dbaafcf64b88438"
URL = "http://127.0.0.1:4416"
ENABLE_ENV = "CHORDAPP_ENABLE_POT"
ROOT = Path(__file__).resolve().parents[2]
_start_lock = threading.Lock()


class PotError(Exception):
    pass


def enabled():
    return (os.environ.get("CHORDAPP_PRIVATE_MODE") == "1"
            and os.environ.get("CHORDAPP_ENABLE_YTDLP") == "1"
            and os.environ.get(ENABLE_ENV) == "1")


def paths():
    base = ROOT / "work" / "pot"
    return base, base / ("provider-" + VERSION)


def plugin_version():
    try:
        return version("bgutil-ytdlp-pot-provider")
    except PackageNotFoundError:
        return None


def _installed():
    _, source = paths()
    try:
        stamp = json.loads((source / ".chordapp-installed.json").read_text())
        return stamp.get("commit") == COMMIT and (source / "server/node_modules").is_dir()
    except (OSError, ValueError):
        return False


def _server_version():
    # 로컬 제어 요청에만 외부 프록시를 사용하지 않는다.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(URL + "/ping", timeout=.7) as response:
            return json.loads(response.read(4096)).get("version")
    except (OSError, ValueError):
        return None


def status():
    requested = enabled()
    plugin = plugin_version()
    server = _server_version() if requested else None
    installed = _installed()
    if not requested:
        state = "disabled"
    elif plugin != VERSION or (server and server != VERSION):
        state = "version_mismatch" if plugin else "not_installed"
    elif server == VERSION:
        state = "ready"
    else:
        state = "stopped" if installed else "not_installed"
    return {"enabled": requested, "state": state, "expected_version": VERSION,
            "plugin_version": plugin, "server_version": server, "installed": installed}


def _env():
    base, _ = paths()
    values = {**os.environ, "DENO_DIR": str(base / "deno-cache"), "DENO_NO_UPDATE_CHECK": "1"}
    # 검색 키는 요청 증명 생성에 필요하지 않다. 네트워크 프록시·CA 설정은 그대로 보존한다.
    values.pop("YOUTUBE_API_KEY", None)
    return values


def _run(command, *, cwd, timeout=300):
    base, _ = paths()
    with (base / "install.log").open("ab") as log:
        proc = subprocess.Popen(command, cwd=cwd, env=_env(), stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            raise PotError("토큰 공급기 설치 시간이 초과되었습니다.") from None
    if code:
        raise PotError("토큰 공급기 설치에 실패했습니다. work/pot/install.log를 확인해 주세요.")


def server_command():
    from .private_youtube import find_deno
    _, source = paths()
    return [find_deno(), "run", "--allow-env", "--allow-net", "--allow-ffi=.", "--allow-read=.",
            "../src/main.ts", "--host", "127.0.0.1", "--port", "4416"]


def install():
    from .private_youtube import find_deno
    if not find_deno() or plugin_version() != VERSION:
        raise PotError("먼저 requirements-private.txt의 고정 버전을 설치해 주세요.")
    base, source = paths()
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    base.chmod(0o700)
    if _installed():
        return
    if not source.exists():
        _run(["git", "clone", "--depth", "1", "--branch", VERSION,
              "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git", str(source)],
             cwd=ROOT, timeout=60)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    except (OSError, subprocess.SubprocessError):
        raise PotError("토큰 공급기 소스 버전을 확인하지 못했습니다.") from None
    if commit != COMMIT:
        raise PotError("토큰 공급기 소스가 고정 버전과 다릅니다. 설치 폴더를 자동으로 덮어쓰지 않았습니다.")
    _run([find_deno(), "install", "--allow-scripts=npm:canvas", "--frozen"], cwd=source / "server")
    # 실제 모듈·네이티브 의존성을 로드한다. 단순 파일 존재로 설치 성공을 판단하지 않는다.
    _run(server_command() + ["--help"], cwd=source / "server/node_modules", timeout=30)
    (source / ".chordapp-installed.json").write_text(json.dumps({"version": VERSION, "commit": COMMIT}))


def ensure_started():
    if not enabled():
        return
    with _start_lock:
        current = status()
        if current["state"] == "ready":
            return
        if current["state"] != "stopped":
            raise PotError("토큰 공급 모듈을 설치하거나 버전을 맞춰야 합니다. 사이트 업데이트·설치 후 다시 시작해 주세요.")
        base, source = paths()
        # 외부 공급기가 토큰 원문을 stdout에 쓰므로 실행 출력을 보관하거나 반환하지 않는다.
        proc = subprocess.Popen(server_command(), cwd=source / "server/node_modules", env=_env(),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(40):
            if _server_version() == VERSION:
                (base / "server.pid").write_text(str(proc.pid))
                return
            if proc.poll() is not None:
                break
            time.sleep(.2)
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            except ProcessLookupError:
                proc.wait()
        raise PotError("토큰 공급 서버가 시작되지 않았습니다. 실행 상태에서 공급 모듈 상태를 확인해 주세요.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("install", "start", "status"))
    args = parser.parse_args()
    try:
        if args.action == "install":
            install()
        elif args.action == "start":
            ensure_started()
        print(json.dumps(status(), ensure_ascii=False))
    except (PotError, OSError) as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
