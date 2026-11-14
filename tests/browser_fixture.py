"""브라우저 검증 전용 서버. 임시 데이터와 가짜 영상 메타데이터를 사용한다."""
import os
from pathlib import Path
import sys
import tempfile

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo))
(repo / "work").mkdir(exist_ok=True)

if __name__ == "__main__":
    import uvicorn
    with tempfile.TemporaryDirectory(prefix="browser-fixture-", dir=repo / "work") as scratch:
        os.chdir(scratch)
        from chordapp.api.server import app, service
        from chordapp.service.youtube import VideoInfo

        class FixtureSearch:
            available = False

            def metadata(self, video_id):
                return VideoInfo(video_id, title="합성 검증 음원", channel="테스트 전용")

        service.search_backend = FixtureSearch()
        if os.environ.get("CHORDAPP_TEST_POT") == "1":
            # 표시 계약만 검증한다. 실제 토큰 생성 성공을 모의 결과로 주장하지 않는다.
            from chordapp.service import pot
            pot.status = lambda: {"enabled": True, "state": "ready", "expected_version": "2.0.2",
                "plugin_version": "2.0.2", "server_version": "2.0.2", "installed": True}
        if os.environ.get("CHORDAPP_TEST_SEARCH") == "1":
            import chordapp.config as config
            from chordapp.service.youtube import DataApiSearch, SearchUnavailable
            config.PROJECT_ROOT = Path.cwd()  # 테스트 키는 임시 작업 폴더에만 저장한다.
            os.environ["CHORDAPP_PRIVATE_MODE"] = "1"
            os.environ["YOUTUBE_API_KEY"] = ""

            def fixture_get(self, endpoint, params):
                if self.api_key != "AIza" + "x" * 35:
                    raise SearchUnavailable("API 키를 확인해 주세요.", "invalid_api_key")
                if endpoint == "search":
                    return {"items": [{"id": {"videoId": "PrivateTest"}}]}
                if params.get("part") == "id":
                    return {"items": []}
                return {"items": [{"id": "PrivateTest", "snippet": {"title": "검색 검증 곡 <테스트>",
                    "channelTitle": "합성 검증", "thumbnails": {}}, "contentDetails": {"duration": "PT31S"}}]}

            DataApiSearch._get = fixture_get
            service.search_backend = DataApiSearch()
        if os.environ.get("CHORDAPP_TEST_PRIVATE") == "1":
            # 공급 네트워크만 합성 음원으로 대체하고 분석·API·화면은 실제 코드로 실행한다.
            from contextlib import contextmanager
            from chordapp.tools.synth import generate_all
            from chordapp.service.private_youtube import PrivateYouTubeProvider
            import shutil

            demo = generate_all(Path("fixture-audio"))[0]

            class FixturePrivateProvider(PrivateYouTubeProvider):
                @property
                def enabled(self): return True
                @property
                def available(self): return True
                @property
                def reason(self): return "합성 음원을 이용한 개인 링크 경로 검증"
                @contextmanager
                def audio(self, video):
                    if os.environ.get("CHORDAPP_TEST_DOWNLOAD_FAILURE") == "bot":
                        import subprocess
                        from chordapp.service.private_youtube import download_failure
                        raise download_failure(subprocess.CompletedProcess([], 1,
                            '[youtube] Downloading webpage', "Sign in to confirm you're not a bot. --cookies-from-browser"))
                    with tempfile.TemporaryDirectory(dir=".") as folder:
                        path = Path(folder) / "audio.wav"
                        shutil.copyfile(demo, path)
                        yield path

            service.private_youtube = FixturePrivateProvider()
            if os.environ.get("CHORDAPP_TEST_CONNECTION_COMPARE") == "1":
                # 두 프로필 모두 실제 공급기 검증·임시 파일 정리를 거치되 외부 요청만 모사한다.
                from chordapp.service import connection_check, pot
                from chordapp.service.private_youtube import PrivateYouTubeProvider
                import subprocess
                for name in ("CHORDAPP_PRIVATE_MODE", "CHORDAPP_ENABLE_YTDLP", "CHORDAPP_ENABLE_POT"):
                    os.environ[name] = "1"
                pot.ensure_started = lambda: None

                class CompareProvider(PrivateYouTubeProvider):
                    def _download(self, command):
                        if self.profile == "current":
                            return subprocess.CompletedProcess(command, 1, "[download] Destination: fixture", "HTTP Error 403: Forbidden")
                        output = Path(command[command.index("-o") + 1].replace("%(ext)s", "wav"))
                        shutil.copyfile(demo, output)
                        return subprocess.CompletedProcess(command, 0, "", "")

                connection_check.PrivateYouTubeProvider = CompareProvider
        meter = os.environ.get("CHORDAPP_TEST_METER")
        if meter:
            import numpy as np
            import soundfile as sf
            import chordapp.analysis.audio as audio
            import chordapp.analysis.beat_tracker as beat_tracker

            class FixtureTracker:
                name = "fixture-meter"
                available = True
                device = "cpu"

                def track(self, path):
                    beats = np.arange(0, sf.info(path).duration, .5)
                    return beat_tracker.BeatGrid(beats, 120, beats[::int(meter)], int(meter), self.name)

            tracker = FixtureTracker()
            audio.default_tracker = lambda: tracker
            beat_tracker.default_tracker = lambda: tracker
        uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("CHORDAPP_TEST_PORT", "8765")))
