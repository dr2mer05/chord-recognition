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
                    with tempfile.TemporaryDirectory(dir=".") as folder:
                        path = Path(folder) / "audio.wav"
                        shutil.copyfile(demo, path)
                        yield path

            service.private_youtube = FixturePrivateProvider()
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
