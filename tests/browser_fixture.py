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
        uvicorn.run(app, host="127.0.0.1", port=8765)
