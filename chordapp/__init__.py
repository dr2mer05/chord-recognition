"""코드 인식 + 화성학 대체 코드 추천 — M1 (MVP).

레이어
  theory/   화성학 엔진. 의존성 0, 순수 파이썬.
  analysis/ 오디오 → 코드 인식 (librosa 필요, 외부 모델은 선택).
  service/  캐시 + DB 조회 + 파이프라인 오케스트레이션.
  api/      FastAPI 서버와 프론트엔드.
"""

__version__ = "0.1.0"

# 프로젝트 루트의 .env를 환경변수로 올린다 (있을 때만).
from .config import ensure_loaded as _ensure_loaded  # noqa: E402

_ensure_loaded()
