"""오디오 분석 레이어 (librosa 필요).

theory 레이어와 달리 여기는 무거운 의존성을 쓰므로,
화성학 엔진만 쓰고 싶을 때는 이 패키지를 import 하지 않으면 된다.
"""

from .pipeline import analyze_audio

__all__ = ["analyze_audio"]
