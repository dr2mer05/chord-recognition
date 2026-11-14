"""캐시 재현성을 위한 실제 설치 환경과 선택된 엔진의 식별 정보."""

from importlib.metadata import PackageNotFoundError, version
import platform


def runtime_identity(vocabulary="standard", separate=False, tempo_override=None):
    from .beat_tracker import default_tracker
    from .recognizers import default_recognizers

    tracker = default_tracker() if not tempo_override else None
    packages = {}
    for name in ("numpy", "librosa", "scipy", "numba", "soundfile", "beat_this", "torch",
                 "torchaudio", "demucs", "autochord", "chord-extractor"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return {
        "python": platform.python_version(), "packages": packages,
        "beat_tracker": tracker.name if tracker else "librosa",
        "beat_model": "File2Beats:default" if tracker else None,
        "device": tracker.device if tracker else "cpu",
        "separated": separate,
        "separation_model": "htdemucs" if separate else None,
        "recognizers": [r.name for r in default_recognizers(vocabulary, separated=separate)],
    }


def cacheable_execution(identity, execution):
    """폴백·부분 실패 결과를 선호 엔진의 정상 결과로 캐시하지 않는다."""
    return (identity["beat_tracker"] == execution.get("beat_tracker")
            and identity["separated"] == execution.get("separated")
            and sorted(identity["recognizers"]) == sorted(execution.get("recognizers", [])))
