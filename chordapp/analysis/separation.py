"""소스 분리 (Demucs).

파이프라인 1-3의 6번 항목이자 2단계. 원래 "1단계에서는 보류"였지만,
실제 상용 녹음에서 인식기 확신도가 0.43(합성 0.83)까지 떨어지는 게 측정되면서
우선순위가 올라갔다. 보컬·드럼이 화성 악기 스펙트럼을 오염시키는 게 주원인이고,
그걸 직접 걷어내는 유일한 수단이다.

파이프라인 문서의 설계를 그대로 따른다:
  - harmonic 스템(`other`)만 코드 인식에 쓴다
  - `bass` 스템으로 슬래시 코드·전위를 판별한다 (저역 CQT 근사보다 훨씬 정확)
  - 비트 추적은 드럼이 있는 원본 믹스로 한다 (드럼을 빼면 오히려 나빠진다)

**배포 주의**: 분리된 스템은 원곡 마스터의 파생물이다. 개인 이용 전용이고
서비스에서 다운로드·재배포 기능을 붙이면 안 된다(파이프라인 0장 경계선).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_MODEL = "htdemucs"
STEM_NAMES = ("drums", "bass", "other", "vocals")
DEFAULT_STEM_DIR = Path("data/stems")

# 분리를 켤지. GPU가 없으면 CPU에서 실시간의 몇 배가 걸리므로 기본은 끔.
ENABLE_ENV = "CHORDAPP_ENABLE_DEMUCS"


class SeparationError(Exception):
    pass


class SeparationUnavailable(SeparationError):
    pass


@dataclass
class Stems:
    """분리 결과. 스템이 없으면 해당 항목이 None이고 호출부는 원본으로 폴백한다."""

    mix: Path
    paths: Dict[str, Path] = field(default_factory=dict)
    model: str = DEFAULT_MODEL
    device: str = "cpu"
    cached: bool = False

    def get(self, name: str) -> Optional[Path]:
        path = self.paths.get(name)
        return path if path and path.is_file() else None

    @property
    def harmonic(self) -> Optional[Path]:
        """화성 악기(기타·건반·스트링) 스템."""
        return self.get("other")

    @property
    def accompaniment(self) -> List[Path]:
        """코드 인식의 주 입력 = 믹스에서 보컬·드럼만 뺀 것 (other + bass).

        `other`만 쓰면 근음이 빠져 오히려 나빠진다. 코드 판정에는 베이스가
        필요하다 — 무엇이 근음인지가 곧 코드 이름이기 때문.
        """
        return [p for p in (self.get("other"), self.get("bass")) if p is not None]

    @property
    def bass(self) -> Optional[Path]:
        return self.get("bass")

    def to_dict(self) -> Dict[str, object]:
        return {
            "model": self.model,
            "device": self.device,
            "cached": self.cached,
            "stems": sorted(self.paths),
        }


class DemucsSeparator:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        out_dir: Path = DEFAULT_STEM_DIR,
        device: Optional[str] = None,
        timeout: float = 1800.0,
    ):
        self.model = model
        self.out_dir = Path(out_dir)
        self._device = device
        self.timeout = timeout

    # -- 가용성 -------------------------------------------------------
    @property
    def installed(self) -> bool:
        try:
            import demucs  # noqa: F401
            import torch  # noqa: F401
        except Exception:
            return False
        return True

    @property
    def enabled(self) -> bool:
        return os.environ.get(ENABLE_ENV, "") == "1" and self.installed

    @property
    def device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def status(self) -> Dict[str, object]:
        return {
            "installed": self.installed,
            "enabled": self.enabled,
            "device": self.device if self.installed else "n/a",
            "model": self.model,
        }

    # -- 실행 ---------------------------------------------------------
    def _stem_dir(self, audio_path: Path) -> Path:
        return self.out_dir / self.model / audio_path.stem

    def _existing(self, audio_path: Path) -> Dict[str, Path]:
        target = self._stem_dir(audio_path)
        found: Dict[str, Path] = {}
        for name in STEM_NAMES:
            candidate = target / f"{name}.wav"
            if candidate.is_file():
                found[name] = candidate
        return found

    def separate(self, audio_path: Path) -> Stems:
        audio_path = Path(audio_path)
        if not self.installed:
            raise SeparationUnavailable(
                "demucs가 설치돼 있지 않다: pip install demucs "
                "(torch가 함께 필요하며, GPU를 쓰려면 CUDA 빌드를 설치해야 한다)"
            )

        # 같은 곡을 두 번 분리할 이유가 없다. 분리가 파이프라인에서 제일 비싸다.
        cached = self._existing(audio_path)
        if len(cached) == len(STEM_NAMES):
            return Stems(mix=audio_path, paths=cached, model=self.model,
                         device=self.device, cached=True)

        self.out_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-m", "demucs.separate",
            "-n", self.model,
            "--device", self.device,
            "-o", str(self.out_dir),
            str(audio_path),
        ]
        try:
            # encoding을 명시하지 않으면 윈도우에서 로케일 코덱(cp949)으로 읽다가
            # demucs의 UTF-8 진행 표시에서 UnicodeDecodeError로 리더 스레드가 죽는다.
            proc = subprocess.run(
                command, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise SeparationError(f"소스 분리 시간 초과 ({self.timeout:.0f}초)") from exc
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-400:]
            raise SeparationError(f"demucs 실패: {tail}")

        found = self._existing(audio_path)
        if not found:
            raise SeparationError(f"스템을 찾을 수 없다: {self._stem_dir(audio_path)}")
        return Stems(mix=audio_path, paths=found, model=self.model, device=self.device)


def default_separator() -> DemucsSeparator:
    return DemucsSeparator()
