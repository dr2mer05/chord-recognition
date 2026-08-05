"""FastAPI 서버.

M1은 동기 처리다. 30초 음원 분석이 CPU에서 2초 남짓이라 아직 큐가 필요 없고,
큐(Celery/RQ)는 M3에서 유튜브 연동으로 입력 길이가 길어질 때 붙인다.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..models import AnalysisResult
from ..service import AnalysisCache, ChordService, LocalJsonDatabase, NullDatabase
from ..service.youtube import (
    AudioFetchDisabled,
    DisabledAudioProvider,
    InvalidVideoUrl,
    SearchUnavailable,
    YouTubeError,
    parse_video_id,
)
from ..theory.chord import Chord
from ..theory.key import Key
from ..theory.substitution import suggest

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "chord_db.json"

ALLOWED_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
MAX_UPLOAD_BYTES = 40 * 1024 * 1024

app = FastAPI(title="코드 인식 & 대체 코드 추천", version="0.1.0")


def _build_service() -> ChordService:
    database = LocalJsonDatabase(DB_PATH) if DB_PATH.exists() else NullDatabase()
    return ChordService(cache=AnalysisCache(DATA_DIR / "cache.sqlite3"), database=database)


service = _build_service()


# --------------------------------------------------------------------------
# 스키마
# --------------------------------------------------------------------------

class YouTubeRequest(BaseModel):
    url: str
    key: Optional[str] = None
    vocabulary: str = "standard"
    limit: int = 6
    max_seconds: Optional[float] = None


class SubstitutionRequest(BaseModel):
    chord: str
    key: str = "C"
    prev: Optional[str] = None
    next: Optional[str] = None
    limit: int = 8


class ProgressionRequest(BaseModel):
    chords: List[str]
    key: Optional[str] = None
    limit: int = 6


# --------------------------------------------------------------------------
# 라우트
# --------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    from ..analysis.recognizers import default_recognizers

    return {
        "status": "ok",
        "recognizers": [r.name for r in default_recognizers()],
        "database": getattr(service.database, "name", "none"),
        "cache": service.cache.stats(),
        **_capabilities(),
    }


def _capabilities() -> dict:
    """프론트엔드가 어떤 입력 경로가 살아 있는지 알아야 안내 문구를 낼 수 있다."""
    from ..config import ENV_FILENAME, PROJECT_ROOT

    provider = service.audio_provider
    return {
        "youtube_search": bool(getattr(service.search_backend, "available", False)),
        "youtube_audio": bool(getattr(provider, "enabled", False)),
        "youtube_audio_reason": "" if getattr(provider, "enabled", False)
        else getattr(provider, "reason", DisabledAudioProvider.reason),
        "env_file": str(PROJECT_ROOT / ENV_FILENAME),
    }


@app.get("/api/capabilities")
def capabilities() -> dict:
    return _capabilities()


@app.get("/api/search")
def search(q: str = Query(..., min_length=1), limit: int = 8) -> JSONResponse:
    """곡 제목/아티스트로 유튜브 영상 후보를 찾는다."""
    try:
        videos = service.search_videos(q, limit=limit)
    except SearchUnavailable as exc:
        raise HTTPException(503, str(exc))
    except YouTubeError as exc:
        raise HTTPException(502, str(exc))
    return JSONResponse({"query": q, "results": [v.to_dict() for v in videos]})


@app.post("/api/youtube")
def analyze_youtube(req: YouTubeRequest) -> JSONResponse:
    """유튜브 링크(또는 영상 ID)로 분석. [A] 캐시 → [B] 코드 DB → [C] 오디오."""
    try:
        if req.key:
            Key.parse(req.key)
        result, info = service.analyze_youtube(
            req.url, key_override=req.key, vocabulary=req.vocabulary,
            max_seconds=req.max_seconds,
        )
    except InvalidVideoUrl as exc:
        raise HTTPException(400, str(exc))
    except AudioFetchDisabled as exc:
        raise HTTPException(501, str(exc))
    except YouTubeError as exc:
        raise HTTPException(502, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    report = service.full_report(result, limit=req.limit, key_override=req.key)
    report["video"] = info.to_dict() if info else {"video_id": parse_video_id(req.url)}
    return JSONResponse(report)


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    key: Optional[str] = Form(None),
    vocabulary: str = Form("standard"),
    limit: int = Form(6),
    max_seconds: Optional[float] = Form(None),
) -> JSONResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"지원하지 않는 형식: {suffix or '(확장자 없음)'}")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_DIR, suffix=suffix) as tmp:
        size = 0
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(413, "파일이 너무 크다 (최대 40MB)")
            tmp.write(chunk)
        temp_path = Path(tmp.name)

    try:
        if key:
            Key.parse(key)  # 잘못된 조성 문자열을 미리 걸러낸다
        result = service.analyze_file(temp_path, key_override=key, vocabulary=vocabulary,
                                      max_seconds=max_seconds)
        result.source = file.filename or temp_path.name
        # 프론트엔드에서 재생하려면 파일이 남아 있어야 한다
        stored = UPLOAD_DIR / f"{temp_path.stem}{suffix}"
        shutil.move(str(temp_path), stored)
        report = service.full_report(result, limit=limit, key_override=key)
        report["audio_url"] = f"/api/audio/{stored.name}"
        return JSONResponse(report)
    except ValueError as exc:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(400, str(exc))
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(500, f"분석 실패: {exc}")


@app.get("/api/audio/{name}")
def audio(name: str) -> FileResponse:
    path = (UPLOAD_DIR / name).resolve()
    if not path.is_file() or UPLOAD_DIR.resolve() not in path.parents:
        raise HTTPException(404, "없는 파일")
    return FileResponse(path)


DEMO_DIR = DATA_DIR / "demo"


def _ensure_demo() -> List[Path]:
    """데모 음원이 없으면 그 자리에서 만든다. 합성이라 몇 초면 된다."""
    wavs = sorted(DEMO_DIR.glob("*.wav"))
    if wavs:
        return wavs
    from ..tools.synth import generate_all

    return sorted(generate_all(DEMO_DIR))


@app.get("/api/demo")
def demo_list() -> JSONResponse:
    """자격증명 없이도 파이프라인 전체를 볼 수 있는 경로.

    정답 라벨(.lab)이 같이 있는 합성 음원이라 인식 결과를 정답과 대조할 수 있다.
    """
    tracks = []
    for wav in _ensure_demo():
        lab = wav.with_suffix(".lab")
        truth = []
        if lab.exists():
            from ..evaluation import load_lab

            truth = [c.name() for _, _, c in load_lab(lab) if c is not None]
        tracks.append({
            "name": wav.stem,
            "truth": truth[:8],
            "audio_url": f"/api/demo/audio/{wav.stem}",
        })
    return JSONResponse({"tracks": tracks})


@app.post("/api/demo/{name}")
def demo_analyze(name: str, limit: int = 6, key: Optional[str] = None) -> JSONResponse:
    wav = (DEMO_DIR / f"{name}.wav").resolve()
    if not wav.is_file() or DEMO_DIR.resolve() not in wav.parents:
        raise HTTPException(404, f"그런 데모가 없다: {name}")

    result = service.analyze_file(wav, key_override=key)
    result.source = f"{name}.wav (합성 데모)"
    report = service.full_report(result, limit=limit, key_override=key)
    report["audio_url"] = f"/api/demo/audio/{name}"

    lab = wav.with_suffix(".lab")
    if lab.exists():
        from ..evaluation import evaluate_result

        scores = evaluate_result(result, lab, name=name).scores
        report["accuracy"] = {k: round(v * 100, 1) for k, v in scores.items()}
    return JSONResponse(report)


@app.get("/api/demo/audio/{name}")
def demo_audio(name: str) -> FileResponse:
    path = (DEMO_DIR / f"{name}.wav").resolve()
    if not path.is_file() or DEMO_DIR.resolve() not in path.parents:
        raise HTTPException(404, "없는 파일")
    return FileResponse(path)


class CorrectionRequest(BaseModel):
    source_key: str
    start: float
    end: float
    chord: str
    original: str = ""


class VerifyRequest(BaseModel):
    source_key: str
    verified: bool = True


@app.post("/api/corrections")
def add_correction(req: CorrectionRequest) -> JSONResponse:
    """틀린 코드 하나를 고친다. 원본 인식 결과는 캐시에 그대로 남는다."""
    try:
        fix = service.corrections.add(
            req.source_key, req.start, req.end, req.chord, req.original
        )
    except ValueError as exc:
        raise HTTPException(400, f"코드 표기를 해석할 수 없다: {exc}")
    return JSONResponse({"ok": True, "correction": fix.to_dict()})


@app.delete("/api/corrections")
def remove_correction(source_key: str = Query(...), start: float = Query(...)) -> JSONResponse:
    removed = service.corrections.remove(source_key, start)
    return JSONResponse({"ok": True, "removed": removed})


@app.post("/api/corrections/verify")
def verify_source(req: VerifyRequest) -> JSONResponse:
    """곡 전체를 확인했다고 표시. 이 곡만 벤치마크에 쓰인다.

    틀린 코드만 골라 고친 표본은 편향돼 있어서 정확도를 계산할 수 없다.
    """
    if service.corrections.source(req.source_key) is None:
        raise HTTPException(404, "등록되지 않은 곡")
    service.corrections.set_verified(req.source_key, req.verified)
    return JSONResponse({"ok": True, "verified": req.verified})


class TempoRequest(BaseModel):
    source_key: str
    bpm: Optional[float] = None       # None이면 자동 추정으로 되돌린다
    beat_offset: Optional[float] = None   # 첫 박의 시각(초). 탭 템포가 채운다


@app.post("/api/corrections/tempo")
def set_tempo(req: TempoRequest) -> JSONResponse:
    """곡별 템포 지정. 자동 추정이 배수를 틀릴 때의 확실한 탈출구.

    69 BPM 발라드를 136으로 잡는 옥타브 오류가 대표적인데, 사용자는 정답을 안다.
    """
    if service.corrections.source(req.source_key) is None:
        raise HTTPException(404, "등록되지 않은 곡")
    try:
        service.corrections.set_tempo(req.source_key, req.bpm, req.beat_offset)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return JSONResponse({"ok": True, "tempo": req.bpm, "beat_offset": req.beat_offset})


class MeterRequest(BaseModel):
    source_key: str
    beats_per_bar: Optional[int] = None    # None이면 기본값(4)으로
    downbeat_offset: Optional[int] = None  # 마디 시작을 몇 박 밀지


@app.post("/api/corrections/meter")
def set_meter(req: MeterRequest) -> JSONResponse:
    """곡별 박자표와 마디 시작 위치.

    박자표 자동 판별은 넣지 않았다 — 저역 온셋의 위상 대비로 구별해 보려 했는데
    정답이 4/4인 곡에서도 6박이 더 높게 나와 신뢰할 수 없었다.
    """
    if service.corrections.source(req.source_key) is None:
        raise HTTPException(404, "등록되지 않은 곡")
    try:
        service.corrections.set_meter(req.source_key, req.beats_per_bar, req.downbeat_offset)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return JSONResponse({"ok": True})


@app.get("/api/corrections/sources")
def correction_sources(verified_only: bool = False) -> JSONResponse:
    return JSONResponse({
        "sources": service.corrections.sources(verified_only=verified_only),
        "stats": service.corrections.stats(),
    })


@app.get("/api/lookup")
def lookup(artist: str = Query(...), title: str = Query(...), limit: int = 6) -> JSONResponse:
    """[B] 공개 코드 DB 조회. HIT면 오디오 분석 없이 즉시 반환."""
    result: Optional[AnalysisResult] = service.lookup(artist, title)
    if result is None:
        raise HTTPException(404, "코드 DB에 없음 — 음원을 업로드해 분석하면 된다")
    return JSONResponse(service.full_report(result, limit=limit))


@app.post("/api/substitutions")
def substitutions(req: SubstitutionRequest) -> JSONResponse:
    """오디오 없이 화성학 엔진만 쓰는 엔드포인트."""
    try:
        key = Key.parse(req.key)
        chord = Chord.parse(req.chord)
        prev = Chord.parse(req.prev) if req.prev else None
        nxt = Chord.parse(req.next) if req.next else None
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    subs = suggest(chord, key, prev=prev, next=nxt, limit=req.limit)
    return JSONResponse({
        "chord": chord.name(key.prefer_flat),
        "key": key.name(),
        "roman": key.roman(chord),
        "function": key.function_of(chord),
        "substitutions": [s.to_dict(key.prefer_flat) for s in subs],
    })


@app.post("/api/progression")
def progression(req: ProgressionRequest) -> JSONResponse:
    """코드 진행을 직접 입력해 대체안을 받는다. 조성 미지정 시 진행에서 추정."""
    try:
        chords = [Chord.parse(c) for c in req.chords]
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    if req.key:
        key = Key.parse(req.key)
    else:
        from ..theory.key import key_from_chords

        key = key_from_chords(chords)

    rows = []
    for i, chord in enumerate(chords):
        prev = chords[i - 1] if i > 0 else None
        nxt = chords[i + 1] if i + 1 < len(chords) else None
        subs = suggest(chord, key, prev=prev, next=nxt, limit=req.limit)
        rows.append({
            "index": i,
            "chord": chord.name(key.prefer_flat),
            "roman": key.roman(chord),
            "function": key.function_of(chord),
            "substitutions": [s.to_dict(key.prefer_flat) for s in subs],
        })
    return JSONResponse({"key": key.name(), "analysis": rows})


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def index() -> FileResponse:
    """프론트엔드. **캐시를 끈다.**

    StaticFiles는 기본적으로 캐시를 허용하는데, 그러면 UI를 고쳐도 사용자
    화면에는 예전 버전이 계속 뜬다. 실제로 코드 분리·마디 격자·재생 동기화를
    다 고친 뒤에도 사용자에게는 옛 화면이 보여서 "여전히 안 된다"는 보고가 나왔다.
    """
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"},
    )


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
