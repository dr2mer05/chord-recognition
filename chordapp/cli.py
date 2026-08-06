"""명령줄 인터페이스.

    python -m chordapp search "밤편지"       유튜브에서 곡 찾기
    python -m chordapp youtube <url>        유튜브 링크로 분석
    python -m chordapp analyze <file>       로컬 파일로 분석
    python -m chordapp subs Am --key C      화성학 엔진만 (오디오 불필요)
    python -m chordapp demo                 데모 음원 생성 (정답 라벨 포함)
    python -m chordapp eval                 정확도 벤치마크
    python -m chordapp serve                웹 서버
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from .theory.chord import Chord
from .theory.key import Key
from .theory.substitution import suggest


def _force_utf8() -> None:
    """윈도우 콘솔 기본 코드페이지에서 한글/기호가 깨지는 걸 막는다."""
    if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("utf"):
        return
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass


# --------------------------------------------------------------------------

def cmd_demo(args: argparse.Namespace) -> int:
    from .tools.synth import generate_all

    out = Path(args.out)
    paths = generate_all(out)
    print(f"{len(paths)}개 생성 → {out.resolve()}")
    for p in paths:
        print(f"  {p.name}  (정답: {p.with_suffix('.lab').name})")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from .service import AnalysisCache, ChordService

    service = ChordService(cache=AnalysisCache(Path(args.cache)), vocabulary=args.vocabulary)
    result = service.analyze_file(
        Path(args.file),
        key_override=args.key,
        detect_inversions=not args.no_inversions,
        separate=args.separate,
        max_seconds=args.max_seconds,
        use_cache=not args.no_cache,
    )

    if args.json:
        print(json.dumps(service.full_report(result, limit=args.limit), ensure_ascii=False, indent=2))
        return 0

    origin = {"cache": "캐시 HIT", "database": "코드 DB HIT", "audio": "오디오 분석"}[result.origin]
    print(f"\n{result.source}")
    print(f"  {origin} · {result.duration:.1f}초 · {result.tempo:.0f} BPM · "
          f"키 {result.key.name()} (확신도 {result.key_confidence:.2f})")
    print(f"  인식기: {', '.join(result.recognizers)}")
    for w in result.warnings:
        print(f"  ! {w}")

    print("\n" + "─" * 72)
    _print_chart(result)

    if not args.no_subs:
        print("\n" + "─" * 72)
        print("대체 코드 제안\n")
        _print_substitutions(service, result, args)
    return 0


def _print_substitutions(service, result, args: argparse.Namespace) -> None:
    rows = service.substitutions(result, limit=args.limit, key_override=args.key)
    shown = 0
    seen = set()
    for row in rows:
        if row["chord"] in seen and not args.all:
            continue
        seen.add(row["chord"])
        print(f"  {row['chord']}  ({row['roman']}, {_fn_ko(row['function'])})  {row['start']:.1f}s~")
        for sub in row["substitutions"]:
            marker = {"replace": "→", "insert_before": "↰", "insert_after": "↳"}[sub["kind"]]
            print(f"      {sub['score']:5.1f} {marker} {sub['chord']:<10s} [{sub['label']}]")
            print(f"            {sub['reason']}")
        print()
        shown += 1
        if args.max_chords and shown >= args.max_chords:
            break


def cmd_config(args: argparse.Namespace) -> int:
    """설정이 제대로 잡혔는지 확인. 키 값 자체는 출력하지 않는다."""
    from .config import ENV_FILENAME, PROJECT_ROOT, status
    from .service.youtube import YtDlpProvider

    info = status()
    print(f"\n설정 파일: {info['env_file']}  ({'있음' if info['env_file_exists'] else '없음'})")
    if not info["env_file_exists"]:
        print(f"  → {PROJECT_ROOT / '.env.example'} 를 {ENV_FILENAME} 로 복사해서 값을 채우면 된다")

    key_state = f"설정됨 ({info['youtube_api_key_hint']})" if info["youtube_api_key_set"] else "없음"
    print(f"\n  YOUTUBE_API_KEY       {key_state}")
    print(f"  → 유튜브 검색:        {'사용 가능' if info['youtube_api_key_set'] else '비활성'}")

    provider = YtDlpProvider()
    print(f"\n  CHORDAPP_ENABLE_YTDLP {'1 (켜짐)' if info['ytdlp_enabled'] else '없음 (꺼짐)'}")
    print(f"  yt-dlp 실행파일       {provider.binary or '없음'}")
    print(f"  → 유튜브 오디오 추출: {'사용 가능' if provider.enabled else '비활성'}")

    from .analysis.separation import ENABLE_ENV, default_separator

    sep = default_separator().status()
    print(f"\n  {ENABLE_ENV} {'1 (켜짐)' if os.environ.get(ENABLE_ENV) == '1' else '없음 (꺼짐)'}")
    print(f"  demucs 설치           {'예' if sep['installed'] else '아니오'}")
    print(f"  연산 장치             {sep['device']}")
    print(f"  → 소스 분리:          {'사용 가능' if sep['enabled'] else '비활성'}")

    print("\n  코드 DB / 캐시 / 로컬 파일 / 데모 경로는 설정 없이 항상 동작한다.\n")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from .service import ChordService
    from .service.youtube import YouTubeError

    service = ChordService()
    try:
        videos = service.search_videos(args.query, limit=args.limit)
    except YouTubeError as exc:
        print(f"검색 실패: {exc}")
        return 1
    if not videos:
        print("결과 없음")
        return 0
    for i, video in enumerate(videos, 1):
        length = f"{int(video.duration // 60)}:{int(video.duration % 60):02d}" if video.duration else "?"
        print(f"{i:2d}. {video.title}")
        print(f"    {video.channel} · {length} · {video.url}")
    return 0


def cmd_youtube(args: argparse.Namespace) -> int:
    from .service import AnalysisCache, ChordService
    from .service.youtube import YouTubeError

    service = ChordService(cache=AnalysisCache(Path(args.cache)), vocabulary=args.vocabulary)
    try:
        result, info = service.analyze_youtube(
            args.url, key_override=args.key, vocabulary=args.vocabulary,
            separate=args.separate,
            max_seconds=args.max_seconds, use_cache=not args.no_cache,
        )
    except YouTubeError as exc:
        print(f"실패: {exc}")
        return 1

    origin = {"cache": "캐시 HIT", "database": "코드 DB HIT", "audio": "오디오 분석"}[result.origin]
    print(f"\n{info.title if info else result.source}")
    if info:
        print(f"  {info.channel} · {info.url}")
    print(f"  {origin} · {result.duration:.1f}초 · {result.tempo:.0f} BPM · "
          f"키 {result.key.name()} (확신도 {result.key_confidence:.2f})")
    for w in result.warnings:
        print(f"  ! {w}")
    print("\n" + "─" * 72)
    _print_chart(result)

    if not args.no_subs:
        print("\n" + "─" * 72)
        print("대체 코드 제안\n")
        _print_substitutions(service, result, args)
    return 0


def _fn_ko(function: str) -> str:
    return {"tonic": "토닉", "subdominant": "서브도미넌트",
            "dominant": "도미넌트", "chromatic": "반음계"}.get(function, function)


MARKS = {"high": "", "mid": "·", "low": "?"}


def _print_chart(result, bars_per_line: int = 4, slot: int = 7) -> None:
    """마디 × 박 격자 코드 차트.

    코드만 나열하면 어느 박에 떨어지는지 알 수 없다. 실제 코드보처럼 박마다
    칸을 주고, 앞 박에서 이어지는 칸은 '/'로 채운다.
    """
    bars = result.bars()
    if not bars:
        print("  (구간 없음)")
        return

    for i in range(0, len(bars), bars_per_line):
        row = bars[i : i + bars_per_line]
        width = max(len(bar["slots"]) * slot for bar in row)

        print("  " + "".join(f"{bar['bar']:<{width + 3}d}" for bar in row).rstrip())
        cells = []
        for bar in row:
            parts = []
            for s in bar["slots"]:
                if s["chord"] is None:
                    parts.append(f"{'/' if s['index'] is not None else '':<{slot}s}")
                else:
                    text = s["chord"] + MARKS[s["grade"]]
                    parts.append(f"{text:<{slot}s}")
            cells.append(f"{''.join(parts).rstrip():<{width}s}")
        print("  | " + " | ".join(cells) + " |")
        print()

    counts = {g: sum(1 for s in result.segments if s.grade == g) for g in ("high", "mid", "low")}
    total = max(len(result.segments), 1)
    print(f"  {len(bars)}마디 · {result.beats_per_bar}박자 · 코드 {len(result.segments)}개 · "
          f"확신도 확실 {counts['high']} / 보통(·) {counts['mid']} / 낮음(?) {counts['low']}"
          f"  (평균 {sum(s.certainty for s in result.segments) / total:.2f})")
    print("  '/'는 앞 박에서 이어지는 칸")


def cmd_subs(args: argparse.Namespace) -> int:
    key = Key.parse(args.key)
    chord = Chord.parse(args.chord)
    prev = Chord.parse(args.prev) if args.prev else None
    nxt = Chord.parse(args.next) if args.next else None

    print(f"\n{chord.name(key.prefer_flat)} in {key.name()}  →  {key.roman(chord)} "
          f"({_fn_ko(key.function_of(chord))})")
    if prev or nxt:
        print(f"문맥: {prev.name() if prev else '-'} → [{chord.name()}] → {nxt.name() if nxt else '-'}")
    print("─" * 72)
    for sub in suggest(chord, key, prev=prev, next=nxt, limit=args.limit):
        marker = {"replace": "→", "insert_before": "앞에 삽입", "insert_after": "뒤에 삽입"}[sub.kind]
        print(f"{sub.score:5.1f}  {sub.chord.name(key.prefer_flat):<10s} {marker:<10s} "
              f"[{sub.label}]  {'·'.join(sub.tags)}")
        print(f"       {sub.reason}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .analysis import analyze_audio
    from .evaluation import aggregate, evaluate_result
    from .tools.synth import generate_all

    data_dir = Path(args.dir)
    wavs = sorted(data_dir.glob("*.wav"))
    if not wavs:
        print(f"{data_dir}에 음원이 없어 데모를 생성한다")
        wavs = generate_all(data_dir)

    reports = []
    for wav in wavs:
        lab = wav.with_suffix(".lab")
        if not lab.exists():
            print(f"  건너뜀 (정답 없음): {wav.name}")
            continue
        # 합성 음원에 소스 분리를 걸면 오히려 나빠진다 (Demucs는 실제 녹음으로
        # 학습돼서 합성음에 아티팩트를 만든다 — half_bar_changes 98.0% → 89.8%).
        # 벤치마크는 분리 없이 도는 게 기본이고, --separate로 강제할 수 있다.
        result = analyze_audio(str(wav), vocabulary=args.vocabulary, separate=args.separate)
        report = evaluate_result(result, lab, name=wav.stem)
        reports.append(report)
        print(report.format())
        print()

    if reports:
        agg = aggregate(reports)
        print("─" * 72)
        print("전체 (길이 가중 평균): " + "  ".join(f"{k}={v * 100:5.1f}%" for k, v in agg.items()))
        print("\n주의: 합성 음원 기준이라 실제 녹음보다 후하게 나온다. "
              "실측은 M2에서 McGill Billboard로.")
    return 0


def cmd_beateval(args: argparse.Namespace) -> int:
    """비트·다운비트 추적을 여러 곡으로 평가한다.

    한 곡을 보고 고치면 과적합된다. 템포 62~160, 3/4·4/4·6/8, 화성 리듬이
    빠른 곡과 느린 곡을 고루 넣은 세트에서 재야 변경이 일반화되는지 알 수 있다.
    """
    import numpy as np

    from .analysis.audio import extract_features
    from .evaluation import BEAT_TOLERANCE, beat_f_measure, tempo_error
    from .tools.synth import BEAT_EVAL_TRACKS, beat_grid, render_track, write_wav

    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, track in enumerate(BEAT_EVAL_TRACKS):
        wav = out_dir / f"{track.name}.wav"
        if not wav.exists() or args.regenerate:
            audio, _ = render_track(track, seed=i)
            write_wav(wav, audio)

        beats_true, downs_true = beat_grid(track)
        feats = extract_features(str(wav), beats_per_bar=track.beats_per_bar)
        beats_est = list(feats.beat_times)
        downs_est = [feats.beat_times[j] for j in np.atleast_1d(feats.downbeat_indices)
                     if j < len(feats.beat_times)]

        beat = beat_f_measure(beats_true, beats_est, BEAT_TOLERANCE)
        down = beat_f_measure(downs_true, downs_est, BEAT_TOLERANCE)
        tempo = tempo_error(track.bpm, feats.tempo)
        rows.append((track, beat, down, tempo, feats))

    print(f"\n비트 추적 평가 — {len(rows)}곡 (±{BEAT_TOLERANCE * 1000:.0f}ms 허용)\n")
    print(f"  {'곡':<20}{'정답':>6}{'검출':>8}  {'비트F':>6}{'마디F':>6}  비고")
    print("  " + "─" * 62)
    for track, beat, down, tempo, feats in rows:
        note = []
        if tempo["octave"]:
            note.append(f"템포 {tempo['ratio']:.2f}배")
        elif tempo["percent"] > 3:
            note.append(f"템포 {tempo['percent']:.0f}%")
        if feats.beats_per_bar != track.beats_per_bar:
            note.append(f"{feats.beats_per_bar}박 오인")
        print(f"  {track.name:<20}{track.bpm:>6.0f}{feats.tempo:>8.1f}  "
              f"{beat['f'] * 100:>5.0f}%{down['f'] * 100:>5.0f}%  {' '.join(note)}")

    mean = lambda key, idx: sum(r[idx][key] for r in rows) / len(rows)
    octaves = sum(1 for r in rows if r[3]["octave"])
    meters = sum(1 for r in rows if r[4].beats_per_bar != r[0].beats_per_bar)
    print("  " + "─" * 62)
    print(f"  {'전체':<20}{'':>6}{'':>8}  {mean('f', 1) * 100:>5.0f}%{mean('f', 2) * 100:>5.0f}%")
    print(f"\n  템포 배수 오류 {octaves}/{len(rows)} · 박자 오인 {meters}/{len(rows)}")
    print(f"  추적기: {rows[0][4].beat_source}\n")
    return 0


def cmd_corrections(args: argparse.Namespace) -> int:
    """교정 현황. 검수 완료한 곡이 곧 실제 녹음 기준 정답 데이터다."""
    from .service.corrections import CorrectionStore

    store = CorrectionStore(Path(args.db))
    stats = store.stats()
    print(f"\n곡 {stats['songs']}개 · 검수 완료 {stats['verified']}개 · 교정 {stats['corrections']}건\n")

    rows = store.sources(verified_only=args.verified_only)
    if not rows:
        print("  아직 없다. 웹 UI에서 코드를 클릭해 고치면 여기에 쌓인다.\n")
        return 0
    for row in rows:
        mark = "[검수완료]" if row["verified"] else "[  진행중]"
        print(f"  {mark} {row['label'] or row['source_key']}")
        print(f"            {row['source_key']} · 교정 {row['corrections']}건 · "
              f"{row['duration']:.0f}초")
    if not stats["verified"]:
        print("\n  * 검수 완료 곡이 없어 벤치마크를 돌릴 수 없다.")
        print("    틀린 코드만 고친 표본은 편향돼 있어서 정확도를 계산할 수 없다 —")
        print("    곡 전체를 확인한 뒤 UI에서 '검수 완료'를 체크해야 한다.")
    print()
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """검수 완료한 실제 곡으로 정확도를 측정한다.

    합성 벤치마크는 98.8%로 포화돼 변별력이 없다. 실제 녹음 기준 숫자는
    사용자가 검수한 곡에서만 나온다.
    """
    from .analysis import analyze_audio
    from .evaluation import aggregate, evaluate
    from .service.corrections import CorrectionStore, apply_corrections
    from .service.pipeline import ChordService

    store = CorrectionStore(Path(args.db))
    service = ChordService(corrections=store)
    sources = store.sources(verified_only=True)
    if not sources:
        print("검수 완료한 곡이 없다. `python -m chordapp corrections` 참고.")
        return 1

    reports = []
    for row in sources:
        audio = Path(row["audio_path"])
        if not audio.is_file():
            print(f"  건너뜀 (음원 없음): {row['label']} → {audio}")
            continue

        raw = analyze_audio(str(audio), separate=args.separate)
        truth = apply_corrections(raw.segments, store.corrections(row["source_key"]))
        report = evaluate(
            [(s.start, s.end, None if s.chord.is_no_chord else s.chord) for s in truth],
            [(s.start, s.end, None if s.chord.is_no_chord else s.chord) for s in raw.segments],
            name=row["label"] or row["source_key"],
        )
        reports.append(report)
        print(report.format())
        print()

    if reports:
        agg = aggregate(reports)
        print("─" * 72)
        print("실제 녹음 기준: " + "  ".join(f"{k}={v * 100:5.1f}%" for k, v in agg.items()))
        print(f"({len(reports)}곡 · 사용자 검수 정답 기준)")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn이 필요하다: pip install 'uvicorn[standard]' fastapi python-multipart")
        return 1
    uvicorn.run("chordapp.api.server:app", host=args.host, port=args.port, reload=args.reload)
    return 0


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chordapp", description="코드 인식 + 대체 코드 추천")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("demo", help="정답 라벨이 붙은 데모 음원 생성")
    p.add_argument("--out", default="data/demo")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("analyze", help="음원 분석")
    p.add_argument("file")
    p.add_argument("--key", help="조성 지정 (예: C, Am). 생략하면 자동 추정")
    p.add_argument("--vocabulary", default="standard", choices=["triads", "standard", "extended"])
    p.add_argument("--limit", type=int, default=5, help="코드당 대체안 개수")
    p.add_argument("--max-chords", type=int, default=8, help="대체안을 보여줄 코드 수")
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--all", action="store_true", help="같은 코드가 반복돼도 전부 출력")
    p.add_argument("--no-subs", action="store_true")
    p.add_argument("--no-inversions", action="store_true")
    p.add_argument("--separate", dest="separate", action="store_true", default=None, help="소스 분리(Demucs) 강제 사용")
    p.add_argument("--no-separate", dest="separate", action="store_false", help="소스 분리 강제 해제")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--cache", default="data/cache.sqlite3")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("config", help="설정 상태 확인 (키 값은 출력하지 않음)")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("search", help="유튜브에서 곡 찾기 (YOUTUBE_API_KEY 필요)")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=8)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("youtube", help="유튜브 링크/영상 ID로 분석")
    p.add_argument("url")
    p.add_argument("--key", help="조성 지정 (예: C, Am). 생략하면 자동 추정")
    p.add_argument("--vocabulary", default="standard", choices=["triads", "standard", "extended"])
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--max-chords", type=int, default=8)
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--no-subs", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--separate", dest="separate", action="store_true", default=None,
                   help="소스 분리(Demucs) 강제 사용")
    p.add_argument("--no-separate", dest="separate", action="store_false",
                   help="소스 분리 강제 해제")
    p.add_argument("--cache", default="data/cache.sqlite3")
    p.set_defaults(func=cmd_youtube)

    p = sub.add_parser("subs", help="화성학 엔진 단독 실행")
    p.add_argument("chord")
    p.add_argument("--key", default="C")
    p.add_argument("--prev")
    p.add_argument("--next")
    p.add_argument("--limit", type=int, default=8)
    p.set_defaults(func=cmd_subs)

    p = sub.add_parser("eval", help="정확도 벤치마크")
    p.add_argument("--dir", default="data/demo")
    p.add_argument("--vocabulary", default="standard")
    p.add_argument("--separate", dest="separate", action="store_true", default=False,
                   help="소스 분리를 켜고 측정 (합성 음원에서는 오히려 나빠진다)")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("beateval", help="비트·다운비트 추적 평가 (여러 곡)")
    p.add_argument("--dir", default="data/beateval")
    p.add_argument("--regenerate", action="store_true", help="음원을 다시 생성")
    p.set_defaults(func=cmd_beateval)

    p = sub.add_parser("corrections", help="사용자 교정 현황")
    p.add_argument("--db", default="data/corrections.sqlite3")
    p.add_argument("--verified-only", action="store_true")
    p.set_defaults(func=cmd_corrections)

    p = sub.add_parser("benchmark", help="검수 완료한 실제 곡으로 정확도 측정")
    p.add_argument("--db", default="data/corrections.sqlite3")
    p.add_argument("--separate", dest="separate", action="store_true", default=None)
    p.add_argument("--no-separate", dest="separate", action="store_false")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("serve", help="웹 서버 실행")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    _force_utf8()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
