"""정답 박자 입력 여부와 실행 환경을 기록하는 합성 비트 평가. 인식 엔진과 독립적이다."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import time

import numpy as np

from .analysis import audio
from .analysis import beat_tracker
from .analysis.constants import ANALYSIS_VERSION
from . import evaluation
from .evaluation import BEAT_TOLERANCE, bar_interval_f_measure, beat_f_measure, tempo_error
from .tools import synth

SCHEMA_VERSION = 1


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    packages = {}
    for name in ('numpy', 'librosa', 'scipy', 'beat-this', 'torch'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'packages': packages,
            'analysis_version': ANALYSIS_VERSION,
            'generator_sha256': fingerprint(synth.__file__),
            'audio_frontend_sha256': fingerprint(audio.__file__),
            'beat_tracker_sha256': fingerprint(beat_tracker.__file__),
            'metrics_sha256': fingerprint(evaluation.__file__),
            'evaluation_protocol_sha256': fingerprint(__file__)}


def fixtures(directory, tracks, regenerate):
    """기존 음원을 정답과 무조건 연결하지 않고 생성 조건과 내용 해시를 확인한다."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / '.beat-fixtures.json'
    if manifest_path.exists() and not regenerate:
        try:
            manifest = json.loads(manifest_path.read_text())
            if not isinstance(manifest, dict):
                raise ValueError('invalid manifest')
        except (OSError, ValueError):
            raise ValueError('평가 음원의 생성 기록을 읽을 수 없습니다. --regenerate로 재생성해 주세요.') from None
    else:
        manifest = {}
    generator = fingerprint(synth.__file__)
    records = {}
    for seed, track in enumerate(tracks):
        path = directory / (track.name + '.wav')
        expected = {'track': json.loads(json.dumps(asdict(track))), 'seed': seed, 'sample_rate': synth.SR,
                    'generator_sha256': generator}
        record = manifest.get(track.name)
        if not path.exists() or regenerate:
            signal, _ = synth.render_track(track, seed=seed)
            synth.write_wav(path, signal)
        elif (not isinstance(record, dict) or any(record.get(k) != v for k, v in expected.items())
              or record.get('audio_sha256') != fingerprint(path)):
            raise ValueError(f'{track.name}: 음원과 정답의 생성 조건을 확인할 수 없습니다. --regenerate로 재생성해 주세요.')
        records[track.name] = {**expected, 'audio_sha256': fingerprint(path)}
    if records != manifest:
        manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    return records


def evaluate_tracks(directory, *, meter_mode='auto', tracker='auto', regenerate=False, tracks=None):
    if meter_mode not in {'auto', 'reference'} or tracker not in {'auto', 'librosa'}:
        raise ValueError('지원하지 않는 평가 설정입니다.')
    tracks = list(synth.BEAT_EVAL_TRACKS if tracks is None else tracks)
    if not tracks:
        raise ValueError('평가할 합성 음원이 없습니다.')
    directory = Path(directory)
    records = fixtures(directory, tracks, regenerate)
    rows = []
    for track in tracks:
        kwargs = {'beats_per_bar': track.beats_per_bar if meter_mode == 'reference' else None}
        if tracker == 'librosa':
            kwargs['tracker'] = None
        started = time.perf_counter()
        feats = audio.extract_features(str(directory / (track.name + '.wav')), **kwargs)
        elapsed = time.perf_counter() - started
        reference_beats, reference_downbeats = synth.beat_grid(track)
        estimated_beats = np.asarray(feats.beat_times, dtype=float)
        indices = np.asarray(feats.downbeat_indices, dtype=int).reshape(-1)
        estimated_downbeats = estimated_beats[indices[(indices >= 0) & (indices < len(estimated_beats))]]
        duration = len(reference_beats) * 60.0 / track.bpm
        # 마디 구간 평가는 음악 끝 경계도 포함한다. 비트·다운비트 F는 기존 정의를 유지한다.
        reference_bar_boundaries = [*reference_downbeats, duration]
        estimated_bar_boundaries = estimated_downbeats[
            (estimated_downbeats >= 0) & (estimated_downbeats <= duration + BEAT_TOLERANCE)]
        rows.append({
            'name': track.name, 'audio_sha256': records[track.name]['audio_sha256'],
            'reference': {'bpm': track.bpm, 'beats_per_bar': track.beats_per_bar,
                          'duration_seconds': duration, 'pulse_unit': 'synthetic_generator_pulse'},
            'estimate': {'bpm': float(feats.tempo), 'beats_per_bar': int(feats.beats_per_bar),
                         'beat_source': feats.beat_source, 'beats_seconds': estimated_beats.tolist(),
                         'downbeats_seconds': estimated_downbeats.tolist()},
            'metrics': {
                'tempo': tempo_error(track.bpm, float(feats.tempo)),
                'beats': beat_f_measure(reference_beats, estimated_beats),
                'downbeats': beat_f_measure(reference_downbeats, estimated_downbeats),
                'bars': bar_interval_f_measure(reference_bar_boundaries, estimated_bar_boundaries),
                'beats_per_bar_match': bool(feats.beats_per_bar == track.beats_per_bar),
            },
            'feature_extraction_seconds': elapsed,
        })
    mean = lambda key: sum(row['metrics'][key]['f'] for row in rows) / len(rows)
    return {
        'schema_version': SCHEMA_VERSION, 'dataset_kind': 'synthetic',
        'created_at': datetime.now(timezone.utc).isoformat(), 'environment': environment(),
        'settings': {'meter_mode': meter_mode, 'reference_meter_supplied': meter_mode == 'reference',
                     'tracker_request': tracker, 'tolerance_seconds': BEAT_TOLERANCE,
                     'event_scoring': 'all_estimates_against_generator_beats; audio_has_1s_release_tail',
                     'bar_scoring': 'consecutive_start_and_end_boundaries_within_annotated_music'},
        'summary': {'tracks': len(rows), 'beat_f_macro': mean('beats'),
                    'downbeat_f_macro': mean('downbeats'), 'bar_f_macro': mean('bars'),
                    'tempo_multiple_errors': sum(row['metrics']['tempo']['octave'] for row in rows),
                    'beats_per_bar_accuracy': (sum(row['metrics']['beats_per_bar_match'] for row in rows) / len(rows)
                                              if meter_mode == 'auto' else None)},
        'tracks': rows,
        'limitations': [
            '합성 평가이며 실제 음악 인식 성능의 증거가 아님',
            'auto는 추적기 및 기본값을 포함한 파이프라인 평가; librosa의 기본값 4도 포함',
            '6/8 이름의 표본도 합성기 6펄스 기준; 실제 겹박자 해석 능력과 구분',
            '변박·템포 변화·못갖춘마디·다양한 악기 및 실제 녹음 조건은 아직 평가하지 않음',
        ],
    }


def format_report(report):
    mode = '자동 (정답 박자 미입력)' if report['settings']['meter_mode'] == 'auto' else '정답 박자 입력 (자동 박자 판정 평가 아님)'
    lines = [f"비트 추적 평가 — {report['summary']['tracks']}곡 · {mode}",
             '합성 음원 · ±70ms · F값은 곡별 평균',
             f"{'곡':<22} {'BPM':>7} {'박/마디':>7} {'비트F':>7} {'다운비트F':>8} {'마디구간F':>8}"]
    for row in report['tracks']:
        ref, est, metrics = row['reference'], row['estimate'], row['metrics']
        lines.append(f"{row['name']:<22} {est['bpm']:7.1f} {est['beats_per_bar']:>2}/{ref['beats_per_bar']:<2}  "
                     + ' '.join(f"{metrics[key]['f'] * 100:7.1f}%" for key in ('beats', 'downbeats', 'bars')))
    summary = report['summary']
    lines.append(f"전체: 비트F {summary['beat_f_macro']*100:.1f}% · 다운비트F {summary['downbeat_f_macro']*100:.1f}% · 마디구간F {summary['bar_f_macro']*100:.1f}%")
    lines.append(f"템포 배수 오류 {summary['tempo_multiple_errors']}/{summary['tracks']}")
    if summary['beats_per_bar_accuracy'] is not None:
        lines.append(f"마디당 펄스 수 일치 (기본값 포함): {summary['beats_per_bar_accuracy']*100:.1f}%")
    lines.append('실행 추적기: ' + ', '.join(sorted({row['estimate']['beat_source'] for row in report['tracks']})))
    lines.extend('주의: ' + item for item in report['limitations'])
    return '\n'.join(lines)
