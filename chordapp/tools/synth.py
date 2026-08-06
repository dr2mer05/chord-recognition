"""정답을 아는 테스트 오디오 생성기.

M1 단계에서 파이프라인 정확도를 실측하려면 정답 라벨이 붙은 오디오가 필요한데,
상용 음원은 저작권 때문에 리포지토리에 못 넣는다. 그래서 합성 음원으로 대신한다.
- 실제 녹음보다 쉬운 신호라 여기 정확도는 상한선으로 읽어야 한다.
- 진짜 벤치마크는 M2에서 McGill Billboard / Isophonics의 .lab으로 돌린다.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from ..theory.chord import QUALITY_INTERVALS, Chord

SR = 22050


@dataclass
class DemoTrack:
    name: str
    key: str
    bpm: float
    beats_per_bar: int
    chords: Sequence[str]
    repeats: int = 2
    chord_beats: int = 0       # 코드 하나가 지속되는 박 수 (0이면 한 마디)

    @property
    def beats_per_chord(self) -> int:
        return self.chord_beats or self.beats_per_bar


def beat_grid(track: DemoTrack) -> "tuple[List[float], List[float]]":
    """이 트랙의 정답 비트·다운비트 시각.

    합성기가 만든 것이므로 오차가 없다. 비트 추적 알고리즘을 평가할 때
    이게 정답 라벨이 된다.
    """
    beat_dur = 60.0 / track.bpm
    total = track.beats_per_chord * len(track.chords) * track.repeats
    beats = [i * beat_dur for i in range(total)]
    downbeats = beats[:: track.beats_per_bar]
    return beats, downbeats


# 비트 추적 평가용 세트. 한 곡에 맞춰 튜닝하는 걸 막으려면 템포·박자·화성 리듬이
# 고루 퍼진 여러 곡에서 재야 한다. 실제 대중음악의 범위를 덮도록 구성했다.
BEAT_EVAL_TRACKS: List[DemoTrack] = [
    # 느린 발라드 (배음으로 끌려가기 쉬운 구간)
    DemoTrack("bpm062_4_4", "C", 62, 4, ["C", "Am", "F", "G"], repeats=4),
    DemoTrack("bpm070_4_4", "G", 70, 4, ["G", "Em", "C", "D"], repeats=4),
    DemoTrack("bpm078_4_4", "F", 78, 4, ["Fmaj7", "Dm7", "Gm7", "C7"], repeats=4),
    # 중간 템포
    DemoTrack("bpm092_4_4", "D", 92, 4, ["D", "Bm", "G", "A"], repeats=4),
    DemoTrack("bpm104_4_4", "A", 104, 4, ["A", "F#m", "D", "E"], repeats=4),
    DemoTrack("bpm118_4_4", "Eb", 118, 4, ["Eb", "Cm", "Ab", "Bb"], repeats=4),
    # 빠른 곡 (절반으로 잡히기 쉬운 구간)
    DemoTrack("bpm132_4_4", "Bb", 132, 4, ["Bb", "Gm", "Eb", "F"], repeats=5),
    DemoTrack("bpm146_4_4", "E", 146, 4, ["E", "C#m", "A", "B"], repeats=5),
    DemoTrack("bpm160_4_4", "C", 160, 4, ["C", "G", "Am", "F"], repeats=6),
    # 3박자 (왈츠)
    DemoTrack("bpm096_3_4", "C", 96, 3, ["C", "F", "G"], repeats=6),
    DemoTrack("bpm138_3_4", "Am", 138, 3, ["Am", "Dm", "E"], repeats=6),
    # 6박자
    DemoTrack("bpm084_6_8", "D", 84, 6, ["D", "G", "A", "D"], repeats=4),
    # 화성 리듬이 빠른 경우 (반 마디마다 전환)
    DemoTrack("fast_harmony_88", "C", 88, 4, ["C", "Am", "F", "G", "Em", "Am", "Dm", "G"],
              repeats=3, chord_beats=2),
    DemoTrack("fast_harmony_124", "G", 124, 4, ["G", "D", "Em", "C", "G", "Bm", "Am", "D"],
              repeats=3, chord_beats=2),
    # 코드가 두 마디씩 지속 (화성 리듬이 느린 경우)
    DemoTrack("slow_harmony_76", "F", 76, 4, ["F", "Dm", "Bb", "C"], repeats=3, chord_beats=8),
    DemoTrack("slow_harmony_128", "A", 128, 4, ["A", "E", "F#m", "D"], repeats=4, chord_beats=8),
]


DEMO_TRACKS: List[DemoTrack] = [
    DemoTrack("pop_I_V_vi_IV", "C", 96, 4, ["C", "G", "Am", "F"], repeats=3),
    DemoTrack("ballad_sevenths", "F", 76, 4, ["Fmaj7", "Dm7", "Gm7", "C7"], repeats=3),
    DemoTrack("minor_loop", "Am", 112, 4, ["Am", "F", "C", "G"], repeats=3),
    DemoTrack("eight_chord_loop", "G", 104, 4, ["G", "Em", "C", "D", "G", "Bm", "Am", "D"], repeats=2),
    # 반 마디마다 바뀌는 빠른 화성 리듬 — 비트 단위 해상도 스트레스 테스트
    DemoTrack("half_bar_changes", "C", 88, 4, ["C", "Am", "F", "G", "Em", "Am", "Dm", "G"],
              repeats=3, chord_beats=2),
]


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _tone(freq: float, dur: float, sr: int, amp: float, harmonics: int = 6) -> np.ndarray:
    """감쇠하는 배음 합성음. 피아노/일렉피아노 비슷한 스펙트럼."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    out = np.zeros(n, dtype=np.float32)
    for h in range(1, harmonics + 1):
        out += (1.0 / h ** 1.4) * np.sin(2 * np.pi * freq * h * t + np.random.uniform(0, 0.2))
    attack = int(0.008 * sr)
    env = np.exp(-t * 2.2)
    if attack > 0:
        env[:attack] *= np.linspace(0.0, 1.0, attack)
    return (out * env * amp).astype(np.float32)


def _voice_chord(chord: Chord) -> Tuple[int, List[int]]:
    """근음(베이스 옥타브)과 상성부 보이싱을 MIDI 번호로."""
    bass_midi = 36 + (chord.bass if chord.bass is not None else chord.root)
    upper: List[int] = []
    for i, semitone in enumerate(QUALITY_INTERVALS[chord.quality]):
        pc = (chord.root + semitone) % 12
        midi = 60 + pc
        if upper and midi <= upper[-1]:
            midi += 12
        upper.append(midi)
    return bass_midi, upper


def _drums(n_samples: int, beat_positions: np.ndarray, sr: int, beats_per_bar: int = 4) -> np.ndarray:
    """대중음악의 기본 드럼 패턴. 킥(1·3박) + 스네어(2·4박) + 하이햇.

    이전 버전은 짝수 박마다 똑같은 킥을 넣어서 마디 첫 박과 셋째 박이
    음향적으로 구별되지 않았다. 그래서 다운비트 추정을 테스트할 수가 없었다
    (추정기가 틀린 게 아니라 음원에 단서가 없었던 것).
    실제 음악처럼 1박 킥을 3박보다 세게 하고 2·4박에 스네어를 넣는다.
    """
    out = np.zeros(n_samples, dtype=np.float32)
    for i, pos in enumerate(beat_positions):
        idx = int(pos * sr)
        if idx >= n_samples:
            break
        beat_in_bar = i % beats_per_bar

        if beat_in_bar == 0 or beat_in_bar == beats_per_bar // 2:
            amp = 0.70 if beat_in_bar == 0 else 0.42   # 마디 첫 박을 더 세게
            dur = int(0.13 * sr)
            t = np.arange(min(dur, n_samples - idx)) / sr
            thump = np.sin(2 * np.pi * 55 * np.exp(-t * 18) * t * 6) * np.exp(-t * 22)
            out[idx : idx + len(thump)] += (thump * amp).astype(np.float32)
        else:
            dur = int(0.09 * sr)                        # 스네어
            length = min(dur, n_samples - idx)
            t = np.arange(length) / sr
            body = np.sin(2 * np.pi * 190 * t) * 0.4 + np.random.randn(length) * 0.6
            out[idx : idx + length] += (body * np.exp(-t * 34) * 0.30).astype(np.float32)

        dur = int(0.05 * sr)                            # 하이햇
        length = min(dur, n_samples - idx)
        t = np.arange(length) / sr
        hat = np.random.randn(length) * np.exp(-t * 90)
        out[idx : idx + length] += (hat * 0.08).astype(np.float32)
    return out


def render_track(track: DemoTrack, sr: int = SR, seed: int = 0) -> Tuple[np.ndarray, List[Tuple[float, float, str]]]:
    """(오디오, [(시작, 끝, 코드라벨)]) 반환."""
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    beat_dur = 60.0 / track.bpm
    chord_dur = beat_dur * track.beats_per_chord

    sequence = list(track.chords) * track.repeats
    total = chord_dur * len(sequence)
    n_samples = int(total * sr) + sr
    audio = np.zeros(n_samples, dtype=np.float32)
    labels: List[Tuple[float, float, str]] = []

    for chord_i, label in enumerate(sequence):
        chord = Chord.parse(label)
        start = chord_i * chord_dur
        labels.append((start, start + chord_dur, label))
        bass_midi, upper = _voice_chord(chord)

        # 박마다 다시 때린다 (지속음만 있으면 비트 추적이 어렵다)
        for beat in range(track.beats_per_chord):
            t0 = start + beat * beat_dur
            idx = int(t0 * sr)
            # 마디 첫 박을 음향적으로 구별 가능하게 만든다.
            #
            # 예전에는 "코드가 시작하는 박"만 세게 쳤는데, 코드가 반 마디마다
            # 바뀌는 곡에서는 1박과 3박이 완전히 똑같아져서 다운비트를 알아낼
            # 단서가 음원에 아예 없었다. 그 상태로 평가하면 어떤 알고리즘도
            # 맞힐 수 없고, 거기 맞춰 튜닝하면 잘못된 픽스처에 과적합된다.
            beat_in_bar = int(round((t0 / beat_dur))) % track.beats_per_bar
            amp_scale = 1.0 if beat == 0 else 0.75
            if beat_in_bar == 0:
                amp_scale *= 1.25          # 마디 첫 박 강세
            # 코드가 바뀐 뒤까지 길게 끌면 인식기가 아니라 음원이 틀린 게 된다.
            # 코드 끝 + 짧은 여운까지만 울리게 자른다.
            remaining = chord_dur - beat * beat_dur + 0.18
            dur = min(beat_dur * 1.8, remaining)
            bass = _tone(midi_to_hz(bass_midi), min(beat_dur * 1.6, remaining), sr,
                         0.30 * amp_scale, harmonics=4)
            audio[idx : idx + len(bass)] += bass[: max(0, n_samples - idx)]
            for midi in upper:
                voice = _tone(
                    midi_to_hz(midi + rng.uniform(-0.04, 0.04)), dur, sr, 0.16 * amp_scale,
                )
                audio[idx : idx + len(voice)] += voice[: max(0, n_samples - idx)]

    beat_positions = np.arange(0, total, beat_dur)
    audio += _drums(n_samples, beat_positions, sr, track.beats_per_bar)
    audio += rng.randn(n_samples).astype(np.float32) * 0.002  # 약한 노이즈 플로어

    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.85
    return audio, labels


def write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def write_lab(path: Path, labels: Sequence[Tuple[float, float, str]]) -> None:
    """MIREX .lab 형식. McGill Billboard / Isophonics와 같은 포맷이라
    나중에 진짜 벤치마크로 갈아탈 때 평가 코드를 그대로 쓴다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{s:.6f}\t{e:.6f}\t{lab}" for s, e, lab in labels]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_all(out_dir: Path, sr: int = SR) -> List[Path]:
    written: List[Path] = []
    for i, track in enumerate(DEMO_TRACKS):
        audio, labels = render_track(track, sr=sr, seed=i)
        wav_path = out_dir / f"{track.name}.wav"
        write_wav(wav_path, audio, sr)
        write_lab(out_dir / f"{track.name}.lab", labels)
        written.append(wav_path)
    return written


if __name__ == "__main__":  # pragma: no cover
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/demo")
    for p in generate_all(target):
        print("생성:", p)
