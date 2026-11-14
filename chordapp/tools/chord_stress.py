"""코드 품질 변화·짧은 화성 리듬·전위의 독립 합성 표본. 기존 데모와 분리한다."""
from pathlib import Path
from .synth import DemoTrack, render_track, write_lab, write_wav


CHORD_STRESS_TRACKS = [
    DemoTrack('same_root_major', 'C', 96, 4, ['C','C7','Cmaj7','C6'], repeats=2),
    DemoTrack('same_root_minor', 'Am', 104, 4, ['Am','Am7','AmMaj7','Am6'], repeats=2),
    DemoTrack('quality_two_beats', 'F', 112, 4, ['F','Fmaj7','F7','F6'], repeats=2, chord_beats=2),
    DemoTrack('same_root_sus', 'D', 100, 4, ['D','Dsus4','Dsus2','D7'], repeats=2),
    DemoTrack('three_pulse_harmony', 'C', 108, 4, ['C','Am','F','G','Em','Dm'], repeats=2, chord_beats=3),
    DemoTrack('one_beat_changes', 'C', 120, 4, ['C','G','Am','F'], repeats=4, chord_beats=1),
    DemoTrack('inversions', 'C', 92, 4, ['C/E','C/G','Am/C','Am/E','F/A','G/B'], repeats=2),
]


def generate_stress(directory):
    """명시한 별도 디렉터리에 결정적 음원과 생성 시점의 정답을 함께 쓴다."""
    directory = Path(directory)
    paths = []
    for index, track in enumerate(CHORD_STRESS_TRACKS):
        signal, labels = render_track(track, seed=100 + index)
        path = directory / (track.name + '.wav')
        write_wav(path, signal)
        write_lab(path.with_suffix('.lab'), labels)
        paths.append(path)
    return paths
