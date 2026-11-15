"""Create or restore a private backup of app data. Stop the app first."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sqlite3
import tarfile
import tempfile
from datetime import datetime, timezone


def backup(data: Path, destination: Path) -> Path:
    data = data.resolve()
    if not data.is_dir():
        raise ValueError('data 디렉터리가 없습니다.')
    destination = destination.expanduser().resolve()
    if destination == data or data in destination.parents:
        raise ValueError('백업은 data 디렉터리 바깥에 저장해야 합니다.')
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    files = sorted(data.rglob('*'))
    if any(path.is_symlink() for path in files):
        raise ValueError('data에 심볼릭 링크가 있어 백업을 중단합니다.')
    for path in files:
        if path.is_file() and path.suffix == '.sqlite3':
            with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as con:
                if con.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                    raise ValueError(f'데이터베이스 검사 실패: {path.name}')
    name = 'chordlab-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.tar.gz'
    target = destination / name
    if target.exists():
        raise FileExistsError('동일한 이름의 백업이 이미 있습니다. 잠시 뒤 다시 시도해 주세요.')
    fd, temp_name = tempfile.mkstemp(prefix='.chordlab-', suffix='.tmp', dir=destination)
    os.close(fd)
    temp = Path(temp_name)
    try:
        with tarfile.open(temp, 'w:gz') as archive:
            archive.add(data, arcname='data', recursive=True)
        with tarfile.open(temp, 'r:gz') as archive:
            for item in archive:
                if item.isfile():
                    stream = archive.extractfile(item)
                    while stream.read(1024 * 1024):
                        pass
        os.chmod(temp, 0o600)
        os.replace(temp, target)
        return target
    finally:
        temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description='앱을 중지한 뒤 개인 데이터를 백업합니다.')
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--destination', type=Path, default=Path.home() / 'chordlab-backups')
    args = parser.parse_args()
    print('백업 완료:', backup(args.data, args.destination))
    print('이 파일에는 음원과 개인 분석 기록이 포함됩니다. 기기 밖의 안전한 장소에 별도로 보관하세요.')


if __name__ == '__main__':
    main()
