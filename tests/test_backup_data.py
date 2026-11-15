import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile


def test_backup_keeps_databases_and_audio_private(tmp_path):
    data = tmp_path / 'data'
    (data / 'uploads').mkdir(parents=True)
    (data / 'uploads' / 'song.wav').write_bytes(b'test audio')
    with sqlite3.connect(data / 'cache.sqlite3') as db:
        db.execute('create table saved (value text)')
        db.execute("insert into saved values ('result')")
    (tmp_path / '.env').write_text('SECRET=not-in-archive')
    output = tmp_path / 'backups'
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'backup_data.py'
    subprocess.run([sys.executable, str(script), '--data', str(data), '--destination', str(output)], check=True)
    archive, = output.glob('*.tar.gz')
    assert os.stat(archive).st_mode & 0o077 == 0
    with tarfile.open(archive) as tar:
        assert tar.extractfile('data/uploads/song.wav').read() == b'test audio'
        assert 'data/cache.sqlite3' in tar.getnames()
        assert all('.env' not in item for item in tar.getnames())
        restored = tmp_path / 'restored'
        restored.mkdir()
        tar.extractall(restored, filter='data')
    with sqlite3.connect(restored / 'data' / 'cache.sqlite3') as db:
        assert db.execute('select value from saved').fetchone()[0] == 'result'
