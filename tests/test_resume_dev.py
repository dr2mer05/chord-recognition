"""로컬 Git 원격으로 자동 갱신·자기 파일 갱신·사용자 작업 보존을 검증한다."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

RESUME = Path(__file__).resolve().parents[1] / 'scripts' / 'resume-dev.sh'


def git(root, *args):
    return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def projects(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    git(source, 'init', '-b', 'main')
    git(source, 'config', 'user.email', 'test@example.invalid')
    git(source, 'config', 'user.name', 'Test')
    (source / 'scripts').mkdir()
    shutil.copyfile(RESUME, source / 'scripts/resume-dev.sh')
    (source / 'scripts/start-dev.sh').write_text('#!/bin/bash\nprintf "v1 %s" "$*" > work/started.txt\n')
    (source / 'scripts/setup-dev.sh').write_text('#!/bin/bash\nexit 90\n')
    (source / 'requirements-cpu.lock').write_text('cpu1\n')
    (source / 'requirements-private.txt').write_text('private1\n')
    (source / '.gitignore').write_text('work/\n.venv/\n')
    (source / 'app.txt').write_text('v1\n')
    git(source, 'add', '.')
    git(source, 'commit', '-m', 'initial')
    remote = tmp_path / 'remote.git'
    subprocess.run(['git', 'clone', '--bare', str(source), str(remote)], check=True, capture_output=True)
    git(source, 'remote', 'add', 'origin', str(remote))
    local = tmp_path / 'local'
    subprocess.run(['git', 'clone', str(remote), str(local)], check=True, capture_output=True)
    (local / '.venv/bin').mkdir(parents=True)
    (local / '.venv/bin/python').symlink_to(sys.executable)
    (local / 'work').mkdir()
    digest = hashlib.sha256(b'cpu1\nprivate1\n').hexdigest()
    (local / 'work/dev-requirements.sha256').write_text(digest+'\n')
    return source, local


def publish(source):
    (source / 'app.txt').write_text('v2\n')
    (source / 'scripts/start-dev.sh').write_text('#!/bin/bash\nprintf "v2 %s" "$*" > work/started.txt\n')
    with (source / 'scripts/resume-dev.sh').open('a') as stream:
        stream.write('# self update\n' * 300)
    git(source, 'add', '.')
    git(source, 'commit', '-m', 'update')
    git(source, 'push', 'origin', 'main')


def resume(local):
    return subprocess.run(['bash', str(local / 'scripts/resume-dev.sh')],
        env={**os.environ, 'CODESPACES':'true'}, capture_output=True, text=True, timeout=15)


def test_resume_updates_without_terminal_commands_and_survives_self_update(projects):
    source, local = projects
    publish(source)
    result = resume(local)
    assert result.returncode == 0, result.stderr
    assert (local / 'app.txt').read_text() == 'v2\n'
    assert (local / 'work/started.txt').read_text() == 'v2 --resumed --restart'
    assert git(source, 'rev-parse', 'HEAD') == git(local, 'rev-parse', 'HEAD')


def test_resume_preserves_user_changes(projects):
    source, local = projects
    publish(source)
    (local / 'app.txt').write_text('my work\n')
    result = resume(local)
    assert result.returncode == 0 and '건너뜁니다' in result.stdout
    assert (local / 'app.txt').read_text() == 'my work\n'
    assert (local / 'work/started.txt').read_text() == 'v1 --resumed --restart'


def test_failed_update_starts_existing_version(projects):
    source, local = projects
    git(local, 'remote', 'set-url', 'origin', str(source / 'missing-remote'))
    result = resume(local)
    assert result.returncode == 0 and '현재 버전' in result.stdout
    assert (local / 'app.txt').read_text() == 'v1\n'
    assert (local / 'work/started.txt').read_text() == 'v1 --resumed --restart'
