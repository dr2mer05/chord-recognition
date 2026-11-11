"""독립 최소 재현의 요청 제한, 음원 검증, 비밀 비노출. 외부 YouTube는 호출하지 않는다."""
from pathlib import Path
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave

import pytest

from scripts import probe_youtube_audio as probe


def write_wav(path, frames=2205):
    with wave.open(str(path), 'wb') as audio:
        audio.setnchannels(1); audio.setsampwidth(2); audio.setframerate(22050)
        audio.writeframes(b'\0\1'*frames)


def ready(monkeypatch):
    monkeypatch.setattr(probe, 'tools', lambda: ('/fixture/deno','/fixture/ffmpeg'))
    monkeypatch.setattr(probe, 'environment', lambda: {'packages': {'yt-dlp':'fixture'}, 'runnable': {'deno':True,'ffmpeg':True}})


def test_fixed_video_and_restricted_download_arguments(tmp_path):
    args=probe.command(probe.DEFAULT_VIDEO,tmp_path,ipv4=False,binaries=('/deno','/ffmpeg'))
    ipv4=probe.command(probe.DEFAULT_VIDEO,tmp_path,ipv4=True,binaries=('/deno','/ffmpeg'))
    assert '--force-ipv4' not in args and '--force-ipv4' in ipv4
    assert '--ignore-config' in args and '--no-plugin-dirs' in args
    assert args[-1]=='https://www.youtube.com/watch?v=BzYnNdJhZQw'
    assert '40M' in args and '!is_live & duration > 0 & duration <= 900' in args
    assert not any('cookies' in v or 'proxy' in v or 'player_client' in v for v in args)
    with pytest.raises(ValueError):
        probe.command('https://example.invalid',tmp_path,ipv4=False,binaries=('/deno','/ffmpeg'))


def test_audio_is_actually_validated_and_deleted(monkeypatch):
    ready(monkeypatch)
    paths=[]
    def download(args):
        path=Path(args[args.index('-o')+1].replace('%(ext)s','wav'))
        write_wav(path);paths.append(path)
        return subprocess.CompletedProcess(args,0,'','')
    monkeypatch.setattr(probe,'download',download)
    report=probe.run(probe.DEFAULT_VIDEO,lambda value:None)
    assert report['rows'][0]['ok'] and report['rows'][0]['duration_seconds']==.1
    assert len(paths)==1 and not paths[0].exists()


def test_success_exit_without_audio_is_not_success(monkeypatch):
    ready(monkeypatch)
    monkeypatch.setattr(probe,'download',lambda args:subprocess.CompletedProcess(args,0,'',''))
    assert probe.probe(probe.DEFAULT_VIDEO,ipv4=False)['code']=='audio_invalid'


def test_proxy_rejection_is_not_retried_or_exposed(monkeypatch):
    ready(monkeypatch)
    calls=[]
    def download(args):
        calls.append(args)
        return subprocess.CompletedProcess(args,1,'','Tunnel connection failed: 403 Forbidden https://user:secret@proxy.invalid/?token=private')
    monkeypatch.setattr(probe,'download',download)
    report=probe.run(probe.DEFAULT_VIDEO,lambda value:None)
    assert len(calls)==1 and report['rows'][0]['code']=='network_proxy_blocked'
    assert report['rows'][0]['http_status']==403
    assert 'secret' not in str(report) and 'proxy.invalid' not in str(report)


def test_bot_failure_is_not_confused_with_successful_ipv4(monkeypatch):
    ready(monkeypatch)
    def download(args):
        if '--force-ipv4' not in args:
            return subprocess.CompletedProcess(args,1,'','Sign in to confirm you are not a bot')
        path=Path(args[args.index('-o')+1].replace('%(ext)s','wav'));write_wav(path)
        return subprocess.CompletedProcess(args,0,'','')
    monkeypatch.setattr(probe,'download',download)
    report=probe.run(probe.DEFAULT_VIDEO,lambda value:None)
    assert [r['code'] for r in report['rows']]==['youtube_bot_check','audio_received']


def test_timeout_still_cleans_request_folder(monkeypatch):
    ready(monkeypatch)
    paths=[]
    def download(args):
        path=Path(args[args.index('-o')+1]).parent;paths.append(path)
        (path/'partial').write_bytes(b'partial')
        raise subprocess.TimeoutExpired(args,90)
    monkeypatch.setattr(probe,'download',download)
    assert probe.probe(probe.DEFAULT_VIDEO,ipv4=False)['code']=='download_timeout'
    assert not paths[0].exists()


def test_missing_dependency_makes_no_network_request(monkeypatch):
    monkeypatch.setattr(probe,'environment',lambda:{'packages':{'deno':None},'runnable':{'deno':False}})
    def unexpected(*args,**kwargs):pytest.fail('network must not run')
    monkeypatch.setattr(probe,'download',unexpected)
    report=probe.run(probe.DEFAULT_VIDEO,lambda value:None)
    assert report['state']=='failed' and report['code']=='dependency_missing'


def test_conversion_stage_and_html_escaping():
    row=probe.failure(subprocess.CompletedProcess([],1,'[download] file\n[ExtractAudio] file','HTTP Error 403'))
    assert row['stage']=='conversion'
    row.update(profile='default',seconds=1,code='<script>')
    html=probe.page({'state':'complete','rows':[row]})
    assert '<script>' not in html and '&lt;script&gt;' in html
    assert 'http-equiv="refresh"' not in html


def test_probe_versions_match_private_installation():
    root=Path(__file__).resolve().parents[1]
    private=set((root/'requirements-private.txt').read_text().splitlines())
    packages=[line for line in (root/'requirements-source-probe.txt').read_text().splitlines() if line and not line.startswith('#')]
    assert len(packages)==4 and all(line in private for line in packages)


def test_real_timeout_reaps_process_and_does_not_pass_search_key(monkeypatch):
    original_spawn=subprocess.Popen
    processes=[]
    monkeypatch.setenv('YOUTUBE_API_KEY','must-not-be-passed')
    monkeypatch.setenv('HTTPS_PROXY','http://fixture-proxy.invalid:8080')
    def spawn(args,**kwargs):
        assert 'YOUTUBE_API_KEY' not in kwargs['env']
        assert kwargs['env']['HTTPS_PROXY']=='http://fixture-proxy.invalid:8080'
        process=original_spawn(args,**kwargs)
        original_communicate=process.communicate
        process.communicate=lambda timeout=None:original_communicate(timeout=.1 if timeout else None)
        processes.append(process)
        return process
    monkeypatch.setattr(probe.subprocess,'Popen',spawn)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            probe.download([sys.executable,'-c','import time; time.sleep(30)'])
        assert processes[0].poll() is not None
    finally:
        for process in processes:
            if process.poll() is None:process.kill();process.wait()


def test_http_report_does_not_allow_download_or_rerun():
    root=Path(__file__).resolve().parents[1]
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    proc=subprocess.Popen([sys.executable,str(root/'scripts/probe_youtube_audio.py'),'serve'],
        env={**os.environ,'PORT':str(port),'SOURCE_PROBE_RUN':'0'},stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base=f'http://127.0.0.1:{port}'
    try:
        for _ in range(50):
            try:
                with opener.open(base+'/healthz',timeout=.3) as response:
                    assert response.read()==b'ok'
                break
            except OSError:time.sleep(.05)
        else:pytest.fail('probe server did not start')
        with opener.open(base+'/report.json',timeout=1) as response:
            assert b'"state": "disabled"' in response.read()
        for url,method,status in (('/audio.wav','GET',404),('/?video=anything','GET',404),('/','POST',501)):
            with pytest.raises(urllib.error.HTTPError) as error:
                opener.open(urllib.request.Request(base+url,method=method),timeout=1)
            assert error.value.code==status
    finally:
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
