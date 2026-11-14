"""고정 공급기 수명주기와 실제 yt-dlp 플러그인의 로컬 HTTP 전달 계약."""
import json
import os
from pathlib import Path
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from chordapp.service import pot
from chordapp.service.private_youtube import PrivateYouTubeError
from test_audio_sources import setup
from test_private_youtube import private


@pytest.fixture
def pot_env(tmp_path, monkeypatch):
    monkeypatch.setattr(pot, 'ROOT', tmp_path)
    for name in ('CHORDAPP_PRIVATE_MODE','CHORDAPP_ENABLE_YTDLP','CHORDAPP_ENABLE_POT'):
        monkeypatch.setenv(name, '1')
    monkeypatch.setattr(pot, 'plugin_version', lambda: pot.VERSION)
    monkeypatch.setattr(pot, '_server_version', lambda: None)
    base, source = pot.paths()
    (source / 'server/node_modules').mkdir(parents=True)
    (source / '.chordapp-installed.json').write_text(json.dumps({'commit':pot.COMMIT}))
    return base, source


def test_matching_installed_provider_waits_for_start(pot_env):
    assert pot.status()['state'] == 'stopped'


def test_provider_disabled_requires_all_personal_flags(pot_env, monkeypatch):
    monkeypatch.delenv('CHORDAPP_PRIVATE_MODE')
    assert pot.status()['state'] == 'disabled'
    pot.ensure_started()


@pytest.mark.parametrize('plugin,server,state', [(None,None,'not_installed'),('1.0.0',None,'version_mismatch'),(pot.VERSION,'1.0.0','version_mismatch'),(pot.VERSION,pot.VERSION,'ready')])
def test_provider_version_contract(pot_env, monkeypatch, plugin, server, state):
    monkeypatch.setattr(pot,'plugin_version',lambda:plugin)
    monkeypatch.setattr(pot,'_server_version',lambda:server)
    assert pot.status()['state'] == state


def test_running_provider_is_reused(pot_env, monkeypatch):
    monkeypatch.setattr(pot,'_server_version',lambda:pot.VERSION)
    def unexpected(*a,**kw): pytest.fail('must reuse healthy provider')
    monkeypatch.setattr(pot.subprocess,'Popen',unexpected)
    pot.ensure_started()


def test_start_is_loopback_only_and_does_not_keep_token_output(pot_env, monkeypatch):
    base, _ = pot_env
    versions=iter([None,pot.VERSION])
    monkeypatch.setattr(pot,'_server_version',lambda:next(versions))
    monkeypatch.setenv('YOUTUBE_API_KEY','not-for-token-server')
    monkeypatch.setenv('HTTPS_PROXY','http://proxy.invalid:8080')
    calls=[]
    class Process:
        pid=12345
    def spawn(command,**kwargs):
        calls.append((command,kwargs));return Process()
    monkeypatch.setattr(pot.subprocess,'Popen',spawn)
    pot.ensure_started()
    command, options=calls[0]
    assert command[command.index('--host')+1]=='127.0.0.1'
    assert command[command.index('--port')+1]=='4416'
    assert options['stdout']==options['stderr']==subprocess.DEVNULL
    assert 'YOUTUBE_API_KEY' not in options['env']
    assert options['env']['HTTPS_PROXY']=='http://proxy.invalid:8080'
    assert (base/'server.pid').read_text()=='12345'


def test_start_failure_is_explicit(pot_env, monkeypatch):
    class Process:
        def poll(self):return 1
    monkeypatch.setattr(pot.subprocess,'Popen',lambda *a,**kw:Process())
    with pytest.raises(pot.PotError,match='시작되지'):
        pot.ensure_started()


def test_wrong_source_commit_is_not_overwritten(pot_env, monkeypatch):
    _,source=pot_env
    (source/'.chordapp-installed.json').unlink()
    monkeypatch.setattr(pot.subprocess,'check_output',lambda *a,**kw:'wrong-sha')
    with pytest.raises(pot.PotError,match='덮어쓰지'):
        pot.install()
    assert source.is_dir()


def test_download_command_has_explicit_token_and_baseline_modes(private, monkeypatch):
    provider=private[5]
    monkeypatch.setenv('CHORDAPP_ENABLE_POT','1')
    command=provider.command('PrivateTest',Path('/tmp'))
    assert 'youtube:player_client=mweb' in command
    assert 'youtubepot-bgutilhttp:base_url=http://127.0.0.1:4416' in command
    assert '--no-plugin-dirs' not in command
    monkeypatch.setenv('CHORDAPP_ENABLE_POT','0')
    baseline=provider.command('PrivateTest',Path('/tmp'))
    assert '--no-plugin-dirs' in baseline and not any('player_client' in arg for arg in baseline)


def test_missing_token_server_does_not_silently_fallback(private, monkeypatch):
    monkeypatch.setenv('CHORDAPP_ENABLE_POT','1')
    def fail():raise pot.PotError('공급기 시작 실패')
    monkeypatch.setattr(pot,'ensure_started',fail)
    with pytest.raises(PrivateYouTubeError) as error:
        with private[5].audio('PrivateTest'):pass
    assert error.value.code=='pot_unavailable' and private[6]==[]


def test_ready_token_provider_preserves_download_cleanup(private, monkeypatch):
    monkeypatch.setenv('CHORDAPP_ENABLE_POT','1')
    monkeypatch.setattr(pot,'ensure_started',lambda:None)
    with private[5].audio('PrivateTest') as path:
        assert path.is_file()
    assert not path.exists()


def test_real_plugin_hands_off_video_binding_over_local_http():
    module=pytest.importorskip('yt_dlp_plugins.extractor.getpot_bgutil_http')
    from yt_dlp import YoutubeDL
    from yt_dlp.extractor.youtube import YoutubeIE
    from yt_dlp.extractor.youtube.pot._director import YoutubeIEContentProviderLogger
    from yt_dlp.extractor.youtube.pot.provider import PoTokenRequest,PoTokenContext
    requests=[]
    fake_token='fixture-only-not-a-real-token'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*a):pass
        def do_GET(self):
            assert self.path=='/ping'
            self.send_response(200);self.end_headers()
            self.wfile.write(json.dumps({'version':pot.VERSION,'server_uptime':1}).encode())
        def do_POST(self):
            assert self.path=='/get_pot'
            requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200);self.end_headers()
            self.wfile.write(json.dumps({'poToken':fake_token}).encode())
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with YoutubeDL({'quiet':True,'no_warnings':True}) as ydl:
            ie=YoutubeIE(ydl)
            provider=module.BgUtilHTTPPTP(ie,YoutubeIEContentProviderLogger(ie,'fixture'),
                {'base_url':[f'http://127.0.0.1:{server.server_port}']})
            request=PoTokenRequest(context=PoTokenContext.GVS,
                innertube_context={'client':{'clientName':'MWEB','clientVersion':'test'}},
                internal_client_name='mweb',video_id='BzYnNdJhZQw',_gvs_bind_to_video_id=True)
            response=provider.request_pot(request)
            assert response.po_token==fake_token
        assert len(requests)==1 and requests[0]['content_binding']=='BzYnNdJhZQw'
        assert requests[0]['disable_tls_verification'] is False
        assert requests[0]['proxy'] is None
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)
