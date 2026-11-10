# Chord Analysis Lab

음원의 코드 진행을 인식하고 로마숫자·화성 기능·대체 코드를 살펴보는 개인 학습용 웹앱입니다.
FastAPI, HTML/JavaScript, SQLite를 사용하며 기본 분석은 Linux CPU에서 실행합니다.

## 실행

Python 3.12에서 검증한 CPU 의존성 조합을 제공합니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.lock
python -m chordapp serve
```

브라우저에서 `http://127.0.0.1:8000`을 엽니다. iPad에서는 본인만 접근할 수 있는
클라우드 개발 환경의 브라우저 미리보기를 사용하세요. 현재는 개인 단일 사용자용이며,
로그인·사용자별 데이터 격리는 없습니다. 공개 서비스로 배포하기 전에는 접근 제어가 필요합니다.

`requirements.txt`는 주요 의존성, `requirements-cpu.lock`은 검증 환경의 전체 고정 버전입니다.
beat_this·Demucs·외부 인식기는 기본 설치에 포함하지 않았습니다. 필요할 때
`requirements-models.txt`를 검토하고 설치하세요. 모델을 설치하면 사용하는 엔진과 비용·결과가 달라집니다.

## YouTube 링크로 분석하기

개인 전용 자동 분석과 별도 음원 연결의 두 경로를 제공합니다. 별도 음원이 등록된 영상은
그 연결을 우선 사용합니다. 해제·만료된 연결은 자동 다운로드로 대체하지 않습니다.

### 개인 전용 자동 분석

개인 Codespaces에서는 시작 스크립트가 포트 8000을 **Private**로 설정한 다음 자동 분석을
활성화합니다. 기존 Codespace를 업데이트할 때는 다음을 실행하세요.

```bash
git pull --ff-only origin feat/authorized-audio-sources
bash scripts/setup-dev.sh
bash scripts/start-dev.sh --restart
```

사이트를 새로고침하고 영상 링크를 입력한 뒤 **분석**을 누릅니다. 검색은 별도로
`YOUTUBE_API_KEY`가 필요하지만 링크 입력에는 필요하지 않습니다.
이 경로는 `yt-dlp`와 FFmpeg·Deno·설치된 EJS 패키지를 사용하며 분석 엔진은 기존과 같습니다.
로컬 환경은 `requirements-private.txt`를 설치하고 `.env`에 다음 두 값을 설정한 뒤 서버를 재시작합니다.

```dotenv
CHORDAPP_PRIVATE_MODE=1
CHORDAPP_ENABLE_YTDLP=1
```

두 설정이 모두 켜져야 동작합니다. 앱 자체에는 로그인이 없으므로 개인 모드는
비공개 포트 또는 본인만 접근 가능한 환경에서 사용하세요. 이 설정 자체가 접근 제어를 제공하지는 않습니다.
개인 이용만으로 다운로드 권한이나 YouTube 약관의 예외가 부여되는 것은 아닙니다.
향후 공개 서비스의 음원 공급 경로는 별도로 검토해야 합니다.

음원은 요청마다 독립적인 임시 폴더에 받아 **분석 성공·실패 후 삭제**하며 다운로드 API는 제공하지 않습니다.
정상 처리와 시간 초과 시 자식 FFmpeg 프로세스도 정리합니다. 서버 강제 종료나 머신 중단 시에는
잔여 임시 파일이 남을 수 있으므로 `work/private-youtube/`를 삭제할 수 있습니다.
분석 결과와 음원 해시를 24시간 동안 재사용해 같은 옵션으로 다시 열 때 다운로드를 생략합니다.
교정은 영상 ID와 음원 해시에 연결하며 원본 인식 결과와 별도로 보존합니다.
템포·박자·분석 옵션·엔진 환경을 변경하거나 캐시가 만료되면 음원을 다시 받아 분석할 수 있습니다.
음원을 보관하지 않으므로 이 경로의 결과는 원본 음원이 필요한 실제 곡 벤치마크에 바로 사용하지 못합니다.

라이브·15분 초과 영상은 지원하지 않습니다. YouTube의 접근 제한·연결 변경이나 네트워크 차단으로
실패할 수 있으며 모든 영상의 이용 가능성을 보장하지 않습니다. 기본 플레이어 클라이언트를 강제 교체하거나
쿠키·프록시 순환 등 추가 우회 로직은 넣지 않았습니다.

### 별도 음원 연결

1. YouTube 탭에 영상 링크를 입력합니다. 검색을 쓰려면 `YOUTUBE_API_KEY`를 설정합니다.
2. 연결된 음원이 없다면 **사용 권한이 있는 음원 연결 · 변경**을 엽니다.
3. 직접 녹음했거나 권리자가 분석·보관을 허락한 별도 원본을 선택하고 권한 근거를 기록합니다.
4. 영상과 같은 녹음·편집본인지 확인하고 필요하면 시작 시차와 사용 기한을 지정합니다.
5. **연결하고 분석**을 누릅니다. 이후에는 같은 링크나 **연결한 음원 다시 열기**로 결과를 엽니다.

영상 시각 = 음원 시각 + 영상 시작 시차입니다. 음원 0초가 영상 5초에 해당하면 `5`를 입력합니다.
다른 편집본·라이브 버전·속도가 다른 영상은 동기화 지원 범위에 포함하지 않습니다.
동일 녹음 여부 확인을 하지 않은 연결은 보관할 수 있지만 영상 분석을 시작하지 않습니다.
영상과 무관한 분석은 **로컬 파일** 탭을 이용하세요.

연결할 수 있는 형식은 WAV·FLAC·MP3·OGG·AIFF이며 최대 40MB·15분입니다.
권한 확인 기록은 사용자 입력이며 앱이 권리를 자동으로 검증하거나 사용 허락을 부여하지 않습니다.
이번 버전에는 일반 상용 음원 공급 카탈로그가 없습니다.

YouTube Data API와 oEmbed는 영상 정보를, 공식 IFrame 플레이어는 영상 재생을 담당합니다.
선택적인 개인 자동 분석은 공식 Data API가 아닌 별도 공급 모듈이며, 공식 API의 오디오 제공 기능이 아닙니다.

- [YouTube 이용약관](https://www.youtube.com/t/terms)
- [YouTube 개발자 정책](https://developers.google.com/youtube/terms/developer-policies)
- [Google 개인정보처리방침](https://policies.google.com/privacy)

영상 정보·플레이어·개인 자동 분석을 사용할 때 YouTube에 요청이 전송됩니다.
별도로 등록한 원본 음원은 이 서버에 보관하고 YouTube에 업로드하지 않습니다.
앱의 분석 결과는 YouTube가 제공한 정보가 아닙니다.

## 저장·교정·재열람

- `data/authorized_audio/`: 내용 해시로 보관한 원본. 별도 다운로드 API는 제공하지 않습니다.
- `data/audio_sources.sqlite3`: 영상 연결, 권한 근거·기한, 버전·시차.
- `data/cache.sqlite3`: 원본 분석과 별도 영상 메타데이터 캐시.
- `data/corrections.sqlite3`: 사용자 교정·템포·박자 지정.
- `work/private-youtube/`: 개인 자동 분석의 요청별 임시 파일. 요청 종료 시 삭제.

동일 음원·연결 버전·분석 옵션·설치 환경에서는 원본 분석을 재사용합니다.
실제 추적기나 소스 분리가 폴백한 결과는 선호 엔진의 정상 결과로 캐시하지 않습니다.
음원 연결을 교체하면 새 버전과 교정 키를 사용하며 기존 교정은 보존하지만 자동 적용하지 않습니다.
과거 `youtube:<id>` 캐시·교정도 삭제하지 않고 새 연결에 자동 승계하지 않습니다.
캐시 사용 여부와 원본 분석 출처는 별도 필드입니다.

코드 칸을 클릭해 교정하고 다시 열어도 원본 인식 결과는 덮어쓰지 않습니다.
연결 해제나 권한 만료는 캐시 조회도 차단합니다. 연결 해제는 파일 삭제가 아닙니다.
저장 데이터를 완전히 지우려면 서버를 멈추고 위 파일·디렉터리를 삭제하세요.
음원과 DB는 `data/` 아래에 두며 저장소에 커밋하지 않습니다.

YouTube 메타데이터는 분석과 분리해 24시간 뒤 갱신합니다. 제목으로 조회한 로컬 코드 DB는
`/api/lookup`에서 참고 진행으로 사용할 수 있지만 영상 동기화 분석을 대신하지 않습니다.

## API

| 경로 | 용도 |
|---|---|
| `GET /api/youtube/source?url=...` | 영상 정보·음원 연결 가능 상태 |
| `POST /api/audio-sources` | 음원과 권한·영상 연결 등록 (multipart) |
| `GET /api/audio-sources` | 연결한 음원 목록 |
| `DELETE /api/audio-sources/{video_id}` | 연결 사용 해제 |
| `POST /api/youtube` | 연결 음원 또는 개인 자동 분석·캐시 조회 |
| `POST /api/analyze` | 로컬 파일 분석 |
| `POST /api/corrections` | 원본과 별도로 교정 저장 |

연결이 없거나 기한 만료·해제·동일 녹음 미확인·파일 유실·내용 변경 시
`/api/youtube`는 HTTP 409와 `detail.code`, `detail.message`를 반환합니다.
API 문서는 실행 후 `/docs`에서 확인할 수 있습니다.
개인 공급 경로의 연결 실패·지원 불가·잘못된 음원은 HTTP 502, 시간 초과는 504,
필수 도구 누락은 503으로 안내하며 `detail.code`와 `detail.message`를 제공합니다.

## 검증과 알려진 한계

```bash
python -m pytest tests -q
python -m chordapp eval
python -m chordapp beateval
```

테스트는 외부 YouTube 요청 없이 공급·권한·캐시·교정·API 동작을 검증합니다.
합성 평가 5곡의 약 98.8%는 실제 음악 정확도의 증거가 아닙니다.
비트 평가 16곡도 합성 데이터이고 정답 박자 정보를 입력하므로 자동 박자 추정 성능과 구분해야 합니다.

기존 beat_this/librosa 및 템플릿·계층형 코드 인식 알고리즘은 이번 단계에서 유지했습니다.
추적기가 판단한 마디당 박 수를 구간 번호·결과·차트까지 보존하며, 수동 박자 지정이 자동 추정보다 우선합니다.
자동으로 되돌리면 추적기의 박자를 다시 사용하고, 추정이 불가능한 경로에서는 기본 4박을 사용합니다.
전체 곡 단일 조성·고정 박자 구조, 알터드 코드 인식 범위는 후속 과제입니다.
기존 `benchmark`는 새 예측에 교정을 얹어 정답을 구성하는 한계가 있으므로
엔진 우열을 주장하는 공식 실제 곡 평가에 사용하지 마세요. 독립 정답 스냅샷이 먼저 필요합니다.

긴 분석은 현재 동기 요청입니다. 작업 큐·중복 실행 방지·취소·진행 상태는 다음 단계입니다.
실제 권한 음원과 동일 버전 영상의 동기화, iPad Safari는 해당 자료와 기기에서 별도 검증해야 합니다.
외부 유료 API나 GPU 서비스는 사용하지 않습니다.

브라우저 검증은 Playwright가 있는 Node 환경에서 실행합니다. 테스트 서버는 임시 DB와 가짜
영상 정보를 사용하고, 테스트는 직접 생성한 합성 음원을 실제 CPU 분석기에 넣습니다.
YouTube IFrame은 모의 플레이어로 바꾸므로 실제 YouTube 동기화 검증을 대신하지 않습니다.

```bash
# 터미널 1: 임시 데이터 전용 검증 서버 (기존 data/에 쓰지 않음)
python tests/browser_fixture.py
# 터미널 2: Playwright·브라우저가 설치된 환경
node tests/browser_flow.cjs
# 시스템 Chromium 사용 시: CHROMIUM_PATH=/usr/bin/chromium node tests/browser_flow.cjs
```

브라우저 도구가 없다면 `npm install --prefix work/browser-tools playwright`로 설치한 뒤
`NODE_PATH="$PWD/work/browser-tools/node_modules"`를 설정하고 Playwright 브라우저를 설치하거나
`CHROMIUM_PATH`로 설치된 Chromium을 지정하세요. 검증 스크린샷은 `work/browser-preview.png`에 저장됩니다.

## iPad·휴대폰에서 Codespaces로 검증

`.devcontainer/devcontainer.json`과 `scripts/`에 Python 3.12 CPU 환경의 자동 설치·실행 구성이 있습니다.
GitHub에 이 변경이 반영된 브랜치에서 다음 순서로 시작합니다.

1. 저장소의 **Code → Codespaces → Create codespace**에서 2코어 환경을 선택합니다.
2. 최초 의존성 설치가 끝나면 앱이 자동 실행됩니다. 이미 만든 Codespace는 컨테이너를 재빌드해야 설정이 적용됩니다.
3. **Ports → 8000 → Open in Browser**를 엽니다. 포트 공개 범위는 **Private**로 유지합니다.
4. 같은 GitHub 계정으로 로그인한 iPad·휴대폰에서 그 HTTPS 주소를 엽니다.
5. 합성 데모로 화면·터치 교정을 확인한 뒤, 개인 링크 분석 또는 별도 원본 음원 연결을 검증합니다.
6. 검증을 마치면 GitHub의 Codespaces 목록에서 **Stop codespace**로 실행을 중지합니다.

시작 스크립트는 Codespaces에서 `gh codespace ports visibility 8000:private`를 실행하고,
비공개 설정에 실패하면 새 서버를 시작하지 않습니다. 서버 로그는 `work/dev-server.log`에 저장합니다.
자동 실행에 실패하면 터미널에서 `bash scripts/setup-dev.sh`와 `bash scripts/start-dev.sh`를 순서대로 실행하세요.
기존 서버가 있으면 중복 실행하지 않습니다.

Codespaces의 포함 사용량은 GitHub 요금제에 따라 다르고 실행 시간·저장 공간에 비용이 붙을 수 있습니다.
유휴 중지 시간을 짧게 설정하고 유료 초과 사용은 예산 설정으로 제한하세요. Codespace를 삭제하면
커밋하지 않은 음원·DB도 없어지므로, 삭제 전 필요한 데이터를 개인 보관소에 백업해야 합니다.

- [비공개 포트 미리보기](https://docs.github.com/en/codespaces/developing-in-a-codespace/forwarding-ports-in-your-codespace)
- [Codespaces 사용량·비용](https://docs.github.com/en/billing/concepts/product-billing/github-codespaces)

화면은 320px·390px 휴대폰과 834px 태블릿 너비에서 터치 편집·재열람을 검증합니다.
터치 환경에서는 코드 칸을 누르면 저장·취소 버튼이 있는 편집창이 열립니다.
실제 iPad Safari와 실제 YouTube 영상 동기화 검증은 별도로 필요합니다.

## 정식 웹사이트·앱으로 운영할 때

현재 개인용 웹앱을 개발·검증한 뒤 다음 환경을 준비합니다. 이번 단계에서 유료 서버를 개설하지 않습니다.

| 구성 | 개인용 초기 운영 | 확대 시 검토 |
|---|---|---|
| 실행 서버 | Linux CPU 서버 하나에서 FastAPI·화면 제공 | 분석 작업자 분리, 동시 처리 수 제한 |
| 외부 접속 | HTTPS 주소, 도메인 또는 관리형 주소 | TLS 갱신·접속 장애 모니터링 |
| 접근 제어 | 본인 로그인, 요청·파일 접근 제한 | 사용자별 데이터·교정 격리 |
| 분석 작업 | 작업 큐, 진행·실패 상태, 시간 제한 | 재시도·취소·작업 재개 |
| 분석 기록 | 영구 디스크의 SQLite와 정기 백업 | 필요해질 때 PostgreSQL 등으로 이전 |
| 음원 보관 | 비공개 영구 저장소, 삭제·기한 관리 | 오브젝트 저장소, 용량·사용자별 한도 |
| 음원 공급 | 허용된 원본과 정확한 영상 버전의 연결 | 권리자·공급 API와의 명시적 이용 조건 |
| 검증·운영 | 고정 의존성, 회귀 테스트, 오류 기록 | 독립 정답 평가·배포 자동화 |
| 앱 형태 | 태블릿·모바일 반응형 웹 | PWA 설치, 필요할 때 네이티브 앱 |

맥미니 등 개인 컴퓨터도 상시 서버로 사용할 수 있습니다. 외부 접속용 네트워크와 HTTPS·인증,
전원·인터넷 유지, 업데이트·백업을 직접 관리해야 합니다. 원격 분석이라면 iPad의 연산 성능과는
무관하게 서버 CPU와 메모리가 분석 속도를 결정합니다. 구매 전 실제 곡의 CPU 시간·메모리를 측정하고
작은 클라우드 서버의 장기 비용과 비교하는 것이 좋습니다. 현재는 GPU나 새 컴퓨터가 필수 조건이 아닙니다.

박자 전달 회귀 검증에서는 추적 결과를 통제해 3박·4박·6박의 결과·차트와 수동 지정 우선순위를 확인합니다.
모델 자체의 박자 인식 정확도를 측정하는 테스트는 아닙니다. 브라우저에서도 확인하려면 검증 서버에
`CHORDAPP_TEST_METER=3`, 브라우저 스크립트에 `CHORDAPP_EXPECTED_METER=3`을 지정하세요.
박자 처리 수정으로 이전 버전의 분석 캐시는 한 번 재계산될 수 있으며, 저장된 교정은 별도로 유지됩니다.
