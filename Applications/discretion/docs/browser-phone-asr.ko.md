# 진행자 폰 브라우저에서 nntrainer ASR 돌리기 (B안 시제품)

**목적·독자:** 진행자가 폰 Chrome으로 앨리스 진행자 화면(`/host`)을 열면, 같은 폰 안에서 nntrainer가
회의 소리를 인식해 그 줄이 앨리스로 가게 하는 시제품의 계획과 진행 기록입니다. 앱(P2) 없이 지금 웹 화면으로
폰 인식을 시험하려는 개발자가 읽습니다.

## 1. 구조

```
진행자 폰
 ├ Chrome: https://dog.agentic-fw.com/alice-dev/host
 │    마이크 → AudioWorklet(16 kHz PCM16, 0.2초)
 │       ├─▶ 서버 /api/audio/chunk        (지금과 같음: 서버 ASR이 정식 전사)
 │       └─▶ http://127.0.0.1:8031/feed   (폰 안에서만)
 │    http://127.0.0.1:8031/poll ─ 줄 ─▶ 서버 /api/audio/compare-lines (ASR 비교 탭)
 └ nntr_room_http (폰 안 프로세스, 127.0.0.1만): room_api 그대로 (Silero·끝 판정·Qwen3-ASR·화자 군집)
```

- 시제품은 **비교 모드**만 합니다. 서버 ASR이 회의 전사를 맡고 폰 줄은 ASR 비교 탭에만 갑니다(#58과 같은 계약).
  회의 전사·카드·회의록은 바뀌지 않습니다. 폰 줄만으로 회의를 하는 모드는 다음 단계입니다.
- Chrome은 공개 사이트가 `127.0.0.1`을 부를 때 권한 창을 띄웁니다("…이 기기의 다른 앱 및 서비스에 액세스").
  사이트마다 한 번 "허용"하면 됩니다(2026-11-02 확인, 5절).

## 2. 폰 서비스 `nntr_room_http`

`room_http.cpp`. `nntr_room`과 같은 `--config`로 모델을 한 번 올리고 루프백에서 HTTP로 받습니다.

| 요청 | 하는 일 |
|---|---|
| `GET /health` | `{"status":"loading"/"ready"/"error", "session":…, "error":…}` |
| `POST /session` | 새 회의: 앞 회의의 인식기를 버리고 새로 만듦. 본문(선택) `{"labels_in_use":[…]}` |
| `POST /feed` | 본문 PCM16 16 kHz 모노. `X-At-Ms`: 이 덩어리의 시작(인식기 시계, ms). `{"rc":0}` 받음, `1` 바쁨(다시 보냄) |
| `GET /poll` | 지난 poll 뒤의 사건 `{"events":[…]}` (`room_api.h`의 line·moved·error) |
| `GET /status` | `nntr_room_status` |
| `POST /flush` | 받은 소리를 끝까지 인식(끝날 때까지 기다림) |

- 127.0.0.1에만 묶습니다. `--origin URL`(여러 번)을 주면 그 출처만 받고, CORS 응답도 그 출처만 허용합니다.
- 인식기 시계: 브라우저가 소리 흐름(`X-Audio-Stream`)마다 시작 위치를 정해 `X-At-Ms`로 줍니다. 줄의 시각을
  흐름 안 시각으로 되돌려 서버에 보냅니다(서버가 그 흐름을 자기 회의 시계에 놓음, `accept_compare`).

## 3. 앨리스 변경

- `POST /api/audio/compare-lines` (진행자 열쇠): `{"stream", "lines":[{n, speaker, start_ms, end_ms, text, doubt?}]}`
  → `RoomSession.accept_compare`. 장치용 `/api/device/compare-lines`와 같은 검사·번호 규칙.
- 진행자 화면(`ambient/ui/pages/listen.js`): 마이크를 켤 때, 이 사이트가 루프백에 닿아도 되고(권한 상태만 물음, 창 없음)
  8031이 답하면 위 흐름을 켭니다. 아니면 지금과 같습니다. `?phone=PORT`는 다른 포트로 시험할 때만.
- 권한은 진행자 본인 자리의 ASR 비교 탭에서 묻습니다(Android에서만 버튼). 사이트 단위라 진행자 화면에도 적용됩니다.

## 4. 단계

1. 폰 서비스 구현, 폰에서 녹음 파일로 `nntr_room`과 줄이 같은지 확인
2. 앨리스 Issue 등록, 서버 API·진행자 화면 구현, 이 PC 시험 서버에서 확인
3. alice-dev 배포(사용자), 진행자 폰 Chrome으로 실제 회의 확인

## 5. 진행 기록

### 0. 가능성 확인 (2026-11-02)

- 폰 Chrome 156(Android 15)에서 `https://dog.agentic-fw.com/alice-dev/` 탭이 `http://127.0.0.1:8031`을 부르자
  권한 창이 뜸. 허용 전 요청은 Chrome 안에서 기다리다 허용과 함께 도착
- 허용 뒤: GET 200(5 ms), PCM 0.2초 POST 200(5 ms, 6,400바이트 그대로), `localhost`도 같음.
  POST는 일반 CORS 사전 확인(OPTIONS)을 거침. 권한 상태 `local-network-access`·`loopback-network` = granted

### 1. 폰 서비스 (2026-11-02)

- `room_http.cpp` → `nntr_room_http` (ndk-build, `libs/arm64-v8a/`, 기존 `libnntr_discretion_jni.so` 그대로 사용)
- 모델 올림 1.4~1.8초(Galaxy S25+). 새 회의마다 인식기를 새로 만들어도 부담 없음
- boda 녹음(약 2분)을 0.2초씩 `/feed`로 넣고 `/flush`·`/poll`: 줄 16개가 같은 파일의 `nntr_room` 결과와
  글·화자·시각까지 모두 같음. 빨리 넣으면 `rc 1`(바쁨)이 57번 나왔고 다시 넣어 빠짐 없음
- 허용하지 않은 출처는 403, 사전 확인(OPTIONS)은 204와 그 출처만 허용하는 CORS 헤더

### 2. 앨리스 연결 (2026-11-02, ALICE Issue #82, 브랜치 `feat/82-phone-browser-asr`)

- 서버 `POST /api/audio/compare-lines`(진행자 열쇠), 진행자 화면이 폰 서비스를 스스로 찾음(`ambient/ui/shared/phone-asr.js`),
  ASR 비교 탭에 권한 버튼. 처음 시험은 `?phone=8031`로 함
- 이 PC 시험 서버(8847, 서버 ASR은 로컬 Whisper small, 언어 모델 없음) + 폰 Chrome으로 처음부터 끝까지:
  진행자 화면을 `https://127.0.0.1:8847/host?phone=8031`로 열고(adb reverse), 마이크 대신 boda 60초를 넣는
  가짜 `getUserMedia`로 회의 시작 → 종료
  - 폰 줄 7개가 ASR 비교 줄로 저장됨. 각 줄의 시작·끝이 서버 자신의 줄과 ms까지 같음(같은 구간 규칙·Silero)
  - 12.7~21.7초 구간은 서버·폰 모두 줄이 없음: 화면이 받은 소리 쪽 문제로 보고 폰 경로 문제로 보지 않음
  - 글은 파일을 바로 넣은 결과와 조금 다름(브라우저가 소리를 다시 표본화함)
- 미실행: 실제 폰 마이크, alice-dev(공개 주소, 권한 창 포함) 위 회의
- 주소 없이 켜기: 127.0.0.1에 있는 페이지는 Chrome이 권한을 "prompt"라고 답해도 루프백 호출이 창 없이 됨 →
  그런 페이지는 허용된 것으로 봄. 8847 진행자 화면에서 자동으로 8031을 찾음 확인
