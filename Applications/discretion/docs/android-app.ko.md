# 회의실 폰 앱 개발 안내 (ALICE 1단계)

진행자 폰 한 대가 회의실 마이크가 되는 안드로이드 앱을 만드는 사람을 위한 문서다.
폰이 회의 소리를 직접 인식하고 화자를 나눈 뒤, ALICE 서버에는 **글 줄만** 보낸다.
소리는 폰 밖으로 나가지 않는다.

앱이 할 일은 지금 PC의 시험 도구 `tools/phone_relay.py`(ALICE 저장소)가 adb로 대신하고 있다.
이 문서는 그 일을 앱 안에서 하는 방법을 적는다.

```
 ┌──────────── 진행자 폰 (앱) ─────────────┐          ┌──── ALICE 서버 ────┐
 │ AudioRecord ─▶ Recognizer.feed()        │          │                    │
 │                  │ (폰 안에서 인식·화자) │  글 줄   │ 회의·카드·회의록   │
 │ Recognizer.poll() ─▶ line 이벤트 ───────┼────────▶ │ POST /lines        │
 │                                         │          │                    │
 │ mapSpeaker / pin / enrollVector ◀───────┼───────── │ GET /speaker-ops   │
 │                                         │  수정    │ (사람의 화자 수정)  │
 └─────────────────────────────────────────┘          └────────────────────┘
```

## 1. 준비물

### 1.1 라이브러리 (arm64-v8a)

`Applications/discretion/jni`에서 `ndk-build`로 만든다(빌드 조건은 `jni/Android.mk` 머리말).
앱의 `jniLibs/arm64-v8a/`에 넣을 파일과 크기는 다음과 같다.

| 파일 | 크기 | 설명 |
|---|---|---|
| `libnntr_discretion_jni.so` | 1.7 MB | 인식기와 JNI 함수 |
| `libquick_ai_core.so` | 81 MB | Qwen3-ASR |
| `libnntrainer.so`, `libccapi-nntrainer.so` | 5 MB | nntrainer |
| `libonnxruntime.so` | 17 MB | **넣지 않아도 된다.** `silero`에 `.onnx`를 줄 때만 필요(ONNX Runtime으로 빌드했을 때). 기본 `.bin`은 인식기가 직접 돌린다 |
| `libc++_shared.so` | 1.7 MB | NDK C++ 런타임 |

Java 클래스는 `java/ai/nntrainer/discretion/Recognizer.java` 하나다.
패키지 이름 `ai.nntrainer.discretion`을 바꾸면 JNI 함수 이름이 맞지 않으니 그대로 둔다.

### 1.2 모델 파일 (약 1.2 GB)

앱 전용 폴더(예: `getFilesDir()`)에 복사해 두고 경로를 설정에 넘긴다.

| 설정 키 | 파일 | 크기 |
|---|---|---|
| `model_dir` | `Qwen3-ASR-1.7B/` 폴더 전체 | 1.1 GB |
| `segmentation_weights` | `sd_segmentation.bin` (PyanNet: 구간 안에서 화자가 바뀌는 곳) | 6 MB |
| `speaker_weights` | 목소리 모델 하나: `speaker_eres2netv2.bin` (기본, 앨리스 서버와 같은 ERes2NetV2) 또는 `speaker_wespeaker.bin` (WeSpeaker ResNet34) | 68 MB / 34 MB |
| `silero` | `silero_vad_v6.bin` (Silero VAD v6, 인식기가 직접 돌림) 또는 `silero_vad_v6.onnx` (ONNX Runtime 필요) | 1.2 MB |

목소리 모델은 넘긴 파일로 정해진다(파일 안의 텐서로 알아본다, 따로 고르는 설정은 없다).

| 목소리 모델 | 목소리 숫자 | 화자 묶기 기준(기본) | 속도 (Galaxy S25+) | 비고 |
|---|---|---|---|---|
| ERes2NetV2 (`speaker_eres2netv2.bin`) | 192개 | 0.66 | 소리 1초에 약 0.2초 | 같은 소리에서 서버와 같은 숫자(cosine 1.0000). float32 |
| WeSpeaker ResNet34 (`speaker_wespeaker.bin`) | 256개 | 0.6 | 줄 하나에 약 55 ms | 8비트. 6명 TTS 회의에서는 화자 구분이 더 좋았다(95.8% 대 69.7%) |

- 모델이 다른 목소리 숫자는 서로 섞어 쓸 수 없다. 아래 표의 "256개"는 ERes2NetV2면 192개다.
- 파일 만들기(nntrainer 저장소 `Applications/SpeakerDiarization`):
  - `split_sd_weights.py speaker_diarization.bin 폴더`: 예전 한 파일을 `sd_segmentation.bin`과 `speaker_wespeaker.bin`으로 나눈다(텐서는 바이트 그대로).
  - `export_eres2netv2.py speaker-eres2netv2.onnx speaker_eres2netv2.bin`: 앨리스 서버의 모델 파일에서 만든다.
  - `export_silero_vad.py silero_vad_v6.onnx silero_vad_v6.bin`: Silero 모델 파일에서 만든다. 같은 소리에서 ONNX Runtime과 블록 확률 차이 5e-6 이하, 폰에서 week1·boda의 줄이 모두 같았다(`docs/silero-vad.ko.md`).
- 예전 한 파일(`sd_weights`: `speaker_diarization.bin`, PyanNet + WeSpeaker)도 계속 받는다. 두 파일 대신 주면 WeSpeaker로 동작한다.

시험 폰의 `/data/local/tmp/nntrainer/discretion/models/Qwen3-ASR-1.7B/` 안에는 `.safetensors`가 다른 폴더를 가리키는 링크로 들어 있다.
앱에 넣을 때는 링크가 아니라 실제 파일을 복사해야 한다.

### 1.3 서버 접속 정보

| 항목 | 받는 곳 |
|---|---|
| 서버 주소 | 예: `https://alice-dev.example` |
| 장치 토큰 | 서버 관리자가 `python -m ambient.admin register-device <회의실 ID>`로 발급한다. 한 번만 보이니 앱 설정에 안전하게 보관한다(Android Keystore 권장). |
| CA 인증서 | 시험 서버처럼 사설 인증서를 쓸 때만 필요하다(`ambient.devcert`가 만든 `ca.pem`). 실제 배포는 공인 인증서라 필요 없다. |

장치 토큰 하나가 회의실 하나에 묶인다. 앱은 모든 요청에 `Authorization: Bearer <토큰>` 헤더를 붙인다.

## 2. 인식기 (`Recognizer`) 사용법

### 2.1 만들기

```java
String config = new JSONObject()
    .put("model_dir", dir + "/Qwen3-ASR-1.7B")
    .put("segmentation_weights", dir + "/sd_segmentation.bin")
    .put("speaker_weights", dir + "/speaker_eres2netv2.bin")   // WeSpeaker: speaker_wespeaker.bin
    .put("silero", dir + "/silero_vad_v6.bin")
    .put("language", "Korean")
    .put("labels_in_use", labelsFromStartReply)   // 3.2 참고, 없으면 빈 배열
    .toString();
Recognizer r = new Recognizer(config);   // 모델을 읽는다(수 초). 실패하면 IllegalStateException
```

나머지 설정(`asr_secs`, `threads`, `vad`, `silero_threshold`, `speaker_threshold`)은 기본값을 쓰면 된다.
뜻은 `room_api.h` 머리말에 있다. `speaker_threshold`의 기본값은 목소리 모델을 따라간다(1.2의 표).

만드는 데 몇 초가 걸리니 UI 스레드에서 부르지 않는다.

### 2.2 소리 넣기

```java
AudioRecord rec = new AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION,
    16000, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufBytes);
short[] buf = new short[1600];             // 100 ms
while (listening) {
  int n = rec.read(buf, 0, buf.length);
  if (n > 0 && r.feed(buf, n, -1) == 1) {  // 1 = 바쁨: 같은 조각을 잠시 뒤 다시 넣는다
    ...
  }
}
```

- 형식은 16 kHz, 모노, PCM16이다. 시험 도구도 `VOICE_RECOGNITION` 입력으로 100 ms씩 넣는다.
- `feed`는 막히지 않는다. 인식은 라이브러리 안의 작업 스레드가 한다.
- 인식이 60초 넘게 밀리면 `feed`가 1(바쁨)을 돌려준다. 그 조각을 버리지 말고 보관했다가 다시 넣는다.
- 세 번째 인자 `atMs`는 그 조각이 회의 시계(회의 시작부터 ms)의 어디인지다. `-1`이면 앞 조각 바로 뒤로 본다.
  앱이 회의 중에 다시 시작했을 때는 첫 조각에 실제 경과 시간을 넣어야 줄 시각이 이어진다(3.6 참고).
- 회의 중에 소리를 계속 받으려면 마이크 권한과 `microphone` 유형의 포그라운드 서비스가 필요하다.

### 2.3 결과 받기

0.2~0.5초마다 `poll()`을 부른다. 지난번 이후의 이벤트가 JSON 배열로 온다.

```json
[
  {"type":"line","id":"audio-3","speaker":"화자 1","start_ms":12700,"end_ms":22600,
   "text":"그러면 이걸 펴서 생각해 보죠.","doubt":[],"info":{"asr_ms":1208,"...":"..."}},
  {"type":"moved","moved":{"audio-1":"화자 2"}},
  {"type":"enrolled","person":"김철수","ok":true}
]
```

| 이벤트 | 앱이 할 일 |
|---|---|
| `line` | 서버로 보낼 줄(3.3). `id`는 서버 줄 번호와 짝지어 둔다(3.4). |
| `moved` | 사람이 고친 것을 폰이 반영했다는 메아리다. 서버에 이미 반영돼 있으니 보내지 않는다. |
| `enrolled` | 회의 중 목소리 등록 결과. 서버에 알린다(3.5). |
| `error` | 기록만 한다. |

- `speaker`가 `null`이면 화자를 정하지 못한 줄이다. 그대로 보낸다.
- `doubt`는 화자가 틀렸을 수 있는 이유다(`alone`, `merged`, `overlap`, `short` 등). 서버가 이 줄 하나만으로 사람을 정하지 않게 하니 그대로 보낸다.
- `info`는 진단용이다. 서버에는 보내지 않는다.

### 2.4 사람의 수정과 목소리 넣기

| 메서드 | 언제 |
|---|---|
| `mapSpeaker("화자 2", "김철수")` | 서버 화면에서 누가 "이건 나예요"를 눌렀을 때. 이후 그 목소리 줄에 이름이 붙는다. |
| `pin("audio-20", "김철수")` | 줄 하나를 고쳤을 때. 사람을 `null`로 주면 "아무도 아님"이다. `false`면 모르는 줄이니 무시한다. |
| `enrollVector("김철수", float[256])` | 서버가 그 사람의 저장된 목소리를 줄 때. 메모리에만 둔다. |
| `beginEnrollment("김철수", 6)` | 회의 중 목소리 등록. 다음 6초가 그 사람 목소리다(받아 적지 않음). `false`면 다른 사람이 등록 중이다. |
| `voiceOf("김철수")` | 회의 끝에 그 사람의 목소리(256개 숫자). `flush()` 뒤에 부른다. 없으면 `null`. |

모든 메서드는 다른 스레드에서 불러도 된다(시험 도구도 수정은 별도 스레드에서 넣는다).

### 2.5 끝내기

```java
r.flush();               // 들은 것을 모두 인식할 때까지 기다린다(수 초, UI 스레드 금지)
String last = r.poll();  // 마지막 줄들
// 기억할 사람들의 voiceOf(...)
r.close();
```

## 3. 서버 장치 API

모든 요청은 HTTPS이고, 본문은 JSON이며, `Authorization: Bearer <장치 토큰>`을 붙인다.
토큰이 틀리면 401이 온다.

| 요청 | 언제 | 본문 → 응답 |
|---|---|---|
| `POST /api/device/meeting/start` | 듣기 시작 | `{}` 또는 `{"model_manifest": …}` → 3.2 |
| `POST /api/device/lines` | 줄이 생길 때마다 | 3.3 |
| `GET /api/device/speaker-ops?after=k` | 0.5초마다 | 3.4 |
| `POST /api/device/enrolled` | `enrolled` 이벤트, 등록 실패 | 3.5 |
| `POST /api/device/voices` | 회의 끝, end 전에 | 3.6 |
| `POST /api/device/meeting/end` | 회의 끝 | `{}` → `meeting_id`와 끝내기 상태(`phase`가 `"ending"`이면 아직 정리 중) |
| `GET /api/device/status` | 상태 확인 | `phase`(`lobby`·`meeting`·`ending`·`ended`), `meeting_id`, `next` |

### 3.1 회의 한 번의 순서

```
0. manifest = Recognizer.modelManifest(speaker_weights 경로)   (모델을 읽지 않음, 3.9)
1. POST meeting/start {"model_manifest": manifest}   (manifest가 ""이면 {})
                                 → meeting_id, next, labels_in_use, ops_next, voices, remember
   409이면: 서버의 목소리 모델이 다르다. speaker_wespeaker.bin으로 바꾸고 {}로 다시 부른다
2. Recognizer 만들기 (labels_in_use 넣어서)
3. voices의 각 사람: enrollVector(person, vector)
   remember의 각 사람: "회의 끝에 목소리를 돌려줄 사람" 목록에 넣기
4. 회의 중 (동시에)
   - 마이크 → feed
   - poll → line 이벤트 → 번호 붙여 POST lines (n = next부터 1씩)
   - GET speaker-ops?after=ops_next → 수정 적용, after를 응답의 next로
5. 끝
   - flush → poll → 남은 줄 POST lines, 서버가 다 받을 때까지(응답 next 확인)
   - remember 사람마다 voiceOf → POST voices
   - POST meeting/end, phase가 "ending"이면 status를 0.5초마다 물어 끝날 때까지 기다린다
```

진행자가 서버 화면에서 회의를 먼저 열었으면 `start`는 그 회의에 이어 붙는다.
앱이 회의를 끝내지 않고 진행자가 화면에서 끝내게 할 수도 있다(시험 도구의 `--keep-open`).

### 3.2 `meeting/start` 응답

```json
{"meeting_id":"meeting-c10d7f9d4d0505df","phase":"meeting","next":0,
 "labels_in_use":[],"ops_next":0,
 "voices":[{"person":"user1","vector":[0.08,0.21,"... 256개"]}],
 "remember":["user1"]}
```

| 칸 | 뜻 |
|---|---|
| `next` | 다음에 보낼 줄 번호. 새 회의면 0이다. |
| `labels_in_use` | 이 회의에서 이미 보인 화자 라벨. 앱이 다시 시작했을 때 새 목소리가 같은 라벨을 받지 않게 한다. |
| `ops_next` | 수정 번호의 시작. 이보다 뒤의 수정만 받는다. |
| `voices` | 참석자의 저장된 목소리. 폰 목소리 모델의 숫자(ERes2NetV2 192개, WeSpeaker 256개). `enrollVector`로 넣는다. |
| `remember` | 목소리 기억을 켠 사람. 회의 끝에 이 사람들의 `voiceOf`를 돌려준다. |

### 3.3 줄 보내기 `POST /api/device/lines`

```json
{"meeting_id":"meeting-c10d7f9d4d0505df",
 "lines":[{"n":0,"speaker":"화자 1","start_ms":900,"end_ms":3500,"text":"시작할게요.","doubt":["alone"]},
          {"n":1,"speaker":"화자 2","start_ms":4300,"end_ms":11600,"text":"네."}]}
```
응답은 `{"accepted":2,"next":2,"meeting_id":..}`이다.

- `n`은 `start`의 `next`부터 줄마다 1씩 늘린다. 한 번 붙인 번호는 바꾸지 않는다.
- 서버는 `next`번 줄만 받는다. 이미 받은 번호는 무시하고, 번호가 비면 그 뒤를 받지 않는다.
  그래서 네트워크가 끊겼을 때는 보낸 줄을 지우지 말고 같은 묶음을 다시 보내면 된다. 중복도 빠짐도 생기지 않는다.
- 응답의 `next`보다 작은 번호는 서버가 받은 것이니 지운다.
- 실패하면 0.5초부터 최대 10초까지 늘려 가며 다시 보낸다. 한 번에 500줄까지 보낼 수 있다(시험 도구는 50줄).
- 409는 회의가 끝났거나 바뀌었다는 뜻이다. 남은 줄을 버리고 멈춘다.
- 400이면 줄 형식이 틀린 것이다. 서버 검사 기준은 다음과 같다.
  - `text`: 1~2000자
  - `speaker`: `null` 또는 1~40자
  - `start_ms`, `end_ms`: 정수, `0 <= start_ms <= end_ms`
  - `doubt`: 8개까지, 각 20자까지

### 3.4 사람의 수정 받기 `GET /api/device/speaker-ops?after=k`

```json
{"meeting_id":"meeting-c10d7f9d4d0505df","next":3,
 "ops":[{"k":1,"op":"map_speaker","label":"화자 2","person":"user1"},
        {"k":2,"op":"pin","n":7,"person":"user1"},
        {"k":3,"op":"record","person":"user1","seconds":6}]}
```

| `op` | 앱이 할 일 |
|---|---|
| `map_speaker` | `mapSpeaker(label, person)` |
| `pin` | 서버 줄 번호 `n`을 폰 줄 ID로 바꿔 `pin(id, person)`. 모르는 번호(앱 재시작 전의 줄)는 버린다. |
| `enroll` | `enrollVector(person, vector)` |
| `remember` | `on`이 true면 "끝에 목소리를 돌려줄 사람"에 넣고, false면 뺀다. |
| `record` | `beginEnrollment(person, seconds)`. `false`면 바로 3.5의 실패를 보낸다. |

- 다음 요청의 `after`는 응답의 `next`로 바꾼다.
- `meeting_id`가 지금 회의와 다르면 무시한다.
- **줄 번호 ↔ 폰 줄 ID 대응**은 앱이 갖고 있어야 한다. 줄을 보낼 때 `n → line.id`를 기록해 둔다(예: `7 → "audio-8"`).
- `pin`의 `person`이 `null`이면 "아무도 아님"이다.

### 3.5 회의 중 목소리 등록 결과 `POST /api/device/enrolled`

```json
{"meeting_id":"meeting-…","person":"user1","ok":true}
{"meeting_id":"meeting-…","person":"user1","ok":false,"error":"다른 참석자가 목소리를 등록 중입니다."}
```

- `enrolled` 이벤트의 `ok`와 `error`를 그대로 보낸다.
- `beginEnrollment`가 `false`일 때도 실패로 보낸다.
- 등록하는 6초 동안은 그 사람 혼자 말해야 한다. 등록 버튼을 누른 사람의 화면에 "다른 분들은 잠시 조용히 해 주세요"라는 안내가 나온다.
- 서버는 30초 동안 결과가 오지 않으면 다른 사람의 등록을 받는다.

### 3.6 회의 끝 목소리 `POST /api/device/voices`

```json
{"meeting_id":"meeting-…","voices":{"user1":[0.08,0.21,"... 256개"]}}
```
응답은 `{"kept":["user1"]}`이다.

- `meeting/end` **전에** 보낸다.
- 서버는 목소리 기억을 켜고 동의한 사람의 것만 저장한다. 다음 회의의 `start` 응답 `voices`로 돌아온다.
- 녹음이 아니라 목소리 특징(숫자 192개 또는 256개)만 오간다.
- ERes2NetV2 폰이면 서버가 그 사람의 기억 목소리 자체를 갱신한다. 서버가 직접 들은 회의에서 쓰는 목소리와 같은 것이다.

### 3.7 앱이 회의 중에 다시 시작했을 때

- `start`를 다시 부르면 같은 회의가 이어진다. `next`, `labels_in_use`, `ops_next`, `voices`가 지금 상태로 온다.
- 새 `Recognizer`를 `labels_in_use`로 만든다. 그래야 새로 잡힌 목소리가 이미 화면에 보인 라벨을 받지 않는다.
- 다시 시작하기 전 줄의 ID는 새 인식기가 모른다. 그 줄을 가리키는 `pin`은 버린다.
- 회의 시계가 이어지게 첫 `feed`의 `atMs`에 회의 시작부터 지난 시간을 넣는다. 앱이 회의 시작 시각을 따로 저장해 둬야 한다.

### 3.8 (선택) ASR 비교: 같은 소리를 서버에도 보내기

ALICE의 **ASR 비교** 탭(ALICE Issue #58)을 쓸 때만 해당한다.
폰이 소리를 서버로 보내 서버 ASR이 정식 전사를 맡고, 폰이 같은 소리로 인식한 줄은 비교용으로만 보낸다.
서버가 소리를 인식하도록 설정돼 있어야 한다(`AMBIENT_ROOM_RECOGNITION=server`).

```
AudioRecord ─┬─▶ Recognizer.feed()  ─▶ line 이벤트 ─▶ POST /api/device/compare-lines (비교용)
             └─▶ POST /api/device/audio/chunk (같은 표본, 서버 ASR이 정식 전사)
```

| 요청 | 내용 |
|---|---|
| `POST /api/device/audio/start` | `meeting/start` 뒤에 한 번. 앱은 소리 흐름 ID(예: UUID)를 하나 정한다. |
| `POST /api/device/audio/chunk` | 본문은 PCM16 원본(5초 이하), 헤더 `X-Audio-Stream: <흐름 ID>`, `X-Audio-Position: <이 조각 첫 표본의 번호>` |
| `POST /api/device/compare-lines` | `{"meeting_id","stream":"<흐름 ID>","lines":[{n,speaker,start_ms,end_ms,text,doubt?}]}`, 번호 규칙은 3.3과 같다. |
| `POST /api/device/audio/stop` | 회의 끝, `meeting/end` 전에 |

- `feed`에 넣은 표본과 `audio/chunk`로 보낸 표본이 같아야 한다. `line` 이벤트의 `start_ms`는 흐름의 0번째 표본부터 센 시간이고, 서버가 그 흐름의 회의 시계 원점만큼 옮겨 놓는다. 줄 경계와 화자 라벨은 맞추지 않는다.
- 서버가 아직 그 흐름의 소리를 받지 못했으면 `compare-lines`가 409이고 응답의 `phase`는 `"meeting"`이다. 잠시 뒤 다시 보낸다. `phase`가 다르면 회의가 끝난 것이다.
- 이 방식에서는 사람의 화자 수정이 폰에 오지 않는다(`speaker-ops` 없음).
- 시험 도구: `nntr_room --pcm-out FIFO`가 인식하는 소리를 그대로 FIFO에 쓰고, ALICE `tools/phone_relay.py --compare`가 그것을 서버로 보낸다.

### 3.9 목소리 모델 알리기 (`model_manifest`, 앨리스 Issue #62)

`speaker_eres2netv2.bin`은 앨리스 서버와 같은 목소리 모델이다. 회의 시작 때 그 사실을 서버에 알리면, 서버에 저장된 목소리(서버 회의에서 등록하거나 기억한 것)를 폰이 그대로 받고, 회의 끝 목소리도 서버 것으로 돌아간다.

- `Recognizer.modelManifest(경로)`(C: `nntr_room_model_manifest`)가 파일 안의 정보를 모델을 읽지 않고 돌려준다. 시험은 `nntr_room --model-manifest speaker_eres2netv2.bin`
- 그 JSON을 그대로 `POST /api/device/meeting/start`의 `model_manifest`로 보낸다. `speech_capabilities`는 보내지 않는다(그건 다른 방식인 v2)
- 서버의 목소리 모델과 다르면 409가 온다. `speaker_wespeaker.bin`으로 인식기를 만들고 본문 `{}`로 다시 시작한다
- 회의 중에 앱이 다시 시작할 때도 같은 `model_manifest`로 부른다. 다른 모델(또는 manifest 없이)로 오면 409다
- WeSpeaker 파일은 manifest가 없다(`""`). 그때는 `{}`로 시작하고 예전처럼 256개 목소리를 주고받는다

## 4. 성능과 확인된 범위

시험 폰은 Galaxy S25+이다.

| 항목 | 값 |
|---|---|
| 모델 읽기 | 1.6초(두 번째부터, 파일 캐시가 있을 때) |
| 처리 속도 | RTF 0.37~0.46(1분 소리를 약 25초에 처리) |
| 줄 하나 인식 | 1.0~2.4초 |
| 녹음 정확도 | 6분 회의(TTS 6명) CER 11.4%, 화자 정확도 95.8%. 2명 실제 대화 120초 CER 22.8%, 화자 86.5% |

**아직 확인하지 않은 것**
- 메모리와 발열, 1시간 넘는 회의
- 배터리(adb 시험은 충전 중이었다)
- 화면 꺼짐, 백그라운드, 네트워크 끊김
- 2026-10-27 실제 회의실 시험에서, 알아듣기 어려운 구간에서 같은 말이 반복되는 줄이 두 번 나왔다. 반복을 잘라 내는 일은 nntrainer 쪽 과제로 남아 있다.

## 5. 참고

| 무엇 | 어디 |
|---|---|
| C API와 이벤트 형식 | `Applications/discretion/room_api.h` |
| Java 클래스 | `Applications/discretion/java/ai/nntrainer/discretion/Recognizer.java` |
| 앱의 일을 대신하는 시험 도구 | ALICE `tools/phone_relay.py` (줄 번호 대응, 수정 받기, 끝 처리) |
| 줄 재전송 | ALICE `ambient/room_client.py`의 `Uplink` |
| 서버 쪽 처리 | ALICE `ambient/web/cloud.py`(`/api/device/*`), `ambient/room.py`(`device_join`, `speaker_ops`) |
| 설계와 결정 | ALICE `docs/findings/phone-recognition-design.md` 8절, `docs/findings/phone-recognition-differences.md` |
| 명령줄 시험 실행 파일 | `nntr_room` (`room_cli.cpp`; `--commands`로 3.4의 수정을 표준 입력으로 받는다) |
