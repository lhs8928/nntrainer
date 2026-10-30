# Silero VAD를 저장소 안 C++로 옮기기

- 요청: 2026-10-30 사용자 ("Silero VAD를 nntrainer로 구현해 보자")
- 목적: 폰의 음성 구간 판정(Silero VAD v6)을 ONNX Runtime 대신 이 저장소의 코드로 돌림
  - 폰 라이브러리에서 ONNX Runtime(`libonnxruntime.so`, 약 18MB)을 뺄 수 있게 함
  - 저장소 밖에서 가져오는 ML 부분을 없앰
- 완료 기준: 같은 소리에서 ONNX Runtime과 블록별 말소리 확률이 같고(최대 차이 1e-5 이하), 폰 전체 결과(week1 38줄, boda 16줄)가 줄 단위로 같음

## 모델 (폰의 `silero_vad_v6.onnx`, opset 18, 파라미터 309,652개, float32)

32ms(512샘플)마다 한 번 실행한다. 앞 블록 끝 64샘플을 문맥으로 붙여 576샘플을 넣는다.

```
입력 [1, 576]
 → Pad(reflect) → STFT 합성곱 (필터 258 = 실수 129 + 허수 129, 길이 256, 간격 128)
 → 크기 √(실수² + 허수²) [129 × T]
 → Conv1d 129→128 (k3, p1, s1) + ReLU
 → Conv1d 128→64  (k3, p1, s2) + ReLU
 → Conv1d 64→64   (k3, p1, s2) + ReLU
 → Conv1d 64→128  (k3, p1, s1) + ReLU
 → LSTM 128 (상태 h·c를 다음 블록으로 넘김)
 → ReLU → Conv1d 128→1 (k1) → Sigmoid → 확률
```

## 방식

- SD 모델들(`Applications/SpeakerDiarization/models/`)과 같이 C++로 직접 구현한다. 행렬곱은 nntrainer의 `__cblas_sgemm`을 쓴다.
- 가중치는 ONNX에서 꺼내 기존 가중치 파일 형식(`weight_loader`, `split_sd_weights.write`)으로 저장한다.
- `voice_gate`의 Silero는 모델 파일로 고른다.
  - `.onnx`: ONNX Runtime으로 빌드했을 때 지금처럼 ONNX Runtime 사용
  - 그 밖(`.bin`): 저장소 구현 사용. ONNX Runtime 없이 빌드해도 Silero가 돈다
  - 그래서 앨리스 쪽 설정(`silero` 키)은 파일 이름만 바꾸면 된다

## 단계

| # | 할 일 | 상태 |
|---|---|---|
| 1 | 기준 만들기: ONNX Runtime으로 블록별 확률 기록 (합성 신호, week1, boda) | 완료 |
| 2 | 변환 스크립트 `Applications/SpeakerDiarization/export_silero_vad.py` | 완료 |
| 3 | C++ 구현 `Applications/SpeakerDiarization/models/silero_vad.{h,cpp}` | 완료 |
| 4 | PC 대조: 1의 기준과 블록별 확률 비교 | 완료 |
| 5 | `voice_gate` 연결 (`.onnx` → ONNX Runtime, `.bin` → 저장소 구현) | 완료 |
| 6 | 빌드: Linux(meson), Android(ONNX Runtime 있음·없음 둘 다) | 완료 |
| 7 | 폰 대조: `nntr_room`으로 week1·boda, ONNX Runtime 판과 줄 비교, 처리 시간 | 완료 |
| 8 | 문서(`android-app.ko.md` 모델 표)와 커밋 | 완료 |

## 진행 기록

### 2026-10-30 1단계: 기준 만들기
- 폰의 모델 파일(`silero_vad_v6.onnx`, sha256 `4cbf549b…`)과 녹음 두 개(week1, boda)를 폰에서 가져옴
- 중간 모양을 ONNX Runtime으로 직접 확인 (블록 하나)
  - 패딩: 양쪽 128씩 반사 → 832샘플
  - STFT 합성곱: 5프레임. 그중 **첫 프레임은 버림**(Slice가 프레임 1부터) → 129 × 4
  - 합성곱 4개를 지나 4 → 4 → 2 → 1 → 1프레임. 그래서 **LSTM은 블록마다 정확히 1스텝**
- 기준 기록: `SileroGate`와 똑같이(int16/32768, 앞 문맥 64샘플, 상태 이어 받기) 블록별 확률을 float32로 저장
  - 합성 신호 30초(잡음, 무음, 진폭이 변하는 사인파, 클리핑): 937블록
  - week1: 10,960블록 (0.5 이상 9,107)
  - boda: 3,750블록, 최대 0.39 (조용한 녹음, RMS 0.0087). 앨리스 기준이 0.3이라 boda도 말소리로 잡힘. 기준 근처 값이라 대조 시험에 좋음
- 도구: 세션 작업 폴더의 `reference.py` (저장소 밖, 시험용)

### 2026-10-30 2단계: 변환 스크립트
- `export_silero_vad.py`: 쓰기 전에 그래프가 예상과 같은지 확인함(패딩 방식·길이, STFT 필터 모양과 간격, 자르는 위치, 합성곱 4개의 모양·간격·패딩, LSTM 크기·방향). 다르면 거절
- 텐서 15개, 파라미터 309,633개(ONNX의 309,652개 중 19개는 패딩·자르기 위치 정수), 파일 1.24MB
  - `stft.basis` [258, 256], `conv0~3.weight/bias`, `lstm.w`·`lstm.r` [512, 128](ONNX 게이트 순서 i·o·f·c 그대로), `lstm.wb`·`lstm.rb` [512], `out.weight` [128], `out.bias` [1]
  - 맨 앞에 `meta.model_manifest`: 모델 이름·버전, ONNX 파일 sha256, 블록·문맥 크기

### 2026-10-30 3단계: C++ 구현
- `SileroVad` (`models/silero_vad.{h,cpp}`): `load(파일)`, `run(512샘플)` → 확률, `reset()`, `manifest()`
  - 앞 문맥 64샘플과 LSTM 상태 h·c를 객체가 들고 다음 호출로 넘김 (`SileroGate`의 ONNX Runtime 판과 같은 방식)
  - 순서: 반사 패딩 128 → STFT 4프레임(첫 프레임 버림) → 크기 → 합성곱 4개 + ReLU → LSTM 1스텝(ONNX 게이트 순서 i·o·f·c, 두 bias를 읽을 때 더함) → ReLU → 1×1 합성곱 → sigmoid
  - 한 블록 계산이 약 75만 번 곱셈·덧셈이라 BLAS 없이 단순 반복문으로 씀. nntrainer 다른 부분에 기대지 않고 `weight_loader`만 씀

### 2026-10-30 4단계: PC 대조
- 시험 도구: 세션 작업 폴더의 `compare.cpp`(저장소 밖). 같은 소리를 512샘플씩 넣어 1단계 기준과 블록마다 비교
- 시험 도구 실수 두 개를 먼저 잡음
  - 기준 파일을 한 바이트씩 float로 읽음 → 차이 129라는 말이 안 되는 값
  - WAV 머리를 44바이트로 가정 → week1은 `LIST` 묶음이 있어 소리가 78바이트부터. `data` 위치를 찾게 고침
- 결과 (float로 더했을 때 → double로 더했을 때)

  | 소리 | 블록 | 최대 차이 | 평균 차이 | 0.3·0.5 기준으로 갈린 블록 |
  |---|---|---|---|---|
  | 합성 신호 30초 | 937 | 5.8e-6 → **2.6e-6** | 1.4e-7 | 0 |
  | week1 (5.8분) | 10,960 | 1.7e-5 → **5.0e-6** | 6.1e-8 | 0 |
  | boda (2분, 조용함) | 3,750 | 2.3e-6 → **1.8e-6** | 1.4e-7 | 0 |

- float로 더하면 week1에서 1.7e-5까지 벌어져 완료 기준(1e-5)을 넘음 → 합성곱·STFT·LSTM의 합을 double로 바꿈. 이 PC에서 블록당 0.23 → 0.29ms (블록은 32ms마다)

### 2026-10-30 5단계: `voice_gate` 연결
- `SileroGate`의 내부를 "512샘플 → 확률" 하나로 묶고 두 구현을 둠
  - `.onnx` 파일: ONNX Runtime(지금과 같은 코드, ONNX Runtime으로 빌드했을 때만)
  - 그 밖의 파일: `SileroVad`(저장소 구현)
  - ONNX Runtime 없이 빌드했는데 `.onnx`를 주면 "`export_silero_vad.py`로 만든 .bin을 주라"는 오류. 지금처럼 에너지 판정으로 대신함(`makeVoiceGate`)
- 버퍼링과 켜짐·꺼짐 판정(기준, 기준 - 0.15)은 두 구현이 같이 씀. 바뀐 것은 모델 계산뿐
- 빌드 파일: `meson.build`, `jni/Android.mk`에 `silero_vad.cpp` 추가, 머리말 설명 고침

### 2026-10-30 6단계: 빌드
- Linux: `build_chk_rpi5`(aarch64 교차 빌드)에서 `nntr_room` 빌드 성공
- Android (NDK 27, `Applications/discretion`에서 `ndk-build`, 결과는 `Applications/discretion/libs/`)
  - ONNX Runtime 있음: 성공. `libnntr_discretion_jni.so` 1,880,040바이트 + `libonnxruntime.so` 18,214,224바이트
  - ONNX Runtime 없음(`ORT_ROOT`를 없는 경로로): 성공. `libnntr_discretion_jni.so` 1,874,152바이트, 필요한 라이브러리 목록에 ONNX Runtime 없음
- 참고: `jni/libs`, `jni/obj`는 예전 빌드가 남은 것(10-28). 지금 빌드 결과는 `libs/`, `obj/`

### 2026-10-30 7단계: 폰 대조 (Galaxy S25+)
- 폰 폴더에 올린 것
  - 새 `libnntr_discretion_jni.so`·`nntr_room`(ONNX Runtime 있음 빌드). 예전 것은 `*.1028`로 보관
  - ONNX Runtime 없는 빌드는 `noort/` 하위 폴더에
  - `silero_vad_v6.bin`
- 실행: `nntr_room`으로 녹음 파일을 최대 속도로 처리, 목소리 모델 ERes2NetV2, 설정에서 `silero`만 바꿈

  | 녹음 | Silero | 줄 | `.onnx`와 비교 | 전체 시간 | ASR 시간 합 |
  |---|---|---|---|---|---|
  | week1 | `.onnx` (ONNX Runtime) | 38 | – | 261 s | 121.7 s |
  | week1 | `.bin` (저장소 구현) | 38 | **모두 같음** | 288 s | 132.5 s |
  | week1 | `.bin`, ONNX Runtime 없는 빌드 | 38 | **모두 같음** | 298 s | 135.5 s |
  | boda | `.onnx` | 16 | – | 85 s | 42.9 s |
  | boda | `.bin` | 16 | **모두 같음** | 87 s | 43.1 s |
  | boda | `.bin`, ONNX Runtime 없는 빌드 | 16 | **모두 같음** | 88 s | 43.6 s |

  - "모두 같음": 줄 번호·화자·시작·끝·글·의심 표시와 화자 판단 정보가 모두 같고, 다른 것은 ASR 처리 시간(`info.asr_ms`)뿐
  - 모든 실행에서 소리 문은 `silero`
- 시간 차이는 VAD 때문이 아님
  - VAD와 상관없는 ASR 시간 합부터 `.bin` 실행에서 11~14초 더 걸림. 실행 순서가 `.onnx` → `.bin` → 없는 빌드라 폰이 데워진 탓으로 봄
  - 저장소 VAD만 따로 잼(대조 도구를 폰용으로 빌드): 블록당 **0.27ms** → week1 전체 10,960블록에 약 3초(전체 시간의 약 1%). 같은 실행에서 ONNX Runtime 기준과의 최대 차이 5.2e-6, 기준으로 갈린 블록 0

### 2026-10-30 8단계: 문서와 커밋
- `android-app.ko.md`: `libonnxruntime.so`는 넣지 않아도 됨(`.onnx`를 줄 때만 필요), `silero` 기본 파일을 `silero_vad_v6.bin`으로, 만드는 방법에 `export_silero_vad.py` 추가, 설정 예시도 `.bin`으로
- 커밋 2개(nntrainer `discretion_alice_v2`, push 안 함)
  - `[SpeakerDiarization]` 모델과 변환 스크립트
  - `[Applications/discretion]` 소리 문 연결, 빌드, 문서

### 2026-10-30 7-2단계: 소리 문을 거친 직후 값 비교 (폰, 사용자 요청)
- 이유: 7단계는 파일 전체 결과(줄)로 본 것이라, 소리 문 바로 뒤의 값을 직접 비교함
- 소리 문 바로 뒤의 값
  - ① 모델 출력: 512샘플(32ms) 블록마다 말소리 확률
  - ② 소리 문 판정: 파이프라인이 넘기는 20ms(320샘플) 프레임마다 켜짐·꺼짐 (기준 이상이면 켜짐, 켜진 뒤 기준 - 0.15 아래로 갈 때까지 유지)
- 방법: 폰용 시험 도구(세션 작업 폴더의 `vad_dump.cpp`, 저장소 밖)가 실제 `voice_gate` 코드의 `makeVoiceGate`로 두 소리 문을 만듦
  - `.onnx` → ONNX Runtime, `.bin` → 저장소 구현
  - 같은 20ms 프레임을 차례로 넣고, 프레임마다 두 쪽의 마지막 블록 확률과 켜짐·꺼짐을 CSV로 기록
- 결과: **모든 프레임에서 켜짐·꺼짐이 같음**

  | 소리 | 프레임 | 기준 | 켜진 프레임 (onnx / bin) | 다른 프레임 | 말소리 구간 수 | 확률 최대 차이 |
  |---|---|---|---|---|---|---|
  | week1 | 17,536 | 0.3 | 15,327 / 15,327 | 0 | 109 / 109 | 6.8e-6 |
  | boda | 6,000 | 0.3 | 23 / 23 | 0 | 2 / 2 | 1.2e-6 |
  | 합성 신호 | 1,500 | 0.3 | 530 / 530 | 0 | 4 / 4 | 2.9e-6 |
  | week1 | 17,536 | 0.5 | 14,789 / 14,789 | 0 | 201 / 201 | 6.8e-6 |
  | boda | 6,000 | 0.5 | 0 / 0 | 0 | 0 / 0 | 1.2e-6 |
  | 합성 신호 | 1,500 | 0.5 | 504 / 504 | 0 | 3 / 3 | 2.9e-6 |

- 함께 본 것: boda(조용한 녹음)는 Silero가 0.46초만 말소리로 봄. 그런데도 16줄이 나오는 것은 앨리스의 "조용한 창" 규칙 때문
  - 규칙: 8초 동안 소리 문이 한 번도 안 열려도 바닥 소음보다 큰 소리가 있었으면 그 구간을 통째로 인식에 보냄(`quietWindow`, 앨리스 `_quiet_window` 이식)
  - 두 모델 모두 같고 앨리스 서버도 같은 동작. 이번 작업과는 별개지만, 조용한 방에서는 Silero보다 이 규칙이 줄을 만든다는 점은 알아 둘 만함

## 남은 것
- 앨리스 쪽: `tools/phone_relay.py`의 기본 `silero` 값이 아직 `silero_vad_v6.onnx`. `.bin`으로 바꿀지는 앨리스 PR로 따로 정함(지금 폰 빌드는 ONNX Runtime이 있어 둘 다 됨)
- 실시간 마이크(`nntr_room mic`)로는 아직 안 돌려 봄. 녹음 파일과 같은 코드 경로지만 실제 회의에서 한 번 확인 필요
