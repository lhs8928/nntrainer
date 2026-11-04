# discretion ASR을 Qwen3-ASR에서 WhisperX(Whisper)로 바꾸기

**목적·독자:** 지금 `Applications/discretion`의 인식(ASR)은 Qwen3-ASR-1.7B 하나에 묶여 있습니다.
이 문서는 그 자리에 WhisperX 방식(Whisper 인식 + 단어 시각 정렬)을 넣기 위해 해야 할 일과 진행 기록입니다.
작업을 이어받는 개발자가 읽습니다.

## 1. "WhisperX로 바꾼다"의 범위

WhisperX(m-bain/whisperX)는 모델 하나가 아니라 아래 다섯 단계를 잇는 파이프라인입니다.
discretion에는 이미 그 중 셋이 있고, **빠진 것은 Whisper 모델 자체와 단어 정렬 모델**입니다.

| WhisperX 단계 | WhisperX가 쓰는 것 | discretion 지금 | 할 일 |
|---|---|---|---|
| 1. 음성 구간(VAD) | pyannote / Silero | `batch_pipeline.cpp` PyanNet Binarize(onset 0.5 / offset 0.363, min-cut), `voice_gate.cpp` Silero v6 | 없음 |
| 2. 구간 묶기(≤ 30초) | `merge_chunks` | `batch_pipeline.cpp` `merge_chunks` 구현 (`chunk_sec` 기본 30) | 없음 |
| 3. 인식 | faster-whisper (CTranslate2, Whisper large-v2/v3) | **Qwen3-ASR-1.7B** (`asr_engine.cpp`) | **Whisper를 nntrainer로 구현** |
| 4. 단어 시각 | wav2vec2 CTC 강제 정렬(언어별 모델) | Qwen3 디코더 11층 attention 최댓값 (`qwen3_asr_causallm.cpp`) | **Whisper용 단어 시각** (5.4절, 5.6절) |
| 5. 화자 붙이기 | pyannote + `assign_word_speakers` | `batch_pipeline.cpp`, `room_diarizer.cpp` | 없음 |

그래서 이 작업은 "nntrainer에 Whisper 인식 모델을 넣고, discretion이 엔진을 골라 쓰게 하는 일"이 중심입니다.

## 2. 지금 ASR이 닿는 곳

엔진은 `AsrEngine`(`asr_engine.h`) 하나이고, 밖에서 쓰는 함수는 넷뿐입니다.

```cpp
bool init(const AsrOptions &opt, std::string &error); // 설정 읽고 그래프 만들고 가중치 올림
size_t segmentSamples() const;                         // 그래프가 받는 길이(고정)
std::string transcribe(std::vector<float> pcm);        // 16 kHz 모노, 정확히 segmentSamples()
std::vector<std::pair<std::string, double>> lastWords() const; // 단어와 구간 안 시각(초)
```

| 파일 | ASR과의 관계 | Whisper로 바꿀 때 |
|---|---|---|
| `asr_engine.h/.cpp` | `quick_ai::Qwen3ASRCausalLM` 생성, Qwen3 프롬프트(`<|audio_pad|>`…, `language X<asr_text>`), 길이별 토큰 예산 | 엔진 구현을 둘로 나눔(5.1절) |
| `asr_lane.h/.cpp` | `AsrOptions`(model_dir, segment_sec 8, language "English", word_timestamps) | `language` 값 변환, `segment_sec` 의미 확인 |
| `batch_pipeline.cpp` | 30초 묶음을 `transcribe` + `lastWords`, 반복 검사(Whisper compression 검사 대신) | Whisper 30초 창과 그대로 맞음 |
| `speech_ops.cpp` | 길이별로 엔진 여러 개(`asr_secs`) | Whisper는 입력이 늘 30초라 엔진 하나면 됨(5.5절) |
| `room_pipeline.cpp`, `room_api.cpp/.h` | `model_dir`, `language`(기본 "Korean"), `asr_provider = "nntrainer:qwen3-asr"` | provider 문자열, 설정 키 추가 |
| `main.cpp`, `room_cli.cpp`, `room_http.cpp`, `room_jni.cpp` | 설정·명령줄 전달 | `--asr-backend` 같은 선택지 |
| `meson.build`, `jni/Android.mk` | `quick_ai/models/qwen3_asr` 소스·헤더 경로 | Whisper 모델 소스 추가 |
| `docs/android-app.ko.md` | 라이브러리·모델 파일 크기(Qwen3 1.1 GB) | 표 갱신 |

Qwen3에만 해당하는 가정(Whisper에서 달라지는 점):

- **언어 이름**: Qwen3은 `"Korean"`, `"English"` 같은 영어 이름을 프롬프트에 넣습니다. Whisper는 `<|ko|>`, `<|en|>` 토큰입니다.
- **입력 길이**: Qwen3은 그래프를 구간 길이(기본 8초)에 맞춰 만듭니다. Whisper 인코더는 30초(멜 3000프레임 → 1500토큰)로 학습되어 짧은 구간도 30초로 채웁니다.
- **단어 시각**: Qwen3은 LLM 디코더가 오디오 토큰을 보는 self-attention에서 뽑습니다. Whisper는 디코더의 cross-attention에서 뽑습니다(openai/whisper `word_timestamps`: 정해진 alignment head + DTW).
- **문맥 없음**: Qwen3도 앞 구간 글을 넣지 않습니다. WhisperX도 `condition_on_previous_text=False`라 같습니다.

## 3. 그대로 쓸 수 있는 것

- **멜 스펙트로그램**: `quick_ai/models/qwen3_asr/audio_preprocessor.cpp`가 이미 Whisper 특징 추출기와 같습니다
  (n_fft 400, hop 160, reflect padding 200, log10·clamp(max−8)·(x+4)/4). `num_mel_bins` 인자로
  large-v3·turbo는 128, tiny~medium·large-v2는 80을 줍니다.
- **nntrainer 레이어**: `conv1d`, `layer_normalization`(bias 있음), `fully_connected`(bias), GELU 활성화가 있습니다.
- **quick_ai**: `CausalLM` 생성 루프, KV 캐시, `mha_core`, HF `tokenizer.json` 읽기, 양자화(Q4_0/QINT8), safetensors 가중치.
- **검증 방식**: `Applications/quick_ai/VERIFICATION_PYTORCH.md`처럼 PyTorch(HF transformers) 출력과 층별 비교.

## 4. 먼저 정할 것

| 항목 | 후보 | 제안 |
|---|---|---|
| Whisper 크기 | large-v3 (1.55B, 디코더 32층) / **large-v3-turbo** (809M, 디코더 4층) / medium (769M) / small (244M) | **large-v3-turbo**. 인코더는 large-v3와 같고 디코더가 4층이라 폰에서 생성이 빠릅니다. 한국어 품질은 7절에서 Qwen3과 비교 |
| 단어 시각 | (a) Whisper cross-attention + DTW / (b) WhisperX처럼 wav2vec2 CTC 강제 정렬 | **(a) 먼저**. 모델이 하나 더 늘지 않고 지금 Qwen3 방식과 비슷합니다. 정확도가 부족하면 (b)를 6단계로 |
| 짧은 구간 | 30초로 채움 / 인코더 길이를 줄임(whisper.cpp `audio_ctx`) | 묶음(batch)은 30초 그대로. 실시간 lane은 7절에서 지연을 재고 정함 |
| 엔진 고르기 | 설정 키 `asr_backend` / `config.json`의 `model_type`으로 자동 | 둘 다: 기본은 자동(`"whisper"`면 Whisper), 키가 있으면 그 값 |

## 5. 작업 단계

### 5.1 엔진 추상화 (동작 변화 없음)

- [ ] `AsrEngine`을 인터페이스로 바꾸고 지금 코드를 `Qwen3AsrEngine`으로 옮김. `asr_lane`, `batch_pipeline`,
      `speech_ops`, `room_pipeline`은 인터페이스만 봄
- [ ] `AsrEngine::create(opt, error)`: `model_dir/config.json`의 `model_type`(또는 `asr_backend`)으로 구현 선택
- [ ] `AsrOptions::language`를 엔진이 해석: Qwen3은 이름 그대로, Whisper는 `"Korean"`→`ko` 표(+ `"ko"` 직접 입력)
- [ ] `asr_provider`를 엔진이 알려 줌(`nntrainer:qwen3-asr` / `nntrainer:whisper-large-v3-turbo`)
- [ ] 확인: 녹음 파일(week1, boda)로 바꾸기 전과 줄이 같음

### 5.2 Whisper 가중치 변환

- [ ] `Applications/quick_ai/res/whisper/whisper-large-v3-turbo/` 아래 `weight_converter.py`, `nntr_config.json`
      (HF `openai/whisper-large-v3-turbo` safetensors → nntrainer 순서·이름, 인코더/디코더 dtype 따로)
- [ ] `tokenizer.json`, `generation_config.json`(`alignment_heads`, `suppress_tokens`, `lang_to_id`) 함께 둠
- [ ] 양자화 조합 정함: 인코더 FC는 QINT8/Q4_0, conv는 FP32부터

### 5.3 Whisper 모델 구현 (`Applications/quick_ai/models/whisper/`)

- [ ] **인코더**: conv1d(k3,s1)+GELU → conv1d(k3,s2)+GELU → 고정 사인 위치 임베딩 더함 →
      pre-LN 블록 ×32(MHA 양방향, q·v bias 있음·k bias 없음, FFN 4×, GELU) → `ln_post`. 출력 1500×1280
- [ ] **디코더**: 토큰 임베딩 + 학습 위치 임베딩(448) → 블록 ×4(causal self-attn + KV 캐시,
      **cross-attn**(K/V는 인코더 출력에서 구간마다 한 번 계산해 둠), FFN) → LN → 토큰 임베딩을 묶은 LM head
- [ ] **cross-attention 지원**: 지금 `mha_core`는 Q·K·V가 같은 줄기에서 오고 RoPE·causal을 전제로 함.
      인코더 출력 K/V를 고정 캐시로 받는 모드(RoPE 끔, 마스크 없음, 증분 디코딩 때 재계산 없음)를 추가하거나
      별도 레이어를 만듦. **가장 큰 기술 위험**
- [ ] PyTorch 대조: 인코더 출력, 첫 디코더 스텝 logits, 탐욕 디코딩 토큰열이 HF와 같음
- [ ] `quick_ai` 팩토리 등록(`WhisperForConditionalGeneration`), `quick_ai` 앱에서 wav 하나 인식

### 5.4 디코딩

- [ ] 시작 토큰 `<|startoftranscript|><|ko|><|transcribe|><|notimestamps|>`, 언어 비우면 언어 감지 한 스텝
- [ ] 탐욕 디코딩, `suppress_tokens`·`begin_suppress` 적용, 최대 224토큰
- [ ] 무음·반복 환각 대응: `no_speech` 확률로 버림, 압축률 검사(지금 `batch_pipeline`의 반복 검사와 합칠지 결정).
      WhisperX처럼 temperature fallback은 기본 끔
- [ ] 단어 시각: `alignment_heads`의 cross-attention을 토큰마다 모아 DTW → 단어 시작 시각 → `lastWords()`

### 5.5 discretion 연결

- [ ] `WhisperAsrEngine` (`asr_engine_whisper.cpp`): `segmentSamples()`는 30초, 짧은 입력은 0으로 채움.
      단어 시각이 채운 부분으로 넘어가지 않게 실제 길이로 자름
- [ ] `speech_ops`: Whisper일 때 길이별 엔진을 하나로
- [ ] `room_api` 설정 키 문서(`room_api.h` 머리말), `main.cpp`·`room_cli` 도움말
- [ ] `meson.build`·`jni/Android.mk`에 Whisper 소스, Qwen3과 함께 빌드(둘 중 하나만 올림)

### 5.6 (선택) wav2vec2 강제 정렬 — WhisperX 4단계 그대로

- [ ] 한국어 CTC 모델 선정(WhisperX 기본: `kresnik/wav2vec2-large-xlsr-korean`, 라이선스 확인)
- [ ] nntrainer로 wav2vec2(conv 특징 추출 + transformer) 구현, CTC 강제 정렬(trellis·backtrack)
- [ ] 5.4의 DTW 시각과 화자 바뀜 위치 정확도 비교 후 채택 여부 결정

### 5.7 안드로이드와 문서

- [ ] `libquick_ai_core.so` 크기, 모델 파일 크기·목록을 `docs/android-app.ko.md`에 반영
- [ ] 시험 폰 경로 `/data/local/tmp/nntrainer/discretion/models/whisper-large-v3-turbo/`

## 6. 위험과 확인할 점

- **실시간 lane 지연**: 8초 구간도 30초 인코더를 돌립니다. Qwen3(8초 그래프)보다 구간당 인코더 비용이 큽니다.
  묶음 모드(`--batch`)와 room 끝 판정 줄 단위에서 먼저 쓰고, 실시간은 재고 정합니다.
- **환각**: Whisper는 무음·잡음에서 글을 지어내는 경향이 있습니다. Silero/PyanNet 게이트 뒤에만 돌리고 `no_speech`를 씁니다.
- **cross-attention 구현**: 5.3절. 지금 quick_ai 모델은 모두 디코더 전용이라 새 길입니다.
- **메모리**: turbo Q4_0 기준 약 0.5 GB 예상(Qwen3 1.1 GB보다 작음). 실측 필요.
- **라이선스**: Whisper 가중치 MIT. 정렬 모델(5.6)은 따로 확인.

## 7. 비교 평가 (바꿀지 정하는 기준)

같은 녹음(week1, boda, 6명 TTS 회의)에서 Qwen3-ASR-1.7B와 나란히:

| 항목 | 재는 법 |
|---|---|
| 글 정확도 | 사람 전사 대비 CER(한국어), WER(영어) |
| 속도 | Galaxy S25+, 소리 1초당 처리 시간(RTF), 30초 묶음 하나 시간 |
| 지연 | room 줄이 끝난 뒤 나오기까지 |
| 메모리 | 모델 올린 뒤 RSS |
| 화자 | 단어 시각 품질 → 줄 화자 정확도(지금 95.8% / 69.7% 표와 같은 방식) |

## 8. 진행 기록

### 0. 조사와 계획 (2026-11-04)

- discretion의 ASR 의존 지점 정리(2절): 엔진 함수 넷, Qwen3 전용 가정 넷(언어 이름, 8초 그래프, self-attn 단어 시각, provider 문자열)
- `batch_pipeline`이 이미 WhisperX의 VAD·묶기·화자 붙이기를 따르고 있어, 남은 일은 Whisper 모델(3단계)과 단어 정렬(4단계)임을 확인
- `audio_preprocessor`가 Whisper 멜과 같은 계산임을 확인(멜 수만 인자로)
- nntrainer에 `conv1d`, `layer_normalization`이 있음. quick_ai `mha_core`에는 cross-attention 모드가 없음 → 5.3절 핵심 작업
- 다음: 4절 결정(모델 크기·단어 시각 방식) 확정 후 5.1 엔진 추상화부터
