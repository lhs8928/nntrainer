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
| Whisper 크기 | **large-v3** (1.55B, 디코더 32층) / large-v3-turbo (809M, 디코더 4층) / medium (769M) / small (244M) | **large-v3로 결정(2026-11-04)**. WhisperX가 권장하는 large 계열이고, 같은 인코더에 디코더가 깊어 turbo보다 인식 품질이 대체로 좋습니다. 디코더가 32층이라 생성이 느린 점은 6절에서 잽니다 |
| 단어 시각 | (a) Whisper cross-attention + DTW / (b) WhisperX처럼 wav2vec2 CTC 강제 정렬 | **둘 다 구현, (a) 먼저로 결정(2026-11-04)**. (a) 구현 완료(9절 2), (b)도 구현(9절 3). 둘 중 무엇을 쓸지는 실제 모델로 화자 줄 정확도를 재고 정함 |
| 짧은 구간 | 30초로 채움 / 인코더 길이를 줄임(whisper.cpp `audio_ctx`) | 묶음(batch)은 30초 그대로. 실시간 lane은 8절에서 지연을 재고 정함 |
| 엔진 고르기 | 설정 키 `asr_backend` / `config.json`의 `model_type`으로 자동 | **자동만 구현**: `config.json`의 `architectures`가 `WhisperForConditionalGeneration`이면 Whisper. 설정 키는 필요할 때 |

## 5. 작업 단계

### 5.1 엔진 고르기 (Qwen3 동작 변화 없음)

계획과 다르게, 인터페이스를 새로 나누지 않고 `AsrEngine` 안에서 고릅니다. 바깥 함수 넷(init / segmentSamples /
transcribe / lastWords)이 그대로라 `asr_lane`, `batch_pipeline`, `speech_ops`는 고칠 곳이 거의 없었습니다.

- [x] `AsrEngine::init`: `config.json`의 `architectures`로 Qwen3-ASR / Whisper 선택 (`asr_engine.cpp`)
- [x] `AsrOptions::language`를 엔진이 해석: Qwen3은 이름 그대로, Whisper는 `"Korean"`→`ko` 표(코드 직접도 받음, 모르는 언어는 init 오류)
- [x] `asr_provider`를 엔진이 알려 줌(`nntrainer:qwen3-asr` / `nntrainer:whisper`). `SpeechModels::asrProvider()`, `SpeechOps::describe()`
- [ ] 확인: 녹음 파일(week1, boda)로 바꾸기 전과 Qwen3 줄이 같음 (Qwen3 가중치가 있는 곳에서)

### 5.2 Whisper 가중치 변환

- [x] `Applications/quick_ai/res/whisper/whisper-large-v3/` 아래 `weight_converter.py`, `nntr_config.json`
      (HF safetensors → nntrainer 이름, 모두 FP32. 인코더 graph의 dtype은 `nntr_config.json`의 `encoder`로 따로)
- [x] `tokenizer.json`, `generation_config.json`(`alignment_heads`, `suppress_tokens`, `lang_to_id`)은 HF 체크포인트 것을 그대로 씀
- [ ] 양자화 조합 정함: 인코더·디코더 FC는 QINT8/Q4_0, conv는 FP32부터 (`nntr_quantize`). 지금은 FP32라 폰에 못 올림(6절)

### 5.3 Whisper 모델 구현 (`Applications/quick_ai/models/whisper/`)

- [x] **인코더** (`WhisperEncoder`, 별도 graph): conv1d(k3,s1)+GELU → conv1d(k3,s2)+GELU → 사인 위치 임베딩 →
      pre-LN 블록(`mha_core` 양방향, q·v bias, k bias 없음, GELU FFN) → `ln_post` → **층마다 cross K/V까지 계산**
- [x] **디코더** (`WhisperCausalLM`): 토큰 임베딩 + 학습 위치 임베딩(`position_embedding`, 위치 기준) → 블록
      (causal self-attn + KV 캐시, `cross_attention`, FFN) → LN → 묶은 LM head
- [x] **cross-attention**: `mha_core`를 고치지 않고 새 레이어 둘로 함 (`layers/cross_attention.*`).
      인코더 graph 끝의 `cross_memory_writer`가 층별 K/V를 이름 붙인 메모리(FP16 빌드는 FP16)에 저장하고,
      디코더의 `cross_attention`이 그 이름(`memory_id`)으로 읽음. 두 graph가 따로라 graph 입력 순서에 기대지 않음
- [x] PyTorch 대조: 멜·cross K·디코딩 8스텝 logits 오차 5e-4 이하, 탐욕 토큰 224개·단어·시각 모두 같음(9절 2)
- [x] `quick_ai` 팩토리 등록(`WhisperForConditionalGeneration`), `nntr_quick_ai <dir> --audio x.wav --language ko`

### 5.4 디코딩

- [x] 시작 토큰 `<|startoftranscript|><|ko|><|transcribe|><|notimestamps|>`, 언어 비우면 SOT 행에서 언어 감지
- [x] 탐욕 디코딩: 끝 토큰 말고 특수 토큰 전부 + `suppress_tokens` 막음, 첫 토큰은 `begin_suppress_tokens`도, 최대 224토큰
- [x] 무음: SOT 행의 `no_speech` 확률 > 0.6 이고 평균 log 확률 < -1 이면 글 없음 (discretion `AsrEngine`, openai 규칙)
- [ ] 압축률 검사(반복 환각)는 아직. `batch_pipeline`의 반복 검사가 있음. temperature fallback은 WhisperX처럼 끔
- [x] 단어 시각 (a): openai/whisper `timing.py`와 같은 단계 (`models/whisper/whisper_timing.*`). 끝 토큰을 한 번 더 넣어
      그 행까지 모음, 프레임 softmax → 행 방향 정규화 → median 7 → head 평균 → DTW → 단어(공백, zh/ja 등은 글자) →
      문장 끝 긴 단어 줄이기 → 구두점 합치기. `lastWords()`는 단어 시작 시각

### 5.5 discretion 연결

- [x] `segmentSamples()`는 `segment_sec`(최대 30초). 호출자가 채운 뒤쪽 0은 떼고 넣어 단어 시각이 소리 안에 있게 함
- [x] `speech_ops`: Whisper면 `asr_secs` 전체에 모델 하나(가장 긴 길이)
- [x] `room_api.h` 머리말, `main.cpp` 도움말
- [x] `meson.build`·`jni/Android.mk`(quick_ai, discretion)에 Whisper 소스와 헤더 경로. Qwen3과 함께 빌드, 설정으로 하나만 올림
- [ ] 폰 빌드(ndk-build) 확인

### 5.6 wav2vec2 강제 정렬 (b) — WhisperX 4단계 그대로

- [ ] 한국어 CTC 모델 확정(WhisperX 기본: `kresnik/wav2vec2-large-xlsr-korean`, 라이선스 확인)
- [x] nntrainer로 wav2vec2 구현 (`models/wav2vec2/wav2vec2_ctc.*`, `layers/wav2vec2_layers.*`): conv 특징 추출(layer/group norm),
      feature projection, group conv 위치 임베딩, encoder(pre-LN XLS-R / post-LN base), CTC head. 30초 graph에서 구간 길이만큼만 계산
- [x] WhisperX 3.8.6 `align()` 그대로 (`models/wav2vec2/ctc_align.*`): 소문자, 공백 `|`, 사전 밖 글자는 와일드카드,
      trellis → backtrack → 같은 토큰 묶기, 1 ms 반올림, 공백 단위 단어(ja·zh는 글자). 문장 나누기(nltk)는 단어가 같아 뺌
- [x] 연결: `nntr_quick_ai --align-model`, discretion `--align-model` / room 설정 `align_model_dir` (Qwen3 전사에도 붙음)
- [ ] 5.4의 DTW 시각과 화자 바뀜 위치 정확도 비교 후 기본값 결정

### 5.7 안드로이드와 문서

- [ ] `libquick_ai_core.so` 크기, 모델 파일 크기·목록을 `docs/android-app.ko.md`에 반영
- [ ] 시험 폰 경로 `/data/local/tmp/nntrainer/discretion/models/whisper-large-v3/`

## 6. 위험과 확인할 점

- **실시간 lane 지연**: 8초 구간도 30초 인코더를 돌립니다. Qwen3(8초 그래프)보다 구간당 인코더 비용이 큽니다.
  묶음 모드(`--batch`)와 room 끝 판정 줄 단위에서 먼저 쓰고, 실시간은 재고 정합니다.
- **환각**: Whisper는 무음·잡음에서 글을 지어내는 경향이 있습니다. Silero/PyanNet 게이트 뒤에만 돌리고 `no_speech`를 씁니다.
- **cross-attention 구현**: 5.3절. 지금 quick_ai 모델은 모두 디코더 전용이라 새 길입니다.
- **메모리**: 지금 변환기는 FP32만 냄: large-v3 가중치 약 6.2 GB라 폰에 못 올림. 양자화(5.2) 뒤 Q4_0 기준 약 0.9~1 GB 예상.
  여기에 구간마다 cross K/V 메모리(32층 × 2 × 1500 × 1280, FP16 245 MB, FP16 없는 빌드는 FP32 490 MB)가 더해짐. 실측 필요.
- **cross-attention 속도**: `cross_attention`은 스칼라 루프(head별 스레드)라 토큰당 약 1.2억 곱셈(large-v3). 폰에서 재고 느리면 NEON/GEMV로
- **생성 속도**: large-v3 디코더는 32층(폭 1280)이라 토큰 하나에 Qwen3-1.7B 디코더(28층, 폭 2048)와 비슷하거나 조금 덜 듭니다.
  여기에 30초 인코더(32층)가 구간마다 더해집니다. 폰에서 RTF를 먼저 재고, 부족하면 디코더 양자화(Q4_0)·인코더 길이 줄이기를 봅니다.
- **정렬 모델 메모리·속도**: XLS-R large(약 3억 파라미터)는 FP32 약 1.2 GB, 구간마다 한 번 더 돎. 특징 추출 conv는
  im2col + sgemm이고 30초 구간이면 첫 층 출력만 약 200 MB. 지금은 FP32만(양자화·FP16 활성값 아직)
- **라이선스**: Whisper 가중치 MIT. 정렬 모델(5.6)은 따로 확인.

## 7. 사용법 (지금)

```
# HF 체크포인트(openai/whisper-large-v3) → nntrainer 가중치
python Applications/quick_ai/res/whisper/whisper-large-v3/weight_converter.py <hf_dir> \
    -o <model_dir>/nntr_whisper_large_v3_fp32.safetensors
cp <hf_dir>/{config,generation_config,tokenizer}.json <model_dir>/
cp Applications/quick_ai/res/whisper/whisper-large-v3/nntr_config.json <model_dir>/

# 한 파일 인식 (30초씩), 단어 시각 함께 출력
nntr_quick_ai <model_dir> --audio x.wav --language ko
# 같은 계산의 PyTorch 기준 (같은 줄 형식)
python Applications/quick_ai/res/whisper/whisper-large-v3/reference_pytorch.py <hf_dir> x.wav --language ko

# discretion: model_dir만 Whisper 폴더로 바꾸면 됨 (language는 "Korean" 또는 "ko")
nntr_discretion <model_dir> --no-sd --stream-file x.wav --language Korean

# 단어 시각 (b): wav2vec2 CTC 체크포인트(예: kresnik/wav2vec2-large-xlsr-korean) 변환
python Applications/quick_ai/res/wav2vec2/weight_converter.py <hf_w2v_dir> -o <w2v_dir>/nntr_wav2vec2.safetensors
cp <hf_w2v_dir>/{config,vocab}.json Applications/quick_ai/res/wav2vec2/nntr_config.json <w2v_dir>/
nntr_quick_ai <model_dir> --audio x.wav --language ko --align-model <w2v_dir>     # Whisper + 정렬
nntr_quick_ai <w2v_dir> --audio x.wav --align-text "글" --language ko             # 정렬만
python Applications/quick_ai/res/wav2vec2/reference_pytorch.py <hf_w2v_dir> x.wav "글" --language ko
nntr_discretion <model_dir> --no-sd --stream-file x.wav --align-model <w2v_dir>
```

## 8. 비교 평가 (바꿀지 정하는 기준)

같은 녹음(week1, boda, 6명 TTS 회의)에서 Qwen3-ASR-1.7B와 나란히:

| 항목 | 재는 법 |
|---|---|
| 글 정확도 | 사람 전사 대비 CER(한국어), WER(영어) |
| 속도 | Galaxy S25+, 소리 1초당 처리 시간(RTF), 30초 묶음 하나 시간 |
| 지연 | room 줄이 끝난 뒤 나오기까지 |
| 메모리 | 모델 올린 뒤 RSS |
| 화자 | 단어 시각 품질 → 줄 화자 정확도(지금 95.8% / 69.7% 표와 같은 방식) |

## 9. 진행 기록

### 0. 조사와 계획 (2026-11-04)

- discretion의 ASR 의존 지점 정리(2절): 엔진 함수 넷, Qwen3 전용 가정 넷(언어 이름, 8초 그래프, self-attn 단어 시각, provider 문자열)
- `batch_pipeline`이 이미 WhisperX의 VAD·묶기·화자 붙이기를 따르고 있어, 남은 일은 Whisper 모델(3단계)과 단어 정렬(4단계)임을 확인
- `audio_preprocessor`가 Whisper 멜과 같은 계산임을 확인(멜 수만 인자로)
- nntrainer에 `conv1d`, `layer_normalization`이 있음. quick_ai `mha_core`에는 cross-attention 모드가 없음 → 5.3절 핵심 작업
- 다음: 4절 결정(모델 크기·단어 시각 방식) 확정 후 5.1 엔진 추상화부터

### 1. 모델 크기 결정 (2026-11-04)

- turbo 대신 **Whisper large-v3**로 함(사용자 결정). 인코더 32층 + 디코더 32층, 멜 128개, 어휘 51866
- 경로·provider 이름을 `whisper-large-v3`로 맞춤. 6절에 디코더 32층의 생성 속도 위험 추가
- 다음: 단어 시각 방식(cross-attention + DTW 먼저) 확인 후 5.1 엔진 추상화

### 2. Whisper 구현과 단어 시각 (a) (2026-11-04)

- quick_ai: `layers/cross_attention.*`(cross_attention, cross_memory_writer), `layers/position_embedding.*`,
  `models/whisper/`(인코더 graph, 디코더, 탐욕 디코딩, DTW 단어 시각), `res/whisper/whisper-large-v3/`(설정, 변환기, PyTorch 기준)
- discretion: `AsrEngine`이 `config.json`으로 Whisper 선택, 언어 이름→코드, 무음 규칙, provider, `speech_ops` 모델 하나
- 검증(x86, `-Denable-fp16=false`): HF를 받을 수 없어 **large-v3와 같은 구조의 작은 랜덤 Whisper**(멜 128, 어휘 51866, 2+2층)와
  openai-whisper 어휘로 만든 실제 토크나이저로 `reference_pytorch.py`와 비교. 3.1초·7.3초 입력, 언어 ko 고정·자동 감지 모두
  **토큰 224개, 글, 단어와 시작·끝 시각이 전부 같음**. 멜·cross K·디코딩 logits 오차 5e-4 이하(KV 캐시 FP16), no_speech·평균 log 확률은 셋째 자리까지
- `nntr_discretion --no-sd`로 같은 모델이 줄을 냄. 모르는 언어는 init에서 멈춤
- 고친 기존 문제 (각자 커밋):
  - `layer_normalization` 증분 경로가 스텝 행을 절대 위치 `[from, to)`에서 읽어, 위치 0이 아닌 스텝(디코딩)에서 틀림 → 0행부터
  - x86 FP16 없는 빌드가 깨져 있었음: `tensor_base.cpp`·`tensor.cpp`의 FP16 safetensors 읽기, quick_ai `rms_norm`·`reshaped_rms_norm`·
    `tie_word_embedding`·Qwen3-ASR의 `_FP16` 변환
- 아직 못 한 것: 실제 large-v3 가중치로 인식(한국어 CER), 양자화, 폰 빌드·속도, Qwen3 줄 회귀 확인(가중치 있는 곳에서), 압축률 검사, 단어 시각 (b)
- 다음: large-v3 실제 가중치로 `reference_pytorch.py`와 대조 → 양자화 → 폰 RTF → (b) wav2vec2 정렬

### 3. 단어 시각 (b): wav2vec2 강제 정렬 (2026-11-04)

- (a)와 (b) 둘 다 구현하고 (a)를 먼저 하기로 함(사용자 결정). (b) 구현
- quick_ai: `layers/wav2vec2_layers.*`(특징 추출기, 위치 conv), `models/wav2vec2/`(Wav2Vec2ForCTC, WhisperX 정렬),
  `res/wav2vec2/`(변환기, 설정, PyTorch 기준). Whisper의 `run()`은 `setAligner()`가 있으면 정렬기로 단어 시각
- discretion: `align_model_dir`이 있으면 엔진(Whisper·Qwen3 모두)이 전사를 정렬해 `lastWords()`로 줌. 실패하거나 구간이
  정렬기 graph보다 길면 모델 자체 시각. 정렬기는 폴더마다 하나만 올려 엔진끼리 나눠 씀
- 기준: 받아 둔 WhisperX 코드를 직접 실행하는 것은 막혀서, 그 정렬 단계를 `res/wav2vec2/reference_pytorch.py`에 옮겨 기준으로 씀
- 검증(x86, `-Denable-fp16=false`, 랜덤 작은 wav2vec2 두 구조: XLS-R식 pre-LN/layer norm/conv bias, base식 post-LN/group norm):
  logits 차이 3e-3 이하(attention 캐시 FP16). 단어와 시작·끝 시각(1 ms)이 **전부 같음**: 3.1초·7.3초, 한국어·영어·숫자·구두점(와일드카드),
  여분 공백, ja 글자 단위, 0.02초 입력, 프레임보다 긴 글(양쪽 모두 실패). Whisper 전사 → 정렬도 같음. discretion 줄의 단어 시각이 정렬 결과로 나옴
- 아직: 실제 한국어 정렬 모델(kresnik)로 확인, (a)와 화자 줄 정확도 비교, 정렬기 양자화·폰 속도

