# Nemotron 3.5 Transcribe on NNTrainer — 분석 및 구현 계획

> 이 문서는 구현을 진행하면서 계속 갱신합니다. 진행 상황은 맨 아래
> **진행 현황** 섹션의 체크박스와 **작업 로그**를 확인하세요.

## 0. 대상 모델

| 항목 | 내용 |
|---|---|
| HF 체크포인트 | `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` (2026-10-28 공개) |
| HF 아키텍처 | `Nemotron3_5TranscribeForConditionalGeneration` (`model_type: nemo_speechlm`) |
| 참조 구현 | `transformers==5.22.2` 의 `models/nemotron3_5_transcribe`, `models/nemotron3_diarization`, `models/nemotron_h`, `models/nemotron_asr_streaming` (feature extractor) |
| 규모 | 전체 ≈ 34B 파라미터, 토큰당 활성 ≈ 4.3B (LLM 30B-A3B + 오디오 인코더 ≈ 0.7B) |
| 입력 | 16 kHz mono 오디오 + 텍스트 프롬프트 (40개 언어, 화자 태그 `<spk:N>`, 이벤트 태그 등) |
| 출력 | 텍스트 (전사/번역/화자 구분 전사), 선택적으로 단어 타임스탬프 (CTC 정렬 헤드) |

> 참고: 이름이 비슷한 `nvidia/nemotron-3.5-asr-streaming-0.6b` (FastConformer + RNNT 스트리밍 ASR)는
> **다른 모델**이며 이 계획의 대상이 아닙니다.

## 1. 모델 구조 상세 분석

```
 waveform(16k) ──► Log-Mel (128 bins, 10ms hop) ──┬──────────────────────────────────────────┐
                                                   │ per-feature 정규화                        │ (정규화 없음)
                                                   ▼                                          ▼
                                    ASR Encoder (32L Transformer, 1280d)      Diarization Model (Sortformer 계열,
                                    · 8프레임 stacking → 80ms/frame              31L Transformer 512d + speaker head)
                                    · chunk 500 + 좌/우 context 50               · chunk 500 / 우 context 50 / 좌 50
                                                   │                             · Arrival-Order Speaker Cache + FIFO
                                                   │ asr_last_hidden_state       ▼
                                                   │                  speaker logits (10ms, num_speakers=8)
                                                   │                             │ sigmoid → 8프레임 평균 → >0.5
                                                   ▼                             ▼
                                         LayerNorm(asr_norm)    +   SinusoidalSpeakerEmb(LayerNorm(activity))
                                                   └──────────────┬──────────────┘
                                                                  ▼
                                                 multi_modal_projector: Linear(1280 → 2688)
                                                                  ▼
          chat prompt 토큰 임베딩 ──(audio_token_id=131072 위치를 오디오 임베딩으로 치환)──►
                                                                  ▼
                       Nemotron-H LLM (Nemotron 3.5 Lightning 30B-A3B, 52 blocks hybrid)
                       pattern: MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME
                         M = Mamba2 mixer (23개), E = MoE (23개), * = GQA attention (6개)
                                                                  ▼
                                               RMSNorm → lm_head(131072) → 텍스트
```

### 1.1 오디오 전처리 (`NemotronAsrStreamingFeatureExtractor`)
- 16 kHz, pre-emphasis 0.97 (첫 샘플 유지, 패딩 영역 0)
- STFT: `n_fft=512`, `win_length=400` (Hann, **symmetric** `periodic=False`, 중앙 정렬 zero-pad 256), `hop=160`
- 파워 스펙트럼 → librosa Slaney mel filterbank (128 bins, 0~8 kHz, `norm="slaney"`) → `log(x + 2^-24)`
- Transcribe 체크포인트는 window/filterbank를 **bfloat16으로 반올림**해서 사용 (`mel_filters_dtype="bfloat16"`)
- 유효 프레임 수 = `audio_len // 160` (center=True 기준) 이후 프레임은 0, 프레임 수는 8의 배수로 오른쪽 패딩
- 오디오 토큰 수 = `ceil(valid_frames / 8)` (80 ms 당 1 토큰)

### 1.2 ASR 인코더 (`Nemotron3_5TranscribeAsrAudioModel`)
| 파라미터 | 값 |
|---|---|
| hidden / FFN / layers / heads | 1280 / 5120 / 32 / 10 (head_dim 128, MHA) |
| 입력 | per-feature 정규화된 mel (평균 0, 분산 1, unbiased std + 1e-5) |
| embedder | 8프레임 stacking(1024) → Linear(1024→1280, no bias) → `input_layer_norm` |
| layer | Pre-LN: `x += o_proj(attn(LN1(x)))`, `x += fc2(gelu(fc1(LN2(x))))` |
| attention | q/k/v Linear(no bias), **o_proj bias 있음**, **q_norm/k_norm = head 단위 LayerNorm(128)**, 양방향(non-causal), padding mask |
| RoPE | theta 1e4, **partial_rotary_factor 0.5** → 앞 64차원만 회전 (rotate_half 쌍은 (i, i+32)) |
| MLP | fc1/fc2 bias 있음, exact GELU |
| 출력 | 최종 LayerNorm → `asr_last_hidden_state` |
| chunking | 500 프레임 chunk, 좌/우 context 50 프레임씩 붙여서 독립 실행 후 chunk 구간만 사용 |

### 1.3 화자 분리 모델 (`Nemotron3DiarizationForAudioFrameClassification`, frozen)
| 파라미터 | 값 |
|---|---|
| 인코더 | 512d / FFN 2048 / 31 layers / 8 heads, ASR 인코더와 동일 구조이나 **q/k norm 없음, full RoPE(64)** |
| 입력 | **정규화하지 않은** log-mel |
| head | `proj` Linear(512→192) → Subpixel upsampler Conv1d(192→192·8, k=3, pad=1) → reshape(10ms 해상도) → ReLU → Linear(192,192) → ReLU → Linear(192→num_speakers) |
| 스트리밍 상태 | Arrival-Order Speaker Cache(188 프레임) + FIFO(40), chunk 500, 좌/우 context 50, update period 300 |
| silence_embeds | 학습된 512d 파라미터 (cache 압축 시 사용) |
| 출력 | 10 ms 해상도 speaker logits |

결합: `sigmoid → 8프레임 평균 → (> 0.5) → speaker_norm(LayerNorm(num_spk)) → @ sinusoidal(num_spk × 1280)` 을
`asr_norm(asr_hidden)` 에 더함.

### 1.4 Projector
`Linear(1280 → 2688, bias)`; 유효 프레임만 꺼내서 프롬프트의 `<|audio|>`(id 131072, LLM vocab 범위 밖 → 0번 임베딩 후 치환) 자리에 순서대로 삽입.

### 1.5 LLM: Nemotron-H (Nemotron 3.5 Lightning 30B-A3B)
| 파라미터 | 값 |
|---|---|
| hidden | 2688, vocab 131072, RMSNorm eps 1e-5, untied lm_head |
| blocks | 52개, 각 block = `x + mixer(RMSNorm(x))` (block 하나에 mixer 하나) |
| **Mamba2 (M)** | heads 64 × head_dim 64 (inner 4096), n_groups 8, d_state 128, conv k=4 (+bias), `in_proj`: 2688 → 4096(z) + 6144(xBC) + 64(dt) = 10304, `A=-exp(A_log)`, `dt=softplus(dt+dt_bias)` (prefill에서 ≥1e-3 clamp), `D` skip, **gated grouped RMSNorm** (`norm(y * silu(z))`, group 512), `out_proj` 4096→2688 |
| **MoE (E)** | 128 routed experts, top-6, **sigmoid router + e_score_correction_bias** (선택용), 선택된 sigmoid 점수 정규화 × `routed_scaling_factor 2.5`, expert = **non-gated** `down(relu²(up(x)))` (inter 1856), shared expert 1개 (inter 3712, relu²) |
| **Attention (\*)** | 32 heads, 2 KV heads (GQA 16), head_dim 128, **RoPE 없음** (위치 정보는 Mamba가 담당), causal |
| 기타 | MTP head(추측 디코딩용)는 로딩 시 무시 |

### 1.6 단어 타임스탬프 헤드 (선택 기능)
`ConvTranspose1d(×2 upsample)` → 2-layer 인코더(ASR layer 구조, 1280d, FFN 640, 16 heads) →
`x + 0.1·LN(h)` → Conv1d(k=1) → 23863 classes (CTC, blank=23862) → forced alignment (torchaudio).

### 1.7 프롬프트
- 기본 프롬프트: `"Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only."`
- user turn content = [text, audio], chat template은 체크포인트(`chat_template.jinja`)에 포함, `enable_thinking=False`
- 오디오 위치에 `<|audio|>` × N 삽입

## 2. NNTrainer 기존 자산 vs 필요한 모듈

| 구성 요소 | 기존 자산 | 필요 작업 |
|---|---|---|
| Embedding / RMSNorm / FC / lm_head | ✅ `embedding_layer`, `rms_norm`, `fully_connected`, `lm_head` | 재사용 |
| GQA attention (no RoPE) + KV cache | ✅ `mha_core` (`rope_theta=0` → RoPE off) | 재사용 |
| 임베딩 직접 입력 (멀티모달) | ✅ LFM2의 `use_embedding` + `run_with_embeddings` 패턴 | 재사용 (Nemotron-H용 구현) |
| **Mamba2 SSM** (conv state + selective scan + gated grouped RMSNorm) | ❌ (`causal_conv1d`는 k=3, bias 없음) | **신규 레이어 `mamba2_ssm`** |
| **Nemotron MoE** (sigmoid+bias router, relu² non-gated, shared expert) | △ `qwen_moe`(softmax, SwiGLU) | **신규 레이어 `nemotron_h_moe`** |
| Hybrid 블록 구성 (M/E/*) | △ LFM2 hybrid 예시 | **신규 모델 `NemotronHCausalLM`** |
| Log-Mel 전처리 (STFT, mel, preemph) | ❌ | **신규 `audio_frontend`** (WAV 로더 포함) |
| 양방향 Transformer 인코더 (LayerNorm, q/k LayerNorm, partial RoPE, GELU, bias, padding mask, 가변 길이 chunk) | ❌ (mha_core는 decoder/KV 지향, partial RoPE 방식 다름) | **신규 `audio_encoder`** (cpu_backend `sgemm` 기반 C++ 런타임) |
| Diarization head + Speaker Cache | ❌ | **신규 `diarization`** |
| 오디오 ↔ LLM 결합, 프롬프트 생성 | ❌ | **신규 `nemotron3_5_transcribe` 파이프라인 + 실행 앱** |
| 가중치 변환 (HF safetensors → nntrainer bin) | 모델별 python 스크립트 예시 | **신규 converter** (LLM bin + audio bin) |
| 검증 | - | HF transformers 참조 출력과 비교하는 tiny-config 테스트 |

### 설계 결정
1. **LLM 은 NNTrainer 그래프(레이어) 방식**으로 구현 → KV cache, 양자화(Q4_0 FC), FSU 등 기존 CausalLM 인프라를 그대로 활용.
2. **오디오 타워는 NNTrainer cpu_backend 커널(`sgemm` 등)을 쓰는 독립 C++ 런타임**으로 구현.
   - 이유: 오디오 길이에 따라 시퀀스 길이가 매번 다르고(chunk 500+context, speaker cache 길이 가변),
     양방향 attention + padding mask + 부분 RoPE + head 단위 LayerNorm 등 decoder용 `mha_core`와 맞지 않음.
     고정 shape 그래프로 만들면 최대 길이로 패딩해야 해서 짧은 오디오에서 낭비가 큼.
   - 1회성(오디오당 한 번) 연산이라 그래프 최적화 이점이 상대적으로 작음.
3. 오디오 임베딩은 LLM 입력 임베딩 시퀀스에 치환하여 `run_with_embeddings` 로 prefill.
4. 정확도 검증은 **HF 참조 구현 + 랜덤 tiny config** 로 단계별(멜, 오디오 임베딩, 로짓, 생성 토큰) 비교.

## 3. 파일 구성 (계획)

```
Applications/CausalLM/
├── layers/
│   ├── mamba2_ssm_layer.{h,cpp}          # Mamba2 SSM core (conv+scan+gated norm, state 유지)
│   └── nemotron_h_moe_layer.{h,cpp}      # sigmoid-router MoE (relu², shared expert 포함)
├── models/nemotron_h/
│   ├── nemotron_h_causallm.{h,cpp}       # Hybrid Mamba/MoE/Attention CausalLM (임베딩 입력 지원)
│   └── meson.build
├── models/nemotron3_5_transcribe/
│   ├── PLAN.md                           # (이 문서)
│   ├── audio_frontend.{h,cpp}            # WAV 로딩, pre-emphasis, STFT, mel, log
│   ├── audio_encoder.{h,cpp}             # ASR 인코더, Diarization, Speaker fusion, Projector
│   ├── nemotron3_5_transcribe.{h,cpp}    # 파이프라인 (오디오→임베딩→LLM 생성)
│   ├── main_transcribe.cpp               # 실행 앱 nntr_nemotron_transcribe
│   └── meson.build
└── res/nemotron3_5_transcribe/
    ├── weight_converter.py               # HF → nntrainer (LLM bin, audio bin, config)
    ├── make_tiny_reference.py            # tiny random 모델 + 참조 출력 생성
    └── README.md                         # 사용법 (nntr_config.json 등 설정 파일은 converter 가 생성)
```

## 4. 단계별 계획

| 단계 | 내용 | 산출물 / 완료 기준 |
|---|---|---|
| P0 | 모델 분석, 모듈 정리, 계획 수립 | 이 문서 |
| P1 | 빌드/참조 환경: nntrainer 빌드, torch+transformers CPU 환경, tiny random 체크포인트 + 참조 텐서 덤프 | `make_tiny_reference.py` |
| P2 | `mamba2_ssm` 레이어 | prefill(scan)/decode(state update) 모두 참조와 일치 |
| P3 | `nemotron_h_moe` 레이어 | 라우팅 결과/출력 일치 |
| P4 | `NemotronHCausalLM` 모델 + 레지스트리 + LLM 가중치 변환 | tiny LLM 로짓/greedy 토큰 일치 |
| P5 | 오디오 전처리 (`audio_frontend`) | log-mel 이 HF feature extractor 와 일치 |
| P6 | ASR 인코더 + chunking | `asr_last_hidden_state` 일치 |
| P7 | Diarization (인코더 + head + speaker cache 오프라인 모드) + speaker 결합 + projector | 오디오 임베딩 일치 |
| P8 | Transcribe 파이프라인 + 앱 + 오디오 가중치 변환 | tiny 모델 end-to-end 생성 토큰이 HF `generate` 와 동일 |
| P9 | 문서화 (README, 실제 체크포인트 실행 방법, 메모리 가이드) | README |
| P10 (후속) | 단어 타임스탬프 헤드, 실 체크포인트 대상 Q4_0 양자화/FSU(slim) 최적화, 스트리밍 | 선택 |

## 5. 리스크 / 제약

- **실 체크포인트 실행 제약**: 이 작업 환경은 huggingface.co 접근이 차단되어 실제 가중치(≈60 GB bf16)를 받을 수 없고,
  RAM 15 GB라 FP32 로딩도 불가 → 정확도는 tiny random 모델로 HF 참조 구현과 비교 검증하고,
  실 체크포인트는 변환 스크립트/설정/실행 방법을 제공 (실행은 충분한 메모리 + Q4_0 양자화 환경 필요).
- chat template 은 체크포인트에만 있으므로 실 실행 시 `tokenizer_config.json`/`chat_template.jinja` 기준으로
  프롬프트 prefix/suffix 를 설정 파일로 지정 (기본값은 Nemotron 계열 ChatML 형식 가정 — 실 체크포인트로 확인 필요).
- 오디오 30분 이상 장시간 입력: speaker cache 압축 로직까지 구현하되 chunk 단위 처리로 메모리 제한.
- 성능: Mamba2 prefill 은 chunked-scan 대신 순차 recurrence(수학적으로 동일)로 우선 구현.

## 6. 진행 현황

- [x] P0 분석 및 계획
- [x] P1 빌드/참조 환경 — nntrainer 빌드(`-Denable-transformer=true`), torch 2.15 + transformers 5.22.2 CPU,
      `make_tiny_reference.py` (tiny random 모델 + 참조 텐서 + tiny tokenizer)
- [x] P2 `mamba2_ssm` 레이어 — conv state / SSM state 유지, chunked prefill + decode 검증
- [x] P3 `nemotron_h_moe` 레이어 — sigmoid router + correction bias, relu², shared expert, expert 별 GEMM
- [x] P4 `NemotronHCausalLM` + LLM 변환 — 텍스트 프롬프트 logits(rel 7.6e-5) / greedy 12 토큰 일치
- [x] P5 오디오 전처리 — log-mel rel 1e-5 이내 일치 (bf16 window/filterbank 포함)
- [x] P6 ASR 인코더 — chunk + context 처리, encoder states rel 4e-6
- [x] P7 Diarization + Projector — speaker cache 압축(11 s 입력에서 6회) 포함, speaker logits rel 2e-6,
      audio embeddings rel 4e-6
- [x] P8 파이프라인 + 앱 + end-to-end 검증 — `nntr_nemotron_transcribe --verify` ALL CHECKS PASSED
      (텍스트 / 2.6 s / 11 s / tokenizer 를 거친 `transcribe()` 전체 경로)
- [x] P9 문서화 — `res/nemotron3_5_transcribe/README.md`
- [ ] P10 (후속) — 일부만 완료. 세부 항목과 순서는 아래 **6.1 남은 작업** 참고
  - [x] LLM Q4_0 양자화 — `nntr_quantize --fc_dtype Q4_0` 지원. Mamba 파라미터/라우터는 1-row FP32 weight 로
        요청해 `Layer::save()` 가 양자화하지 않도록 하고, expert/FC/lm_head 만 Q4_0.
        검증: Q4_0 격자에 snap 한 가중치로 FP32 vs Q4_0 실행 시 greedy 48 토큰 중 47 개 일치
        (나머지 1 개는 Q4_0 GEMM 의 활성값 Q8 양자화 영향)

> **현재 상태 요약**: P0~P9 는 "작동하는 첫 구현 + tiny 랜덤 모델로 HF 참조와 수치 일치" 수준이다.
> 실제 30B 체크포인트로는 한 번도 실행하지 않았고, 성능 최적화·모바일 메모리 대응·WER 평가는 하지 않았다.

### 6.1 남은 작업

우선순위 순서. 각 항목의 **완료 기준**을 만족해야 체크한다.

#### R1. 실제 체크포인트 실행 (최우선, 다른 모든 항목의 전제)
- [ ] R1-1 체크포인트 확보 및 변환
  - huggingface.co 접근 가능 + RAM ≥ 160 GB 머신에서 `weight_converter.py` 실행
    (NeMo export 이름 매핑은 transformers `conversion_mapping.py` 기준으로만 작성됨 → 실제 키 목록과 대조)
  - 변환기가 무시한 키(`IGNORED`)와 매핑되지 않은 키를 로그로 남기도록 보강
  - 완료 기준: 모든 LLM/오디오 텐서가 shape 오류 없이 변환되고 unmapped 키 0 개(MTP, preprocessor 등 의도한 것 제외)
- [ ] R1-2 프롬프트 형식 확인
  - processor 파일로 `chat_template_parts()` 가 렌더링되는지 확인 (현재 tiny 는 ChatML fallback 만 검증됨)
  - HF `apply_transcription_request()` 가 만든 `input_ids` 와 `buildPromptIds()` 결과를 바이트 단위로 비교
  - `enable_thinking=False` 처리, BOS 유무, `<|audio|>` 앞뒤 공백/개행 확인
  - 완료 기준: 기본 프롬프트·사용자 프롬프트 각 3 개에서 prompt ids 완전 일치
- [ ] R1-3 FP32 레퍼런스 대조 (고메모리 서버, x86)
  - 실 가중치로 HF transformers 와 단계별 비교: log-mel, ASR 출력, speaker logits, audio embeds, prefill logits
  - 기존 `--verify` 를 실 체크포인트용으로 확장 (참조 덤프 스크립트 `make_real_reference.py` 추가)
  - 완료 기준: 각 단계 rel 1e-3 이내, 30 s 한국어/영어 샘플 각 2 개에서 greedy 출력 문자열 일치
- [ ] R1-4 Q4_0 실행
  - `nntr_quantize --fc_dtype Q4_0 --lmhead_dtype Q4_0` 후 동일 샘플 실행
  - 완료 기준: 전사 결과가 FP32 와 의미상 동일(WER 차이 R2 에서 측정), 크래시/NaN 없음

#### R2. 정확도 평가
- [ ] R2-1 평가 스크립트: wav 목록 → `nntr_nemotron_transcribe` → 전사 결과 수집 (배치 실행 모드 추가 필요,
      현재 앱은 실행마다 모델을 다시 로드함)
- [ ] R2-2 한국어 WER/CER (예: KsponSpeech eval, Zeroth-Korean test) — HF transformers 결과와 함께 측정
- [ ] R2-3 영어 WER (LibriSpeech test-clean 일부) — 모델 카드 수치와 비교
- 완료 기준: NNTrainer FP32 vs HF 의 CER 차이 0.1 %p 이내, Q4_0 의 열화폭을 표로 기록

#### R3. 기능 보완
- [ ] R3-1 Android / 앱 빌드 목록에 새 소스 추가 — `jni/Android.mk`, `app_build/meson.build` 에
      `mamba2_ssm_layer.cpp`, `nemotron_h_moe_layer.cpp`, `models/nemotron_h`, `models/nemotron3_5_transcribe`
      가 빠져 있음. 완료 기준: `build_android.sh` 빌드 성공
- [ ] R3-2 Windows 빌드 확인 (`std::filesystem`, `std::erf` 등 사용) — CI Windows 셀 통과
- [ ] R3-3 장시간 오디오: 현재 전체 오디오 임베딩을 한 번에 prefill → `max_seq_len` 이 12.5 × 초 이상 필요.
      30 분 입력(≈22,500 토큰)에서 KV cache / 활성 메모리 측정, 필요 시 오디오를 구간 분할해 여러 번 전사
- [ ] R3-4 리샘플러: 현재 선형 보간 → 윈도드 sinc 등 품질 보장되는 방식으로 교체 (16 kHz 아닌 입력 대응)
- [ ] R3-5 앱 기능: 여러 파일 연속 처리(모델 1 회 로드), 스트리밍 출력 콜백, 사용자 프롬프트 파일 입력
- [ ] R3-6 단어 타임스탬프: word timestamp encoder(ConvTranspose ×2 + 2층) + CTC head(23,863 class) +
      forced alignment(Viterbi) + 화자 매칭(`_match_tags_to_speakers`, Hungarian). alignment tokenizer 필요
- [ ] R3-7 Mamba prefill 의 dt clamp: 현재 "이번 step 길이 > 1" 일 때만 clamp → prefill 마지막 chunk 가
      1 토큰이면 HF 와 다를 수 있음. prefill/decode 구분을 레이어에 명시적으로 전달하도록 수정

#### R4. 성능 최적화 (x86 → ARM 순)
- [ ] R4-1 프로파일링: 30 s 오디오 기준 단계별 시간(전처리 / ASR / diarization / prefill / decode tok/s) 측정표 작성
- [ ] R4-2 Mamba2 prefill 을 순차 recurrence → chunked scan(SSD, chunk 128) GEMM 화. decode 경로는 head 단위
      벡터화(NEON/AVX2). 완료 기준: prefill 수치 rel 1e-4 유지, prefill 속도 ≥ 5 배
- [ ] R4-3 MoE: router GEMV 를 스칼라 루프 → sgemv, expert 별 gather 버퍼 재사용(매 호출 Tensor 할당 제거),
      decode 시 6 expert 병렬화
- [ ] R4-4 오디오 인코더: attention 을 head 단위 병렬화, 활성 버퍼 재사용, GELU/LayerNorm 벡터화,
      FP16 가중치(ARM) 옵션
- [ ] R4-5 KV cache: 52 레이어 전부 할당 중 → attention 6 레이어만 할당 (`getKVCacheRows` 를 레이어별로 0 반환
      가능하게 수정)
- [ ] R4-6 Mamba state / conv state 를 FP16 으로 둘지 정밀도 영향 평가

#### R5. 메모리 / 모바일
- [ ] R5-1 변환 경로 메모리: `nntr_quantize` 가 FP32 모델 전체(≈130 GB)를 로드함 → `weight_converter.py` 에서
      Q4_0 을 직접 기록(블록 양자화 + ISA repack 포맷 재현)하거나 레이어 단위 스트리밍 양자화 도구 작성
- [ ] R5-2 오디오 타워 가중치 FP16/Q8 저장 (현재 FP32 2.8 GB)
- [ ] R5-3 slim / FSU expert on-demand 로딩 (`qwen3_cached_slim_moe` 방식 이식). 목표: 상주 메모리 ≤ 6 GB
- [ ] R5-4 임베딩 테이블 Q6_K/Q4_0 (현재 FP32/FP16 side file)
- [ ] R5-5 Android 실기기 측정: peak RSS, 30 s 오디오 처리 시간, 발열/스로틀링
- 참고: Q4_0 만으로는 LLM ≈ 17 GB + 오디오 2.8 GB → 모바일 불가. R5-3 없이는 모바일 목표 달성 불가

#### R6. 테스트 / CI
- [ ] R6-1 `mamba2_ssm`, `nemotron_h_moe` gtest 단위 테스트 추가 (참조 텐서 고정 파일 사용, torch 불필요)
- [ ] R6-2 tiny 모델 end-to-end 를 CI 에서 돌릴 수 있도록 참조 파일을 작은 크기로 커밋하거나 생성 단계 분리
- [ ] R6-3 `check_count` 영향 확인 및 테스트 수 갱신

#### 예상 일정 (1 인 기준, 대략)

| 묶음 | 기간 | 비고 |
|---|---|---|
| R1 실 체크포인트 실행 | 1~2 주 | 고메모리 서버 필요. 이름 매핑/템플릿 문제 발생 시 증가 |
| R2 정확도 평가 | 1 주 | 데이터셋 준비 포함 |
| R3 기능 보완 | 2~3 주 | 타임스탬프(R3-6) 가 절반 |
| R4 성능 최적화 | 3~5 주 | SSD chunked scan, NEON |
| R5 메모리/모바일 | 3~5 주 | slim expert 로딩이 핵심 |
| R6 테스트/CI | 1 주 | |
| 합계 | 약 11~17 주 | |

### 검증 결과 (tiny 모델, `--verify`)

| 항목 | 결과 |
|---|---|
| text-only: last logits / greedy 12 tokens | rel 7.6e-5 / 일치 |
| short (2.6 s, 33 audio tokens, ASR 2 chunks) | mel rel 5.7e-6, ASR rel 3.7e-6, speaker rel 1.6e-6, embeds rel 3.9e-6, logits rel 2.7e-5, tokens 일치 |
| long (11 s, 138 audio tokens, 7 chunks, cache 압축 6회, 147-token prompt → chunked prefill) | mel rel 9.5e-6, ASR rel 4.2e-6, speaker rel 1.8e-6, embeds rel 3.8e-6, logits rel 1.9e-5, tokens 일치 |
| transcribe(): tokenizer + 프롬프트 템플릿 + 생성 | prompt ids / greedy tokens 일치 |

## 7. 작업 로그

- 2026-11-09: transformers 5.22.2 참조 구현 분석 완료, 계획 수립.
- 2026-11-09: P1 — HF 접근 차단으로 PyPI 의 transformers wheel 에서 참조 코드 확보, tiny 참조 생성기 작성.
  tiny diarization head 가 bias 지배로 상수 출력 → gain/bias 조정하여 화자 활동이 실제로 바뀌도록 수정.
- 2026-11-09: P2~P4 — `mamba2_ssm`, `nemotron_h_moe`, `NemotronHCausalLM` 구현. LLM 은 임베딩 입력 그래프 +
  임베딩 테이블 side file 구조(오디오 임베딩 삽입용). 변환기는 safetensors 를 텐서 단위로 스트리밍하며
  NeMo export 이름(`perception.*`, `llm.*`, fused `w_qkv`, stacked experts)과 HF 이름 모두 지원.
- 2026-11-09: P5~P8 — 오디오 전처리/인코더/diarization/파이프라인/앱 구현, 전 단계 참조 일치 확인.
  참조 생성 시 HF `generate(pad_token_id=0)` 가 prompt 의 id 0 을 padding 으로 가리는 문제 발견 →
  참조 쪽에 명시적 attention mask 전달로 수정 (NNTrainer 구현 문제 아님).
- 2026-11-09: P9 — README 작성.
- 2026-11-09: P10(양자화) — 커스텀 레이어 weight 를 기본 `Layer::save()` 가 전부 양자화하던 문제를
  1-row FP32 요청으로 해결, `nntr_quantize` 에 Nemotron-H 등록/레이어 이름 매핑 추가.
  tiny 랜덤 모델은 Q4_0 에 매우 민감(HF 에서 Q4_0 시뮬레이션만 해도 logits rel 0.84)하여
  Q4_0 격자 snap 가중치로 FP32/Q4_0 동치성을 확인.
- 2026-11-11: 남은 작업을 R1~R6 으로 구체화 (실 체크포인트 실행, 정확도 평가, 기능 보완, 성능, 메모리/모바일,
  테스트). Android/앱 빌드 목록 누락(R3-1), Mamba dt clamp edge case(R3-7), KV cache 전 레이어 할당(R4-5) 확인.
