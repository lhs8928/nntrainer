# Nemotron 3.5 Transcribe on NNTrainer

[`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)
is a promptable speech-to-text LLM: a speech recognition encoder and a speaker
diarization model run in parallel on the log-mel spectrogram, their outputs are
fused and projected into the token space of a Nemotron-H language model
(hybrid Mamba2 / MoE / attention, 30B total, ~3B active), which writes the
transcript.

| Piece | Where |
|---|---|
| Log-mel front-end, WAV reader | `models/nemotron3_5_transcribe/audio_frontend.*` |
| ASR encoder, diarization + speaker cache, speaker fusion, projector | `models/nemotron3_5_transcribe/audio_encoder.*` |
| Pipeline (prompt, audio splicing, generation) | `models/nemotron3_5_transcribe/nemotron3_5_transcribe.*` |
| App `nntr_nemotron_transcribe` | `models/nemotron3_5_transcribe/main_transcribe.cpp` |
| Nemotron-H LM (`NemotronHForCausalLM`) | `models/nemotron_h/` |
| `mamba2_ssm`, `nemotron_h_moe` layers | `layers/` |
| Design notes and progress | `models/nemotron3_5_transcribe/PLAN.md` |

## 1. Build

```bash
git submodule sync && git submodule update --init --depth 1
meson setup build -Denable-transformer=true
ninja -C build Applications/CausalLM/nntr_nemotron_transcribe
```

## 2. Convert a checkpoint

Python needs `torch`, `transformers>=5.22`, `safetensors` and `numpy`.

```bash
python Applications/CausalLM/res/nemotron3_5_transcribe/weight_converter.py \
    --model_path /path/to/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B \
    --out_dir    /path/to/nntr-nemotron-transcribe \
    --embedding_dtype float16
```

The converter streams the safetensors one tensor at a time and accepts both
the NeMo export naming of the Hub checkpoint (`perception.*`, `llm.*`, fused
`w_qkv`, stacked experts) and the transformers naming. It writes

| File | Content |
|---|---|
| `nntr_nemotron_h.bin` | LLM weights in NNTrainer graph order (FP32) |
| `nntr_nemotron_h_embedding.bin` | token embedding table (FP32 or FP16) |
| `nntr_audio.bin` | audio tower + projector (FP32, self describing) |
| `config.json`, `generation_config.json` | LLM config |
| `nntr_config.json` | runtime settings (`init_seq_len`, `max_seq_len`, `num_to_generate`, tokenizer path, ...) |
| `transcribe_config.json` | front-end / encoder / diarization settings and the chat prompt pieces |

The chat prompt (`transcribe_config.json` -> `prompt`) is rendered from the
checkpoint's chat template when its processor files are present; otherwise a
ChatML fallback is written and the converter says so. Check it against the
checkpoint before relying on the output.

## 3. Transcribe

```bash
./build/Applications/CausalLM/nntr_nemotron_transcribe /path/to/nntr-nemotron-transcribe speech.wav
# with a custom instruction
./build/Applications/CausalLM/nntr_nemotron_transcribe /path/to/nntr-nemotron-transcribe meeting.wav \
    "Transcribe this audio with punctuation and capitalization. It adds speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription."
```

Input: any RIFF/WAVE file (PCM or float, mono or multi-channel). Audio that is
not at 16 kHz is resampled linearly; resample offline for best quality. One
audio token covers 80 ms, so `max_seq_len` must hold `12.5 x seconds` plus the
prompt and the transcript.

### Low-memory conversion and run (what was used on a 15 GB machine)

Convert straight from the BF16 checkpoint to Q4_0, one tensor at a time, and
put the routed experts in their own file that the MoE layers memory-map (only
the experts the router picks are read):

```bash
ninja -C build Applications/CausalLM/nntr_stream_quantize
NNTR_STREAM_QUANTIZE=$PWD/build/Applications/CausalLM/nntr_stream_quantize \
python Applications/CausalLM/res/nemotron3_5_transcribe/weight_converter.py \
    --model_path /path/to/checkpoint_json_and_tokenizer_files \
    --safetensors_url https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B/resolve/main/model.safetensors \
    --out_dir /path/to/out --fc_dtype q4_0 --split_experts --embedding_dtype bfloat16
```

`--safetensors_url` reads the weights with HTTP range requests, so the 64 GB
source is never stored; drop it to read local `*.safetensors` instead. The
output is ~1.1 GB of LLM weights, ~16.5 GB of experts, a 0.7 GB embedding
table and the 2.8 GB audio tower; conversion peaks at ~2.5 GB of memory.

On a 4-core x86 machine with 15 GB of RAM and slow storage this transcribes a
6 s LibriSpeech utterance correctly in ~2 minutes from a cold page cache
(most of it paging experts in); see `PLAN.md` R1-4 for numbers.

### Quantize the language model with nntr_quantize

The FP32 LLM weights of the 30B checkpoint are ~130 GB. Quantize the
projections to Q4_0 with the generic tool; the Mamba2 parameters, the MoE
routers and the norms stay FP32:

```bash
./build/Applications/CausalLM/nntr_quantize /path/to/nntr-nemotron-transcribe \
    --fc_dtype Q4_0 --lmhead_dtype Q4_0
```

Without `--output` the quantized `.bin` is written next to the FP32 one and
`nntr_config.json` is updated to point at it (the audio, embedding and
transcribe settings are kept). With `--output <dir>`, copy
`nntr_nemotron_h_embedding.bin`, `nntr_audio.bin` and `transcribe_config.json`
into `<dir>` as well.

`nntr_quantize` loads the FP32 model first, so the conversion itself needs
about as much memory as the FP32 file. At run time the Q4_0 model takes
roughly 17 GB for the LLM (mostly the 23 x 128 experts), 0.7 GB for an FP16
embedding table and 2.8 GB for the FP32 audio tower.

## 4. Verify against the reference implementation

`make_tiny_reference.py` builds a small, randomly initialised model with every
architectural feature of the real one (chunked encoders, partial RoPE, q/k
norms, the diarization speaker cache and its compression, all three LLM block
types), and dumps the HuggingFace reference outputs. `--verify` compares every
stage:

```bash
python Applications/CausalLM/res/nemotron3_5_transcribe/make_tiny_reference.py --out_dir /tmp/tiny
python Applications/CausalLM/res/nemotron3_5_transcribe/weight_converter.py \
    --model_path /tmp/tiny/hf --out_dir /tmp/tiny/nntr \
    --init_seq_len 64 --max_seq_len 512 --num_to_generate 12
./build/Applications/CausalLM/nntr_nemotron_transcribe --verify /tmp/tiny/nntr /tmp/tiny
```

`--quant_friendly` makes every LLM projection a multiple of the Q4_0 block so
the quantized path can be exercised on the tiny model as well.

Expected: `ALL CHECKS PASSED` (log-mel, ASR encoder states, speaker logits,
audio embeddings, last-position logits and 12 greedy tokens for a text-only
prompt, a 2.6 s and an 11 s recording, and the whole `transcribe()` path with
the tokenizer).

### With the real checkpoint, without the 60 GB language model

The audio tower can be checked against transformers with the real weights on
a small machine: fetch only the `perception.*` tensors (~1.4 GB), then

```bash
python weight_converter.py --model_path <dir> --out_dir <dir>/nntr --audio_only
python make_audio_reference.py --model_path <dir> --out_dir <dir>/ref a.wav b.wav
./build/Applications/CausalLM/nntr_nemotron_transcribe --verify-audio <dir>/nntr <dir>/ref
```

`--verify-llm <model_dir> <ref_dir>` does the same for the language model
alone (tokenizer, prefill logits, greedy tokens), e.g. with a checkpoint cut
down to one block of each type.

## Not supported yet

- Word timestamps (CTC alignment head) and the MTP speculative-decoding head.
- On-demand (slim / FSU) expert loading for low-memory devices.
- Streaming input; the whole recording is encoded before generation.
