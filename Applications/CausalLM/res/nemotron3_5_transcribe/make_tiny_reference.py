## SPDX-License-Identifier: Apache-2.0
## @file   make_tiny_reference.py
## @brief  Build a tiny, randomly initialised Nemotron 3.5 Transcribe model with
##         the HuggingFace reference implementation and dump the reference
##         tensors the NNTrainer port is checked against.
##
## The tiny model keeps every architectural feature of the real checkpoint
## (feature stacking, partial RoPE, q/k LayerNorm, diarization chunks with the
## speaker cache, Mamba2 / MoE / attention hybrid LLM, ...) with small sizes,
## so the whole pipeline can be verified on a laptop.
##
## Usage:
##   python make_tiny_reference.py --out_dir /path/to/tiny
##
## Outputs (in --out_dir):
##   hf/                    HF checkpoint (save_pretrained)
##   audio_short.wav        ~2.6 s 16 kHz PCM16 test signal
##   audio_long.wav         ~11 s test signal (several diarization chunks,
##                          speaker cache compression)
##   ref_<name>.bin         float32 / int32 raw tensors
##   ref_index.json         shapes / dtypes of the dumped tensors

import argparse
import json
import math
import os
import sys
import wave

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import (
    Nemotron3_5TranscribeConfig,
    Nemotron3_5TranscribeForConditionalGeneration,
)
from transformers.models.nemotron_asr_streaming.feature_extraction_nemotron_asr_streaming import (
    NemotronAsrStreamingFeatureExtractor,
)

VOCAB = 1000
AUDIO_TOKEN_ID = VOCAB  # outside of the vocabulary, as in the real checkpoint


def tiny_config():
    return Nemotron3_5TranscribeConfig(
        audio_config=dict(
            asr_config=dict(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
            ),
            diarization_config=dict(
                audio_config=dict(
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=2,
                    num_attention_heads=2,
                ),
                head_config=dict(hidden_size=16, num_speakers=4, audio_hidden_size=32),
                streaming_config=dict(num_speakers=4, speaker_cache_length=16),
                chunk_length=20,
                chunk_left_context=5,
                chunk_right_context=5,
                fifo_length=8,
                speaker_cache_update_period=12,
            ),
        ),
        text_config=dict(
            hidden_size=64,
            hybrid_override_pattern="MEM*EM*E",
            mamba_num_heads=4,
            mamba_head_dim=16,
            n_groups=2,
            ssm_state_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            n_routed_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=32,
            moe_shared_expert_intermediate_size=48,
            routed_scaling_factor=2.5,
            vocab_size=VOCAB,
            max_position_embeddings=4096,
            eos_token_id=2,
        ),
        word_timestamp_config=dict(
            hidden_size=64,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=50,
            blank_token_id=49,
        ),
        audio_token_id=AUDIO_TOKEN_ID,
    )


def randomize(model, seed):
    """Perturb every parameter so that no weight keeps a trivial init value."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            noise = torch.randn(p.shape, generator=g) * 0.1
            if name.endswith("A_log"):
                p.copy_(torch.log(torch.rand(p.shape, generator=g) * 4 + 0.5))
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.5 - 1.0)
            elif "embedder.projection" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
            elif "diarization_model.classifier.out_proj.bias" in name:
                # centre the logits so that every speaker turns on and off
                p.copy_(torch.tensor([2.3, 3.2, 0.9, 0.2])[: p.numel()])
            elif "diarization_model" in name and name.endswith("weight") and (
                "classifier" in name or "upsampler" in name or ".model.proj." in name
            ):
                # a speaker head with enough gain for frame-to-frame changes:
                # exercises the activity threshold and the speaker cache
                p.copy_(torch.randn(p.shape, generator=g) * 0.25)
            else:
                p.add_(noise)
        for name, b in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                b.copy_(torch.randn(b.shape, generator=g) * 0.05)


def make_wave(seconds, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(int(16000 * seconds)) / 16000.0
    sig = np.zeros_like(t)
    # a few "speakers": tones with slowly varying amplitude in different spans
    for k, (f, start, stop) in enumerate(
        [(220.0, 0.0, 0.55), (523.0, 0.3, 0.8), (1250.0, 0.6, 1.0)]
    ):
        env = ((t >= start * seconds) & (t < stop * seconds)).astype(np.float64)
        env *= 0.5 + 0.5 * np.sin(2 * math.pi * (0.7 + k) * t)
        sig += 0.25 * env * np.sin(2 * math.pi * f * t + k)
    sig += 0.02 * rng.standard_normal(t.shape)
    pcm = np.clip(np.round(sig * 32767), -32768, 32767).astype(np.int16)
    return pcm


def write_wav(path, pcm):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())


SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]


def make_tokenizer(path):
    """Word-level tokenizer over the tiny vocabulary, with the chat special
    tokens of the prompt template and the audio placeholder (id == VOCAB)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from tokenizers import AddedToken

    words = (
        "audio transcription . language : as spoken wording verbatim , including "
        "fillers and repetitions formatting punctuation capitalization output "
        "transcript only user assistant"
    ).split()
    vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
    for w in words:
        vocab.setdefault(w, len(vocab))
    for i in range(len(vocab), VOCAB):
        vocab[f"w{i}"] = i
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.normalizer = None
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
    )
    tok.decoder = decoders.WordPiece(prefix="##")
    tok.add_special_tokens(
        [AddedToken(t, special=True) for t in SPECIAL_TOKENS[3:]]
        + [AddedToken("<|audio|>", special=True)]
    )
    assert tok.token_to_id("<|audio|>") == AUDIO_TOKEN_ID
    tok.save(path)
    return tok


class Dumper:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.index = {}

    def __call__(self, name, tensor):
        arr = tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
        if arr.dtype.kind == "f":
            arr = arr.astype(np.float32)
        else:
            arr = arr.astype(np.int32)
        arr.tofile(os.path.join(self.out_dir, f"ref_{name}.bin"))
        self.index[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}

    def save(self, extra):
        self.index["_meta"] = extra
        with open(os.path.join(self.out_dir, "ref_index.json"), "w") as f:
            json.dump(self.index, f, indent=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_new_tokens", type=int, default=12)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    config = tiny_config()
    model = Nemotron3_5TranscribeForConditionalGeneration(config).eval().float()
    randomize(model, args.seed)
    model.save_pretrained(os.path.join(args.out_dir, "hf"), safe_serialization=True)

    tokenizer = make_tokenizer(os.path.join(args.out_dir, "hf", "tokenizer.json"))

    fe = NemotronAsrStreamingFeatureExtractor(feature_size=128, mel_filters_dtype="bfloat16")
    dump = Dumper(args.out_dir)
    meta = {"audio_token_id": AUDIO_TOKEN_ID, "vocab_size": VOCAB}

    # ---------------------------------------------------------------- LLM only
    g = torch.Generator().manual_seed(7)
    text_ids = torch.randint(3, VOCAB, (1, 9), generator=g)
    dump("text_ids", text_ids[0])
    with torch.no_grad():
        logits = model(input_ids=text_ids).logits
        dump("text_logits", logits[0])
        gen = model.generate(
            input_ids=text_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
            # explicit mask: token 0 in a prompt is not padding
            attention_mask=torch.ones_like(text_ids),
        )
        dump("text_generated", gen[0, text_ids.shape[1] :])

    # ---------------------------------------------------------------- audio
    for tag, seconds in (("short", 2.6), ("long", 11.0)):
        pcm = make_wave(seconds, seed=11 if tag == "short" else 13)
        write_wav(os.path.join(args.out_dir, f"audio_{tag}.wav"), pcm)
        audio = pcm.astype(np.float32) / 32768.0
        feats = fe(audio, sampling_rate=16000, return_tensors="pt")
        mel = feats["input_features"]
        mask = feats["attention_mask"].long()
        pad = -mel.shape[1] % 8
        mel = torch.nn.functional.pad(mel, (0, 0, 0, pad))
        mask = torch.nn.functional.pad(mask, (0, pad))
        dump(f"{tag}_mel", mel[0])
        dump(f"{tag}_mel_mask", mask[0])

        with torch.no_grad():
            tower = model.model.audio_tower
            out = tower(mel, input_features_mask=mask)
            dump(f"{tag}_asr_hidden", out.asr_last_hidden_state[0])
            dump(f"{tag}_speaker_logits", out.speaker_logits[0])
            dump(f"{tag}_audio_hidden", out.last_hidden_state[0])
            feats_out = model.model.get_audio_features(mel, mask)
            audio_embeds = feats_out.pooler_output
            dump(f"{tag}_audio_embeds", audio_embeds)

            n_audio = audio_embeds.shape[0]
            prefix = torch.randint(3, VOCAB, (5,), generator=g)
            suffix = torch.randint(3, VOCAB, (4,), generator=g)
            ids = torch.cat(
                [prefix, torch.full((n_audio,), AUDIO_TOKEN_ID), suffix]
            )[None]
            dump(f"{tag}_input_ids", ids[0])
            logits = model(input_ids=ids, input_features=mel, input_features_mask=mask).logits
            dump(f"{tag}_logits", logits[0])
            gen = model.generate(
                input_ids=ids,
                input_features=mel,
                input_features_mask=mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=0,
                attention_mask=torch.ones_like(ids),
            )
            dump(f"{tag}_generated", gen[0, ids.shape[1] :])
        meta[f"{tag}_num_audio_tokens"] = int(n_audio)

        if tag == "short":
            # the whole transcribe() path: chat prompt built with the
            # tokenizer and the converter's prompt template, default prompt
            from weight_converter import chat_template_parts

            parts = chat_template_parts(os.path.join(args.out_dir, "hf"))
            head = tokenizer.encode(
                parts["prefix"] + parts["default_prompt"] + parts["between"],
                add_special_tokens=False,
            ).ids
            tail = tokenizer.encode(parts["suffix"], add_special_tokens=False).ids
            ids = torch.tensor([head + [AUDIO_TOKEN_ID] * n_audio + tail])
            dump("transcribe_prompt_ids", ids[0])
            with torch.no_grad():
                gen = model.generate(
                    input_ids=ids,
                    input_features=mel,
                    input_features_mask=mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=None,
                    pad_token_id=0,
                    attention_mask=torch.ones_like(ids),
                )
            dump("transcribe_generated", gen[0, ids.shape[1] :])

    dump.save(meta)
    print("reference written to", args.out_dir)


if __name__ == "__main__":
    main()
