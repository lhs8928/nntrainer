# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file   reference_pytorch.py
# @date   04 November 2026
# @brief  PyTorch reference of nntrainer's Whisper decoding and word times
# @author Hyeonseok Lee <hs89.lee@samsung.com>
#
# Decodes one segment (the first 30 s of a 16 kHz mono wav) the way
# models/whisper/whisper_causallm.cpp does - start of transcript, language,
# transcribe, no timestamps; greedy; every special token but the end of text
# suppressed, plus suppress_tokens and, for the first token,
# begin_suppress_tokens - and times the words with openai/whisper's
# find_alignment steps on the alignment heads. Prints the same lines as
# `nntr_quick_ai <model_dir> --audio <wav> --language <code>`, so the two
# outputs can be compared line by line.
#
#   python reference_pytorch.py <hf_checkpoint_dir> <wav> [--language ko]
#       [--encoder-context audio [--context-pad-sec S] [--context-min-sec S]]
#   python reference_pytorch.py --make-tiny <dir>   # random small checkpoint
#
# Needs torch, transformers and openai-whisper (its tokenizer and timing
# helpers). --make-tiny writes a randomly initialized Whisper with the
# large-v3 layout (128 mels, 51866 tokens) and few small layers, its
# generation_config.json and the tokenizer.json converted from
# openai-whisper's vocabulary: a checkpoint to test the conversion and the
# C++ model without downloading large-v3.
##

import argparse
import json
import math
import os
import wave

import numpy as np
import torch
from transformers import (WhisperConfig, WhisperFeatureExtractor,
                          WhisperForConditionalGeneration)
from whisper.tokenizer import LANGUAGES, get_tokenizer
import whisper.timing as timing


def make_tiny(out):
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    cfg = WhisperConfig(vocab_size=51866, num_mel_bins=128, d_model=256,
                        encoder_layers=2, encoder_attention_heads=4,
                        encoder_ffn_dim=1024, decoder_layers=2,
                        decoder_attention_heads=4, decoder_ffn_dim=1024,
                        init_std=0.3)
    WhisperForConditionalGeneration(cfg).save_pretrained(out)
    tok = get_tokenizer(True, num_languages=100)
    lang_to_id = {f"<|{c}|>": tok.encoding.encode_single_token(f"<|{c}|>")
                  for c in list(LANGUAGES)[:100]}
    gen = {"decoder_start_token_id": tok.sot, "eos_token_id": tok.eot,
           "bos_token_id": tok.eot, "pad_token_id": tok.eot,
           "lang_to_id": lang_to_id,
           "task_to_id": {"translate": tok.translate,
                          "transcribe": tok.transcribe},
           "no_timestamps_token_id": tok.no_timestamps,
           "suppress_tokens": list(tok.non_speech_tokens),
           "begin_suppress_tokens": [220, tok.eot],
           "alignment_heads": [[0, 1], [1, 0], [1, 3]], "max_length": 448}
    with open(os.path.join(out, "generation_config.json"), "w") as f:
        json.dump(gen, f, indent=1)

    import whisper
    from tokenizers import AddedToken
    from transformers.convert_slow_tokenizer import TikTokenConverter
    vocab = os.path.join(os.path.dirname(whisper.__file__), "assets",
                         "multilingual.tiktoken")
    hf_tok = TikTokenConverter(vocab_file=vocab,
                               pattern=tok.encoding._pat_str).converted()
    specials = sorted(tok.encoding._special_tokens.items(), key=lambda kv: kv[1])
    hf_tok.add_special_tokens(
        [AddedToken(s, special=True, normalized=False) for s, _ in specials])
    hf_tok.save(os.path.join(out, "tokenizer.json"))
    print(f"wrote a random tiny Whisper to {out}")


def load_wav(path):
    with wave.open(path) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("expects 16 kHz mono 16-bit wav")
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return pcm.astype(np.float32) / 32768.0


def encoder_positions(samples, context, pad_sec, min_sec, n_ctx=1500):
    """whisper_causallm.cpp: the whole window, or the audio's positions"""
    if context != "audio":
        return n_ctx
    pos = math.ceil(samples / 320) + math.ceil(pad_sec * 50)
    pos = max(pos, math.ceil(min_sec * 50))
    return int(min(max(pos, 1), n_ctx))


def encode(model, mel, positions):
    """the encoder over its first positions only (whisper.cpp's audio_ctx)"""
    enc = model.model.encoder
    x = torch.nn.functional.gelu(enc.conv1(mel))
    x = torch.nn.functional.gelu(enc.conv2(x)).permute(0, 2, 1)[:, :positions]
    x = x + enc.embed_positions.weight[:positions]
    for layer in enc.layers:
        x = layer(x, None)
        x = x[0] if isinstance(x, tuple) else x
    return enc.layer_norm(x)


def transcribe(ckpt, wav, language, sample_len=224, context="full",
               pad_sec=0.0, min_sec=0.0):
    model = WhisperForConditionalGeneration.from_pretrained(
        ckpt, attn_implementation="eager").eval()
    gen = json.load(open(os.path.join(ckpt, "generation_config.json")))
    eot = gen["eos_token_id"]
    lang_to_id = gen["lang_to_id"]
    heads = gen["alignment_heads"]
    audio = load_wav(wav)[:480000]
    fe = WhisperFeatureExtractor(feature_size=model.config.num_mel_bins)
    mel = fe(audio, sampling_rate=16000, return_tensors="pt").input_features

    positions = encoder_positions(len(audio), context, pad_sec, min_sec)
    with torch.no_grad():
        enc = encode(model, mel, positions)

        def run(ids):
            o = model(encoder_outputs=(enc,),
                      decoder_input_ids=torch.tensor([ids]),
                      output_attentions=True)
            return o.logits[0], o.cross_attentions

        sot = gen["decoder_start_token_id"]
        l0 = run([sot])[0][-1].double()
        lse = torch.logsumexp(l0, 0)
        no_speech_id = gen.get("no_speech_token_id",
                               get_tokenizer(True, num_languages=100).no_speech)
        no_speech = float(torch.exp(l0[no_speech_id] - lse))
        if language:
            lang_id = lang_to_id[f"<|{language}|>"]
        else:
            ids = list(lang_to_id.values())
            lang_id = ids[int(torch.argmax(l0[ids]))]
        code = [c for c, i in lang_to_id.items() if i == lang_id][0][2:-2]
        prompt = [sot, lang_id, gen["task_to_id"]["transcribe"],
                  gen["no_timestamps_token_id"]]

        tokens, sum_lp = [], 0.0
        while True:
            lg = run(prompt + tokens)[0][-1].clone()
            lg[eot + 1:] = -np.inf
            lg[gen.get("suppress_tokens", [])] = -np.inf
            if not tokens:
                lg[gen.get("begin_suppress_tokens", [])] = -np.inf
            nxt = int(torch.argmax(lg))
            sum_lp += float(lg[nxt] - torch.logsumexp(lg, 0))
            if nxt == eot:
                break
            tokens.append(nxt)
            if len(tokens) >= sample_len:
                break

        tok = get_tokenizer(True, num_languages=100, language=code)
        words = []
        if tokens:
            cross = run(prompt + tokens + [eot])[1]
            frames = len(audio) // 160
            w = torch.stack([cross[l][0, h] for l, h in heads])
            # softmax over the segment's frames == renormalized probabilities
            w = torch.log(w[:, :, : min(frames // 2, positions)].double()).softmax(dim=-1)
            std, mean = torch.std_mean(w, dim=-2, keepdim=True, unbiased=False)
            w = timing.median_filter(((w - mean) / std).float(), 7)
            matrix = w.mean(axis=0)[3:-1]
            text_idx, time_idx = timing.dtw(-matrix.double())
            wl, wt = tok.split_to_word_tokens(tokens + [eot])
            if len(wt) > 1:
                bounds = np.pad(np.cumsum([len(t) for t in wt[:-1]]), (1, 0))
                jumps = np.pad(np.diff(text_idx), (1, 0),
                               constant_values=1).astype(bool)
                jt = time_idx[jumps] / 50
                al = [timing.WordTiming(a, b, s, e, 0.0) for a, b, s, e in
                      zip(wl, wt, jt[bounds[:-1]], jt[bounds[1:]])]
                d = np.array([a.end - a.start for a in al])
                d = d[d.nonzero()]
                if len(d):
                    mx = min(0.7, float(np.median(d))) * 2
                    for i in range(1, len(al)):
                        if al[i].end - al[i].start > mx:
                            if al[i].word in ".。!！?？":
                                al[i].end = al[i].start + mx
                            elif al[i - 1].word in ".。!！?？":
                                al[i].start = al[i].end - mx
                timing.merge_punctuations(al, "\"'“¿([{-",
                                          "\"'.。,，!！?？:：”)]}、")
                words = [(a.word, a.start, a.end) for a in al if a.word]

    text = tok.decode(tokens).lstrip(" ")
    print(f"[Whisper] 0.00s ({code}, no_speech {no_speech:.4g}, avg_logprob "
          f"{sum_lp / (len(tokens) + 1):.4g}): {text}")
    for word, s, e in words:
        print(f"  {s:.2f} - {e:.2f}  {word}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?")
    p.add_argument("wav", nargs="?")
    p.add_argument("--language", default="")
    p.add_argument("--make-tiny", metavar="DIR")
    p.add_argument("--encoder-context", default="full", choices=["full", "audio"])
    p.add_argument("--context-pad-sec", type=float, default=0.0)
    p.add_argument("--context-min-sec", type=float, default=0.0)
    a = p.parse_args()
    if a.make_tiny:
        make_tiny(a.make_tiny)
    else:
        transcribe(a.checkpoint, a.wav, a.language, context=a.encoder_context,
                   pad_sec=a.context_pad_sec, min_sec=a.context_min_sec)


if __name__ == "__main__":
    main()
