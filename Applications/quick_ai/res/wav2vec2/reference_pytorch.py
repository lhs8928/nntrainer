# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file   reference_pytorch.py
# @date   04 November 2026
# @brief  PyTorch reference of nntrainer's WhisperX alignment (wav2vec2 CTC)
# @author Hyeonseok Lee <hs89.lee@samsung.com>
#
# Aligns a transcript to a 16 kHz mono wav (one segment from 0 to its end)
# with a HuggingFace Wav2Vec2ForCTC checkpoint, the way whisperx/alignment.py
# (WhisperX 3.8.6) does for one segment: the waveform alone through the model
# (no processor, no mask), log_softmax, characters lower-cased with spaces as
# "|", unknown characters as a wildcard (best non-blank score), trellis,
# backtrack, merge repeats, times rounded to 1 ms, words on spaces. WhisperX's
# sentence split (nltk punkt) is left out: it only regroups the same words.
# Prints the lines of
#   nntr_quick_ai <model_dir> --audio <wav> --align-text <text> --language ko
#
#   python reference_pytorch.py <hf_checkpoint_dir> <wav> "<text>" [--language ko]
#   python reference_pytorch.py --make-tiny <dir> [--post-ln] "<text>"
#
# --make-tiny writes a random small checkpoint whose vocab.json holds the
# characters of <text> but its digits and punctuation (they become
# wildcards); --post-ln makes it a base-style model (post-LN encoder,
# GroupNorm feature encoder, no conv bias) instead of the XLS-R layout.
##

import argparse
import json
import math
import os
import wave

import numpy as np
import torch
from transformers import Wav2Vec2Config, Wav2Vec2ForCTC

NO_SPACE = ["ja", "zh"]


def make_tiny(out, text, post_ln):
    os.makedirs(out, exist_ok=True)
    chars = sorted({c for c in text.lower() if c.isalpha()})
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4}
    for c in chars:
        vocab[c] = len(vocab)
    torch.manual_seed(0)
    cfg = Wav2Vec2Config(
        vocab_size=len(vocab), hidden_size=64, num_hidden_layers=2,
        num_attention_heads=4, intermediate_size=128, conv_dim=(32,) * 7,
        num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=4,
        do_stable_layer_norm=not post_ln,
        feat_extract_norm="group" if post_ln else "layer",
        conv_bias=not post_ln, initializer_range=0.2, pad_token_id=0)
    Wav2Vec2ForCTC(cfg).save_pretrained(out)
    with open(os.path.join(out, "vocab.json"), "w") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"wrote a random tiny wav2vec2 to {out}")


def load_wav(path):
    with wave.open(path) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError("expects 16 kHz mono 16-bit wav")
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return pcm.astype(np.float32) / 32768.0


def get_trellis(emission, tokens, blank_id):
    num_frame, num_tokens = emission.size(0), len(tokens)
    trellis = torch.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = torch.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")
    for t in range(num_frame):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emission[t, blank_id],
            trellis[t, :-1] + emission[t, tokens])
    return trellis


def backtrack(trellis, emission, tokens, blank_id):
    j = trellis.size(1) - 1
    t_start = torch.argmax(trellis[:, j]).item()
    path = []
    for t in range(t_start, 0, -1):
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]
        prob = emission[t - 1, tokens[j - 1] if changed > stayed else blank_id].exp().item()
        path.append((j - 1, t - 1, prob))
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        return None
    return path[::-1]


def merge_repeats(path):
    i1, i2, segs = 0, 0, []
    while i1 < len(path):
        while i2 < len(path) and path[i1][0] == path[i2][0]:
            i2 += 1
        score = sum(path[k][2] for k in range(i1, i2)) / (i2 - i1)
        segs.append((path[i1][1], path[i2 - 1][1] + 1, score))
        i1 = i2
    return segs


def align(ckpt, wav, text, language, normalize=False):
    model = Wav2Vec2ForCTC.from_pretrained(ckpt).eval()
    # WhisperX lower-cases the dictionary ("[PAD]" -> "[pad]", the blank)
    dictionary = {c.lower(): i for c, i in
                  json.load(open(os.path.join(ckpt, "vocab.json"))).items()}
    audio = torch.from_numpy(load_wav(wav)[None])
    t1, t2 = 0.0, audio.shape[1] / 16000

    num_leading = len(text) - len(text.lstrip())
    num_trailing = len(text) - len(text.rstrip())
    clean_char, clean_cdx = [], []
    for cdx, char in enumerate(text):
        c = char.lower()
        if language not in NO_SPACE:
            c = c.replace(" ", "|")
        if cdx < num_leading or cdx > len(text) - num_trailing - 1:
            continue
        if c in dictionary or c not in (" ", "|"):
            clean_char.append(c)
            clean_cdx.append(cdx)
    if not clean_char:
        return []

    seg = audio
    if normalize:  # the feature extractor's do_normalize (not WhisperX's)
        seg = (seg - seg.mean()) / torch.sqrt(seg.var(unbiased=False) + 1e-7)
    if seg.shape[-1] < 400:
        seg = torch.nn.functional.pad(seg, (0, 400 - seg.shape[-1]))
    with torch.inference_mode():
        emission = torch.log_softmax(model(seg).logits, dim=-1)[0]
    blank_id = 0
    for ch, code in dictionary.items():
        if ch in ("[pad]", "<pad>"):
            blank_id = code
    text_clean = "".join(clean_char)
    if any(c not in dictionary for c in text_clean):
        mask = torch.ones(emission.size(1), dtype=torch.bool)
        mask[blank_id] = False
        emission = torch.cat(
            [emission, emission[:, mask].max(dim=1).values.unsqueeze(1)], dim=1)
        wildcard = emission.size(1) - 1
        tokens = [dictionary.get(c, wildcard) for c in text_clean]
    else:
        tokens = [dictionary[c] for c in text_clean]
    trellis = get_trellis(emission, tokens, blank_id)
    path = backtrack(trellis, emission, tokens, blank_id)
    if path is None:
        return []
    segs = merge_repeats(path)
    ratio = (t2 - t1) / (trellis.size(0) - 1)

    chars, word_idx = [], 0
    for cdx, char in enumerate(text):
        start = end = None
        if cdx in clean_cdx:
            s = segs[clean_cdx.index(cdx)]
            start, end = round(s[0] * ratio + t1, 3), round(s[1] * ratio + t1, 3)
        chars.append((char, start, end, word_idx))
        if language in NO_SPACE or cdx == len(text) - 1 or text[cdx + 1] == " ":
            word_idx += 1
    words = []
    for w in sorted({c[3] for c in chars}):
        wc = [c for c in chars if c[3] == w]
        word = "".join(c[0] for c in wc).strip()
        if not word:
            continue
        timed = [c for c in wc if c[0] != " " and c[1] is not None]
        if timed:
            words.append((word, min(c[1] for c in timed), max(c[2] for c in timed)))
        else:
            words.append((word, math.nan, math.nan))
    return words


def main():
    p = argparse.ArgumentParser()
    p.add_argument("args", nargs="*")
    p.add_argument("--language", default="")
    p.add_argument("--make-tiny", metavar="DIR")
    p.add_argument("--post-ln", action="store_true")
    p.add_argument("--normalize", action="store_true",
                   help="normalize the waveform first (nntr_config normalize_input)")
    a = p.parse_args()
    if a.make_tiny:
        make_tiny(a.make_tiny, a.args[0], a.post_ln)
        return
    ckpt, wav, text = a.args
    for word, s, e in align(ckpt, wav, text, a.language, a.normalize):
        print(f"  {s:.3f} - {e:.3f}  {word}")


if __name__ == "__main__":
    main()
