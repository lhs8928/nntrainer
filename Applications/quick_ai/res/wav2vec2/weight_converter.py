# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file   weight_converter.py
# @date   04 November 2026
# @brief  HuggingFace Wav2Vec2ForCTC checkpoint -> nntrainer safetensors
# @author Hyeonseok Lee <hs89.lee@samsung.com>
#
# Usage:
#   python weight_converter.py <hf_checkpoint_dir> -o <model_dir>/nntr_wav2vec2.safetensors
#
# For WhisperX's alignment models, e.g. kresnik/wav2vec2-large-xlsr-korean.
# The names follow models/wav2vec2/wav2vec2_ctc.cpp; fully connected weights
# are stored transposed ([in, out]), conv kernels as [out, in, 1, k], the
# positional conv's weight norm folded into its weight, all float32. Copy
# config.json and vocab.json of the checkpoint next to the output, with
# nntr_config.json from this folder.
##

import argparse
import glob
import os

import numpy as np


def load_state_dict(checkpoint_dir):
    state = {}
    files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if files:
        from safetensors import safe_open
        for path in files:
            with safe_open(path, framework="np") as f:
                for key in f.keys():
                    state[key] = f.get_tensor(key).astype(np.float32)
    else:
        import torch
        path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        for key, t in torch.load(path, map_location="cpu").items():
            state[key] = t.float().numpy()
    return state


def pos_conv_weight(state, p):
    """the weight of the weight-normalized positional conv (dim=2)"""
    if p + "weight" in state:
        return state[p + "weight"]
    if p + "parametrizations.weight.original0" in state:
        g = state[p + "parametrizations.weight.original0"]
        v = state[p + "parametrizations.weight.original1"]
    else:
        g, v = state[p + "weight_g"], state[p + "weight_v"]
    norm = np.sqrt((v.astype(np.float64) ** 2).sum(axis=(0, 1), keepdims=True))
    return (v * (g / norm)).astype(np.float32)


def convert(state):
    out = {}

    def put(name, array, transpose=False):
        array = array.T if transpose else array
        out[name] = np.ascontiguousarray(array, dtype=np.float32)

    def linear(nntr, hf):
        put(nntr + ":weight", state[hf + ".weight"], transpose=True)
        put(nntr + ":bias", state[hf + ".bias"])

    def norm(nntr, hf):
        put(nntr + ":gamma", state[hf + ".weight"])
        put(nntr + ":beta", state[hf + ".bias"])

    w2v = "wav2vec2."
    fe = w2v + "feature_extractor.conv_layers."
    n_conv = len({k.split(".")[3] for k in state if k.startswith(fe)})
    for i in range(n_conv):
        w = state[f"{fe}{i}.conv.weight"]  # [out, in, k]
        put(f"feature_encoder:conv{i}_weight",
            w.reshape(w.shape[0], w.shape[1], 1, w.shape[2]))
        if f"{fe}{i}.conv.bias" in state:
            put(f"feature_encoder:conv{i}_bias", state[f"{fe}{i}.conv.bias"])
        if f"{fe}{i}.layer_norm.weight" in state:
            put(f"feature_encoder:conv{i}_norm_gamma",
                state[f"{fe}{i}.layer_norm.weight"])
            put(f"feature_encoder:conv{i}_norm_beta",
                state[f"{fe}{i}.layer_norm.bias"])

    norm("feature_projection_norm", w2v + "feature_projection.layer_norm")
    linear("feature_projection", w2v + "feature_projection.projection")

    pc = w2v + "encoder.pos_conv_embed.conv."
    w = pos_conv_weight(state, pc)  # [C, C / groups, k]
    put("pos_conv:weight", w.reshape(w.shape[0], w.shape[1], 1, w.shape[2]))
    put("pos_conv:bias", state[pc + "bias"])
    norm("encoder_norm", w2v + "encoder.layer_norm")

    enc = w2v + "encoder.layers."
    n_layers = len({k.split(".")[3] for k in state if k.startswith(enc)})
    for i in range(n_layers):
        hf = f"{enc}{i}."
        p = f"enc{i}_"
        linear(p + "q", hf + "attention.q_proj")
        linear(p + "k", hf + "attention.k_proj")
        linear(p + "v", hf + "attention.v_proj")
        linear(p + "attn_out", hf + "attention.out_proj")
        norm(p + "attn_norm", hf + "layer_norm")
        linear(p + "ffn_up", hf + "feed_forward.intermediate_dense")
        linear(p + "ffn_down", hf + "feed_forward.output_dense")
        norm(p + "ffn_norm", hf + "final_layer_norm")
    linear("lm_head", "lm_head")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    from safetensors.numpy import save_file
    weights = convert(load_state_dict(args.checkpoint_dir))
    save_file(weights, args.output)
    print(f"wrote {len(weights)} tensors to {args.output}")


if __name__ == "__main__":
    main()
