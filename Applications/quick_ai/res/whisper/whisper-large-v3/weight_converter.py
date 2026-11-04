# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file   weight_converter.py
# @date   04 November 2026
# @brief  HuggingFace Whisper checkpoint -> nntrainer safetensors
# @author Hyeonseok Lee <hs89.lee@samsung.com>
#
# Usage:
#   python weight_converter.py <hf_checkpoint_dir> -o <model_dir>/<file>.safetensors
#
# The names follow models/whisper/whisper_causallm.cpp. Fully connected
# weights are stored transposed ([in, out]), the conv kernels as
# [out, in, 1, k]. Every tensor is written as float32; quantize afterwards
# with nntr_quantize if needed. Copy config.json, generation_config.json and
# tokenizer.json of the checkpoint next to the output, with nntr_config.json
# from this folder.
##

import argparse
import glob
import os

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def load_state_dict(checkpoint_dir):
    state = {}
    files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {checkpoint_dir}")
    for path in files:
        with safe_open(path, framework="np") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key).astype(np.float32)
    # some checkpoints store the weights without the "model." prefix
    return {k if k.startswith(("model.", "proj_out")) else "model." + k: v
            for k, v in state.items()}


def convert(state):
    out = {}

    def put(name, array, transpose=False):
        array = array.T if transpose else array
        out[name] = np.ascontiguousarray(array, dtype=np.float32)

    def linear(nntr, hf, bias=True):
        put(nntr + ":weight", state[hf + ".weight"], transpose=True)
        if bias:
            put(nntr + ":bias", state[hf + ".bias"])

    def norm(nntr, hf):
        put(nntr + ":gamma", state[hf + ".weight"])
        put(nntr + ":beta", state[hf + ".bias"])

    enc = "model.encoder."
    for i in (1, 2):
        w = state[f"{enc}conv{i}.weight"]  # [out, in, k]
        put(f"encoder_conv{i}:filter", w.reshape(w.shape[0], w.shape[1], 1, w.shape[2]))
        put(f"encoder_conv{i}:bias", state[f"{enc}conv{i}.bias"])
    put("encoder_pos_embed:weight", state[enc + "embed_positions.weight"])

    n_enc = len({k.split(".")[3] for k in state if k.startswith(enc + "layers.")})
    for i in range(n_enc):
        hf = f"{enc}layers.{i}."
        p = f"enc{i}_"
        norm(p + "attn_norm", hf + "self_attn_layer_norm")
        linear(p + "q", hf + "self_attn.q_proj")
        linear(p + "k", hf + "self_attn.k_proj", bias=False)
        linear(p + "v", hf + "self_attn.v_proj")
        linear(p + "attn_out", hf + "self_attn.out_proj")
        norm(p + "ffn_norm", hf + "final_layer_norm")
        linear(p + "ffn_up", hf + "fc1")
        linear(p + "ffn_down", hf + "fc2")
    norm("encoder_norm", enc + "layer_norm")

    dec = "model.decoder."
    put("embedding0:Embedding", state[dec + "embed_tokens.weight"])
    put("decoder_pos_embed:weight", state[dec + "embed_positions.weight"])
    n_dec = len({k.split(".")[3] for k in state if k.startswith(dec + "layers.")})
    for i in range(n_dec):
        hf = f"{dec}layers.{i}."
        p = f"dec{i}_"
        norm(p + "self_attn_norm", hf + "self_attn_layer_norm")
        linear(p + "self_q", hf + "self_attn.q_proj")
        linear(p + "self_k", hf + "self_attn.k_proj", bias=False)
        linear(p + "self_v", hf + "self_attn.v_proj")
        linear(p + "self_out", hf + "self_attn.out_proj")
        norm(p + "cross_attn_norm", hf + "encoder_attn_layer_norm")
        linear(p + "cross_q", hf + "encoder_attn.q_proj")
        # the encoder graph projects its output to these keys and values
        linear(p + "cross_k", hf + "encoder_attn.k_proj", bias=False)
        linear(p + "cross_v", hf + "encoder_attn.v_proj")
        linear(p + "cross_out", hf + "encoder_attn.out_proj")
        norm(p + "ffn_norm", hf + "final_layer_norm")
        linear(p + "ffn_up", hf + "fc1")
        linear(p + "ffn_down", hf + "fc2")
    norm("decoder_norm", dec + "layer_norm")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    weights = convert(load_state_dict(args.checkpoint_dir))
    save_file(weights, args.output)
    print(f"wrote {len(weights)} tensors to {args.output}")


if __name__ == "__main__":
    main()
