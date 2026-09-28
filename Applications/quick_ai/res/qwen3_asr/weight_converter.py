# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @file weight_converter.py
## @brief weight conversion script for qwen3-asr model
## @author Hyeonseok Lee <hs89.lee@samsung.com>
##
## Converts a HuggingFace Qwen3-ASR-1.7B model into the nntrainer weight format,
## supporting both the legacy binary (.bin) layout and the safetensors layout.

import argparse
import glob
import json
import os
import struct
import math
import numpy as np
import torch

def get_sinusoid_embedding(length, channels, max_timescale=10000):
    """Generate fixed 1D Sinusoidal Position Embeddings."""
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = np.exp(-log_timescale_increment * np.arange(channels // 2))
    scaled_time = np.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    pos_embed = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    return pos_embed.astype(np.float32)

def tensor_to_numpy(tensor, dtype, transpose=False):
    if transpose:
        tensor = tensor.permute(1, 0)
    tensor = tensor.detach()
    if tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)
    arr = tensor.cpu().numpy()
    np_dtype = np.dtype(dtype)
    if arr.dtype != np_dtype:
        arr = arr.astype(np_dtype)
    return np.ascontiguousarray(arr)

def convert_weights(checkpoint_dir, output_path, dtype="float32", audio_seq_len=9):
    print(f"Loading checkpoint from: {checkpoint_dir}")
    
    # Locate weight files
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    state_dict = {}
    if os.path.exists(index_path):
        from safetensors import safe_open
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        unique_files = set(weight_map.values())
        for f_name in unique_files:
            f_path = os.path.join(checkpoint_dir, f_name)
            with safe_open(f_path, framework="pt", device="cpu") as f_handle:
                for k in f_handle.keys():
                    state_dict[k] = f_handle.get_tensor(k)
    else:
        # Fallback to single safetensors or .bin
        sf_path = os.path.join(checkpoint_dir, "model.safetensors")
        if os.path.exists(sf_path):
            from safetensors import safe_open
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
        else:
            bin_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
            if os.path.exists(bin_path):
                state_dict = torch.load(bin_path, map_location="cpu")
            else:
                raise FileNotFoundError("No weights files found in checkpoint directory.")

    print("Weights loaded. Preparing nntrainer payload...")

    # Output file writers
    is_safetensors = output_path.endswith(".safetensors")
    nntr_weights = {}

    def add_weight(nntr_name, tensor, transpose=False, force_dtype=None):
        target_dtype = force_dtype if force_dtype is not None else dtype
        arr = tensor_to_numpy(tensor, target_dtype, transpose)
        nntr_weights[nntr_name] = arr
        return arr

    # 1. Input embedding (embedding0:Embedding)
    # PyTorch key: model.embed_tokens.weight or thinker.model.embed_tokens.weight
    emb_key = "thinker.model.embed_tokens.weight" if "thinker.model.embed_tokens.weight" in state_dict else "model.embed_tokens.weight"
    if emb_key in state_dict:
        add_weight("embedding0:Embedding", state_dict[emb_key])

    # 2. Audio Tower Subsampler weights (independent sub-model)
    add_weight("audio_tower_conv1:filter", state_dict["thinker.audio_tower.conv2d1.weight"], force_dtype="float32")
    add_weight("audio_tower_conv1:weight", state_dict["thinker.audio_tower.conv2d1.weight"], force_dtype="float32")
    add_weight("audio_tower_conv1:bias", state_dict["thinker.audio_tower.conv2d1.bias"], force_dtype="float32")
    add_weight("audio_tower_conv2:filter", state_dict["thinker.audio_tower.conv2d2.weight"], force_dtype="float32")
    add_weight("audio_tower_conv2:weight", state_dict["thinker.audio_tower.conv2d2.weight"], force_dtype="float32")
    add_weight("audio_tower_conv2:bias", state_dict["thinker.audio_tower.conv2d2.bias"], force_dtype="float32")
    add_weight("audio_tower_conv3:filter", state_dict["thinker.audio_tower.conv2d3.weight"], force_dtype="float32")
    add_weight("audio_tower_conv3:weight", state_dict["thinker.audio_tower.conv2d3.weight"], force_dtype="float32")
    add_weight("audio_tower_conv3:bias", state_dict["thinker.audio_tower.conv2d3.bias"], force_dtype="float32")
    add_weight("audio_tower_conv_out:weight", state_dict["thinker.audio_tower.conv_out.weight"], transpose=True, force_dtype="float32")

    # 3. Audio Tower Positional Embedding (13 tokens for 100-frame chunk in FP32)
    pos_key = "thinker.audio_tower.positional_embedding.positional_embedding"
    if pos_key in state_dict:
        pos_tensor = state_dict[pos_key][:13, :]
    else:
        pos_tensor = torch.tensor(get_sinusoid_embedding(13, 1024))
    add_weight("audio_tower_pos_embed:weights", pos_tensor, force_dtype="float32")

    # 4. Audio Tower 24 Attention Encoder Blocks
    for i in range(24):
        prefix_hf = f"thinker.audio_tower.layers.{i}."
        prefix_nntr = f"audio_tower_layer{i}_"
        
        # Self-attn layer norm
        add_weight(prefix_nntr + "attention_norm:gamma", state_dict[prefix_hf + "self_attn_layer_norm.weight"])
        add_weight(prefix_nntr + "attention_norm:bias", state_dict[prefix_hf + "self_attn_layer_norm.bias"])
        
        # QKV projections
        add_weight(prefix_nntr + "qkv_q:weight", state_dict[prefix_hf + "self_attn.q_proj.weight"], transpose=True)
        add_weight(prefix_nntr + "qkv_q:bias", state_dict[prefix_hf + "self_attn.q_proj.bias"])
        add_weight(prefix_nntr + "qkv_k:weight", state_dict[prefix_hf + "self_attn.k_proj.weight"], transpose=True)
        add_weight(prefix_nntr + "qkv_k:bias", state_dict[prefix_hf + "self_attn.k_proj.bias"])
        add_weight(prefix_nntr + "qkv_v:weight", state_dict[prefix_hf + "self_attn.v_proj.weight"], transpose=True)
        add_weight(prefix_nntr + "qkv_v:bias", state_dict[prefix_hf + "self_attn.v_proj.bias"])

        # Attention out
        add_weight(prefix_nntr + "attention_out:weight", state_dict[prefix_hf + "self_attn.out_proj.weight"], transpose=True)
        add_weight(prefix_nntr + "attention_out:bias", state_dict[prefix_hf + "self_attn.out_proj.bias"])

        # FFN norm
        add_weight(prefix_nntr + "ffn_norm:gamma", state_dict[prefix_hf + "final_layer_norm.weight"])
        add_weight(prefix_nntr + "ffn_norm:bias", state_dict[prefix_hf + "final_layer_norm.bias"])

        # FFN MLP (up/down)
        add_weight(prefix_nntr + "ffn_up:weight", state_dict[prefix_hf + "fc1.weight"], transpose=True)
        add_weight(prefix_nntr + "ffn_up:bias", state_dict[prefix_hf + "fc1.bias"])
        add_weight(prefix_nntr + "ffn_down:weight", state_dict[prefix_hf + "fc2.weight"], transpose=True)
        add_weight(prefix_nntr + "ffn_down:bias", state_dict[prefix_hf + "fc2.bias"])

    # 5. Audio Tower Post-LN & Projections
    add_weight("audio_tower_ln_post:gamma", state_dict["thinker.audio_tower.ln_post.weight"])
    add_weight("audio_tower_ln_post:bias", state_dict["thinker.audio_tower.ln_post.bias"])

    add_weight("audio_tower_proj1:weight", state_dict["thinker.audio_tower.proj1.weight"], transpose=True)
    add_weight("audio_tower_proj1:bias", state_dict["thinker.audio_tower.proj1.bias"])
    add_weight("audio_tower_proj2:weight", state_dict["thinker.audio_tower.proj2.weight"], transpose=True)
    add_weight("audio_tower_proj2:bias", state_dict["thinker.audio_tower.proj2.bias"])

    # 6. Text Transformer Decoder Blocks (28 layers)
    for i in range(28):
        prefix_hf = f"thinker.model.layers.{i}."
        prefix_nntr = f"layer{i}_"

        # Attention Norm
        add_weight(prefix_nntr + "attention_norm:gamma", state_dict[prefix_hf + "input_layernorm.weight"])

        # Q, K, V Projections and Q/K Norm (interlaced in NNTrainer graph order)
        add_weight(prefix_nntr + "wq:weight", state_dict[prefix_hf + "self_attn.q_proj.weight"], transpose=True)
        add_weight(prefix_nntr + "q_norm:gamma", state_dict[prefix_hf + "self_attn.q_norm.weight"])
        add_weight(prefix_nntr + "wk:weight", state_dict[prefix_hf + "self_attn.k_proj.weight"], transpose=True)
        add_weight(prefix_nntr + "k_norm:gamma", state_dict[prefix_hf + "self_attn.k_norm.weight"])
        add_weight(prefix_nntr + "wv:weight", state_dict[prefix_hf + "self_attn.v_proj.weight"], transpose=True)
        add_weight(prefix_nntr + "attention_out:weight", state_dict[prefix_hf + "self_attn.o_proj.weight"], transpose=True)

        # FFN Norm
        add_weight(prefix_nntr + "ffn_norm:gamma", state_dict[prefix_hf + "post_attention_layernorm.weight"])

        # SwiGLU MLP: in NNTrainer SwiGLU, input 0 is ffn_up and input 1 is ffn_gate
        add_weight(prefix_nntr + "ffn_up:weight", state_dict[prefix_hf + "mlp.up_proj.weight"], transpose=True)
        add_weight(prefix_nntr + "ffn_gate:weight", state_dict[prefix_hf + "mlp.gate_proj.weight"], transpose=True)
        add_weight(prefix_nntr + "ffn_down:weight", state_dict[prefix_hf + "mlp.down_proj.weight"], transpose=True)

    # 7. Final RMS Norm
    add_weight("output_norm:gamma", state_dict["thinker.model.norm.weight"])

    # 8. LM Head (Tied or Untied)
    lm_head_key = "thinker.lm_head.weight" if "thinker.lm_head.weight" in state_dict else "lm_head.weight"
    if lm_head_key in state_dict and is_safetensors:
        add_weight("output_of_causallm:Embedding", state_dict[lm_head_key], transpose=True)

    # Save to file
    if is_safetensors:
        from safetensors.numpy import save_file
        save_file(nntr_weights, output_path)
        print(f"Saved weights to safetensors layout: {output_path}")
    else:
        # legacy positional binary save (ordered by keys mapped in flat array)
        with open(output_path, "wb") as f_out:
            for k, val in nntr_weights.items():
                f_out.write(val.tobytes())
        print(f"Saved weights to positional binary layout: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", help="Directory containing HuggingFace checkpoint safetensors")
    parser.add_argument("-o", "--output", required=True, help="Output weights file (.safetensors or .bin)")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"], help="Export data type")
    parser.add_argument("--audio_seq_len", type=int, default=9, help="Downsampled audio sequence length for positional binary")
    args = parser.parse_args()
    convert_weights(args.checkpoint_dir, args.output, args.dtype, args.audio_seq_len)
