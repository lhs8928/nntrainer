## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
##
## @file weight_converter.py
## @brief weight conversion script for vit model (timm version)
## @author SeungBaek Hong <sb92.hong@samsung.com>

import argparse
import numpy as np
import safetensors.torch
import torch


def load_state_dict(model_path):
    """Load a timm checkpoint from safetensors or PyTorch bin format."""
    if model_path.endswith(".safetensors"):
        state_dict = safetensors.torch.load_file(model_path)
    else:
        state_dict = torch.load(model_path, map_location="cpu")

    if isinstance(state_dict, dict):
        for key in ("state_dict", "model"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break

    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }

def save_weight(weight, dtype, file, transpose=False):
    """Save weight tensor to nntrainer format.

    Args:
        weight: PyTorch tensor or numpy array
        dtype: numpy dtype (e.g., np.float32)
        file: open file object
        transpose: whether to transpose (PyTorch uses OI, nntrainer uses IO)
    """
    if isinstance(weight, np.ndarray):
        array = weight
    else:
        array = weight.detach().cpu().numpy()

    if transpose:
        # PyTorch: [out_dim, in_dim, ...] -> nntrainer: [in_dim, out_dim, ...]
        if array.ndim >= 2:
            array = array.T

    array.astype(dtype).tofile(file)

def convert_timm_vit_to_nntrainer(model_path, output_path, dtype=np.float32):
    """Convert timm ViT weights to nntrainer format."""

    print(f"Loading model from: {model_path}")
    state_dict = load_state_dict(model_path)

    print(f"Converting weights to: {output_path}")

    with open(output_path, 'wb') as f:
        # 1. Patch embedding (Conv2D)
        # PyTorch: [out_channels, in_channels, H, W] = [768, 3, 16, 16]
        print("  Processing patch embedding...")
        patch_weight = state_dict['patch_embed.proj.weight']
        save_weight(patch_weight, dtype, f, transpose=False)  # Conv2D: no transpose

        if 'patch_embed.proj.bias' in state_dict:
            save_weight(state_dict['patch_embed.proj.bias'], dtype, f)

        # 2. Position embedding
        # PyTorch: [1, 196, 768] -> reshape to [1, 1, 196, 768] for nntrainer
        print("  Processing position embedding...")
        pos_embed_reshaped = state_dict['pos_embed'].unsqueeze(1)  # [1, 1, 196, 768]
        save_weight(pos_embed_reshaped, dtype, f, transpose=False)

        # 3. Transformer blocks
        num_layers = 12  # vit_base
        print(f"  Processing {num_layers} transformer blocks...")
        for i in range(num_layers):
            layer_prefix = f'blocks.{i}.'

            # LayerNorm 1
            save_weight(state_dict[layer_prefix + 'norm1.weight'], dtype, f)
            save_weight(state_dict[layer_prefix + 'norm1.bias'], dtype, f)

            # Attention QKV (split for nntrainer Q->K->V layer order)
            # PyTorch: [3*768, 768] -> split into Q, K, V
            qkv_weight = state_dict[layer_prefix + 'attn.qkv.weight']
            qkv_bias = state_dict[layer_prefix + 'attn.qkv.bias']

            dim = 768  # DIM
            Q_weight = qkv_weight[:dim, :]      # [768, 768]
            K_weight = qkv_weight[dim:2*dim, :] # [768, 768]
            V_weight = qkv_weight[2*dim:, :]    # [768, 768]

            Q_bias = qkv_bias[:dim]
            K_bias = qkv_bias[dim:2*dim]
            V_bias = qkv_bias[2*dim:]

            # nntrainer order follows TimmViTTransformer::createAttention():
            # q -> k -> v.
            save_weight(Q_weight, dtype, f, transpose=True)  # attn.q
            save_weight(Q_bias, dtype, f)
            save_weight(K_weight, dtype, f, transpose=True)  # attn.k
            save_weight(K_bias, dtype, f)
            save_weight(V_weight, dtype, f, transpose=True)  # attn.v
            save_weight(V_bias, dtype, f)

            # Attention Output
            save_weight(state_dict[layer_prefix + 'attn.proj.weight'], dtype, f, transpose=True)
            save_weight(state_dict[layer_prefix + 'attn.proj.bias'], dtype, f)

            # LayerNorm 2
            save_weight(state_dict[layer_prefix + 'norm2.weight'], dtype, f)
            save_weight(state_dict[layer_prefix + 'norm2.bias'], dtype, f)

            # MLP FC1
            save_weight(state_dict[layer_prefix + 'mlp.fc1.weight'], dtype, f, transpose=True)
            save_weight(state_dict[layer_prefix + 'mlp.fc1.bias'], dtype, f)

            # MLP FC2
            save_weight(state_dict[layer_prefix + 'mlp.fc2.weight'], dtype, f, transpose=True)
            save_weight(state_dict[layer_prefix + 'mlp.fc2.bias'], dtype, f)

            print(f"    Layer {i+1}/{num_layers} done")

        # 4. Final normalization
        print("  Processing final normalization...")
        save_weight(state_dict['norm.weight'], dtype, f)
        save_weight(state_dict['norm.bias'], dtype, f)

        # 5. Attention Pool (for "global_pool": "map")
        if 'attn_pool.latent' in state_dict:
            print("  Processing attention pool...")
            # Learnable latent query
            save_weight(state_dict['attn_pool.latent'], dtype, f, transpose=False)

            # Query projection
            save_weight(state_dict['attn_pool.q.weight'], dtype, f, transpose=True)
            save_weight(state_dict['attn_pool.q.bias'], dtype, f)

            # Key-Value projection (fused)
            save_weight(state_dict['attn_pool.kv.weight'], dtype, f, transpose=True)
            save_weight(state_dict['attn_pool.kv.bias'], dtype, f)

            # Attention output projection
            save_weight(state_dict['attn_pool.proj.weight'], dtype, f, transpose=True)
            save_weight(state_dict['attn_pool.proj.bias'], dtype, f)

            # LayerNorm
            save_weight(state_dict['attn_pool.norm.weight'], dtype, f)
            save_weight(state_dict['attn_pool.norm.bias'], dtype, f)

            # MLP
            save_weight(state_dict['attn_pool.mlp.fc1.weight'], dtype, f, transpose=True)
            save_weight(state_dict['attn_pool.mlp.fc1.bias'], dtype, f)
            save_weight(state_dict['attn_pool.mlp.fc2.weight'], dtype, f, transpose=True)
            save_weight(state_dict['attn_pool.mlp.fc2.bias'], dtype, f)

        # 6. Head (none for embedding-only model)
        print("  Skipping head (num_classes=0)")

    print(f"\nConversion complete!")
    print(f"  Total weights saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                       default="model.safetensors",
                       help="Input model path (.safetensors or .bin)")
    parser.add_argument("--output", type=str,
                       default="nntr_vit_base_fp32.bin",
                       help="Output nntrainer weight file")
    parser.add_argument("--dtype", type=str, default="float32",
                       help="Data type (float32 or float16)")

    args = parser.parse_args()

    dtype_map = {
        'float32': np.float32,
        'float16': np.float16
    }
    dtype = dtype_map.get(args.dtype, np.float32)

    convert_timm_vit_to_nntrainer(args.input, args.output, dtype)
