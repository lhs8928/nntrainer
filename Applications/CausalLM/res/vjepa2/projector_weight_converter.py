## SPDX-License-Identifier: Apache-2.0
## Copyright (C) 2026 Jungwon-Lee <jungone.lee@samsung.com>
##
## @file projector_weight_converter.py
## @brief Weight conversion for V-JEPA 2.1 Projector/Merger (vision → LLM embedding).
## @author Jungwon-Lee <jungone.lee@samsung.com>
##
## Extracts the merger weights from a HuggingFace VoRA merged checkpoint
## and writes them in the nntrainer binary format expected by VjepaProjector.
##
## The merger graph (VjepaProjector::constructModel) creates layers in this
## order, so the weight file must follow the same order:
##
##   merger_ln1  (LayerNorm: weight[3072], bias[3072])
##   merger_fc1  (FC: weight[3072,3072], bias[3072])
##   merger_gelu1 (no weights)
##   merger_fc2  (FC: weight[3072,1536], bias[1536])
##   merger_gelu2 (no weights)
##   merger_ln2  (LayerNorm: weight[1536], bias[1536])
##   merger_fc3  (FC: weight[1536,1024], bias[1024])
##   merger_ln3  (LayerNorm: weight[1024], bias[1024])
##
## HuggingFace key mapping (VoRA Lfm2VLVJepa21BModel):
##   merger.0.weight / merger.0.bias     → merger_ln1
##   merger.1.weight / merger.1.bias     → merger_fc1
##   merger.3.weight / merger.3.bias     → merger_fc2
##   merger.5.weight / merger.5.bias     → merger_ln2
##   merger.6.weight / merger.6.bias     → merger_fc3
##   merger.7.weight / merger.7.bias     → merger_ln3

import argparse
import numpy as np

try:
    from safetensors.torch import load_file
except ImportError:
    load_file = None


def save_weight(weight, dtype, file, transpose=False):
    """Save a tensor to nntrainer format (optionally transposing OI -> IO)."""
    if isinstance(weight, np.ndarray):
        array = weight
    else:
        # Convert to float32 first to handle bfloat16 tensors (numpy doesn't
        # support bfloat16 natively)
        array = weight.detach().cpu().float().numpy()
    if transpose and array.ndim >= 2:
        array = array.T
    array.astype(dtype).tofile(file)


def convert_from_safetensors(model_path, output_path, dtype):
    """Convert merger weights from a safetensors checkpoint."""
    if load_file is None:
        raise ImportError("safetensors package is required. Install with: pip install safetensors")

    sd = load_file(model_path)

    # Find merger keys
    merger_keys = {k for k in sd.keys() if "merger" in k.lower() or "projector" in k.lower()}

    if not merger_keys:
        # Print available keys for debugging
        print("No merger/projector keys found. Available keys (first 30):")
        for i, k in enumerate(sorted(sd.keys())):
            if i >= 30:
                break
            print(f"  {k} {tuple(sd[k].shape)}")
        raise RuntimeError("Could not find merger keys in checkpoint.")

    print("Found merger/projector keys:")
    for k in sorted(merger_keys):
        print(f"  {k} {tuple(sd[k].shape)}")

    # Build key mapping for the Sequential merger:
    #   0: LayerNorm(3072)
    #   1: Linear(3072→3072)
    #   2: GELU
    #   3: Linear(3072→1536)
    #   4: GELU
    #   5: LayerNorm(1536)
    #   6: Linear(1536→1024)
    #   7: LayerNorm(1024)

    # Try different key patterns
    def find_key(patterns):
        for p in patterns:
            for k in merger_keys:
                if p in k.lower():
                    return k
        return None

    # LayerNorm1 (index 0)
    ln1_w = find_key(["merger.0.weight"])
    ln1_b = find_key(["merger.0.bias"])

    # FC1 (index 1)
    fc1_w = find_key(["merger.1.weight"])
    fc1_b = find_key(["merger.1.bias"])

    # FC2 (index 3)
    fc2_w = find_key(["merger.3.weight"])
    fc2_b = find_key(["merger.3.bias"])

    # LayerNorm2 (index 5)
    ln2_w = find_key(["merger.5.weight"])
    ln2_b = find_key(["merger.5.bias"])

    # FC3 (index 6)
    fc3_w = find_key(["merger.6.weight"])
    fc3_b = find_key(["merger.6.bias"])

    # LayerNorm3 (index 7)
    ln3_w = find_key(["merger.7.weight"])
    ln3_b = find_key(["merger.7.bias"])

    keys = {
        "ln1_w": ln1_w, "ln1_b": ln1_b,
        "fc1_w": fc1_w, "fc1_b": fc1_b,
        "fc2_w": fc2_w, "fc2_b": fc2_b,
        "ln2_w": ln2_w, "ln2_b": ln2_b,
        "fc3_w": fc3_w, "fc3_b": fc3_b,
        "ln3_w": ln3_w, "ln3_b": ln3_b,
    }

    missing = [k for k, v in keys.items() if v is None]
    if missing:
        print(f"\nMissing key mappings: {missing}")
        print("Available merger keys:")
        for k in sorted(merger_keys):
            print(f"  {k} {tuple(sd[k].shape)}")
        raise RuntimeError(f"Could not find all merger keys. Missing: {missing}")

    print(f"\nKey mapping:")
    for name, key in keys.items():
        print(f"  {name}: {key} {tuple(sd[key].shape)}")

    print(f"\nWriting merger weights to: {output_path}")
    with open(output_path, "wb") as f:
        # merger_ln1: LayerNorm(3072)
        save_weight(sd[ln1_w], dtype, f)  # weight
        save_weight(sd[ln1_b], dtype, f)  # bias

        # merger_fc1: FC(3072→3072)
        save_weight(sd[fc1_w], dtype, f, transpose=True)  # weight
        save_weight(sd[fc1_b], dtype, f)  # bias

        # merger_fc2: FC(3072→1536)
        save_weight(sd[fc2_w], dtype, f, transpose=True)  # weight
        save_weight(sd[fc2_b], dtype, f)  # bias

        # merger_ln2: LayerNorm(1536)
        save_weight(sd[ln2_w], dtype, f)  # weight
        save_weight(sd[ln2_b], dtype, f)  # bias

        # merger_fc3: FC(1536→1024)
        save_weight(sd[fc3_w], dtype, f, transpose=True)  # weight
        save_weight(sd[fc3_b], dtype, f)  # bias

        # merger_ln3: LayerNorm(1024)
        save_weight(sd[ln3_w], dtype, f)  # weight
        save_weight(sd[ln3_b], dtype, f)  # bias

    print("Merger weight conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert V-JEPA 2.1 Merger weights to nntrainer format")
    parser.add_argument("--input", type=str, required=True,
                        help="Input checkpoint (safetensors or PyTorch .pt/.bin)")
    parser.add_argument("--output", type=str, default="nntr_projector_fp32.bin",
                        help="Output nntrainer weight file")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16"])
    args = parser.parse_args()

    dtype = {"float32": np.float32, "float16": np.float16}[args.dtype]

    if args.input.endswith(".safetensors"):
        convert_from_safetensors(args.input, args.output, dtype)
    else:
        import torch
        sd = torch.load(args.input, map_location="cpu")
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        elif isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        # For PyTorch checkpoints, we need to save as safetensors first
        # or handle directly. For simplicity, save to temp safetensors.
        from safetensors.torch import save_file as save_safetensors
        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            tmp_path = tmp.name

        # Filter only merger keys
        merger_sd = {k: v for k, v in sd.items() if "merger" in k.lower() or "projector" in k.lower()}
        if not merger_sd:
            raise RuntimeError("No merger/projector keys found in PyTorch checkpoint")

        save_safetensors(merger_sd, tmp_path)
        try:
            convert_from_safetensors(tmp_path, args.output, dtype)
        finally:
            os.unlink(tmp_path)
