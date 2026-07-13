# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @file generate_xlm_roberta_reference.py
## @brief Generate golden fixtures for the XLMRobertaForMaskedLM differential test.
##
## Builds a tiny XLM-RoBERTa encoder (matching the nntrainer XLMRobertaForMaskedLM
## graph), saves its weights in nntrainer order, runs a HuggingFace XLMRobertaModel
## forward pass, and stores the raw last-hidden-state (encode() returns the encoder
## output with no pooling) flattened as the reference.
##
## nntrainer weight order (graph construction order in bert_transformer.cpp):
##   word_emb, position_emb, token_type_emb, embedding LayerNorm(gamma,beta),
##   per layer: Wq(w,b) Wk(w,b) Wv(w,b) Wo(w,b) attn-LN(g,b)
##              ffn_fc1(w,b) ffn_down(w,b) ffn-LN(g,b)
## Fully-connected weights are transposed ([out,in] -> [in,out]); embeddings and
## LayerNorm params are saved as-is.
##
## Key differences from BERT:
##   - type_vocab_size = 1 (token_type_embeddings table shape: [1, hidden])
##   - layer_norm_eps = 1e-5
##   - architectures: XLMRobertaForMaskedLM
##   - position_ids passed explicitly as [0,1,...] to match nntrainer encode()
##     (XLM-RoBERTa HF default uses offset=2, but nntrainer uses 0-based)
##
## Usage:
##   python3 generate_xlm_roberta_reference.py [--out <dir>] [--seed <int>]
##
## Requirements: torch >= 2.0, transformers (XLM-RoBERTa support)

import argparse
import json
import pathlib

import numpy as np
import torch
from transformers import XLMRobertaConfig, XLMRobertaModel

THIS_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUT = THIS_DIR.parent / "xlm_roberta_tiny"

# Tiny XLM-RoBERTa dimensions.
HIDDEN = 64
N_LAYERS = 2
N_HEADS = 4
INTERMEDIATE = 64
VOCAB = 32
MAX_POS = 8
TYPE_VOCAB = 1    # XLM-RoBERTa uses type_vocab_size=1
LN_EPS = 1e-5     # XLM-RoBERTa uses 1e-5, BERT uses 1e-12
SEQ_LEN = 4

INPUT_IDS = [1, 4, 2, 3]
PROMPT = "hello tok4 world tok3"

TINY_TOKENIZER = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {"id": 31, "content": "<eos>", "single_word": False, "lstrip": False,
         "rstrip": False, "normalized": False, "special": True}
    ],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "WordLevel",
        "vocab": {
            "<unk>": 0, "hello": 1, "world": 2,
            **{f"tok{i}": i for i in range(3, 31)},
            "<eos>": 31,
        },
        "unk_token": "<unk>",
    },
}


def build_model(seed: int) -> XLMRobertaModel:
    torch.manual_seed(seed)
    config = XLMRobertaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        intermediate_size=INTERMEDIATE,
        max_position_embeddings=MAX_POS,
        type_vocab_size=TYPE_VOCAB,
        layer_norm_eps=LN_EPS,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=0,
        # Disable the XLM-RoBERTa position offset so HF uses 0-based position_ids
        # (matching nntrainer encode() which sets position_ids = [0, 1, 2, ...])
        position_embedding_type="absolute",
    )
    model = XLMRobertaModel(config, add_pooling_layer=False)
    model.eval()
    return model


def save_xlm_roberta_for_nntrainer(params: dict, f) -> None:
    """Save weights in nntrainer graph construction order (same as BERT)."""
    def save(t):
        np.array(t.detach().float().numpy(), dtype="float32").tofile(f)

    def save_fc(prefix):  # transpose weight, then bias
        save(params[f"{prefix}.weight"].T)
        save(params[f"{prefix}.bias"])

    def save_ln(prefix):  # gamma, beta as-is
        save(params[f"{prefix}.weight"])
        save(params[f"{prefix}.bias"])

    # Embeddings (as-is) + embedding LayerNorm
    save(params["embeddings.word_embeddings.weight"])
    save(params["embeddings.position_embeddings.weight"])
    save(params["embeddings.token_type_embeddings.weight"])
    save_ln("embeddings.LayerNorm")

    for i in range(N_LAYERS):
        p = f"encoder.layer.{i}."
        save_fc(p + "attention.self.query")
        save_fc(p + "attention.self.key")
        save_fc(p + "attention.self.value")
        save_fc(p + "attention.output.dense")
        save_ln(p + "attention.output.LayerNorm")
        save_fc(p + "intermediate.dense")
        save_fc(p + "output.dense")
        save_ln(p + "output.LayerNorm")


def run_embedding(model: XLMRobertaModel, input_ids: list) -> list:
    ids = torch.tensor([input_ids], dtype=torch.long)
    # Pass explicit 0-based position_ids to match nntrainer encode()
    pos_ids = torch.arange(len(input_ids), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        out = model(ids, position_ids=pos_ids)
    # No pooling in nntrainer: encode() returns raw encoder output,
    # i.e. the full [seq_len, hidden] last hidden state, flattened row-major.
    hidden = out.last_hidden_state[0]   # [seq_len, hidden]
    return hidden.float().reshape(-1).tolist()


def write_configs(out_dir: pathlib.Path, bin_name: str,
                  tokenizer_path: str) -> None:
    config_json = {
        "architectures": ["XLMRobertaForMaskedLM"],
        "model_type": "xlm-roberta",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "num_hidden_layers": N_LAYERS,
        "num_attention_heads": N_HEADS,
        "intermediate_size": INTERMEDIATE,
        "max_position_embeddings": MAX_POS,
        "type_vocab_size": TYPE_VOCAB,
        "layer_norm_eps": LN_EPS,
        "hidden_act": "gelu",
        "is_causal": False,
        "tie_word_embeddings": False,
    }
    generation_config_json = {"bos_token_id": 0, "eos_token_id": 31}
    nntr_config_json = {
        "bad_word_ids": [],
        "batch_size": 1,
        "embedding_dtype": "FP32",
        "fc_layer_dtype": "FP32",
        # init_seq_len must exceed SEQ_LEN by at least one (MHA constraint)
        "init_seq_len": SEQ_LEN + 1,
        "lmhead_dtype": "FP32",
        "max_seq_len": MAX_POS,
        "model_file_name": bin_name,
        "model_tensor_type": "FP32-FP32",
        "model_type": "Embedding",
        "num_to_generate": 1,
        "tokenizer_file": tokenizer_path,
        "is_causal": False,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)
    with open(out_dir / "generation_config.json", "w") as f:
        json.dump(generation_config_json, f, indent=2)
    with open(out_dir / "nntr_config.json", "w") as f:
        json.dump(nntr_config_json, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate XLMRobertaForMaskedLM tiny reference fixtures")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir: pathlib.Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[generate] output dir: {out_dir}")

    model = build_model(args.seed)
    print(f"[generate] model params: {sum(p.numel() for p in model.parameters()):,}")

    bin_name = "nntr_xlm_roberta_tiny_fp32.bin"
    with open(out_dir / bin_name, "wb") as f:
        save_xlm_roberta_for_nntrainer(model.state_dict(), f)
    print(f"[converter] saved {out_dir / bin_name} "
          f"({(out_dir / bin_name).stat().st_size / 1024:.1f} KB)")

    tokenizer_path = out_dir / "tokenizer.json"
    with open(tokenizer_path, "w") as f:
        json.dump(TINY_TOKENIZER, f, indent=2)

    write_configs(out_dir, bin_name, str(tokenizer_path))
    print("[generate] configs written")

    ref_embedding = run_embedding(model, INPUT_IDS)
    with open(out_dir / "reference_embedding.json", "w") as f:
        json.dump(ref_embedding, f)
    print(f"[generate] reference embedding: {len(ref_embedding)} values "
          f"(seq={SEQ_LEN} x hidden={HIDDEN})")

    with open(out_dir / "input_ids.json", "w") as f:
        json.dump(INPUT_IDS, f)

    import transformers
    meta = {
        "seed": args.seed,
        "input_ids": INPUT_IDS,
        "prompt": PROMPT,
        "embedding_atol": 2e-2,
        "cosine_min": 0.999,
        "embedding_atol_q40": 0.1,
        "cosine_min_q40": 0.99,
        "pooling": "none",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[generate] meta.json written (transformers={transformers.__version__})")

    print(f"\n[generate] done. Commit: {out_dir}")


if __name__ == "__main__":
    main()
