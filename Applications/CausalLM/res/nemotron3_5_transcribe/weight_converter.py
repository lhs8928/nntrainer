## SPDX-License-Identifier: Apache-2.0
## @file   weight_converter.py
## @brief  Convert a Nemotron 3.5 Transcribe checkpoint to the NNTrainer format
##
## Reads the safetensors file(s) one tensor at a time, so converting the 30B
## checkpoint does not need the whole model in memory. Both the original NeMo
## export (`perception.*`, `llm.*`, fused `w_qkv`, stacked experts) and the
## HuggingFace transformers naming (`model.audio_tower.*`, ...) are accepted.
##
## Outputs (in --out_dir):
##   nntr_nemotron_h.bin          LLM weights in NNTrainer graph order
##   nntr_nemotron_h_embedding.bin  LLM token embedding table [vocab, hidden]
##   nntr_audio.bin               audio tower + projector (self describing)
##   config.json                  LLM config read by NemotronHCausalLM
##   generation_config.json
##   nntr_config.json             runtime settings (edit paths / lengths)
##   transcribe_config.json       audio front-end / encoder / prompt settings
##
## Usage:
##   python weight_converter.py --model_path <hf dir> --out_dir <out dir>

import argparse
import glob
import json
import os
import re
import struct

import numpy as np
import torch
from safetensors import safe_open

AUDIO_MAGIC = b"NNTRAUD1"

# NeMo export -> transformers names (transformers' `conversion_mapping.py`,
# `nemo_speechlm` entry).
_DIAR = r"^perception\.encoder\.diarization_model\.sortformer_modules\."
RENAMES = [
    (r"^perception\.encoder\.asr_encoder\.", "model.audio_tower.asr_encoder."),
    (
        r"^perception\.encoder\.diarization_model\.encoder\.",
        "model.audio_tower.diarization_model.model.audio_tower.",
    ),
    (r"^ctc_timestamp\.decoder\.transformer\.", "model.word_timestamp_encoder."),
    (r"^ctc_timestamp\.decoder\.upsample\.", "model.word_timestamp_encoder.upsampler."),
    (r"^ctc_timestamp\.decoder\.decoder_layers\.0\.", "word_timestamp_head."),
    (_DIAR + r"encoder_proj\.", "model.audio_tower.diarization_model.model.proj."),
    (_DIAR + r"subpixel_upsample\.", "model.audio_tower.diarization_model.model.upsampler.conv."),
    (_DIAR + r"first_hidden_to_hidden\.", "model.audio_tower.diarization_model.classifier.dense."),
    (_DIAR + r"single_hidden_to_spks\.", "model.audio_tower.diarization_model.classifier.out_proj."),
    (_DIAR + r"learnable_sil_emb", "model.audio_tower.diarization_model.silence_embeds"),
    (r"^perception\.encoder\.asr_norm\.", "model.audio_tower.asr_norm."),
    (r"^perception\.encoder\.diar_norm\.", "model.audio_tower.speaker_norm."),
    (r"^perception\.proj\.", "model.multi_modal_projector."),
    (r"^llm\.model\.embed_tokens\.", "model.language_model.embeddings."),
    (r"^llm\.model\.norm\.", "model.language_model.norm_f."),
    (r"^llm\.model\.layers\.", "model.language_model.layers."),
    (r"^llm\.lm_head\.", "lm_head."),
    (r"\.pre_encode\.proj\.", ".embedder.projection."),
    (r"\.embed_norm\.", ".input_layer_norm."),
    (r"\.final_norm\.", ".layer_norm."),
    (r"\.norm1\.", ".layer_norm1."),
    (r"\.norm2\.", ".layer_norm2."),
    (r"\.attn\.out_proj\.", ".self_attn.o_proj."),
    (r"\.attn\.q_norm\.", ".self_attn.q_norm."),
    (r"\.attn\.k_norm\.", ".self_attn.k_norm."),
    (r"\.ffn\.net\.0\.", ".mlp.fc1."),
    (r"\.ffn\.net\.3\.", ".mlp.fc2."),
    (r"\.mixer\._fp32_params\.A_log", ".mixer.A_log"),
    (r"\.mixer\._fp32_params\.D", ".mixer.D"),
    (r"\.mixer\._fp32_params\.dt_bias", ".mixer.dt_bias"),
]

IGNORED = [
    r"^llm\.mtp\.",
    r"^perception\.preprocessor\.",
    r"^perception\.encoder\.diarization_model\.preprocessor\.",
    r"\._extra_state$",
    r"^ctc_timestamp\.decoder\.residual_scale$",
    r"^ctc_timestamp\.decoder\.upsample_factor$",
    r"^ctc_timestamp\.tokenizer\.",
    r"^mtp\.",
    # an unused speaker head of the diarization model (transformers ignores
    # it too)
    r"^perception\.encoder\.diarization_model\.sortformer_modules\.hidden_to_spks\.",
]


def canonical_name(key):
    for pattern, repl in RENAMES:
        key = re.sub(pattern, repl, key)
    return key


class CheckpointReader:
    """Lazy tensor access by transformers name, split / transposed on demand."""

    def __init__(self, model_path):
        files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
        if not files:
            raise FileNotFoundError(f"no *.safetensors in {model_path}")
        self.handles = [safe_open(f, framework="pt") for f in files]
        self.entries = {}  # name -> (handle, key, op)
        self._cache_key = None
        self._cache_val = None
        for h in self.handles:
            for key in h.keys():
                if any(re.search(p, key) for p in IGNORED):
                    continue
                self._register(h, key)

    def _register(self, h, key):
        name = canonical_name(key)
        shape = h.get_slice(key).get_shape()
        if name.endswith(".attn.w_qkv.weight"):
            base = name[: -len(".attn.w_qkv.weight")] + ".self_attn."
            for i, proj in enumerate(("q_proj", "k_proj", "v_proj")):
                self.entries[base + proj + ".weight"] = (h, key, ("chunk", i, 3))
        elif name.endswith(".mixer.experts.gate_and_up_projs"):
            # NeMo: (E, in, out) -> per expert (out, in)
            base = name[: -len("gate_and_up_projs")]
            for e in range(shape[0]):
                self.entries[f"{base}{e}.up_proj.weight"] = (h, key, ("expert_t", e))
        elif name.endswith(".mixer.experts.down_projs"):
            base = name[: -len("down_projs")]
            for e in range(shape[0]):
                self.entries[f"{base}{e}.down_proj.weight"] = (h, key, ("expert_t", e))
        elif re.search(r"\.mixer\.experts\.(up|down)_proj$", name):
            # transformers stacked layout: (E, out, in)
            base, proj = name.rsplit(".", 1)
            for e in range(shape[0]):
                self.entries[f"{base}.{e}.{proj}.weight"] = (h, key, ("expert", e))
        else:
            self.entries[name] = (h, key, None)

    def __contains__(self, name):
        return name in self.entries

    def names(self):
        return list(self.entries)

    def _raw(self, h, key):
        if self._cache_key != (id(h), key):
            self._cache_val = h.get_tensor(key)
            self._cache_key = (id(h), key)
        return self._cache_val

    def get(self, name):
        h, key, op = self.entries[name]
        t = self._raw(h, key)
        if op is None:
            out = t
        elif op[0] == "chunk":
            out = torch.chunk(t, op[2], dim=0)[op[1]]
        elif op[0] == "expert_t":
            out = t[op[1]].transpose(0, 1)
        elif op[0] == "expert":
            out = t[op[1]]
        else:
            raise ValueError(op)
        return out.to(torch.float32).contiguous()


def json_safe(obj):
    """json.dump writes inf / nan as bare literals, which strict JSON parsers
    (nlohmann) reject: store them as null."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(json_safe(obj), f, indent=2, allow_nan=False)


def load_config(model_path):
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(model_path)


def layer_types(text_cfg):
    types = getattr(text_cfg, "layers_block_type", None) or text_cfg.layer_types
    m = {"linear_attention": "M", "moe": "E", "full_attention": "*", "mlp": "-"}
    return [m[t] for t in types]


class Writer:
    def __init__(self, path, dtype):
        self.f = open(path, "wb")
        self.dtype = dtype
        self.count = 0

    def write(self, tensor, dtype=None):
        arr = tensor.detach().cpu().numpy().astype(dtype or self.dtype)
        arr.tofile(self.f)
        self.count += arr.size

    def close(self):
        self.f.close()


def convert_llm(reader, text_cfg, out_dir, fc_dtype, embedding_dtype):
    """Weights in the order NemotronHCausalLM::constructModel() requests them."""
    pre = "model.language_model."
    types = layer_types(text_cfg)
    hidden = text_cfg.hidden_size

    emb = reader.get(pre + "embeddings.weight")
    emb.numpy().astype(embedding_dtype).tofile(
        os.path.join(out_dir, "nntr_nemotron_h_embedding.bin")
    )

    w = Writer(os.path.join(out_dir, "nntr_nemotron_h.bin"), fc_dtype)
    f32 = np.float32

    def fc(name):  # torch Linear [out, in] -> nntrainer [in, out]
        w.write(reader.get(name).transpose(0, 1))
        bias = name[: -len("weight")] + "bias"
        if bias in reader:
            w.write(reader.get(bias), f32)

    for i, t in enumerate(types):
        lp = f"{pre}layers.{i}."
        w.write(reader.get(lp + "norm.weight"), f32)
        mx = lp + "mixer."
        if t == "M":
            fc(mx + "in_proj.weight")
            conv = reader.get(mx + "conv1d.weight")  # [C, 1, K]
            w.write(conv[:, 0, :].transpose(0, 1), f32)  # [K, C]
            if mx + "conv1d.bias" in reader:
                w.write(reader.get(mx + "conv1d.bias"), f32)
            else:
                w.write(torch.zeros(conv.shape[0]), f32)
            w.write(reader.get(mx + "dt_bias"), f32)
            w.write(reader.get(mx + "A_log"), f32)
            w.write(reader.get(mx + "D"), f32)
            w.write(reader.get(mx + "norm.weight"), f32)
            fc(mx + "out_proj.weight")
        elif t == "E":
            w.write(reader.get(mx + "gate.weight").transpose(0, 1), f32)
            w.write(reader.get(mx + "gate.e_score_correction_bias"), f32)
            for e in range(text_cfg.n_routed_experts):
                w.write(reader.get(f"{mx}experts.{e}.up_proj.weight").transpose(0, 1))
                w.write(reader.get(f"{mx}experts.{e}.down_proj.weight").transpose(0, 1))
            if text_cfg.n_shared_experts:
                w.write(reader.get(mx + "shared_experts.up_proj.weight").transpose(0, 1))
                w.write(reader.get(mx + "shared_experts.down_proj.weight").transpose(0, 1))
        elif t == "*":
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                fc(f"{mx}{proj}.weight")
        elif t == "-":
            w.write(torch.zeros(hidden, 1), f32)  # router of the 1-expert MoE
            w.write(torch.zeros(1), f32)
            w.write(reader.get(mx + "up_proj.weight").transpose(0, 1))
            w.write(reader.get(mx + "down_proj.weight").transpose(0, 1))
        print(f"  llm layer {i:2d} [{t}] done")
    w.write(reader.get(pre + "norm_f.weight"), f32)
    w.write(reader.get("lm_head.weight").transpose(0, 1))
    w.close()
    print(f"LLM: {w.count} parameters written")


def convert_audio(reader, config, out_dir):
    prefixes = (
        "model.audio_tower.",
        "model.multi_modal_projector.",
        "model.word_timestamp_encoder.",
        "word_timestamp_head.",
    )
    names = sorted(n for n in reader.names() if n.startswith(prefixes))
    index, offset, blobs = {}, 0, []
    for n in names:
        arr = reader.get(n).numpy().astype(np.float32)
        index[n] = {"offset": offset, "shape": list(arr.shape)}
        offset += arr.nbytes
        blobs.append(arr)
    header = json.dumps({"tensors": index}).encode()
    with open(os.path.join(out_dir, "nntr_audio.bin"), "wb") as f:
        f.write(AUDIO_MAGIC)
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        for arr in blobs:
            arr.tofile(f)
    print(f"audio: {len(names)} tensors, {offset / 2**20:.1f} MiB")


def chat_template_parts(model_path):
    """Prompt text around the audio placeholder, from the checkpoint's chat
    template when the processor files are there."""
    try:
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_path)
        conv = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "@@PROMPT@@"},
                    {"type": "audio", "path": "dummy.wav"},
                ],
            }
        ]
        text = proc.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        audio_token = proc.audio_token
        before_prompt, rest = text.split("@@PROMPT@@")
        between, after = rest.split(audio_token, 1)
        return {
            "prefix": before_prompt,
            "between": between,
            "suffix": after,
            "audio_token": audio_token,
            "default_prompt": proc.default_transcription_prompt,
            "source": "chat_template",
        }
    except Exception as e:  # tiny test checkpoints have no processor files
        print(f"note: chat template not rendered ({type(e).__name__}: {e}); "
              "using the ChatML fallback, check it against the checkpoint")
        return {
            "prefix": "<|im_start|>user\n",
            "between": "",
            "suffix": "<|im_end|>\n<|im_start|>assistant\n<think></think>",
            "audio_token": "<|audio|>",
            "default_prompt": (
                "Audio transcription. Language: as spoken. Wording: verbatim, "
                "including fillers and repetitions. Formatting: punctuation and "
                "capitalization. Output: transcript only."
            ),
            "source": "fallback",
        }


def transcribe_config(config, model_path):
    """settings of the audio front-end, the audio tower and the chat prompt"""
    text = config.text_config
    audio = config.audio_config
    diar = audio.diarization_config
    tc = {
        "audio_token_id": config.audio_token_id,
        "text_hidden_size": text.hidden_size,
        "asr": audio.asr_config.to_dict(),
        "diarization": diar.to_dict(),
        "speaker_activity_threshold": audio.speaker_activity_threshold,
        "feature_extractor": {
            "sampling_rate": 16000,
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "num_mel_bins": audio.asr_config.num_mel_bins,
            "preemphasis": 0.97,
            "mel_filters_dtype": "bfloat16",
        },
        "prompt": chat_template_parts(model_path),
    }
    for fe_name in ("processor_config.json", "preprocessor_config.json"):
        fe_path = os.path.join(model_path, fe_name)
        if not os.path.exists(fe_path):
            continue
        with open(fe_path) as f:
            fe = json.load(f)
        fe = fe.get("feature_extractor", fe)
        for k in ("sampling_rate", "n_fft", "win_length", "hop_length", "preemphasis",
                  "mel_filters_dtype"):
            if k in fe:
                tc["feature_extractor"][k] = fe[k]
        if "feature_size" in fe:
            tc["feature_extractor"]["num_mel_bins"] = fe["feature_size"]
        break
    return tc


def write_configs(config, model_path, out_dir, fc_dtype_name, emb_dtype_name, args):
    text = config.text_config
    llm_cfg = text.to_dict()
    llm_cfg["architectures"] = ["NemotronHForCausalLM"]
    llm_cfg["num_hidden_layers"] = len(layer_types(text))
    llm_cfg["layers_block_type"] = list(
        getattr(text, "layers_block_type", None) or text.layer_types
    )
    llm_cfg["rope_theta"] = 0
    dump_json(llm_cfg, os.path.join(out_dir, "config.json"))

    gen = {"bos_token_id": text.bos_token_id, "eos_token_id": text.eos_token_id,
           "do_sample": False}
    gen_path = os.path.join(model_path, "generation_config.json")
    if os.path.exists(gen_path):
        with open(gen_path) as f:
            src = json.load(f)
        for k in ("bos_token_id", "eos_token_id"):
            if src.get(k) is not None:
                gen[k] = src[k]
    if isinstance(gen["eos_token_id"], int):
        gen["eos_token_id"] = [gen["eos_token_id"]]
    dump_json(gen, os.path.join(out_dir, "generation_config.json"))

    tok = os.path.join(model_path, "tokenizer.json")
    nntr = {
        "model_type": "CausalLM",
        "model_tensor_type": f"{fc_dtype_name}-FP32",
        "model_file_name": "nntr_nemotron_h.bin",
        "embedding_file_name": "nntr_nemotron_h_embedding.bin",
        "embedding_file_dtype": emb_dtype_name,
        "fc_layer_dtype": fc_dtype_name,
        "embedding_dtype": "FP32",
        "lmhead_dtype": fc_dtype_name,
        "bad_word_ids": [],
        "fsu": False,
        "num_to_generate": args.num_to_generate,
        "init_seq_len": args.init_seq_len,
        "max_seq_len": args.max_seq_len,
        "batch_size": 1,
        "tokenizer_file": tok if os.path.exists(tok) else None,
        "audio_file_name": "nntr_audio.bin",
        "transcribe_config_file": "transcribe_config.json",
    }
    dump_json(nntr, os.path.join(out_dir, "nntr_config.json"))

    dump_json(transcribe_config(config, model_path),
              os.path.join(out_dir, "transcribe_config.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fc_dtype", default="float32", choices=["float32"],
                        help="dtype of the LLM projection weights")
    parser.add_argument("--embedding_dtype", default="float32",
                        choices=["float32", "float16"])
    parser.add_argument("--init_seq_len", type=int, default=512)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--num_to_generate", type=int, default=1024)
    parser.add_argument(
        "--audio_only",
        action="store_true",
        help="write only nntr_audio.bin and transcribe_config.json (the "
        "checkpoint may then hold just the perception.* tensors), for "
        "nntr_nemotron_transcribe --verify-audio",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    config = load_config(args.model_path)
    reader = CheckpointReader(args.model_path)

    fc_dtype = {"float32": np.float32}[args.fc_dtype]
    emb_dtype = {"float32": np.float32, "float16": np.float16}[args.embedding_dtype]
    convert_audio(reader, config, args.out_dir)
    if args.audio_only:
        dump_json(transcribe_config(config, args.model_path),
                  os.path.join(args.out_dir, "transcribe_config.json"))
        print("done (audio only):", args.out_dir)
        return
    convert_llm(reader, config.text_config, args.out_dir, fc_dtype, emb_dtype)
    write_configs(config, args.model_path, args.out_dir, "FP32",
                  "FP32" if args.embedding_dtype == "float32" else "FP16", args)
    print("done:", args.out_dir)


if __name__ == "__main__":
    main()
