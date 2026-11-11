## SPDX-License-Identifier: Apache-2.0
## @file   make_audio_reference.py
## @brief  Reference outputs of the Nemotron 3.5 Transcribe audio tower (with
##         the real checkpoint) for `nntr_nemotron_transcribe --verify-audio`.
##
## Only the audio tower and the projector are built, in FP32 as the
## checkpoint runs them, so this needs ~3 GB of memory and only the
## `perception.*` tensors of the checkpoint (the 60 GB language model is not
## read). The checkpoint directory may hold a subset safetensors file with
## just those tensors.
##
## Usage:
##   python make_audio_reference.py --model_path <hf dir> --out_dir <dir> \
##       a.wav b.wav ...
##
## WAV files must be 16 kHz; the C++ side reads the same files.

import argparse
import json
import os
import sys
import wave

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoConfig  # noqa: E402
from transformers.models.nemotron3_5_transcribe.modeling_nemotron3_5_transcribe import (  # noqa: E402
    Nemotron3_5TranscribeAudioModel,
)
from transformers.models.nemotron_asr_streaming.feature_extraction_nemotron_asr_streaming import (  # noqa: E402
    NemotronAsrStreamingFeatureExtractor,
)

from weight_converter import CheckpointReader  # noqa: E402


def read_wav(path):
    with wave.open(path, "rb") as w:
        if w.getframerate() != 16000 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: need 16 kHz PCM16")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        data = data.reshape(-1, w.getnchannels()).astype(np.float64).mean(1)
    return (data / 32768.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("wavs", nargs="+")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model_path)
    reader = CheckpointReader(args.model_path)

    tower = Nemotron3_5TranscribeAudioModel(config.audio_config).eval().float()
    state = {}
    for name in tower.state_dict():
        full = "model.audio_tower." + name
        if full in reader:
            state[name] = reader.get(full)
    missing = tower.load_state_dict(state, strict=False)
    real_missing = [k for k in missing.missing_keys if "speaker_embedding" not in k]
    if real_missing or missing.unexpected_keys:
        raise RuntimeError(f"missing {real_missing}, unexpected {missing.unexpected_keys}")
    projector = torch.nn.Linear(
        config.audio_config.hidden_size, config.text_config.hidden_size
    )
    projector.load_state_dict(
        {
            "weight": reader.get("model.multi_modal_projector.weight"),
            "bias": reader.get("model.multi_modal_projector.bias"),
        }
    )

    fe = NemotronAsrStreamingFeatureExtractor(
        feature_size=config.audio_config.asr_config.num_mel_bins,
        mel_filters_dtype="bfloat16",
    )
    sub = config.audio_config.subsampling_factor

    index = {"_meta": {"wavs": []}}

    def dump(name, t):
        arr = t.detach().cpu().numpy().astype(np.float32)
        arr.tofile(os.path.join(args.out_dir, f"ref_{name}.bin"))
        index[name] = {"shape": list(arr.shape), "dtype": "float32"}

    for i, path in enumerate(args.wavs):
        tag = f"wav{i}"
        audio = read_wav(path)
        feats = fe(audio, sampling_rate=16000, return_tensors="pt")
        mel, mask = feats["input_features"], feats["attention_mask"].long()
        pad = -mel.shape[1] % sub
        mel = torch.nn.functional.pad(mel, (0, 0, 0, pad))
        mask = torch.nn.functional.pad(mask, (0, pad))
        with torch.no_grad():
            out = tower(mel, input_features_mask=mask)
            n_valid = (int(mask.sum()) + sub - 1) // sub
            embeds = projector(out.last_hidden_state[0, :n_valid])
        dump(f"{tag}_mel", mel[0])
        dump(f"{tag}_asr_hidden", out.asr_last_hidden_state[0])
        dump(f"{tag}_speaker_logits", out.speaker_logits[0])
        dump(f"{tag}_audio_embeds", embeds)
        probs = out.speaker_logits[0].sigmoid().unflatten(0, (-1, sub)).mean(1)
        active = (probs > config.audio_config.speaker_activity_threshold).float()
        index["_meta"]["wavs"].append(
            {
                "tag": tag,
                "path": os.path.abspath(path),
                "seconds": len(audio) / 16000.0,
                "audio_tokens": n_valid,
                "speaker_activity": active.mean(0).tolist(),
            }
        )
        print(f"{path}: {len(audio) / 16000.0:.1f} s, {n_valid} audio tokens, "
              f"speaker activity {np.round(active.mean(0).numpy(), 2).tolist()}")

    with open(os.path.join(args.out_dir, "ref_index.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
