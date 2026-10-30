#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file   export_silero_vad.py
# @author Hyeonseok Lee <hs89.lee@samsung.com>
# @date   30 October 2026
# @brief  Export Silero VAD v6 (the 16 kHz ONNX graph discretion's voice gate
#         loads: inputs input[N, 576], h, c) into the NNTrainer binary format
#         that models/silero_vad.cpp reads.
#
# The graph, checked here before anything is written:
#   Pad(reflect, 128 each side) -> Conv(258 STFT filters, kernel 256, stride
#   128) -> drop the first frame -> magnitude of 129 real/imaginary pairs ->
#   4 x (Conv1d k3 p1, strides 1 2 2 1, ReLU) -> LSTM 128 -> ReLU ->
#   Conv1d 128->1 -> Sigmoid.
# Tensors keep the ONNX layouts: lstm.w / lstm.r are [4*128, 128] in ONNX gate
# order i, o, f, c; lstm.wb / lstm.rb are the two halves of the ONNX bias.
# The file starts with "meta.model_manifest" (JSON: model, sha256 of the ONNX
# file, block and context sizes) so a phone can tell which model it holds.
#
# Usage:
#   python3 export_silero_vad.py silero_vad_v6.onnx silero_vad_v6.bin
##

import argparse
import hashlib
import json

import numpy as np
import onnx
from onnx import helper, numpy_helper

from split_sd_weights import write  # the file format of models/weight_loader.cpp

HIDDEN = 128
CONVS = [(129, 128, 1), (128, 64, 2), (64, 64, 2), (64, 128, 1)]  # in, out, stride


def attrs(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def tensors(model):
    """Named float32 arrays, after checking the graph is the expected one."""
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    nodes = list(model.graph.node)
    kinds = [n.op_type for n in nodes]
    convs = [n for n in nodes if n.op_type == "Conv"]
    pad = next(n for n in nodes if n.op_type == "Pad")
    lstm = next(n for n in nodes if n.op_type == "LSTM")
    slices = [n for n in nodes if n.op_type == "Slice"]
    if kinds.count("Conv") != 6 or kinds.count("LSTM") != 1:
        raise ValueError(f"unexpected graph: {kinds}")
    if attrs(pad)["mode"] != b"reflect" or init[pad.input[1]].tolist() != [0, 128, 0, 128]:
        raise ValueError("expected reflect padding of 128 samples on each side")
    stft = convs[0]
    if attrs(stft)["strides"] != [128] or init[stft.input[1]].shape != (258, 1, 256):
        raise ValueError("expected 258 STFT filters of 256 samples, stride 128")
    for s, real in zip(slices, (True, False)):  # [0:129] / [129:] channels, frames from 1
        starts = init[s.input[1]].tolist()
        if starts != ([0, 1] if real else [129, 1]):
            raise ValueError(f"unexpected STFT slice {starts}")
    out = {"stft.basis": init[stft.input[1]].reshape(258, 256)}
    for i, (node, (cin, cout, stride)) in enumerate(zip(convs[1:5], CONVS)):
        a = attrs(node)
        w = init[node.input[1]]
        if w.shape != (cout, cin, 3) or a["strides"] != [stride] or a["pads"] != [1, 1]:
            raise ValueError(f"conv {i}: {w.shape} {a}")
        out[f"conv{i}.weight"], out[f"conv{i}.bias"] = w, init[node.input[2]]
    if attrs(lstm).get("hidden_size") != HIDDEN or attrs(lstm).get("direction", b"forward") != b"forward":
        raise ValueError(f"unexpected LSTM {attrs(lstm)}")
    w, r, b = (init[lstm.input[k]] for k in (1, 2, 3))
    out["lstm.w"], out["lstm.r"] = w.reshape(4 * HIDDEN, HIDDEN), r.reshape(4 * HIDDEN, HIDDEN)
    out["lstm.wb"], out["lstm.rb"] = b.reshape(2, 4 * HIDDEN)
    last = convs[5]
    out["out.weight"], out["out.bias"] = init[last.input[1]].reshape(HIDDEN), init[last.input[2]]
    return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", help="silero_vad_v6.onnx (inputs input, h, c)")
    parser.add_argument("out", help="output .bin")
    args = parser.parse_args()
    model = onnx.load(args.onnx)
    names = [i.name for i in model.graph.input]
    if names != ["input", "h", "c"]:
        raise ValueError(f"expected inputs input, h, c; got {names}")
    named = tensors(model)
    with open(args.onnx, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    manifest = {"name": "silero-vad", "version": "v6", "weights_sha256": digest,
                "sample_rate": 16000, "block": 512, "context": 64}
    meta = json.dumps(manifest, sort_keys=True).encode("utf-8")
    write(args.out, [("meta.model_manifest", (len(meta),), meta)] +
          [(n, a.shape, a.tobytes()) for n, a in named.items()])
    params = sum(a.size for a in named.values())
    print(f"{args.out}: {len(named)} tensors, {params} parameters, manifest {manifest}")


if __name__ == "__main__":
    main()
