#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file   export_eres2netv2.py
# @author Hyeonseok Lee <hs89.lee@samsung.com>
# @date   28 October 2026
# @brief  Export the ERes2NetV2 speaker model (sherpa-onnx release of
#         3D-Speaker iic/speech_eres2netv2_sv_zh-cn_16k-common) into the
#         NNTrainer binary format that models/eres2netv2.cpp reads.
#
# The file starts with "meta.model_manifest", the voice model as ALICE names it
# (its ModelFingerprint: name, the sha256 of the ONNX file the weights come
# from, sherpa-onnx's preprocessing, dimension, normalization), as UTF-8 JSON
# bytes: the weights are those floats and models/eres2netv2.cpp gives the same
# numbers, so the phone can say it holds the server's model.
#
# The ONNX file has its batch norms folded into the convolutions. Every Conv
# and the final Gemm keep their float32 weights; each is named by the module
# path of the 3D-Speaker model, taken from the ONNX node's output name
# ('/layer3/layer3.0/conv1/Conv_output_0' -> 'layer3.0.conv1.weight').
#
# Usage:
#   python3 export_eres2netv2.py speaker-eres2netv2.onnx speaker_eres2netv2.bin
##

import argparse
import hashlib
import json

import numpy as np
import onnx
from onnx import numpy_helper

from split_sd_weights import write  # the file format of models/weight_loader.cpp

EXPECTED = 181  # 90 convolutions and the linear layer, weight and bias (layer3_ds has no bias)


def tensors(model):
    """{module.weight / module.bias: float32 array} of the ONNX graph."""
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    out = {}
    for node in model.graph.node:
        if node.op_type not in ("Conv", "Gemm"):
            continue
        if node.op_type == "Gemm":
            module = "seg_1"
            if [a.i for a in node.attribute if a.name == "transB"] != [1]:
                raise ValueError("the linear layer is expected as Gemm(transB=1)")
        else:
            parts = node.output[0].strip("/").split("/")[:-1]
            if len(parts) > 1 and parts[1].startswith(parts[0] + "."):
                parts = parts[1:]  # '/layer1/layer1.0/...' repeats the stage
            module = ".".join(parts)
            pads = [a.ints for a in node.attribute if a.name == "pads"]
            kernel = init[node.input[1]].shape[-1]
            if pads and list(pads[0]) != [kernel // 2] * 4:
                raise ValueError(f"{module}: padding other than kernel // 2")
        out[module + ".weight"] = init[node.input[1]].astype(np.float32)
        if len(node.input) > 2:
            out[module + ".bias"] = init[node.input[2]].astype(np.float32)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx", help="speaker-eres2netv2.onnx (sherpa-onnx release)")
    parser.add_argument("out", help="output .bin")
    args = parser.parse_args()
    model = onnx.load(args.onnx)
    meta = {p.key: p.value for p in model.metadata_props}
    if meta.get("output_dim") != "192" or meta.get("feature_normalize_type") != "global-mean":
        raise ValueError(f"not the expected ERes2NetV2 release: {meta}")
    named = tensors(model)
    if len(named) != EXPECTED:
        raise ValueError(f"{len(named)} tensors, expected {EXPECTED}")
    with open(args.onnx, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    manifest = {"name": "eres2netv2", "weights_sha256": digest,
                "preprocessing": "sherpa-pcm16le-mono16k-f32-v1",
                "dimension": 192, "normalization": "l2-v1"}  # ambient.speech.providers.SherpaSpeaker
    meta = json.dumps(manifest, sort_keys=True).encode("utf-8")
    write(args.out, [("meta.model_manifest", (len(meta),), meta)] +
          [(n, a.shape, np.ascontiguousarray(a).tobytes()) for n, a in named.items()])
    params = sum(a.size for a in named.values())
    print(f"{args.out}: {len(named)} tensors, {params} parameters, manifest {manifest}")


if __name__ == "__main__":
    main()
