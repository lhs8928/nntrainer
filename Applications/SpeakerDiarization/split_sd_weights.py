#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file   split_sd_weights.py
# @author Hyeonseok Lee <hs89.lee@samsung.com>
# @date   28 October 2026
# @brief  Split the all-in-one speaker_diarization.bin into the segmentation
#         model and the voice model, so the voice model can be swapped
#         (models/batch_diarizer.h, BatchDiarizer::init with two files).
#
#   sd_segmentation.bin   PyanNet (sincnet., lstm., linear., classifier.)
#   speaker_wespeaker.bin WeSpeaker ResNet34 (resnet.), its fbank set-up
#                         (frontend.) and the PLDA trained on its embeddings
#                         (xvec., plda.)
# The other voice model, ERes2NetV2, comes from export_eres2netv2.py. Every
# tensor is copied byte for byte (the INT8 ones stay INT8).
#
# Usage:
#   python3 split_sd_weights.py speaker_diarization.bin OUT_DIR
##

import argparse
import os
import struct

SEGMENTATION = ("sincnet.", "lstm.", "linear.", "classifier.")
WESPEAKER = ("resnet.", "frontend.", "xvec.", "plda.")


def read(path):
    """[(name, dims, raw bytes)] of a file in the format of weight_loader.cpp."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"NNTRDIAR":
        raise ValueError(f"{path} is not an NNTrainer diarization weight file")
    count = struct.unpack_from("<I", data, 8)[0]
    pos, entries = 12, []
    for _ in range(count):
        length = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        name = data[pos:pos + length].decode("utf-8")
        pos += length
        ndim = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        dims = struct.unpack_from(f"<{ndim}I", data, pos)
        pos += 4 * ndim
        size, offset = struct.unpack_from("<QQ", data, pos)
        pos += 16
        entries.append((name, dims, size, offset))
    base = pos + (64 - pos % 64) % 64
    return [(n, d, data[base + o:base + o + s]) for n, d, s, o in entries]


def write(path, tensors):
    """The same format: header (padded to 64 bytes), then the tensors, each
    starting on 64 bytes (an odd-sized INT8 tensor never leaves the next float
    tensor unaligned)."""
    header = bytearray(b"NNTRDIAR") + struct.pack("<I", len(tensors))
    offset, blobs = 0, []
    for name, dims, raw in tensors:
        encoded = name.encode("utf-8")
        header += struct.pack("<I", len(encoded)) + encoded
        header += struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}I", *dims)
        header += struct.pack("<QQ", len(raw), offset)
        pad = (64 - len(raw) % 64) % 64
        blobs.append(raw + b"\x00" * pad)
        offset += len(raw) + pad
    header += b"\x00" * ((64 - len(header) % 64) % 64)
    with open(path, "wb") as f:
        f.write(header)
        for blob in blobs:
            f.write(blob)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", help="speaker_diarization.bin")
    parser.add_argument("out_dir")
    args = parser.parse_args()
    tensors = read(args.weights)
    parts = {"sd_segmentation.bin": [t for t in tensors if t[0].startswith(SEGMENTATION)],
             "speaker_wespeaker.bin": [t for t in tensors if t[0].startswith(WESPEAKER)]}
    left = len(tensors) - sum(len(p) for p in parts.values())
    if left:
        names = [t[0] for t in tensors if not t[0].startswith(SEGMENTATION + WESPEAKER)]
        raise ValueError(f"{left} tensors belong to neither part: {names[:5]}")
    os.makedirs(args.out_dir, exist_ok=True)
    for name, part in parts.items():
        path = os.path.join(args.out_dir, name)
        write(path, part)
        print(f"{path}: {len(part)} tensors, {os.path.getsize(path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
