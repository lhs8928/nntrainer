# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file   quantize_weights.py
# @date   05 November 2026
# @brief  nntrainer Whisper FP32 weights -> FC layers in FP16, Q4_0 or Q8_0
# @author Hyeonseok Lee <hs89.lee@samsung.com>
##
"""nntrainer Whisper FP32 safetensors -> FC weights of the encoder graph / decoder in FP16, Q4_0 or Q8_0.

Q4_0: ggml blocks repacked for ARM (q4_0x4, interleave 8), as nntrainer's saver does for ISA::ARM.
Q8_0: plain ggml block_q8_0 rows [out][in/32] (fp16 scale + 32 int8 per block), no repack.
Biases of quantized FC layers FP32; everything else FP16 (norms, embedding = LM head, positions, conv).
    python quantize_weights.py <fp32.safetensors> <out_dir> <enc_dtype> <dec_dtype>
writes <out_dir>/nntr_whisper_large_v3_e<enc>_d<dec>.safetensors and nntr_config.json (copy the other jsons).
"""
import json, os, re, sys
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

FC = re.compile(r'^(enc\d+_(q|k|v|attn_out|ffn_up|ffn_down)|dec\d+_(self_q|self_k|self_v|self_out|cross_q|cross_k|cross_v|cross_out|ffn_up|ffn_down))$')
ENC = re.compile(r'^(enc\d+_|dec\d+_cross_[kv]$)')  # FC layers of the encoder graph


def q4_0(w):
    """ggml quantize_row_q4_0_ref per row of w (M, K), repacked to q4_0x4."""
    M, K = w.shape
    x = w.reshape(M, K // 32, 32).astype(np.float32)
    idx = np.argmax(np.abs(x), axis=2)
    mx = np.take_along_axis(x, idx[..., None], axis=2)[..., 0]
    d = mx / -8.0
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1), 0.0).astype(np.float32)
    q = np.minimum(15, (x * inv[..., None] + 8.5).astype(np.int8)).astype(np.uint8)
    qs = (q[..., :16] | (q[..., 16:] << 4)).astype(np.uint8)
    d = d.astype(np.float16)
    nb = K // 32
    d4 = d.reshape(M // 4, 4, nb).transpose(0, 2, 1)
    q4 = qs.reshape(M // 4, 4, nb, 16).transpose(0, 2, 1, 3) ^ 0x88
    out = np.concatenate([d4.copy().view(np.uint8).reshape(M // 4, nb, 8),
                          q4[..., :8].reshape(M // 4, nb, 32), q4[..., 8:].reshape(M // 4, nb, 32)], axis=2)
    return np.ascontiguousarray(out).reshape(-1)


def q8_0(w):
    """ggml quantize_row_q8_0_ref per row of w (M, K): block_q8_0 rows."""
    M, K = w.shape
    x = w.reshape(M, K // 32, 32).astype(np.float32)
    amax = np.abs(x).max(axis=2)
    d = amax / 127.0
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1), 0.0).astype(np.float32)
    qs = np.round(x * inv[..., None]).astype(np.int8)
    out = np.concatenate([d.astype(np.float16)[..., None].view(np.uint8).reshape(M, K // 32, 2),
                          qs.view(np.uint8)], axis=2)
    return np.ascontiguousarray(out).reshape(-1)


src, out_dir, enc_dt, dec_dt = sys.argv[1], sys.argv[2], sys.argv[3].upper(), sys.argv[4].upper()
name = f'nntr_whisper_large_v3_e{enc_dt.lower()}_d{dec_dt.lower()}.safetensors'
os.makedirs(out_dir, exist_ok=True)
out, count = {}, {}
with safe_open(src, 'np') as f:
    meta = f.metadata()
    for k in f.keys():
        a = f.get_tensor(k)
        layer, kind = k.split(':')
        dt = (enc_dt if ENC.match(layer) else dec_dt) if FC.match(layer) else 'FP16'
        if dt in ('Q4_0', 'Q8_0') and kind == 'weight':
            out[k] = (q4_0 if dt == 'Q4_0' else q8_0)(np.ascontiguousarray(a.T))  # stored (in, out)
        elif dt in ('Q4_0', 'Q8_0') and kind == 'bias':
            out[k] = a.astype(np.float32)
        else:
            out[k] = a.astype(np.float16)
        count[dt] = count.get(dt, 0) + 1
save_file(out, os.path.join(out_dir, name), metadata=meta)


def tensor_type(dt):
    # "Q8_0-FP16" is no model_tensor_type, and FP16 activations take only Q4_0/Q6_K weights:
    # a Q8_0 part runs FP32 activations (FP32 Q8_0 GEMM), slow on the phone (RTF 13.3, boda)
    return 'FP16-FP32' if dt == 'Q8_0' else f'{dt}-FP16'


cfg = json.load(open(os.path.join(os.path.dirname(src), 'nntr_config.json')))
cfg.update({'model_tensor_type': tensor_type(dec_dt), 'fc_layer_dtype': dec_dt, 'embedding_dtype': 'FP16',
            'lmhead_dtype': 'FP16', 'encoder': {'model_tensor_type': tensor_type(enc_dt), 'fc_layer_dtype': enc_dt},
            'model_file_name': name})
json.dump(cfg, open(os.path.join(out_dir, 'nntr_config.json'), 'w'), indent=2)
print(name, count)
