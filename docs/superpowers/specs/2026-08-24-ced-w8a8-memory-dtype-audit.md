# CED Audio Detection — Peak Memory & Dtype Audit

**Date**: 2026-08-24
**Branch**: `feat/ced-w8a8-int8-resident`
**Device (Android)**: SM-S938N (Snapdragon 8 Elite), arm64-v8a, Android 16
**Host (x86)**: x86-64, gcc 11.4, release, `NNTR_NUM_THREADS=4`, V3 memory planner
**Model**: CED-tiny + xi-vector, 12 encoder blocks, embed_dim 192, 3 heads,
mlp_hidden 768, QKV 576, 12 classes. Weights: FP32 20.9 MB, Q8_0 6.0 MB.

## 1. Configurations run

| config | `model_tensor_type` | weight dtype | activation storage | `NNTR_W8A8` |
|---|---|---|---|---|
| FP32 | `FP32-FP32` | FP32 | FP32 everywhere | off |
| W8A32 (compute-only) | `Q8_0-FP32` | Q8_0 ch-wise | FP32 everywhere; int8 only inside GEMM (on-the-fly) | off |
| W8A8-resident (this work) | `Q8_0-FP32` | Q8_0 ch-wise | MLP edge int8; attention FP32 | on |

## 2. Peak RSS measured

### x86 (1s windows, dog_tizen.wav, 16 windows)

| config | peak RSS | per-window | realtime | worst prob diff vs FP32-ref | non-finite |
|---|---|---|---|---|---|
| FP32 | **47.8 MB** | 17.8 ms | 0.018x | 2.62e-04 (PASS) | 0 |
| W8A32 | **32.5 MB** | 26.5 ms | 0.026x | (0.031 known) | 0 |
| W8A8-resident | **33.1 MB** | 22.2 ms | 0.022x | 0.0367 | 0 |

### Android (3s windows, barking_only, 13 windows)

| config | peak RSS | per-window | realtime |
|---|---|---|---|
| FP32 | n/a (no FP32 model on device) | — | — |
| W8A32 | **63.3 MB** | 24.3 ms | 0.021x |
| W8A8-resident | **63.3 MB** | 28.6 ms | 0.025x |

**Android full-suite (4 wavs, 67 windows × 12 classes = 804 scores, W8A8-resident):**
detections 100% identical (67/67), top-1 97.0%, 0 non-finite, max diff 0.037,
mean 0.0028, 0 over tol(0.05). ALL CHECKS PASSED.

## 3. Why peak RSS does not drop (the core finding)

W8A8-resident makes the MLP edge (`ffn_up` out → `ffn_gelu` → `ffn_down` in)
int8, which cuts the **sum** of activations 38% (1s: 3456→2160 KB; 3s:
10368→6480 KB across 12 blocks). But **peak** RSS is unchanged because the V3
memory planner already overlaps the activation pool to near its minimum, so the
peak is set by the largest set of **concurrently-live** tensors — and those are
the FP32 attention islands, not the MLP edge.

### Per-layer activation footprint (1s, 24 tokens, one block)

| layer | dims | W8A32 dtype | W8A8 dtype | FP32 bytes | int8 bytes |
|---|---|---|---|---|---|
| attention_norm out / qkv in | 24×192 | FP32 | FP32 | 18 KB | 4 KB |
| **qkv out** | 24×576 | FP32 | **FP32** | 54 KB | 13 KB |
| mha_core out | 24×192 | FP32 | FP32 | 18 KB | 4 KB |
| out_proj out / att_residual | 24×192 | FP32 | FP32 | 18 KB | 4 KB |
| ffn_norm out / ffn_up in | 24×192 | FP32 | FP32 | 18 KB | 4 KB |
| **ffn_up out** | 24×768 | FP32 | **QINT8** | 72 KB | 18 KB |
| **ffn_gelu out** | 24×768 | FP32 | **QINT8** | 72 KB | 18 KB |
| ffn_down out | 24×192 | FP32 | FP32 | 18 KB | 4 KB |

The int8-resident edges (ffn_up/gelu, 768-dim) ARE the largest single tensors,
but at peak they coexist with the FP32 attention chain (qkv 54 KB + norms +
residuals ~110 KB per block). Because the residual additions force the attention
tensors to live alongside the MLP input/output, the MLP int8 win is shadowed at
peak by the FP32 attention footprint that is alive at the same instant.

### Implication

Lowering peak RSS requires shrinking the **attention** activation footprint
(qkv 576-dim, out_proj, softmax) — not the MLP edge. The attention path was kept
FP32 by design (softmax amplifies quantization error; YOLOv7's W8A16 collapsed
38/87 on a 50-layer conv chain for exactly this reason). Two routes, both
larger than this branch:

- **Q8_0-FP16** (FP16 attention storage): the FC/activation W8A8 code is
  dtype-aware for FP16, but the CED graph's xi-pooling / attention reshape
  fails at binding (`view tensor type != source`) — needs a graph-level pass.
- **qkv/out_proj int8-resident**: extends the QINT8 region into attention;
  softmax stays FP32 but its input would be int8-quantized. Accuracy risk,
  needs an S0 simulation gate first.

## 4. Where the 63 MB goes (Android, W8A8-resident, 3s)

Breakdown is approximate (peak RSS includes runtime + mmap overhead):

| component | bytes | dtype |
|---|---|---|
| Q8_0 weights | 6.0 MB | int8 (channel-wise) |
| FP32 bias + norm params | ~1 MB | FP32 |
| attention activations (peak-live, 12 blk × 3s) | ~5–8 MB | FP32 |
| MLP int8 activations (peak-live) | ~1–2 MB | QINT8 |
| front-end STFT/mel buffers (3s, 301 frames) | ~3–4 MB | FP32 |
| nntrainer runtime + mmap pool overhead | ~40 MB | — |

The runtime/overhead dominates on Android, not the activations — which is why a
38% activation-sum cut moves peak by <1%. The reference C runtime's 18 MB
benefits from a far smaller runtime footprint and no V3 pool overhead, not from
a dtype difference.

## 5. Speed note

W8A8-resident is **slower** than W8A32 on both platforms (x86 22 vs 27 ms; Android
29 vs 24 ms). The int8-resident path's dequant→GELU→requant epilogue and the
channel-wise GEMM's per-output-channel scale multiply cost more than the
on-the-fly W8A32 GEMM saves, for these small GEMMs (M=24–72, K=192, N=768).
YOLOv7 saw the same (its GEMMs were also small); the win there was activation
*traffic* (4×) on a conv-heavy net, which the V3 planner already neutralizes
here. So W8A8-resident on CED is a memory-pursuit, not a speed-pursuit — and on
current evidence it is accuracy-neutral but neither faster nor smaller at peak.

## 6. Summary table

| metric | FP32 | W8A32 | W8A8-resident |
|---|---|---|---|
| weight size | 20.9 MB | 6.0 MB | 6.0 MB |
| MLP activation dtype | FP32 | FP32 | **QINT8** |
| attention activation dtype | FP32 | FP32 | FP32 |
| x86 peak RSS | 47.8 MB | 32.5 MB | 33.1 MB |
| Android peak RSS | — | 63.3 MB | 63.3 MB |
| detection parity | (ref) | identical | identical |
| non-finite | 0 | 0 | 0 |
| worst prob diff vs FP32 | 2.6e-04 | 0.031 | 0.037 |
