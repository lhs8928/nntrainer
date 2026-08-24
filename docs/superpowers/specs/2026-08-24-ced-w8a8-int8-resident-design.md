# CED Audio Detection — Int8-Resident W8A8 (MLP-only region)

**Date**: 2026-08-24
**Branch**: `audio-detection`
**Target**: CED-tiny + xi-vector audio detector (12-class, ViT over mel patches)

## Problem

Current "W8A8" on the audio detector is W8A32 in storage: weights are int8
(channel-wise, Q8_0) and compute is int8 (`gemmPerChannelA8` quantizes the FP32
activation on the fly), **but activations are stored FP32** between layers
(`model_tensor_type: FP32-FP32`, `buildModelTensorType` hardcodes `-FP32` at
`quantize.cpp:184`). So the activation pool stays at FP32 size and Android RSS
is 63 MB vs the reference C runtime's 18 MB. Compute is fast (13.9 ms/window)
but memory does not benefit from the 8-bit path.

YOLOv7Pose already solved this: `NNTR_W8A8=1` makes every Q8_0 conv emit a
per-tensor-scale QINT8 activation (CharTensor with an inline FP32 scale) that
concat/addition/pool/upsample consume directly — activations become 4× smaller
and the pool drops 39 MB → 18 MB. That mechanism is conv-only, though: the FC
layer (`fc_layer.cpp`) has no `NNTR_W8A8` path, and CED is a ViT (FC + attention),
not a conv net.

## Goal

Make the detector's MLP sub-block carry int8-resident activations so the
activation pool shrinks toward the reference runtime, at FP32-level accuracy.

- **Weights**: QINT8 channel-wise (unchanged, already Q8_0 + `gemmPerChannelA8`).
- **Activations**: QINT8 tensor-wise, symmetric (`amax/127`), inline FP32 scale.
- **Region**: MLP only — `ffn_up` out → `gelu` → `ffn_down` in, inside every
  encoder block. Attention (`qkv_*`, `mha_core`, `out_proj`), normalization,
  residual addition and the head stay FP32.
- **Gate**: the existing `verify_e2e.sh` end-to-end probability diff against the
  PyTorch reference stays within budget (current FP32 baseline 3.29e-04; W8A8
  MLP must not collapse it), with **0 non-finite**.

## Why MLP-only

- Attention (softmax over logits) amplifies small quantization error; keep it
  FP32 to avoid the YOLOv7 W8A16-style collapse (38/87 there).
- MLP FCs are the big, numerically tame edges: `ffn_up` 192→768, `ffn_down`
  768→192, 12 blocks. Their activations (101–252 tokens × 768) dominate the
  intermediate pool, so int8-residency there is where the memory is.
- GELU (not SiLU) spans both signs, so use symmetric per-tensor scales — the
  conv path's SiLU-specific affine (`kActOff`) correction does not apply.

## Design

### 1. FC layer gains a W8A8 path (port of conv2d's mechanism)

Env-gated by `NNTR_W8A8` (same flag YOLOv7 uses). The FC is the universal
boundary op for the int8 region, exactly as conv is for YOLOv7.

**a) finalize output dtype** (`fc_layer.cpp` finalize, next to the existing
`setTensorType` at line ~92):
- When `NNTR_W8A8` is set and the FC's weight dtype is Q8_0 and the layer name
  matches an MLP FC (`ffn_up`/`ffn_down`), set the output dim to `QINT8`.
- An FC that is NOT an MLP FC (qkv/out_proj/head) always outputs FP32, and if
  its input was QINT8 it dequantizes (so `ffn_down` out → FP32 for the residual
  addition). Mirrors `conv2d_layer.cpp:1164`/`:1165`.

MLP-FC eligibility is decided by **name** in the graph builder
(`timm_vit_transformer.cpp` already names them `layerN_ffn_up`/`ffn_down`); the
framework just provides the mechanics, as in YOLOv7's design §2.

**b) forwarding epilogue** (`fc_layer.cpp` forwarding, after `input_.dot()`):
- `dot()` writes FP32 into the output buffer (the gemm path already does
  int8→int32→FP32 epilogue; for a QINT8 *input* it reads the inline scale and
  skips the on-the-fly quantize — see (c)).
- For a QINT8 output FC: apply bias (folded, FP32), then (if `ffn_up`) no fused
  activation here — GELU is a separate layer — so just amax over the FP32 output,
  pick `scale = amax/127`, quantize to `out.getData<int8_t>()`, write
  `out.getScale<float>()[0] = scale`. `ffn_down` has no activation either;
  because it exits the region, it stays FP32 out (no quantize).
- This is the conv epilogue (`conv2d_layer.cpp:1999`–`2055`) reduced to the
  no-fused-activation case.

**c) QINT8 input consumption** (`float_tensor.cpp` Q8_0 case, ~line 1044):
- `gemmPerChannelA8` currently takes `const float *A` and quantizes on the fly.
  When the input tensor is already QINT8, read `A.getScale<float>()[0]` and the
  int8 data directly (skip the amax+quantize). For an FP32 input (region entry
  at `ffn_up`, whose input is `ffn_norm`'s FP32 output) keep the on-the-fly path.

### 2. GELU QINT8 pass-through (`activation_layer.cpp`)

- When `NNTR_W8A8` is set and the input is QINT8: dequantize via the inline
  scale into a thread-local FP32 scratch, run FP32 GELU, amax, requantize to
  int8 with `scale = amax/127`, write the scale. Symmetric because GELU output
  spans both signs. This is the YOLOv7 SiLU epilogue pattern adapted to GELU.
- FP32-input GELU (none in the MLP region, but defensively) is unchanged.

### 3. `buildModelTensorType` and the env preset

- `quantize.cpp:184` `buildModelTensorType` keeps producing `Q8_0-FP32` on disk
  (the file format is unchanged; residency is a runtime decision). The detector
  runner sets `NNTR_W8A8=1` at launch — same as YOLOv7's `YOLO_TENSOR_TYPE=w8a8`
  preset, which leaves `model_tensor_type: FP32-FP32` in the config and flips the
  runtime path. No enum-table change needed.

### 4. TensorPool QScheme

- The pool currently hardcodes `PER_CHANNEL_AFFINE` for non-weight tensors
  (YOLOv7 design §3.1). MLP activation tensors need `PER_TENSOR_AFFINE` so the
  CharTensor carries one FP32 scale. Plumb the scheme through (or default
  per-tensor for activation groups). This is the same fix YOLOv7 lists.

## int8 region boundary (per encoder block)

```
ffn_norm(LN, FP32) ─┐
                    ▼
              ffn_up(FC, Q8_0 w)  ── out: QINT8 ──┐
                                                 ▼
                                          gelu (QINT8 in/out)
                                                 │
                                                 ▼
             ffn_down(FC, Q8_0 w) ◄── QINT8 in
                    │
                    ▼  (dequantize to FP32 at ffn_down output)
               FP32 ──► ffn_residual(addition, FP32)
```

Attention path (qkv_*, mha_core, out_proj, attention_residual) and the head are
untouched FP32 islands.

## Staged plan with accuracy gate at every stage (YOLOv7 discipline)

- **S0 — PyTorch simulation (no C++).** Fake-quantize the `ffn_up` out / `gelu`
  out / `ffn_down` in edges of the reference model to per-tensor symmetric int8,
  run `verify_e2e.sh`'s reference extraction, measure worst prob diff. Gate:
  within the W8A8 budget (~3e-2, the existing W8A8 compute-only number is the
  ceiling; ideally much tighter since only MLP edges quantize). If it fails,
  stop and revisit (per-edge exceptions) before any C++.
- **S1 — FC QINT8 emit/consume (x86).** `NNTR_W8A8` path in `fc_layer.cpp` +
  `float_tensor.cpp` + `activation_layer.cpp`; one MLP edge end-to-end under the
  flag. Parity vs FP32 within S0-predicted delta, 0 non-finite.
- **S2 — region rollout.** All 12 blocks' MLP edges resident; detector run with
  `NNTR_W8A8=1`. `verify_e2e.sh` gate on x86.
- **S3 — Android.** Build, push via `verify_android.sh`, confirm the 804-score
  diff stays in budget and measure RSS (expect a drop from 63 MB toward the
  reference's 18 MB; the attention FP32 island caps the reduction).

## Memory expectation

Only MLP activations go int8 (101–252 tokens × 768 × ~24 int8 edges), so the
reduction is partial but should be the bulk of the intermediate pool — attention
activations (576-dim qkv, softmax) stay FP32. YOLOv7 hit 4× on a conv net; CED
will do less because of the FP32 attention island, but the MLP edges are the
largest intermediate tensors. Actual number measured at S3.

## Risks

- **R1 — GELU per-tensor scale accuracy.** GELU's output range is wider than
  SiLU's; per-tensor symmetric may lose resolution on one side. S0 settles this
  for free. Fallback: keep `ffn_up` out FP32 and only resident the `ffn_down` in
  edge (smaller win, safer).
- **R2 — TensorPool QScheme plumbing.** Same risk YOLOv7 notes (R3); batch is 1
  throughout for the detector, sidestepping slice-scale semantics.
- **R3 — `ffn_down` output dtype boundary.** Must dequantize to FP32 so the
  residual addition is FP32+FP32. Covered by the "non-MLP FC → FP32" rule.
