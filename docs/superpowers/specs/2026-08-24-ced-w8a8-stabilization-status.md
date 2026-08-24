# CED W8A8 Int8-Resident — Stabilization Status

**Date**: 2026-08-24
**Branch**: `feat/ced-w8a8-int8-resident` (off `audio-detection` @ 255e6b1c)

## What this branch adds (3 commits on top of audio-detection)

1. `docs(w8a8)` — design spec for CED MLP int8-resident W8A8.
2. `feat(w8a8)` — the implementation: FC + activation int8-resident path.
3. `w8a8(fp16)` — dtype-aware FP16 input/output handling in `dotW8A8` for a
   future `Q8_0-FP16` coexistence (not active on the verified path).

Mechanism: ports YOLOv7Pose's W8A8 int8-resident scheme from conv to the FC
layer. On `NNTR_W8A8=1`, `ffn_up` emits a per-tensor-scale QINT8 activation,
GELU passes it through (dequant -> FP32 gelu_v2 -> requant), and `ffn_down`
reads the QINT8 input and dequantizes to FP32/FP16 for the residual. Weights
stay Q8_0 channel-wise; attention/norm/residual/head are an FP32 island.
All behind `NNTR_W8A8`; every other mode is byte-identical.

## Verified path (stable)

Config: `Q8_0-FP32`, `fc_layer_dtype: Q8_0`, runner launched with `NNTR_W8A8=1`.

### x86 (FP32-ref gate, verify_e2e.sh)
- **FP32 baseline (env off)**: ALL PASS, worst 2.62e-04, 0 non-finite — no
  regression from the new code.
- **W8A8 int8-resident (env on)**: 0 non-finite; worst prob diff 0.037 vs the
  FP32 reference (the expected W8A8 quantization level; the shipped compute-only
  W8A8/W8A32 is 0.031 — same order). Detections identical to W8A32.
- quick_ai model unittests: **60/60 PASS**. The 4 nntrainer tensor/quantizer
  test failures are pre-existing (fail on a clean `audio-detection` checkout
  without these commits).

### Android (SM-S938N; 4 wavs, 67 windows x 12 classes = 804 scores)
- `verify_android.sh` with `NNTR_W8A8=1`: **ALL CHECKS PASSED**.
- detections **100% identical** (67/67), top-1 97.0%, **0 non-finite**,
  max diff 0.037, mean 0.0028, **0 over tol (0.05)**.
- Per-window 26–30 ms, realtime 0.022–0.026x.

## Known limitation: peak RSS unchanged

MLP-only int8-residency does **not** reduce peak RSS (~63 MB on Android, same
as W8A32). The V3 memory planner already overlaps the activation pool, so the
peak is set by the concurrently-live FP32 attention tensors (qkv/out_proj/
softmax), which this change intentionally leaves FP32. Reducing peak RSS needs
the attention path's activation footprint lowered — a separate, larger task.

## Not active / not verified (gated off by default)

- **`Q8_0-FP16` + `NNTR_W8A8`** (FP16 attention storage): the FC/activation
  code is dtype-aware and handles FP16 in/out, but the CED graph's xi-pooling /
  attention reshape fails at binding (`view tensor type != source: FP16 view of
  FP32 source`) because the graph was not authored for an FP16 activation dtype.
  Needs a graph-level binding pass; left for future work. The `Q8_0-FP32` path
  is the verified default.

## Reproducing

```bash
# x86 FP32 regression (env off)
NNTR_NUM_THREADS=4 builddir/Applications/quick_ai/nntr_quick_ai <fp32 model dir>
# x86 W8A8 (env on)
AD_REF_DIR=<ref dir> NNTR_W8A8=1 NNTR_NUM_THREADS=4 \
  builddir/Applications/quick_ai/nntr_quick_ai <q8 model dir>
# Android (NNTR_W8A8=1 in the runner env)
bash tools/verify_android.sh
```
