// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   wespeaker_resnet34.cpp
 * @date   29 September 2026
 * @brief  WeSpeaker ResNet34 Speaker Embedding Model in NNTrainer (Native W8A8 & W4 Quantized)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "wespeaker_resnet34.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>

#if (defined(__ARM_FEATURE_DOTPROD) && __ARM_FEATURE_DOTPROD) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#if defined(USE_BLAS)
#include <cblas_interface.h>
#define RUN_SGEMM(M, N, K, A, B, C) nntrainer::__cblas_sgemm(0, false, false, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N)
#else
static inline void matmul_fallback(size_t M, size_t N, size_t K, const float *A, const float *B, float *C) {
  for (size_t m = 0; m < M; ++m) {
    const float *a_row = A + m * K;
    float *c_row = C + m * N;
    for (size_t k = 0; k < K; ++k) {
      float a_val = a_row[k];
      const float *b_row = B + k * N;
      for (size_t n = 0; n < N; ++n) {
        c_row[n] += a_val * b_row[n];
      }
    }
  }
}
#define RUN_SGEMM(M, N, K, A, B, C) do { \
  std::memset(C, 0, (M) * (N) * sizeof(float)); \
  matmul_fallback(M, N, K, A, B, C); \
} while(0)
#endif

namespace speaker_diarization {

static inline void add_residual_relu_fast(float *out, const float *res, size_t count) {
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
  const float32x4_t vzero = vdupq_n_f32(0.0f);
  size_t i = 0;
  for (; i + 16 <= count; i += 16) {
    float32x4_t o0 = vaddq_f32(vld1q_f32(out + i + 0), vld1q_f32(res + i + 0));
    float32x4_t o1 = vaddq_f32(vld1q_f32(out + i + 4), vld1q_f32(res + i + 4));
    float32x4_t o2 = vaddq_f32(vld1q_f32(out + i + 8), vld1q_f32(res + i + 8));
    float32x4_t o3 = vaddq_f32(vld1q_f32(out + i + 12), vld1q_f32(res + i + 12));

    vst1q_f32(out + i + 0, vmaxq_f32(o0, vzero));
    vst1q_f32(out + i + 4, vmaxq_f32(o1, vzero));
    vst1q_f32(out + i + 8, vmaxq_f32(o2, vzero));
    vst1q_f32(out + i + 12, vmaxq_f32(o3, vzero));
  }
  for (; i + 4 <= count; i += 4) {
    float32x4_t o = vaddq_f32(vld1q_f32(out + i), vld1q_f32(res + i));
    vst1q_f32(out + i, vmaxq_f32(o, vzero));
  }
  for (; i < count; ++i) {
    float val = out[i] + res[i];
    out[i] = (val > 0.0f) ? val : 0.0f;
  }
#else
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < count; ++i) {
    float val = out[i] + res[i];
    out[i] = (val > 0.0f) ? val : 0.0f;
  }
#endif
}

static inline std::vector<int8_t> transpose_KN_to_NK(const int8_t *src, size_t K, size_t N) {
  if (!src) return {};
  std::vector<int8_t> dst(N * K);
  for (size_t k = 0; k < K; ++k) {
    for (size_t n = 0; n < N; ++n) {
      dst[n * K + k] = src[k * N + n];
    }
  }
  return dst;
}

/**
 * @brief 4-row micro-tiled INT8 dot product kernel (Seungbaek Hong's dotI8Tile optimization).
 *
 * Keeps accumulators live in vector registers and loads weights ONCE across up to 4
 * activation rows, reducing weight memory traffic by 75%.
 */
static inline void dotI8Tile4(const int8_t *A, size_t lda, unsigned int rows,
                             const int8_t *w_col, size_t K, int32_t *out) {
#if (defined(__ARM_FEATURE_DOTPROD) && __ARM_FEATURE_DOTPROD)
  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  int32x4_t acc2 = vdupq_n_s32(0);
  int32x4_t acc3 = vdupq_n_s32(0);

  size_t k = 0;
  for (; k + 16 <= K; k += 16) {
    const int8x16_t w0 = vld1q_s8(w_col + k);
    acc0 = vdotq_s32(acc0, vld1q_s8(A + 0 * lda + k), w0);
    if (rows > 1) acc1 = vdotq_s32(acc1, vld1q_s8(A + 1 * lda + k), w0);
    if (rows > 2) acc2 = vdotq_s32(acc2, vld1q_s8(A + 2 * lda + k), w0);
    if (rows > 3) acc3 = vdotq_s32(acc3, vld1q_s8(A + 3 * lda + k), w0);
  }

  out[0] = vaddvq_s32(acc0);
  if (rows > 1) out[1] = vaddvq_s32(acc1);
  if (rows > 2) out[2] = vaddvq_s32(acc2);
  if (rows > 3) out[3] = vaddvq_s32(acc3);

  for (; k < K; ++k) {
    int32_t wv = static_cast<int32_t>(w_col[k]);
    out[0] += static_cast<int32_t>(A[0 * lda + k]) * wv;
    if (rows > 1) out[1] += static_cast<int32_t>(A[1 * lda + k]) * wv;
    if (rows > 2) out[2] += static_cast<int32_t>(A[2 * lda + k]) * wv;
    if (rows > 3) out[3] += static_cast<int32_t>(A[3 * lda + k]) * wv;
  }
#elif (defined(__ARM_NEON) || defined(__ARM_NEON__))
  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  int32x4_t acc2 = vdupq_n_s32(0);
  int32x4_t acc3 = vdupq_n_s32(0);

  size_t k = 0;
  for (; k + 16 <= K; k += 16) {
    const int8x16_t w0 = vld1q_s8(w_col + k);
    const int8x8_t w0_l = vget_low_s8(w0);
    const int8x8_t w0_h = vget_high_s8(w0);

    const int8x16_t a0 = vld1q_s8(A + 0 * lda + k);
    acc0 = vpadalq_s16(acc0, vmull_s8(vget_low_s8(a0), w0_l));
    acc0 = vpadalq_s16(acc0, vmull_s8(vget_high_s8(a0), w0_h));

    if (rows > 1) {
      const int8x16_t a1 = vld1q_s8(A + 1 * lda + k);
      acc1 = vpadalq_s16(acc1, vmull_s8(vget_low_s8(a1), w0_l));
      acc1 = vpadalq_s16(acc1, vmull_s8(vget_high_s8(a1), w0_h));
    }
    if (rows > 2) {
      const int8x16_t a2 = vld1q_s8(A + 2 * lda + k);
      acc2 = vpadalq_s16(acc2, vmull_s8(vget_low_s8(a2), w0_l));
      acc2 = vpadalq_s16(acc2, vmull_s8(vget_high_s8(a2), w0_h));
    }
    if (rows > 3) {
      const int8x16_t a3 = vld1q_s8(A + 3 * lda + k);
      acc3 = vpadalq_s16(acc3, vmull_s8(vget_low_s8(a3), w0_l));
      acc3 = vpadalq_s16(acc3, vmull_s8(vget_high_s8(a3), w0_h));
    }
  }

  out[0] = vaddvq_s32(acc0);
  if (rows > 1) out[1] = vaddvq_s32(acc1);
  if (rows > 2) out[2] = vaddvq_s32(acc2);
  if (rows > 3) out[3] = vaddvq_s32(acc3);

  for (; k < K; ++k) {
    int32_t wv = static_cast<int32_t>(w_col[k]);
    out[0] += static_cast<int32_t>(A[0 * lda + k]) * wv;
    if (rows > 1) out[1] += static_cast<int32_t>(A[1 * lda + k]) * wv;
    if (rows > 2) out[2] += static_cast<int32_t>(A[2 * lda + k]) * wv;
    if (rows > 3) out[3] += static_cast<int32_t>(A[3 * lda + k]) * wv;
  }
#else
  out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0;
  for (size_t k = 0; k < K; ++k) {
    int32_t wv = static_cast<int32_t>(w_col[k]);
    out[0] += static_cast<int32_t>(A[0 * lda + k]) * wv;
    if (rows > 1) out[1] += static_cast<int32_t>(A[1 * lda + k]) * wv;
    if (rows > 2) out[2] += static_cast<int32_t>(A[2 * lda + k]) * wv;
    if (rows > 3) out[3] += static_cast<int32_t>(A[3 * lda + k]) * wv;
  }
#endif
}

/**
 * @brief Completely unbranched 4x4 register micro-tiled INT8 GEMM kernel for full tiles.
 *
 * Eliminates all runtime conditional branches inside the innermost loop for the >98% of tiles
 * where cur_rows == 4 and cur_cols == 4.
 */
static inline void dotI8Tile4x4_full(const int8_t *A, size_t lda,
                                    const int8_t *W, size_t ldw,
                                    size_t K, int32_t out[4][4]) {
#if (defined(__ARM_FEATURE_DOTPROD) && __ARM_FEATURE_DOTPROD)
  int32x4_t acc00 = vdupq_n_s32(0), acc01 = vdupq_n_s32(0), acc02 = vdupq_n_s32(0), acc03 = vdupq_n_s32(0);
  int32x4_t acc10 = vdupq_n_s32(0), acc11 = vdupq_n_s32(0), acc12 = vdupq_n_s32(0), acc13 = vdupq_n_s32(0);
  int32x4_t acc20 = vdupq_n_s32(0), acc21 = vdupq_n_s32(0), acc22 = vdupq_n_s32(0), acc23 = vdupq_n_s32(0);
  int32x4_t acc30 = vdupq_n_s32(0), acc31 = vdupq_n_s32(0), acc32 = vdupq_n_s32(0), acc33 = vdupq_n_s32(0);

  const int8_t *a0_ptr = A + 0 * lda;
  const int8_t *a1_ptr = A + 1 * lda;
  const int8_t *a2_ptr = A + 2 * lda;
  const int8_t *a3_ptr = A + 3 * lda;

  const int8_t *w0_ptr = W + 0 * ldw;
  const int8_t *w1_ptr = W + 1 * ldw;
  const int8_t *w2_ptr = W + 2 * ldw;
  const int8_t *w3_ptr = W + 3 * ldw;

  size_t k = 0;
  for (; k + 16 <= K; k += 16) {
    const int8x16_t a0 = vld1q_s8(a0_ptr + k);
    const int8x16_t a1 = vld1q_s8(a1_ptr + k);
    const int8x16_t a2 = vld1q_s8(a2_ptr + k);
    const int8x16_t a3 = vld1q_s8(a3_ptr + k);

    const int8x16_t w0 = vld1q_s8(w0_ptr + k);
    acc00 = vdotq_s32(acc00, a0, w0);
    acc10 = vdotq_s32(acc10, a1, w0);
    acc20 = vdotq_s32(acc20, a2, w0);
    acc30 = vdotq_s32(acc30, a3, w0);

    const int8x16_t w1 = vld1q_s8(w1_ptr + k);
    acc01 = vdotq_s32(acc01, a0, w1);
    acc11 = vdotq_s32(acc11, a1, w1);
    acc21 = vdotq_s32(acc21, a2, w1);
    acc31 = vdotq_s32(acc31, a3, w1);

    const int8x16_t w2 = vld1q_s8(w2_ptr + k);
    acc02 = vdotq_s32(acc02, a0, w2);
    acc12 = vdotq_s32(acc12, a1, w2);
    acc22 = vdotq_s32(acc22, a2, w2);
    acc32 = vdotq_s32(acc32, a3, w2);

    const int8x16_t w3 = vld1q_s8(w3_ptr + k);
    acc03 = vdotq_s32(acc03, a0, w3);
    acc13 = vdotq_s32(acc13, a1, w3);
    acc23 = vdotq_s32(acc23, a2, w3);
    acc33 = vdotq_s32(acc33, a3, w3);
  }

  out[0][0] = vaddvq_s32(acc00);
  out[0][1] = vaddvq_s32(acc01);
  out[0][2] = vaddvq_s32(acc02);
  out[0][3] = vaddvq_s32(acc03);

  out[1][0] = vaddvq_s32(acc10);
  out[1][1] = vaddvq_s32(acc11);
  out[1][2] = vaddvq_s32(acc12);
  out[1][3] = vaddvq_s32(acc13);

  out[2][0] = vaddvq_s32(acc20);
  out[2][1] = vaddvq_s32(acc21);
  out[2][2] = vaddvq_s32(acc22);
  out[2][3] = vaddvq_s32(acc23);

  out[3][0] = vaddvq_s32(acc30);
  out[3][1] = vaddvq_s32(acc31);
  out[3][2] = vaddvq_s32(acc32);
  out[3][3] = vaddvq_s32(acc33);

  for (; k < K; ++k) {
    int32_t a0v = static_cast<int32_t>(a0_ptr[k]);
    int32_t a1v = static_cast<int32_t>(a1_ptr[k]);
    int32_t a2v = static_cast<int32_t>(a2_ptr[k]);
    int32_t a3v = static_cast<int32_t>(a3_ptr[k]);

    int32_t w0v = static_cast<int32_t>(w0_ptr[k]);
    int32_t w1v = static_cast<int32_t>(w1_ptr[k]);
    int32_t w2v = static_cast<int32_t>(w2_ptr[k]);
    int32_t w3v = static_cast<int32_t>(w3_ptr[k]);

    out[0][0] += a0v * w0v; out[0][1] += a0v * w1v; out[0][2] += a0v * w2v; out[0][3] += a0v * w3v;
    out[1][0] += a1v * w0v; out[1][1] += a1v * w1v; out[1][2] += a1v * w2v; out[1][3] += a1v * w3v;
    out[2][0] += a2v * w0v; out[2][1] += a2v * w1v; out[2][2] += a2v * w2v; out[2][3] += a2v * w3v;
    out[3][0] += a3v * w0v; out[3][1] += a3v * w1v; out[3][2] += a3v * w2v; out[3][3] += a3v * w3v;
  }
#else
  for (unsigned int r = 0; r < 4; ++r) {
    for (unsigned int c = 0; c < 4; ++c) {
      out[r][c] = 0;
      for (size_t k = 0; k < K; ++k) {
        out[r][c] += static_cast<int32_t>(A[r * lda + k]) * static_cast<int32_t>(W[c * ldw + k]);
      }
    }
  }
#endif
}

/**
 * @brief 4x4 register micro-tiled INT8 GEMM kernel (Seungbaek Hong's YOLOv7 4x4 micro-kernel).
 *
 * Keeps 16 accumulators in NEON registers and computes a 4-row x 4-column tile:
 * - 4 activation row vectors are loaded once and reused across 4 weight columns (75% savings).
 * - 4 weight column vectors are loaded once and reused across 4 activation rows (75% savings).
 * Total 16 vdotq_s32 (SDOT) operations per loop iteration, fully utilizing Cortex-A76 registers.
 */
static inline void dotI8Tile4x4(const int8_t *A, size_t lda, unsigned int rows,
                               const int8_t *W, size_t ldw, unsigned int cols,
                               size_t K, int32_t out[4][4]) {
#if (defined(__ARM_FEATURE_DOTPROD) && __ARM_FEATURE_DOTPROD)
  int32x4_t acc00 = vdupq_n_s32(0), acc01 = vdupq_n_s32(0), acc02 = vdupq_n_s32(0), acc03 = vdupq_n_s32(0);
  int32x4_t acc10 = vdupq_n_s32(0), acc11 = vdupq_n_s32(0), acc12 = vdupq_n_s32(0), acc13 = vdupq_n_s32(0);
  int32x4_t acc20 = vdupq_n_s32(0), acc21 = vdupq_n_s32(0), acc22 = vdupq_n_s32(0), acc23 = vdupq_n_s32(0);
  int32x4_t acc30 = vdupq_n_s32(0), acc31 = vdupq_n_s32(0), acc32 = vdupq_n_s32(0), acc33 = vdupq_n_s32(0);

  size_t k = 0;
  for (; k + 16 <= K; k += 16) {
    const int8x16_t a0 = vld1q_s8(A + 0 * lda + k);
    const int8x16_t a1 = (rows > 1) ? vld1q_s8(A + 1 * lda + k) : vdupq_n_s8(0);
    const int8x16_t a2 = (rows > 2) ? vld1q_s8(A + 2 * lda + k) : vdupq_n_s8(0);
    const int8x16_t a3 = (rows > 3) ? vld1q_s8(A + 3 * lda + k) : vdupq_n_s8(0);

    const int8x16_t w0 = vld1q_s8(W + 0 * ldw + k);
    acc00 = vdotq_s32(acc00, a0, w0);
    acc10 = vdotq_s32(acc10, a1, w0);
    acc20 = vdotq_s32(acc20, a2, w0);
    acc30 = vdotq_s32(acc30, a3, w0);

    if (cols > 1) {
      const int8x16_t w1 = vld1q_s8(W + 1 * ldw + k);
      acc01 = vdotq_s32(acc01, a0, w1);
      acc11 = vdotq_s32(acc11, a1, w1);
      acc21 = vdotq_s32(acc21, a2, w1);
      acc31 = vdotq_s32(acc31, a3, w1);
    }
    if (cols > 2) {
      const int8x16_t w2 = vld1q_s8(W + 2 * ldw + k);
      acc02 = vdotq_s32(acc02, a0, w2);
      acc12 = vdotq_s32(acc12, a1, w2);
      acc22 = vdotq_s32(acc22, a2, w2);
      acc32 = vdotq_s32(acc32, a3, w2);
    }
    if (cols > 3) {
      const int8x16_t w3 = vld1q_s8(W + 3 * ldw + k);
      acc03 = vdotq_s32(acc03, a0, w3);
      acc13 = vdotq_s32(acc13, a1, w3);
      acc23 = vdotq_s32(acc23, a2, w3);
      acc33 = vdotq_s32(acc33, a3, w3);
    }
  }

  out[0][0] = vaddvq_s32(acc00);
  if (cols > 1) out[0][1] = vaddvq_s32(acc01);
  if (cols > 2) out[0][2] = vaddvq_s32(acc02);
  if (cols > 3) out[0][3] = vaddvq_s32(acc03);

  if (rows > 1) {
    out[1][0] = vaddvq_s32(acc10);
    if (cols > 1) out[1][1] = vaddvq_s32(acc11);
    if (cols > 2) out[1][2] = vaddvq_s32(acc12);
    if (cols > 3) out[1][3] = vaddvq_s32(acc13);
  }
  if (rows > 2) {
    out[2][0] = vaddvq_s32(acc20);
    if (cols > 1) out[2][1] = vaddvq_s32(acc21);
    if (cols > 2) out[2][2] = vaddvq_s32(acc22);
    if (cols > 3) out[2][3] = vaddvq_s32(acc23);
  }
  if (rows > 3) {
    out[3][0] = vaddvq_s32(acc30);
    if (cols > 1) out[3][1] = vaddvq_s32(acc31);
    if (cols > 2) out[3][2] = vaddvq_s32(acc32);
    if (cols > 3) out[3][3] = vaddvq_s32(acc33);
  }

  for (; k < K; ++k) {
    for (unsigned int c = 0; c < cols; ++c) {
      int32_t wv = static_cast<int32_t>(W[c * ldw + k]);
      for (unsigned int r = 0; r < rows; ++r) {
        out[r][c] += static_cast<int32_t>(A[r * lda + k]) * wv;
      }
    }
  }
#else
  for (unsigned int r = 0; r < rows; ++r) {
    for (unsigned int c = 0; c < cols; ++c) {
      out[r][c] = 0;
      for (size_t k = 0; k < K; ++k) {
        out[r][c] += static_cast<int32_t>(A[r * lda + k]) * static_cast<int32_t>(W[c * ldw + k]);
      }
    }
  }
#endif
}

static inline void quantize_f32_to_i8(const float *src, int8_t *dst, size_t count, float inv_scale) {
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
  const float32x4_t vscale = vdupq_n_f32(inv_scale);
  size_t i = 0;
  for (; i + 16 <= count; i += 16) {
    float32x4_t f0 = vmulq_f32(vld1q_f32(src + i + 0), vscale);
    float32x4_t f1 = vmulq_f32(vld1q_f32(src + i + 4), vscale);
    float32x4_t f2 = vmulq_f32(vld1q_f32(src + i + 8), vscale);
    float32x4_t f3 = vmulq_f32(vld1q_f32(src + i + 12), vscale);

    int16x4_t h0 = vqmovn_s32(vcvtnq_s32_f32(f0));
    int16x4_t h1 = vqmovn_s32(vcvtnq_s32_f32(f1));
    int16x4_t h2 = vqmovn_s32(vcvtnq_s32_f32(f2));
    int16x4_t h3 = vqmovn_s32(vcvtnq_s32_f32(f3));

    int8x8_t b0 = vqmovn_s16(vcombine_s16(h0, h1));
    int8x8_t b1 = vqmovn_s16(vcombine_s16(h2, h3));

    vst1q_s8(dst + i, vcombine_s8(b0, b1));
  }
  for (; i + 8 <= count; i += 8) {
    float32x4_t f0 = vmulq_f32(vld1q_f32(src + i + 0), vscale);
    float32x4_t f1 = vmulq_f32(vld1q_f32(src + i + 4), vscale);

    int16x4_t h0 = vqmovn_s32(vcvtnq_s32_f32(f0));
    int16x4_t h1 = vqmovn_s32(vcvtnq_s32_f32(f1));

    int8x8_t b0 = vqmovn_s16(vcombine_s16(h0, h1));
    vst1_s8(dst + i, b0);
  }
  for (; i < count; ++i) {
    float q = std::round(src[i] * inv_scale);
    dst[i] = static_cast<int8_t>(std::clamp(q, -128.0f, 127.0f));
  }
#else
  for (size_t i = 0; i < count; ++i) {
    float q = std::round(src[i] * inv_scale);
    dst[i] = static_cast<int8_t>(std::clamp(q, -128.0f, 127.0f));
  }
#endif
}

// NHWC im2col with direct single-pass INT8 quantization (Zero float buffer)
static inline void im2col_nhwc_i8(const float *data_im, size_t in_c,
                                 size_t in_h, size_t in_w, size_t k_h, size_t k_w,
                                 size_t pad, size_t stride, float inv_scale, int8_t *data_col_i8) {
  size_t out_h = (in_h + 2 * pad - k_h) / stride + 1;
  size_t out_w = (in_w + 2 * pad - k_w) / stride + 1;
  size_t K = k_h * k_w * in_c;

  // Fast path for standard 3x3 stride-1 pad-1 convolution
  if (k_h == 3 && k_w == 3 && stride == 1 && pad == 1) {
    #pragma omp parallel for schedule(static)
    for (size_t oh = 0; oh < out_h; ++oh) {
      for (size_t ow = 0; ow < out_w; ++ow) {
        int8_t *col_row = data_col_i8 + (oh * out_w + ow) * K;
        for (size_t kh = 0; kh < 3; ++kh) {
          int ih = static_cast<int>(oh + kh) - 1;
          int8_t *dst_kh = col_row + kh * (3 * in_c);
          if (ih >= 0 && ih < static_cast<int>(in_h)) {
            if (ow >= 1 && ow + 1 < in_w) {
              const float *src = data_im + (static_cast<size_t>(ih) * in_w + (ow - 1)) * in_c;
              quantize_f32_to_i8(src, dst_kh, 3 * in_c, inv_scale);
            } else {
              for (size_t kw = 0; kw < 3; ++kw) {
                int iw = static_cast<int>(ow + kw) - 1;
                int8_t *dst = dst_kh + kw * in_c;
                if (iw >= 0 && iw < static_cast<int>(in_w)) {
                  const float *src = data_im + (static_cast<size_t>(ih) * in_w + static_cast<size_t>(iw)) * in_c;
                  quantize_f32_to_i8(src, dst, in_c, inv_scale);
                } else {
                  std::memset(dst, 0, in_c);
                }
              }
            }
          } else {
            std::memset(dst_kh, 0, 3 * in_c);
          }
        }
      }
    }
    return;
  }

  #pragma omp parallel for schedule(static)
  for (size_t oh = 0; oh < out_h; ++oh) {
    for (size_t ow = 0; ow < out_w; ++ow) {
      int8_t *col_row = data_col_i8 + (oh * out_w + ow) * K;
      for (size_t kh = 0; kh < k_h; ++kh) {
        int ih = static_cast<int>(oh * stride + kh) - static_cast<int>(pad);
        for (size_t kw = 0; kw < k_w; ++kw) {
          int iw = static_cast<int>(ow * stride + kw) - static_cast<int>(pad);
          int8_t *dst = col_row + (kh * k_w + kw) * in_c;
          if (ih >= 0 && ih < static_cast<int>(in_h) && iw >= 0 && iw < static_cast<int>(in_w)) {
            const float *src = data_im + (static_cast<size_t>(ih) * in_w + static_cast<size_t>(iw)) * in_c;
            quantize_f32_to_i8(src, dst, in_c, inv_scale);
          } else {
            std::memset(dst, 0, in_c);
          }
        }
      }
    }
  }
}

// NHWC im2col: gathers input patches into Col[out_h * out_w, k_h * k_w * in_c]
static inline void im2col_nhwc(const float *data_im, size_t in_c,
                              size_t in_h, size_t in_w, size_t k_h, size_t k_w,
                              size_t pad, size_t stride, float *data_col) {
  size_t out_h = (in_h + 2 * pad - k_h) / stride + 1;
  size_t out_w = (in_w + 2 * pad - k_w) / stride + 1;
  size_t K = k_h * k_w * in_c;

  // Fast path for standard 3x3 stride-1 pad-1 convolution
  if (k_h == 3 && k_w == 3 && stride == 1 && pad == 1) {
    #pragma omp parallel for schedule(static)
    for (size_t oh = 0; oh < out_h; ++oh) {
      for (size_t ow = 0; ow < out_w; ++ow) {
        float *col_row = data_col + (oh * out_w + ow) * K;
        for (size_t kh = 0; kh < 3; ++kh) {
          int ih = static_cast<int>(oh + kh) - 1;
          float *dst_kh = col_row + kh * (3 * in_c);
          if (ih >= 0 && ih < static_cast<int>(in_h)) {
            if (ow >= 1 && ow + 1 < in_w) {
              const float *src = data_im + (static_cast<size_t>(ih) * in_w + (ow - 1)) * in_c;
              std::memcpy(dst_kh, src, 3 * in_c * sizeof(float));
            } else {
              for (size_t kw = 0; kw < 3; ++kw) {
                int iw = static_cast<int>(ow + kw) - 1;
                float *dst = dst_kh + kw * in_c;
                if (iw >= 0 && iw < static_cast<int>(in_w)) {
                  const float *src = data_im + (static_cast<size_t>(ih) * in_w + static_cast<size_t>(iw)) * in_c;
                  std::memcpy(dst, src, in_c * sizeof(float));
                } else {
                  std::memset(dst, 0, in_c * sizeof(float));
                }
              }
            }
          } else {
            std::memset(dst_kh, 0, 3 * in_c * sizeof(float));
          }
        }
      }
    }
    return;
  }

  #pragma omp parallel for schedule(static)
  for (size_t oh = 0; oh < out_h; ++oh) {
    for (size_t ow = 0; ow < out_w; ++ow) {
      float *col_row = data_col + (oh * out_w + ow) * K;
      for (size_t kh = 0; kh < k_h; ++kh) {
        int ih = static_cast<int>(oh * stride + kh) - static_cast<int>(pad);
        for (size_t kw = 0; kw < k_w; ++kw) {
          int iw = static_cast<int>(ow * stride + kw) - static_cast<int>(pad);
          float *dst = col_row + (kh * k_w + kw) * in_c;
          if (ih >= 0 && ih < static_cast<int>(in_h) && iw >= 0 && iw < static_cast<int>(in_w)) {
            const float *src = data_im + (static_cast<size_t>(ih) * in_w + static_cast<size_t>(iw)) * in_c;
            std::memcpy(dst, src, in_c * sizeof(float));
          } else {
            std::memset(dst, 0, in_c * sizeof(float));
          }
        }
      }
    }
  }
}

void WeSpeakerResNet34::runConv2d(const float *in, float *out, size_t in_c, size_t out_c,
                                 size_t in_h, size_t in_w, size_t k_h, size_t k_w,
                                 size_t stride, size_t pad,
                                 const float *weights_fp32, const int8_t *weights_i8_trans,
                                 const float *scales, const float *bias, float act_scale,
                                 bool apply_relu) {
  size_t out_h = (in_h + 2 * pad - k_h) / stride + 1;
  size_t out_w = (in_w + 2 * pad - k_w) / stride + 1;
  size_t M = out_h * out_w;
  size_t N = out_c;
  size_t K = in_c * k_h * k_w;

  // Native W8A8 Quantized GEMM with Seungbaek Hong's YOLOv7 Indirect Conv architecture
  if (weights_i8_trans && scales && act_scale > 0.0f) {
    float inv_act_scale = 1.0f / act_scale;
    thread_local static std::vector<int8_t> a_i8;

    if (k_h == 1 && k_w == 1 && stride == 1 && pad == 0) {
      if (a_i8.size() < M * in_c) a_i8.resize(M * in_c);
      #pragma omp parallel for schedule(static)
      for (size_t m = 0; m < M; ++m) {
        quantize_f32_to_i8(in + m * in_c, a_i8.data() + m * in_c, in_c, inv_act_scale);
      }
    } else if (k_h == 1 && k_w == 1 && stride > 1 && pad == 0) {
      if (a_i8.size() < M * in_c) a_i8.resize(M * in_c);
      #pragma omp parallel for schedule(static)
      for (size_t oh = 0; oh < out_h; ++oh) {
        const float *in_row = in + (oh * stride) * in_w * in_c;
        int8_t *sub_row = a_i8.data() + oh * out_w * in_c;
        for (size_t ow = 0; ow < out_w; ++ow) {
          quantize_f32_to_i8(in_row + (ow * stride) * in_c, sub_row + ow * in_c, in_c, inv_act_scale);
        }
      }
    } else {
      if (a_i8.size() < M * K) a_i8.resize(M * K);
      im2col_nhwc_i8(in, in_c, in_h, in_w, k_h, k_w, pad, stride, inv_act_scale, a_i8.data());
    }

    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; m += 4) {
      size_t cur_rows = std::min(static_cast<size_t>(4), M - m);
      const int8_t *a_tile = a_i8.data() + m * K;

      for (size_t n = 0; n < N; n += 4) {
        size_t cur_cols = std::min(static_cast<size_t>(4), N - n);
        const int8_t *w_tile = weights_i8_trans + n * K;

        int32_t acc[4][4];
        if (cur_rows == 4 && cur_cols == 4) {
          dotI8Tile4x4_full(a_tile, K, w_tile, K, K, acc);
        } else {
          dotI8Tile4x4(a_tile, K, cur_rows, w_tile, K, cur_cols, K, acc);
        }

        for (size_t c = 0; c < cur_cols; ++c) {
          size_t out_n = n + c;
          float eff_scale = act_scale * scales[out_n];
          float b = bias ? bias[out_n] : 0.0f;

          for (size_t r = 0; r < cur_rows; ++r) {
            float val = static_cast<float>(acc[r][c]) * eff_scale + b;
            out[(m + r) * N + out_n] = apply_relu ? std::max(0.0f, val) : val;
          }
        }
      }
    }
    return;
  }

  // Fallback: FP32 BLAS GEMM
  thread_local static std::vector<float> col_data;
  const float *a_gemm = in;

  if (k_h == 1 && k_w == 1 && stride == 1 && pad == 0) {
    a_gemm = in;
  } else if (k_h == 1 && k_w == 1 && stride > 1 && pad == 0) {
    thread_local static std::vector<float> subsampled;
    if (subsampled.size() < M * in_c) {
      subsampled.resize(M * in_c);
    }
    #pragma omp parallel for schedule(static)
    for (size_t oh = 0; oh < out_h; ++oh) {
      const float *in_row = in + (oh * stride) * in_w * in_c;
      float *sub_row = subsampled.data() + oh * out_w * in_c;
      for (size_t ow = 0; ow < out_w; ++ow) {
        std::memcpy(sub_row + ow * in_c, in_row + (ow * stride) * in_c, in_c * sizeof(float));
      }
    }
    a_gemm = subsampled.data();
  } else {
    if (col_data.size() < M * K) {
      col_data.resize(M * K);
    }
    im2col_nhwc(in, in_c, in_h, in_w, k_h, k_w, pad, stride, col_data.data());
    a_gemm = col_data.data();
  }

  // Fallback: FP32 BLAS GEMM
  RUN_SGEMM(M, N, K, a_gemm, weights_fp32, out);

  if (bias) {
    #pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; ++m) {
      float *out_row = out + m * N;
      if (apply_relu) {
        #pragma omp simd
        for (size_t n = 0; n < N; ++n) {
          float val = out_row[n] + bias[n];
          out_row[n] = (val > 0.0f) ? val : 0.0f;
        }
      } else {
        #pragma omp simd
        for (size_t n = 0; n < N; ++n) {
          out_row[n] += bias[n];
        }
      }
    }
  } else if (apply_relu) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < M * N; ++i) {
      out[i] = (out[i] > 0.0f) ? out[i] : 0.0f;
    }
  }
}

static inline void copy_i8_vec(int8_t *dst, const int8_t *src, size_t count) {
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
  size_t i = 0;
  for (; i + 64 <= count; i += 64) {
    int8x16_t v0 = vld1q_s8(src + i + 0);
    int8x16_t v1 = vld1q_s8(src + i + 16);
    int8x16_t v2 = vld1q_s8(src + i + 32);
    int8x16_t v3 = vld1q_s8(src + i + 48);
    vst1q_s8(dst + i + 0, v0);
    vst1q_s8(dst + i + 16, v1);
    vst1q_s8(dst + i + 32, v2);
    vst1q_s8(dst + i + 48, v3);
  }
  for (; i + 16 <= count; i += 16) {
    vst1q_s8(dst + i, vld1q_s8(src + i));
  }
  for (; i < count; ++i) {
    dst[i] = src[i];
  }
#else
  std::memcpy(dst, src, count);
#endif
}

static inline void zero_i8_vec(int8_t *dst, size_t count) {
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
  const int8x16_t vzero = vdupq_n_s8(0);
  size_t i = 0;
  for (; i + 64 <= count; i += 64) {
    vst1q_s8(dst + i + 0, vzero);
    vst1q_s8(dst + i + 16, vzero);
    vst1q_s8(dst + i + 32, vzero);
    vst1q_s8(dst + i + 48, vzero);
  }
  for (; i + 16 <= count; i += 16) {
    vst1q_s8(dst + i, vzero);
  }
  for (; i < count; ++i) {
    dst[i] = 0;
  }
#else
  std::memset(dst, 0, count);
#endif
}

// Cache-blocked pure byte copy im2col for strip of rows [m0, m1)
static inline void im2col_nhwc_copy_i8_block(
    const int8_t *data_im, size_t in_c,
    size_t in_h, size_t in_w,
    size_t m0, size_t m1, int8_t *a_tile_buf) {
  size_t K = 9 * in_c;
  size_t row_span = 3 * in_c;

  for (size_t m = m0; m < m1; ++m) {
    size_t oh = m / in_w;
    size_t ow = m % in_w;
    int8_t *col_row = a_tile_buf + (m - m0) * K;

    for (size_t kh = 0; kh < 3; ++kh) {
      int ih = static_cast<int>(oh + kh) - 1;
      int8_t *dst_kh = col_row + kh * row_span;
      if (ih >= 0 && ih < static_cast<int>(in_h)) {
        if (ow >= 1 && ow + 1 < in_w) {
          const int8_t *src = data_im + (static_cast<size_t>(ih) * in_w + (ow - 1)) * in_c;
          copy_i8_vec(dst_kh, src, row_span);
        } else {
          for (size_t kw = 0; kw < 3; ++kw) {
            int iw = static_cast<int>(ow + kw) - 1;
            int8_t *dst = dst_kh + kw * in_c;
            if (iw >= 0 && iw < static_cast<int>(in_w)) {
              copy_i8_vec(dst, data_im + (static_cast<size_t>(ih) * in_w + static_cast<size_t>(iw)) * in_c, in_c);
            } else {
              zero_i8_vec(dst, in_c);
            }
          }
        }
      } else {
        zero_i8_vec(dst_kh, row_span);
      }
    }
  }
}

// Cache-blocked single-pass im2col + quantization for strip of rows [m0, m1)
static inline void im2col_nhwc_i8_block(
    const float *data_im, size_t in_c,
    size_t in_h, size_t in_w, size_t k_h, size_t k_w,
    size_t pad, size_t stride, float inv_scale,
    size_t out_h, size_t out_w,
    size_t m0, size_t m1, int8_t *a_tile_buf) {
  size_t K = k_h * k_w * in_c;

  if (k_h == 3 && k_w == 3 && stride == 1 && pad == 1) {
    for (size_t m = m0; m < m1; ++m) {
      size_t oh = m / out_w;
      size_t ow = m % out_w;
      int8_t *col_row = a_tile_buf + (m - m0) * K;

      for (size_t kh = 0; kh < 3; ++kh) {
        int ih = static_cast<int>(oh + kh) - 1;
        int8_t *dst_kh = col_row + kh * (3 * in_c);
        if (ih >= 0 && ih < static_cast<int>(in_h)) {
          if (ow >= 1 && ow + 1 < in_w) {
            const float *src = data_im + (static_cast<size_t>(ih) * in_w + (ow - 1)) * in_c;
            quantize_f32_to_i8(src, dst_kh, 3 * in_c, inv_scale);
          } else {
            for (size_t kw = 0; kw < 3; ++kw) {
              int iw = static_cast<int>(ow + kw) - 1;
              int8_t *dst = dst_kh + kw * in_c;
              if (iw >= 0 && iw < static_cast<int>(in_w)) {
                const float *src = data_im + (static_cast<size_t>(ih) * in_w + static_cast<size_t>(iw)) * in_c;
                quantize_f32_to_i8(src, dst, in_c, inv_scale);
              } else {
                std::memset(dst, 0, in_c);
              }
            }
          }
        } else {
          std::memset(dst_kh, 0, 3 * in_c);
        }
      }
    }
    return;
  }

  for (size_t m = m0; m < m1; ++m) {
    size_t oh = m / out_w;
    size_t ow = m % out_w;
    int8_t *col_row = a_tile_buf + (m - m0) * K;

    for (size_t kh = 0; kh < k_h; ++kh) {
      int ih = static_cast<int>(oh * stride + kh) - static_cast<int>(pad);
      for (size_t kw = 0; kw < k_w; ++kw) {
        int iw = static_cast<int>(ow * stride + kw) - static_cast<int>(pad);
        int8_t *dst = col_row + (kh * k_w + kw) * in_c;
        if (ih >= 0 && ih < static_cast<int>(in_h) && iw >= 0 && iw < static_cast<int>(in_w)) {
          const float *src = data_im + (static_cast<size_t>(ih) * in_w + static_cast<size_t>(iw)) * in_c;
          quantize_f32_to_i8(src, dst, in_c, inv_scale);
        } else {
          std::memset(dst, 0, in_c);
        }
      }
    }
  }
}

void WeSpeakerResNet34::runConv2d_F32toI8(
    const float *in, int8_t *out_i8, size_t in_c, size_t out_c,
    size_t in_h, size_t in_w, size_t k_h, size_t k_w,
    size_t stride, size_t pad,
    const int8_t *weights_i8_trans,
    const float *scales, const float *bias,
    float in_act_scale, float next_act_scale) {
  size_t out_h = (in_h + 2 * pad - k_h) / stride + 1;
  size_t out_w = (in_w + 2 * pad - k_w) / stride + 1;
  size_t M = out_h * out_w;
  size_t N = out_c;
  size_t K = in_c * k_h * k_w;

  float inv_act_scale = 1.0f / in_act_scale;
  float eff_scale_buf[256];
  float eff_bias_buf[256];
  float inv_next_scale = 1.0f / next_act_scale;
  for (size_t n = 0; n < N; ++n) {
    eff_scale_buf[n] = (in_act_scale * scales[n]) * inv_next_scale;
    eff_bias_buf[n] = (bias ? bias[n] : 0.0f) * inv_next_scale;
  }

  constexpr size_t M_CHUNK = 128;

  #pragma omp parallel for schedule(static)
  for (size_t m0 = 0; m0 < M; m0 += M_CHUNK) {
    size_t m1 = std::min(m0 + M_CHUNK, M);
    size_t local_M = m1 - m0;

    // Small L1/L2 cache-resident buffer: at most 128 * K bytes (36 KB for layer 1!)
    thread_local static std::vector<int8_t> a_tile_buf;
    if (a_tile_buf.size() < M_CHUNK * K) a_tile_buf.resize(M_CHUNK * K);

    im2col_nhwc_i8_block(in, in_c, in_h, in_w, k_h, k_w, pad, stride, inv_act_scale,
                         out_h, out_w, m0, m1, a_tile_buf.data());

    for (size_t lm = 0; lm < local_M; lm += 4) {
      size_t cur_rows = std::min(static_cast<size_t>(4), local_M - lm);
      const int8_t *a_tile = a_tile_buf.data() + lm * K;
      size_t global_m = m0 + lm;

      for (size_t n = 0; n < N; n += 4) {
        size_t cur_cols = std::min(static_cast<size_t>(4), N - n);
        const int8_t *w_tile = weights_i8_trans + n * K;

        int32_t acc[4][4];
        if (cur_rows == 4 && cur_cols == 4) {
          dotI8Tile4x4_full(a_tile, K, w_tile, K, K, acc);
        } else {
          dotI8Tile4x4(a_tile, K, cur_rows, w_tile, K, cur_cols, K, acc);
        }

        for (size_t c = 0; c < cur_cols; ++c) {
          size_t out_n = n + c;
          float eff_scale = eff_scale_buf[out_n];
          float b = eff_bias_buf[out_n];

          for (size_t r = 0; r < cur_rows; ++r) {
            float val = static_cast<float>(acc[r][c]) * eff_scale + b;
            float relu_val = std::max(0.0f, val);
            out_i8[(global_m + r) * N + out_n] = static_cast<int8_t>(
                std::clamp(static_cast<int>(std::round(relu_val)), -128, 127));
          }
        }
      }
    }
  }
}

void WeSpeakerResNet34::runConv2d_I8toF32(
    const int8_t *in_i8, float *out, size_t in_c, size_t out_c,
    size_t in_h, size_t in_w, size_t k_h, size_t k_w,
    size_t stride, size_t pad,
    const int8_t *weights_i8_trans,
    const float *scales, const float *bias,
    float act_scale, bool apply_relu) {
  size_t out_h = (in_h + 2 * pad - k_h) / stride + 1;
  size_t out_w = (in_w + 2 * pad - k_w) / stride + 1;
  size_t M = out_h * out_w;
  size_t N = out_c;
  size_t K = in_c * k_h * k_w;

  float eff_scale_buf[256];
  float eff_bias_buf[256];
  for (size_t n = 0; n < N; ++n) {
    eff_scale_buf[n] = act_scale * scales[n];
    eff_bias_buf[n] = bias ? bias[n] : 0.0f;
  }

  constexpr size_t M_CHUNK = 128;

  #pragma omp parallel for schedule(static)
  for (size_t m0 = 0; m0 < M; m0 += M_CHUNK) {
    size_t m1 = std::min(m0 + M_CHUNK, M);
    size_t local_M = m1 - m0;

    // Small L1/L2 cache-resident buffer: at most 128 * K bytes (36 KB for layer 1!)
    thread_local static std::vector<int8_t> a_tile_buf;
    if (a_tile_buf.size() < M_CHUNK * K) a_tile_buf.resize(M_CHUNK * K);

    im2col_nhwc_copy_i8_block(in_i8, in_c, in_h, in_w, m0, m1, a_tile_buf.data());

    for (size_t lm = 0; lm < local_M; lm += 4) {
      size_t cur_rows = std::min(static_cast<size_t>(4), local_M - lm);
      const int8_t *a_tile = a_tile_buf.data() + lm * K;
      size_t global_m = m0 + lm;

      for (size_t n = 0; n < N; n += 4) {
        size_t cur_cols = std::min(static_cast<size_t>(4), N - n);
        const int8_t *w_tile = weights_i8_trans + n * K;

        int32_t acc[4][4];
        if (cur_rows == 4 && cur_cols == 4) {
          dotI8Tile4x4_full(a_tile, K, w_tile, K, K, acc);
        } else {
          dotI8Tile4x4(a_tile, K, cur_rows, w_tile, K, cur_cols, K, acc);
        }

        for (size_t c = 0; c < cur_cols; ++c) {
          size_t out_n = n + c;
          float eff_scale = eff_scale_buf[out_n];
          float b = eff_bias_buf[out_n];

          for (size_t r = 0; r < cur_rows; ++r) {
            float val = static_cast<float>(acc[r][c]) * eff_scale + b;
            out[(global_m + r) * N + out_n] = apply_relu ? std::max(0.0f, val) : val;
          }
        }
      }
    }
  }
}

void WeSpeakerResNet34::runBasicBlock(const float *in, float *out, size_t in_h, size_t in_w,
                                     const BasicBlockWeights &block, size_t &out_h, size_t &out_w) {
  out_h = (in_h + 2 * 1 - 3) / block.stride + 1;
  out_w = (in_w + 2 * 1 - 3) / block.stride + 1;
  size_t total_out = block.out_channels * out_h * out_w;

  thread_local static std::vector<int8_t> conv1_out_i8;
  thread_local static std::vector<float> ds_out;

  if (conv1_out_i8.size() < total_out) conv1_out_i8.resize(total_out);

  // 1. conv1 + bias1 + relu + direct quantize to int8 (Fused W8A8)
  runConv2d_F32toI8(in, conv1_out_i8.data(), block.in_channels, block.out_channels,
                    in_h, in_w, 3, 3, block.stride, 1,
                    block.conv1_w_i8_trans.data(), block.conv1_s, block.conv1_b,
                    block.conv1_act_scale, block.conv2_act_scale);

  // 2. conv2 + bias2 from int8 (Fused W8A8, no relu before residual addition)
  runConv2d_I8toF32(conv1_out_i8.data(), out, block.out_channels, block.out_channels,
                    out_h, out_w, 3, 3, 1, 1,
                    block.conv2_w_i8_trans.data(), block.conv2_s, block.conv2_b,
                    block.conv2_act_scale, /*apply_relu=*/false);

  // 3. Shortcut / Residual addition + final ReLU
  if (block.downsample) {
    if (ds_out.size() < total_out) ds_out.resize(total_out);
    runConv2d(in, ds_out.data(), block.in_channels, block.out_channels,
              in_h, in_w, 1, 1, block.stride, 0,
              block.downsample_conv_w, block.downsample_w_i8_trans.data(),
              block.downsample_conv_s, block.downsample_conv_b, block.downsample_act_scale,
              /*apply_relu=*/false);
    add_residual_relu_fast(out, ds_out.data(), total_out);
  } else {
    add_residual_relu_fast(out, in, total_out);
  }
}

bool WeSpeakerResNet34::init(const WeightLoader &loader) {
  conv1_w_ = loader.getTensor("resnet.conv1.weight");
  conv1_w_i8_ = loader.getTensorI8("resnet.conv1.weight_i8");
  if (conv1_w_i8_) {
    conv1_w_i8_trans_ = transpose_KN_to_NK(conv1_w_i8_, 3 * 3 * 1, 32);
  }
  conv1_s_ = loader.getTensor("resnet.conv1.scale");
  conv1_b_ = loader.getTensor("resnet.conv1.bias");
  const float *c1_act = loader.getTensor("resnet.conv1.act_scale");
  if (c1_act) conv1_act_scale_ = *c1_act;

  auto load_stage = [&](const std::string &stage_name, size_t num_blocks,
                        size_t in_c, size_t out_c, size_t init_stride) {
    std::vector<BasicBlockWeights> blocks(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b) {
      std::string pfx = "resnet." + stage_name + "." + std::to_string(b);
      size_t b_in = (b == 0) ? in_c : out_c;

      blocks[b].conv1_w = loader.getTensor(pfx + ".conv1.weight");
      blocks[b].conv1_w_i8 = loader.getTensorI8(pfx + ".conv1.weight_i8");
      if (blocks[b].conv1_w_i8) {
        blocks[b].conv1_w_i8_trans = transpose_KN_to_NK(blocks[b].conv1_w_i8, 3 * 3 * b_in, out_c);
      }
      blocks[b].conv1_s = loader.getTensor(pfx + ".conv1.scale");
      blocks[b].conv1_b = loader.getTensor(pfx + ".conv1.bias");
      const float *c1_act = loader.getTensor(pfx + ".conv1.act_scale");
      if (c1_act) blocks[b].conv1_act_scale = *c1_act;

      blocks[b].conv2_w = loader.getTensor(pfx + ".conv2.weight");
      blocks[b].conv2_w_i8 = loader.getTensorI8(pfx + ".conv2.weight_i8");
      if (blocks[b].conv2_w_i8) {
        blocks[b].conv2_w_i8_trans = transpose_KN_to_NK(blocks[b].conv2_w_i8, 3 * 3 * out_c, out_c);
      }
      blocks[b].conv2_s = loader.getTensor(pfx + ".conv2.scale");
      blocks[b].conv2_b = loader.getTensor(pfx + ".conv2.bias");
      const float *c2_act = loader.getTensor(pfx + ".conv2.act_scale");
      if (c2_act) blocks[b].conv2_act_scale = *c2_act;

      if (b == 0) {
        blocks[b].in_channels = in_c;
        blocks[b].out_channels = out_c;
        blocks[b].stride = init_stride;
        blocks[b].downsample = (in_c != out_c || init_stride != 1);
        if (blocks[b].downsample) {
          blocks[b].downsample_conv_w = loader.getTensor(pfx + ".shortcut.0.weight");
          blocks[b].downsample_conv_w_i8 = loader.getTensorI8(pfx + ".shortcut.0.weight_i8");
          if (blocks[b].downsample_conv_w_i8) {
            blocks[b].downsample_w_i8_trans = transpose_KN_to_NK(blocks[b].downsample_conv_w_i8, 1 * 1 * in_c, out_c);
          }
          blocks[b].downsample_conv_s = loader.getTensor(pfx + ".shortcut.0.scale");
          blocks[b].downsample_conv_b = loader.getTensor(pfx + ".shortcut.0.bias");
          const float *ds_act = loader.getTensor(pfx + ".shortcut.0.act_scale");
          if (ds_act) blocks[b].downsample_act_scale = *ds_act;
        }
      } else {
        blocks[b].in_channels = out_c;
        blocks[b].out_channels = out_c;
        blocks[b].stride = 1;
        blocks[b].downsample = false;
      }
    }
    return blocks;
  };

  layer1_ = load_stage("layer1", 3, 32, 32, 1);
  layer2_ = load_stage("layer2", 4, 32, 64, 2);
  layer3_ = load_stage("layer3", 6, 64, 128, 2);
  layer4_ = load_stage("layer4", 3, 128, 256, 2);

  seg_1_w_ = loader.getTensor("resnet.seg_1.weight");
  seg_1_w_i4_ = loader.getTensorI8("resnet.seg_1.weight_i4");
  seg_1_s_ = loader.getTensor("resnet.seg_1.scale");
  seg_1_b_ = loader.getTensor("resnet.seg_1.bias");
  const float *s1_act = loader.getTensor("resnet.seg_1.act_scale");
  if (s1_act) seg_1_act_scale_ = *s1_act;

  return (conv1_w_ && conv1_b_ && seg_1_w_ && seg_1_b_);
}

void WeSpeakerResNet34::forwardBackbone(const float *fbank, float *out_feat,
                                       size_t chunk_idx, DebugTensorCallback debug_cb) {
  std::string pfx = "emb_chunk" + std::to_string(chunk_idx) + "_";

  // Pre-allocated thread-local ping-pong buffers to eliminate 336 dynamic heap allocations
  thread_local static std::vector<float> feat_buf0;
  thread_local static std::vector<float> feat_buf1;
  constexpr size_t MAX_FEAT_SIZE = 32 * 80 * 998;
  if (feat_buf0.size() < MAX_FEAT_SIZE) feat_buf0.resize(MAX_FEAT_SIZE);
  if (feat_buf1.size() < MAX_FEAT_SIZE) feat_buf1.resize(MAX_FEAT_SIZE);

  float *src_feat = feat_buf0.data();
  float *dst_feat = feat_buf1.data();

  // 1. Stem Conv1 + Bias1 + ReLU (Fused W8A8): [80, 998, 1] -> [80, 998, 32]
  runConv2d(fbank, src_feat, 1, 32, 80, 998, 3, 3, 1, 1,
            conv1_w_, conv1_w_i8_trans_.data(), conv1_s_, conv1_b_, conv1_act_scale_, /*apply_relu=*/true);
  if (debug_cb) debug_cb(pfx + "resnet_conv1", src_feat, {1, 80, 998, 32});

  size_t curr_h = 80;
  size_t curr_w = 998;

  auto run_stage = [&](const std::vector<BasicBlockWeights> &blocks) {
    for (size_t b = 0; b < blocks.size(); ++b) {
      size_t next_h = 0, next_w = 0;
      runBasicBlock(src_feat, dst_feat, curr_h, curr_w, blocks[b], next_h, next_w);
      std::swap(src_feat, dst_feat);
      curr_h = next_h;
      curr_w = next_w;
    }
  };

  // 2. layer1: 3 blocks (32 channels, 80 x 998)
  run_stage(layer1_);
  if (debug_cb) debug_cb(pfx + "resnet_layer1", src_feat, {1, curr_h, curr_w, 32});

  // 3. layer2: 4 blocks (64 channels, 40 x 499)
  run_stage(layer2_);
  if (debug_cb) debug_cb(pfx + "resnet_layer2", src_feat, {1, curr_h, curr_w, 64});

  // 4. layer3: 6 blocks (128 channels, 20 x 250)
  run_stage(layer3_);
  if (debug_cb) debug_cb(pfx + "resnet_layer3", src_feat, {1, curr_h, curr_w, 128});

  // 5. layer4: 3 blocks (256 channels, 10 x 125)
  run_stage(layer4_);
  if (debug_cb) debug_cb(pfx + "resnet_layer4", src_feat, {1, curr_h, curr_w, 256});

  std::memcpy(out_feat, src_feat, 256 * 10 * 125 * sizeof(float));
}

std::vector<float> WeSpeakerResNet34::forwardPool(const float *feat, const float *mask,
                                                 size_t chunk_idx, size_t spk_idx, DebugTensorCallback debug_cb) {
  std::string pfx = "emb_chunk" + std::to_string(chunk_idx) + "_spk" + std::to_string(spk_idx) + "_";
  constexpr size_t NUM_FRAMES = 125;
  constexpr size_t FEAT_DIM = 2560; // 256 * 10

  float w[NUM_FRAMES];
  float v1 = 1e-8f;
  float v2 = 0.0f;

  for (size_t t = 0; t < NUM_FRAMES; ++t) {
    size_t src_idx = static_cast<size_t>(t * 589 / 125);
    float weight_val = mask ? mask[src_idx] : 1.0f;
    w[t] = weight_val;
    v1 += weight_val;
    v2 += weight_val * weight_val;
  }

  std::vector<float> stats(5120, 0.0f);
  float var_denom = v1 - (v2 / v1) + 1e-8f;

  for (size_t d = 0; d < FEAT_DIM; ++d) {
    size_t c = d / 10;
    size_t h = d % 10;

    // Weighted mean across time frames in NHWC layout [H=10, W=125, C=256]
    float sum_feat = 0.0f;
    for (size_t t = 0; t < NUM_FRAMES; ++t) {
      float val = feat[(h * NUM_FRAMES + t) * 256 + c];
      sum_feat += val * w[t];
    }
    float mean_val = sum_feat / v1;
    stats[d] = mean_val;

    // Weighted variance & std
    float var_sum = 0.0f;
    for (size_t t = 0; t < NUM_FRAMES; ++t) {
      float val = feat[(h * NUM_FRAMES + t) * 256 + c];
      float diff = val - mean_val;
      var_sum += (diff * diff) * w[t];
    }
    float std_val = std::sqrt(std::max(0.0f, var_sum / var_denom));
    stats[FEAT_DIM + d] = std_val;
  }
  if (debug_cb) debug_cb(pfx + "resnet_pool_tstp", stats.data(), {1, 5120});

  // seg_1: Linear(5120 -> 256) W4 Quantized or FP32
  std::vector<float> embedding(256, 0.0f);
  if (seg_1_w_i4_ && seg_1_s_) {
    // Native W4 computation: stats (float) * int4 * scale
    #pragma omp parallel for schedule(static)
    for (size_t oc = 0; oc < 256; ++oc) {
      const int8_t *w_row_i4 = seg_1_w_i4_ + oc * 5120;
      float acc = 0.0f;
      #pragma omp simd reduction(+:acc)
      for (size_t ic = 0; ic < 5120; ++ic) {
        acc += static_cast<float>(w_row_i4[ic]) * stats[ic];
      }
      embedding[oc] = acc * seg_1_s_[oc] + seg_1_b_[oc];
    }
  } else {
    // FP32 baseline
    #pragma omp parallel for schedule(static)
    for (size_t oc = 0; oc < 256; ++oc) {
      const float *w_row = seg_1_w_ + oc * 5120;
      float acc = seg_1_b_[oc];
      #pragma omp simd reduction(+:acc)
      for (size_t ic = 0; ic < 5120; ++ic) {
        acc += w_row[ic] * stats[ic];
      }
      embedding[oc] = acc;
    }
  }
  if (debug_cb) debug_cb(pfx + "spk_embedding", embedding.data(), {1, 256});

  return embedding;
}

} // namespace speaker_diarization
