// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   int8_gemm.h
 * @date   24 August 2026
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  W8A8 GEMM with channel-wise weight scales and a tensor-wise
 * activation scale.
 *
 * The weight is a Q8_0 byte stream whose blocks all carry the same scale within
 * an output channel (written by nntr_quantize_q8_0_per_channel), so there is
 * exactly one weight scale per channel. The activation is quantized with a
 * single scale for the whole tensor. Together those two properties make the
 * inner loop a pure int8 dot product accumulating into int32, with one scalar
 * multiply per output element at the end -- no per-block scale arithmetic,
 * which is what the block-scaled Q8_0 x Q8_0 kernel has to carry.
 */

#ifndef __NNTR_INT8_GEMM_H__
#define __NNTR_INT8_GEMM_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <thread_manager.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#if defined(ARMV7) && ARMV7
#include <armv7_neon.h>
#endif
#endif

namespace nntrainer {
namespace int8_gemm {

/** Q8_0 block geometry: fp16 scale followed by 32 int8 quants. */
constexpr unsigned int QK = 32;
constexpr unsigned int BLOCK_BYTES = 2 + QK;

/**
 * @brief Decode an fp16 bit pattern to FP32.
 */
inline float fp16BitsToFp32(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1fu;
  const uint32_t man = h & 0x3ffu;
  uint32_t f;
  if (exp == 0)
    f = sign | (man ? (0x38800000u | (man << 13)) : 0u);
  else if (exp == 31)
    f = sign | 0x7f800000u | (man << 13);
  else
    f = sign | ((exp + 112u) << 23) | (man << 13);
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

/**
 * @brief A weight prepared once for the channel-wise kernel.
 */
struct PerChannelWeight {
  bool usable = false;      /**< false when the stream is not channel-wise */
  std::vector<float> scale; /**< [N] one scale per output channel */
};

/**
 * @brief Prepare (and cache) a weight for the channel-wise kernel.
 *
 * Only the per-channel scales are extracted and cached, N floats per weight.
 * The int8 values are read in place from the Q8_0 stream: each block holds its
 * 32 quants contiguously, so the dot product walks block by block and steps
 * over the 2-byte scale between them. An earlier version gathered the values
 * into a contiguous [N, K] copy, which cost a second full set of weights in
 * memory (5.3 MB on this model) for no measurable speed.
 *
 * `usable` is false when any row's blocks disagree on their scale, i.e. the
 * stream is a genuine per-block Q8_0 file. Callers must then fall back to the
 * block-scaled kernel: reading it as channel-wise would silently drop every
 * scale but the first of each row.
 */
inline const PerChannelWeight &prepareWeight(const void *B, unsigned int N,
                                             unsigned int K) {
  static std::mutex mtx;
  static std::unordered_map<const void *, PerChannelWeight> cache;
  std::lock_guard<std::mutex> lk(mtx);
  auto it = cache.find(B);
  if (it != cache.end())
    return it->second;

  const unsigned int nb = K / QK;
  const char *base = static_cast<const char *>(B);
  PerChannelWeight w;

  if (nb == 0 || K % QK != 0)
    return cache.emplace(B, std::move(w)).first->second;

  w.scale.resize(N);
  for (unsigned int n = 0; n < N; ++n) {
    const char *src = base + (size_t)n * nb * BLOCK_BYTES;
    uint16_t d0;
    std::memcpy(&d0, src, 2);
    for (unsigned int b = 1; b < nb; ++b) {
      uint16_t d;
      std::memcpy(&d, src + (size_t)b * BLOCK_BYTES, 2);
      if (d != d0) {
        w.scale.clear();
        w.scale.shrink_to_fit();
        return cache.emplace(B, std::move(w)).first->second;
      }
    }
    w.scale[n] = fp16BitsToFp32(d0);
  }
  w.usable = true;
  return cache.emplace(B, std::move(w)).first->second;
}

/**
 * @brief int8 dot product of `len` elements accumulating in int32.
 */
inline int32_t dotI8(const int8_t *a, const int8_t *b, unsigned int len) {
#if defined(__ARM_FEATURE_DOTPROD)
  int32x4_t acc0 = vdupq_n_s32(0);
  int32x4_t acc1 = vdupq_n_s32(0);
  unsigned int i = 0;
  for (; i + 32 <= len; i += 32) {
    acc0 = vdotq_s32(acc0, vld1q_s8(a + i), vld1q_s8(b + i));
    acc1 = vdotq_s32(acc1, vld1q_s8(a + i + 16), vld1q_s8(b + i + 16));
  }
  for (; i + 16 <= len; i += 16)
    acc0 = vdotq_s32(acc0, vld1q_s8(a + i), vld1q_s8(b + i));
  int32_t sum = vaddvq_s32(vaddq_s32(acc0, acc1));
  for (; i < len; ++i)
    sum += (int32_t)a[i] * (int32_t)b[i];
  return sum;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  int32x4_t acc = vdupq_n_s32(0);
  unsigned int i = 0;
  for (; i + 16 <= len; i += 16) {
    const int8x16_t va = vld1q_s8(a + i);
    const int8x16_t vb = vld1q_s8(b + i);
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(va), vget_low_s8(vb)));
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
  }
  int32_t sum = vaddvq_s32(acc);
  for (; i < len; ++i)
    sum += (int32_t)a[i] * (int32_t)b[i];
  return sum;
#else
  int32_t sum = 0;
  for (unsigned int i = 0; i < len; ++i)
    sum += (int32_t)a[i] * (int32_t)b[i];
  return sum;
#endif
}


/**
 * @brief Accumulate up to 4 activation rows against one Q8_0 weight row.
 *
 * `out[r]` receives the full int32 dot product of activation row r with the
 * weight row, over all `nb` blocks.
 *
 * Two things matter here and neither is expressible by calling dotI8 per
 * block. First, the accumulators stay live across the whole reduction: a
 * per-block call has to fold its vector down to a scalar every 32 values, and
 * for K=768 that is 24 horizontal reductions per output element instead of
 * one. Second, the weight block is loaded once and used by every row in the
 * tile, so a 4-row tile cuts weight traffic by four -- with M=72 the old loop
 * re-read each weight row 72 times.
 *
 * The Q8_0 block is 34 bytes (fp16 scale then 32 quants), so `wq` is only
 * 2-byte aligned. Loading it as int8 keeps the access byte-wise; an int16 or
 * int64 typed load would let the compiler emit an alignment-qualified vld1
 * that faults here.
 */
inline void dotI8Tile(const int8_t *A, size_t lda, unsigned int rows,
                      const char *wrow, unsigned int nb, int32_t *out) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  int32x4_t acc0 = vdupq_n_s32(0), acc1 = vdupq_n_s32(0);
  int32x4_t acc2 = vdupq_n_s32(0), acc3 = vdupq_n_s32(0);

  for (unsigned int b = 0; b < nb; ++b) {
    const int8_t *wq =
      reinterpret_cast<const int8_t *>(wrow + (size_t)b * BLOCK_BYTES + 2);
    const int8x16_t w0 = vld1q_s8(wq);
    const int8x16_t w1 = vld1q_s8(wq + 16);
    const size_t off = (size_t)b * QK;

#if defined(__ARM_FEATURE_DOTPROD)
#define NNTR_I8_ROW(acc, r)                                                    \
  do {                                                                         \
    const int8_t *aq = A + (size_t)(r) * lda + off;                            \
    acc = vdotq_s32(acc, vld1q_s8(aq), w0);                                     \
    acc = vdotq_s32(acc, vld1q_s8(aq + 16), w1);                                \
  } while (0)
#else
#define NNTR_I8_ROW(acc, r)                                                    \
  do {                                                                         \
    const int8_t *aq = A + (size_t)(r) * lda + off;                            \
    const int8x16_t x0 = vld1q_s8(aq);                                          \
    const int8x16_t x1 = vld1q_s8(aq + 16);                                     \
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(x0), vget_low_s8(w0)));         \
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(x0), vget_high_s8(w0)));       \
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(x1), vget_low_s8(w1)));         \
    acc = vpadalq_s16(acc, vmull_s8(vget_high_s8(x1), vget_high_s8(w1)));       \
  } while (0)
#endif

    NNTR_I8_ROW(acc0, 0);
    if (rows > 1)
      NNTR_I8_ROW(acc1, 1);
    if (rows > 2)
      NNTR_I8_ROW(acc2, 2);
    if (rows > 3)
      NNTR_I8_ROW(acc3, 3);
#undef NNTR_I8_ROW
  }

  out[0] = vaddvq_s32(acc0);
  if (rows > 1)
    out[1] = vaddvq_s32(acc1);
  if (rows > 2)
    out[2] = vaddvq_s32(acc2);
  if (rows > 3)
    out[3] = vaddvq_s32(acc3);
#else
  for (unsigned int r = 0; r < rows; ++r) {
    int32_t acc = 0;
    for (unsigned int b = 0; b < nb; ++b) {
      const int8_t *wq =
        reinterpret_cast<const int8_t *>(wrow + (size_t)b * BLOCK_BYTES + 2);
      const int8_t *aq = A + (size_t)r * lda + (size_t)b * QK;
      for (unsigned int t = 0; t < QK; ++t)
        acc += (int32_t)aq[t] * (int32_t)wq[t];
    }
    out[r] = acc;
  }
#endif
}

/**
 * @brief The shared body of the two channel-wise kernels.
 */
inline void gemmPerChannelBody(unsigned int M, unsigned int N, unsigned int K,
                               const int8_t *Aq, float a_scale, const void *B,
                               const std::vector<float> &wscale, float *C) {
  const unsigned int nb = K / QK;
  const char *wbase = static_cast<const char *>(B);
  auto &tm = ThreadManager::Global();
  tm.parallel_for(0, N, [&](size_t n) {
    const char *wrow = wbase + (size_t)n * nb * BLOCK_BYTES;
    const float s = a_scale * wscale[n];
    int32_t acc[4];
    unsigned int m = 0;
    for (; m + 4 <= M; m += 4) {
      dotI8Tile(Aq + (size_t)m * K, K, 4, wrow, nb, acc);
      C[(size_t)m * N + n] = (float)acc[0] * s;
      C[(size_t)(m + 1) * N + n] = (float)acc[1] * s;
      C[(size_t)(m + 2) * N + n] = (float)acc[2] * s;
      C[(size_t)(m + 3) * N + n] = (float)acc[3] * s;
    }
    if (m < M) {
      const unsigned int rem = M - m;
      dotI8Tile(Aq + (size_t)m * K, K, rem, wrow, nb, acc);
      for (unsigned int r = 0; r < rem; ++r)
        C[(size_t)(m + r) * N + n] = (float)acc[r] * s;
    }
  });
}

/**
 * @brief Quantize a whole activation block with one shared scale.
 *
 * @return the scale that maps the int8 values back to FP32
 */
inline float quantizeActivation(const float *A, size_t count,
                                std::vector<int8_t> &out) {
  float amax = 0.0f;
  for (size_t i = 0; i < count; ++i)
    amax = std::max(amax, std::fabs(A[i]));
  const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
  const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;

  out.resize(count);
  int8_t *q = out.data();
  for (size_t i = 0; i < count; ++i) {
    const float r = std::round(A[i] * inv);
    q[i] = (int8_t)std::max(-127.0f, std::min(127.0f, r));
  }
  return scale;
}

/**
 * @brief W8A8 GEMM: FP32 activation [M, K] x channel-wise Q8_0 weight [N, K]
 * -> FP32 output [M, N].
 *
 * @param M activation rows
 * @param N output channels
 * @param K reduction length, must be a multiple of 32
 * @param A activation, row-major [M, K]
 * @param B weight, Q8_0 blocks laid out per output channel
 * @param C output, row-major [M, N]
 * @retval false the weight is not channel-wise; nothing was written and the
 *         caller must use the block-scaled kernel
 */
inline bool gemmPerChannelA8(unsigned int M, unsigned int N, unsigned int K,
                             const float *A, const void *B, float *C) {
  const PerChannelWeight &w = prepareWeight(B, N, K);
  if (!w.usable)
    return false;

  // One scratch buffer per thread, reused across calls: the quantized
  // activation is the only per-call allocation this kernel would otherwise do.
  static thread_local std::vector<int8_t> scratch;
  const float a_scale = quantizeActivation(A, (size_t)M * K, scratch);
  const int8_t *Aq = scratch.data();

  gemmPerChannelBody(M, N, K, Aq, a_scale, B, w.scale, C);
  return true;
}

/**
 * @brief W8A8 GEMM with a pre-quantized int8 activation.
 *
 * Same kernel as gemmPerChannelA8 but the activation arrives already quantized
 * to int8 with a known per-tensor scale (the int8-resident path: the previous
 * layer emitted a QINT8 tensor whose inline scale is a_scale). Skipping the
 * amax+quantize here is what makes activations stay int8 between layers.
 *
 * @param Aq activation int8 data, row-major [M, K]
 * @param a_scale per-tensor FP32 activation scale
 * @retval false the weight is not channel-wise; nothing was written
 */
inline bool gemmPerChannelA8_q8in(unsigned int M, unsigned int N, unsigned int K,
                                  const int8_t *Aq, float a_scale,
                                  const void *B, float *C) {
  const PerChannelWeight &w = prepareWeight(B, N, K);
  if (!w.usable)
    return false;

  gemmPerChannelBody(M, N, K, Aq, a_scale, B, w.scale, C);
  return true;
}

} // namespace int8_gemm
} // namespace nntrainer

#endif /* __NNTR_INT8_GEMM_H__ */
