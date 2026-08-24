// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   cpu_backend.h
 * @date   16 August 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Conditional header file to support unsupported intrinsics on armv7l
 *
 */
#ifndef ARMV7_NEON_
#define ARMV7_NEON_

#include <arm_neon.h>
#include <cmath>

/**
 * @brief macro for vfmaq_n_f32
 *
 */
#define vfmaq_n_f32(a, b, n) vaddq_f32(a, vmulq_f32(b, vmovq_n_f32(n)))

/**
 * @brief vdivq_f32 macro
 *
 * @param a a for a / b
 * @param b b for a / b
 * @return float32x4_t
 */
static inline float32x4_t vdivq_f32(float32x4_t a, float32x4_t b) {
  float32x4_t ret;
  for (unsigned int i = 0; i < 4; ++i) {
    ret[i] = a[i] / b[i];
  }
  return ret;
}

/**
 * @brief vsqrtq_f32 macro
 *
 * @param a input vector
 * @return float32x4_t
 */
static inline float32x4_t vsqrtq_f32(float32x4_t a) {
  float32x4_t ret;
  for (unsigned int i = 0; i < 4; ++i) {
    ret[i] = std::sqrt(a[i]);
  }
  return ret;
}

/**
 * @brief vmaxvq_f32 macro
 *
 * @param a input vector
 * @return float
 */
static inline float vmaxvq_f32(float32x4_t a) {
  float ret = a[0];
  for (unsigned int i = 1; i < 4; ++i) {
    if (ret < a[i])
      ret = a[i];
  }
  return ret;
}

/**
 * @brief vaddvq_f32
 *
 * @param a input vector
 * @return float32_t
 */
static inline float32_t vaddvq_f32(float32x4_t a) {
  float32_t ret = a[0];
  for (unsigned int i = 1; i < 4; ++i) {
    ret += a[i];
  }
  return ret;
}

/**
 * @brief vaddvq_f32
 *
 * @param a input vector
 * @return uint32_t
 */
static inline uint32_t vaddvq_u32(uint32x4_t a) {
  uint32_t ret = a[0];
  for (unsigned int i = 1; i < 4; ++i) {
    ret += a[i];
  }
  return ret;
}

/**
 * @brief vcvtaq_s32_f32 for armv7l
 *
 * AArch64 has FCVTAS (round half away from zero); A32 Advanced SIMD does not
 * expose it through an intrinsic, so round in scalar and narrow.
 *
 * @param a input vector
 * @return int32x4_t
 */
static inline int32x4_t vcvtaq_s32_f32(float32x4_t a) {
  int32x4_t ret;
  for (unsigned int i = 0; i < 4; ++i) {
    ret[i] = std::lround(a[i]);
  }
  return ret;
}

/**
 * @brief vpmaxq_f32 for armv7l
 *
 * A32 only has the 64-bit vpmax_f32, so pair the halves of each operand and
 * recombine. Lane order matches AArch64.
 *
 * @param a first input vector
 * @param b second input vector
 * @return float32x4_t
 */
static inline float32x4_t vpmaxq_f32(float32x4_t a, float32x4_t b) {
  return vcombine_f32(vpmax_f32(vget_low_f32(a), vget_high_f32(a)),
                      vpmax_f32(vget_low_f32(b), vget_high_f32(b)));
}

/**
 * @brief vpaddq_f32 for armv7l
 *
 * A32 only has the 64-bit vpadd_f32; pair the halves of each operand and
 * recombine. Lane order matches AArch64.
 *
 * @param a first input vector
 * @param b second input vector
 * @return float32x4_t
 */
static inline float32x4_t vpaddq_f32(float32x4_t a, float32x4_t b) {
  return vcombine_f32(vpadd_f32(vget_low_f32(a), vget_high_f32(a)),
                      vpadd_f32(vget_low_f32(b), vget_high_f32(b)));
}

static inline int32x4_t vcvtnq_s32_f32(float32x4_t a) {
  int32x4_t ret;
  for (unsigned int i = 0; i < 4; ++i) {
    ret[i] = std::lround(a[i]);
  }
  return ret;
}

#ifdef ENABLE_FP16
/**
 * @brief vfmaq_n_f16 for armv7l
 *
 * A32 has the vector fp16 arithmetic (with -march=armv8.2-a+fp16) but not the
 * by-scalar forms, so expand to a broadcast multiply-add. Signature matches
 * the ACLE intrinsic: every call site passes three arguments.
 *
 * @param a addend
 * @param b vector multiplicand
 * @param n scalar multiplicand
 * @return float16x8_t
 */
static inline float16x8_t vfmaq_n_f16(float16x8_t a, float16x8_t b,
                                      float16_t n) {
  return vaddq_f16(a, vmulq_f16(b, vmovq_n_f16(n)));
}

/**
 * @brief vfma_n_f16 for armv7l
 *
 * 64-bit counterpart of vfmaq_n_f16 above.
 *
 * @param a addend
 * @param b vector multiplicand
 * @param n scalar multiplicand
 * @return float16x4_t
 */
static inline float16x4_t vfma_n_f16(float16x4_t a, float16x4_t b,
                                     float16_t n) {
  return vadd_f16(a, vmul_f16(b, vmov_n_f16(n)));
}

/**
 * @brief by-lane fp16 multiply-add for armv7l
 *
 * A32 exposes the vector fp16 arithmetic but none of the by-lane forms, so
 * broadcast the selected lane and fall back to multiply-add. `lane` must be a
 * literal, exactly as the ACLE intrinsics require.
 */
#define vfma_lane_f16(a, b, v, lane)                                           \
  vadd_f16((a), vmul_f16((b), vmov_n_f16((v)[(lane)])))
#define vfmaq_lane_f16(a, b, v, lane)                                          \
  vaddq_f16((a), vmulq_f16((b), vmovq_n_f16((v)[(lane)])))
#define vfma_laneq_f16(a, b, v, lane)                                          \
  vadd_f16((a), vmul_f16((b), vmov_n_f16((v)[(lane)])))
#define vfmaq_laneq_f16(a, b, v, lane)                                         \
  vaddq_f16((a), vmulq_f16((b), vmovq_n_f16((v)[(lane)])))

/**
 * @brief vzip1_f16 / vzip2_f16 for armv7l
 *
 * A32 only has the paired vzip_f16, which returns both halves at once; AArch64
 * splits it into the two single-result forms these kernels use.
 */
/**
 * @brief vaddvq_u16 for armv7l
 *
 * @param a input vector
 * @return uint16_t horizontal sum
 */
static inline uint16_t vaddvq_u16(uint16x8_t a) {
  uint16_t ret = a[0];
  for (unsigned int i = 1; i < 8; ++i)
    ret = (uint16_t)(ret + a[i]);
  return ret;
}

/**
 * @brief vcvt_high_f32_f16 for armv7l
 *
 * AArch64 widens the upper half in place; A32 has to extract it first.
 *
 * @param a input vector
 * @return float32x4_t of the upper four lanes
 */
static inline float32x4_t vcvt_high_f32_f16(float16x8_t a) {
  return vcvt_f32_f16(vget_high_f16(a));
}

/**
 * @brief vdivq_f16 for armv7l
 *
 * Neither A32 nor A64 Advanced SIMD has a vector fp16 divide instruction, so
 * this is lane-wise, matching the vdivq_f32 shim above.
 *
 * @param a dividend
 * @param b divisor
 * @return float16x8_t
 */
static inline float16x8_t vdivq_f16(float16x8_t a, float16x8_t b) {
  // Via fp32, for the same gcc A32 ICE reason as vmaxvq_f16 above.
  float32x4_t al = vcvt_f32_f16(vget_low_f16(a)), ah = vcvt_f32_f16(vget_high_f16(a));
  float32x4_t bl = vcvt_f32_f16(vget_low_f16(b)), bh = vcvt_f32_f16(vget_high_f16(b));
  return vcombine_f16(vcvt_f16_f32(vdivq_f32(al, bl)),
                      vcvt_f16_f32(vdivq_f32(ah, bh)));
}

/**
 * @brief vsqrtq_f16 for armv7l
 *
 * @param a input vector
 * @return float16x8_t lane-wise square root
 */
static inline float16x8_t vsqrtq_f16(float16x8_t a) {
  float32x4_t lo = vcvt_f32_f16(vget_low_f16(a));
  float32x4_t hi = vcvt_f32_f16(vget_high_f16(a));
  return vcombine_f16(vcvt_f16_f32(vsqrtq_f32(lo)), vcvt_f16_f32(vsqrtq_f32(hi)));
}

/**
 * @brief vpaddq_f16 for armv7l
 *
 * A32 only has the 64-bit vpadd_f16; pair the halves and recombine.
 *
 * @param a first input vector
 * @param b second input vector
 * @return float16x8_t
 */
static inline float16x8_t vpaddq_f16(float16x8_t a, float16x8_t b) {
  return vcombine_f16(vpadd_f16(vget_low_f16(a), vget_high_f16(a)),
                      vpadd_f16(vget_low_f16(b), vget_high_f16(b)));
}

static inline float16x4_t vzip1_f16(float16x4_t a, float16x4_t b) {
  return vzip_f16(a, b).val[0];
}
static inline float16x4_t vzip2_f16(float16x4_t a, float16x4_t b) {
  return vzip_f16(a, b).val[1];
}

/**
 * @brief vmaxvq_f16 macro
 *
 * @param a input vector
 * @return float16_t
 */
static inline float16_t vmaxvq_f16(float16x8_t a) {
  // Reduce through fp32: a scalar __fp16 loop here makes gcc 14.2's A32
  // backend ICE ("unrecognizable insn") once the vectorizer gets hold of it.
  float32x4_t lo = vcvt_f32_f16(vget_low_f16(a));
  float32x4_t hi = vcvt_f32_f16(vget_high_f16(a));
  float32x4_t m = vmaxq_f32(lo, hi);
  float32x2_t m2 = vpmax_f32(vget_low_f32(m), vget_high_f32(m));
  m2 = vpmax_f32(m2, m2);
  return (float16_t)vget_lane_f32(m2, 0);
}

/**
 * @brief vminvq_f16 for armv7l
 *
 * @param a input vector
 * @return float16_t horizontal minimum
 */
static inline float16_t vminvq_f16(float16x8_t a) {
  float32x4_t lo = vcvt_f32_f16(vget_low_f16(a));
  float32x4_t hi = vcvt_f32_f16(vget_high_f16(a));
  float32x4_t m = vminq_f32(lo, hi);
  float32x2_t m2 = vpmin_f32(vget_low_f32(m), vget_high_f32(m));
  m2 = vpmin_f32(m2, m2);
  return (float16_t)vget_lane_f32(m2, 0);
}
#endif
#endif
