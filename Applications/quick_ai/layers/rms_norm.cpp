// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rms_norm.cpp
 * @date   19 July 2023
 * @brief  Implementation of custom RMS normalization function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <cpu_backend.h>
#include <iostream>

#include "rms_norm.h"

namespace quick_ai {

static constexpr size_t SINGLE_INOUT_IDX = 0;

// Global cache tensor to prevent output_norm activations from being recycled/overwritten by TensorPool
nntrainer::Tensor output_norm_cache;
unsigned int cache_max_height = 66; // Dynamic max timestep height for the cache

void RMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);

  if (context.getName().find("output_norm") != std::string::npos) {
    cache_max_height = 1024;
  }

  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();

  // gamma is unquantized and stored as FP32 in the bin. Request it as FP32
  // regardless of the activation dtype; declaring it FP16 reinterprets the
  // on-disk FP32 bytes as FP16 and corrupts gamma (≈FP16-max garbage). The
  // FP16 forward path casts gamma down to FP16 at the multiply site.
  auto gamma_dtype = (context.getWeightDataType() == nntrainer::TensorDim::DataType::Q4_0)
    ? context.getActivationDataType()
    : context.getWeightDataType();
  nntrainer::TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    nntrainer::TensorDim::TensorType(context.getFormat(), gamma_dtype));
  wt_idx[RMSParams::gamma] = context.requestWeight(
    gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", true);
}

void RMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {}

void RMSNormLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  unsigned int b_size = in_dim.batch();

  // Cache gamma once in out_step datatype to avoid per-step heap allocation & conversions
  if (cached_gamma.empty() || cached_gamma.getDataType() != out_step_dim.getDataType()) {
    nntrainer::TensorDim g_dim = gamma.getDim();
    g_dim.setDataType(out_step_dim.getDataType());
    cached_gamma = nntrainer::Tensor(g_dim, true);
    if (gamma.getDataType() != out_step_dim.getDataType()) {
      if (out_step_dim.getDataType() == ml::train::TensorDim::DataType::FP16) {
        const float *src = gamma.getData<float>();
        _FP16 *dst = cached_gamma.getData<_FP16>();
        for (unsigned int i = 0; i < gamma.getDim().getDataLen(); ++i) dst[i] = (_FP16)src[i];
      } else {
        const _FP16 *src = gamma.getData<_FP16>();
        float *dst = cached_gamma.getData<float>();
        for (unsigned int i = 0; i < gamma.getDim().getDataLen(); ++i) dst[i] = (float)src[i];
      }
    } else {
      cached_gamma.copy(gamma);
    }
  }

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      const auto &dim = in_step.getDim();
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        in_step.getData<float>(), out_step.getData<float>(), dim.height(),
        dim.width(), epsilon);
      out_step.multiply_i(cached_gamma);
#ifdef ENABLE_FP16
    } else if (in_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
      const auto &dim = in_step.getDim();
      const size_t H = dim.height();
      const size_t W = dim.width();
      const _FP16 *in_ptr = in_step.getData<_FP16>();
      _FP16 *out_ptr = out_step.getData<_FP16>();
      const _FP16 *g_ptr = cached_gamma.getData<_FP16>();

      for (size_t h = 0; h < H; ++h) {
        const _FP16 *rowX = in_ptr + h * W;
        _FP16 *rowY = out_ptr + h * W;

        size_t i = 0;
        float32x4_t acc0 = vdupq_n_f32(0.F);
        float32x4_t acc1 = vdupq_n_f32(0.F);

        for (; i + 8 <= W; i += 8) {
          float16x8_t h8 = vld1q_f16(rowX + i);
          float32x4_t f0 = vcvt_f32_f16(vget_low_f16(h8));
          float32x4_t f1 = vcvt_f32_f16(vget_high_f16(h8));
          acc0 = vfmaq_f32(acc0, f0, f0);
          acc1 = vfmaq_f32(acc1, f1, f1);
        }
        if (i + 4 <= W) {
          float16x4_t h4 = vld1_f16(rowX + i);
          float32x4_t f = vcvt_f32_f16(h4);
          acc0 = vfmaq_f32(acc0, f, f);
          i += 4;
        }
        float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
        for (; i < W; ++i) {
          float x = (float)rowX[i];
          sum += x * x;
        }

        float mean_single = sum / (float)W;
        float scale_single = 1.F / std::sqrt(mean_single + epsilon);
        float16x8_t scale_v = vdupq_n_f16((__fp16)scale_single);
        float16x4_t scale_v4 = vdup_n_f16((__fp16)scale_single);

        i = 0;
        for (; i + 8 <= W; i += 8) {
          float16x8_t xh = vld1q_f16(rowX + i);
          float16x8_t gh = vld1q_f16(g_ptr + i);
          vst1q_f16(rowY + i, vmulq_f16(vmulq_f16(xh, scale_v), gh));
        }
        if (i + 4 <= W) {
          float16x4_t xh4 = vld1_f16(rowX + i);
          float16x4_t gh4 = vld1_f16(g_ptr + i);
          vst1_f16(rowY + i, vmul_f16(vmul_f16(xh4, scale_v4), gh4));
          i += 4;
        }
        for (; i < W; ++i) {
          rowY[i] = (__fp16)((float)rowX[i] * scale_single * (float)g_ptr[i]);
        }
      }
#endif
    } else {
      throw std::invalid_argument(
        "Error: not yet implemented for this data type");
    }

    if (context.getName() == "output_norm") {
      if (from == 0) {
        // Allocate with full cache_max_height once during prefill!
        ml::train::TensorDim cache_dim = out.getDim();
        cache_dim.height(cache_max_height);
        output_norm_cache = nntrainer::Tensor(cache_dim, true);
        output_norm_cache.setZero();
      }
      ml::train::TensorDim slice_dim = out_step.getDim();
      nntrainer::Tensor slice = output_norm_cache.getSharedDataTensor(
        slice_dim, from * out_dim.width());
      slice.copyData(out_step);
    }

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << "gamma:" << gamma << std::endl;
#endif
  }
}

void RMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
  if (context.getName().find("output_norm") != std::string::npos) {
    cache_max_height = 1024;
  }
}

void RMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new RMSNormLayer();
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace quick_ai
