// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   mamba2_ssm_layer.cpp
 * @date   09 November 2026
 * @brief  Mamba2 state-space core (Nemotron-H "M" block, inference only)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Claude <noreply@anthropic.com>
 * @bug    No known bugs except for NYI items
 */

#include "mamba2_ssm_layer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <thread_manager.h>

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

namespace {

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

/** numerically stable softplus, matching torch.nn.functional.softplus
 *  (threshold 20 -> identity) */
inline float softplus(float x) {
  if (x > 20.0f)
    return x;
  return std::log1p(std::exp(x));
}

} // namespace

Mamba2SSMLayer::Mamba2SSMLayer() :
  LayerImpl(),
  mamba_props(props::MambaNumHeads(), props::MambaHeadDim(),
              props::MambaNumGroups(), props::MambaStateSize(),
              props::MambaConvKernel(), props::MambaNormEps(),
              props::MambaDtMin()) {
  weight_idx.fill(std::numeric_limits<unsigned int>::max());
  tensor_idx.fill(std::numeric_limits<unsigned int>::max());
}

void Mamba2SSMLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "[Mamba2SSM] requires exactly one input (the in_proj output)";

  n_heads = std::get<props::MambaNumHeads>(mamba_props).get();
  head_dim = std::get<props::MambaHeadDim>(mamba_props).get();
  n_groups = std::get<props::MambaNumGroups>(mamba_props).get();
  d_state = std::get<props::MambaStateSize>(mamba_props).get();
  kernel = std::get<props::MambaConvKernel>(mamba_props).get();
  norm_eps = std::get<props::MambaNormEps>(mamba_props).get();
  dt_min = std::get<props::MambaDtMin>(mamba_props).get();

  NNTR_THROW_IF(n_heads % n_groups != 0, std::invalid_argument)
    << "[Mamba2SSM] mamba_num_heads must be divisible by mamba_n_groups";
  NNTR_THROW_IF(kernel < 1, std::invalid_argument)
    << "[Mamba2SSM] mamba_conv_kernel must be positive";

  d_inner = n_heads * head_dim;
  conv_dim = d_inner + 2 * n_groups * d_state;
  in_width = d_inner + conv_dim + n_heads;

  const nntrainer::TensorDim &in_dim = context.getInputDimensions()[0];
  NNTR_THROW_IF(in_dim.channel() != 1, std::invalid_argument)
    << "[Mamba2SSM] input must be laid out as [B, 1, T, W]";
  NNTR_THROW_IF(in_dim.width() != in_width, std::invalid_argument)
    << "[Mamba2SSM] input width " << in_dim.width() << " != expected "
    << in_width << " (d_inner + conv_dim + num_heads)";

  nntrainer::TensorDim out_dim = in_dim;
  out_dim.width(d_inner);
  out_dim.setDataType(context.getActivationDataType());
  context.setOutputDimensions({out_dim});

  auto fp32_dim = [&](unsigned int h, unsigned int w) {
    return nntrainer::TensorDim(
      {1, 1, h, w},
      {context.getFormat(), ml::train::TensorDim::DataType::FP32});
  };

  // Weight order is the order of the binary file (see weight_converter.py).
  // Every weight is a single FP32 row (the conv taps [kernel, conv_dim] are
  // flattened), which Layer::save() never quantizes: the SSM parameters stay
  // FP32 when the rest of the model is quantized.
  weight_idx[CONV_W] = context.requestWeight(
    fp32_dim(1, kernel * conv_dim), nntrainer::Initializer::NONE,
    nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f, "conv_weight", false);
  weight_idx[CONV_B] = context.requestWeight(
    fp32_dim(1, conv_dim), nntrainer::Initializer::ZEROS,
    nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f, "conv_bias", false);
  weight_idx[DT_BIAS] = context.requestWeight(
    fp32_dim(1, n_heads), nntrainer::Initializer::ZEROS,
    nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f, "dt_bias", false);
  weight_idx[A_LOG] = context.requestWeight(
    fp32_dim(1, n_heads), nntrainer::Initializer::ZEROS,
    nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f, "A_log", false);
  weight_idx[D_SKIP] = context.requestWeight(
    fp32_dim(1, n_heads), nntrainer::Initializer::ONES,
    nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f, "D", false);
  weight_idx[NORM_W] = context.requestWeight(
    fp32_dim(1, d_inner), nntrainer::Initializer::ONES,
    nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f, "norm_weight", false);

  const unsigned int B = in_dim.batch();
  // conv window: the last (kernel - 1) xBC rows, oldest first
  tensor_idx[CONV_STATE] = context.requestTensor(
    nntrainer::TensorDim(
      {B, 1, std::max(1u, kernel - 1), conv_dim},
      {context.getFormat(), ml::train::TensorDim::DataType::FP32}),
    "conv_state", nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
  // ssm state h[head][p][n]
  tensor_idx[SSM_STATE] = context.requestTensor(
    nntrainer::TensorDim(
      {B, 1, n_heads, head_dim * d_state},
      {context.getFormat(), ml::train::TensorDim::DataType::FP32}),
    "ssm_state", nntrainer::Initializer::ZEROS, false,
    nntrainer::TensorLifespan::MAX_LIFESPAN);
}

void Mamba2SSMLayer::runSegment(const float *in, float *out, float *conv_state,
                                float *ssm_state, unsigned int len,
                                bool clamp_dt, const float *conv_w,
                                const float *conv_b, const float *dt_bias,
                                const float *A_log, const float *D,
                                const float *norm_w) {
  const unsigned int K = kernel;
  const unsigned int hist = K > 1 ? K - 1 : 0;
  const unsigned int heads_per_group = n_heads / n_groups;
  const unsigned int group_size = d_inner / n_groups; // gated-norm group
  const size_t head_state = static_cast<size_t>(head_dim) * d_state;

  std::vector<float> xbc(conv_dim);
  std::vector<float> dt(n_heads);
  std::vector<float> dA(n_heads);
  std::vector<float> A(n_heads);
  for (unsigned int h = 0; h < n_heads; ++h)
    A[h] = -std::exp(A_log[h]);

  auto &tm = nntrainer::ThreadManager::Global();

  for (unsigned int t = 0; t < len; ++t) {
    const float *row = in + static_cast<size_t>(t) * in_width;
    const float *z = row;                           // [d_inner]
    const float *xbc_raw = row + d_inner;           // [conv_dim]
    const float *dt_raw = row + d_inner + conv_dim; // [n_heads]
    float *y = out + static_cast<size_t>(t) * d_inner;

    // ---- depthwise causal conv1d + bias + silu
    // out[c] = sum_{k<K-1} w[k][c] * state[k][c] + w[K-1][c] * x[c] + b[c]
    for (unsigned int c = 0; c < conv_dim; ++c) {
      float acc = conv_b[c] +
                  conv_w[static_cast<size_t>(hist) * conv_dim + c] * xbc_raw[c];
      for (unsigned int k = 0; k < hist; ++k)
        acc += conv_w[static_cast<size_t>(k) * conv_dim + c] *
               conv_state[static_cast<size_t>(k) * conv_dim + c];
      xbc[c] = silu(acc);
    }
    // shift the conv window and append the raw input row
    if (hist > 0) {
      if (hist > 1)
        std::memmove(conv_state, conv_state + conv_dim,
                     sizeof(float) * static_cast<size_t>(hist - 1) * conv_dim);
      std::memcpy(conv_state + static_cast<size_t>(hist - 1) * conv_dim,
                  xbc_raw, sizeof(float) * conv_dim);
    }

    const float *x = xbc.data();            // [n_heads*head_dim]
    const float *Bm = xbc.data() + d_inner; // [G, N]
    const float *Cm = xbc.data() + d_inner + n_groups * d_state; // [G, N]

    for (unsigned int h = 0; h < n_heads; ++h) {
      float d = softplus(dt_raw[h] + dt_bias[h]);
      if (clamp_dt)
        d = std::max(d, dt_min);
      dt[h] = d;
      dA[h] = std::exp(d * A[h]);
    }

    // ---- selective state update, one head per task
    tm.parallel_for(0, n_heads, [&](size_t hh) {
      const unsigned int h = static_cast<unsigned int>(hh);
      const unsigned int g = h / heads_per_group;
      const float *b = Bm + static_cast<size_t>(g) * d_state;
      const float *cvec = Cm + static_cast<size_t>(g) * d_state;
      float *S = ssm_state + h * head_state;
      const float decay = dA[h];
      const float dth = dt[h];
      for (unsigned int p = 0; p < head_dim; ++p) {
        const float xv = x[h * head_dim + p];
        const float dbx = dth * xv;
        float *Sp = S + static_cast<size_t>(p) * d_state;
        float acc = 0.0f;
        for (unsigned int n = 0; n < d_state; ++n) {
          const float s = Sp[n] * decay + dbx * b[n];
          Sp[n] = s;
          acc += s * cvec[n];
        }
        y[h * head_dim + p] = acc + D[h] * xv;
      }
    });

    // ---- gated grouped RMSNorm: w * rmsnorm_group(y * silu(z))
    for (unsigned int g = 0; g < n_groups; ++g) {
      float *yg = y + static_cast<size_t>(g) * group_size;
      const float *zg = z + static_cast<size_t>(g) * group_size;
      double ss = 0.0;
      for (unsigned int i = 0; i < group_size; ++i) {
        yg[i] *= silu(zg[i]);
        ss += static_cast<double>(yg[i]) * yg[i];
      }
      const float inv =
        1.0f / std::sqrt(static_cast<float>(ss / group_size) + norm_eps);
      const float *wg = norm_w + static_cast<size_t>(g) * group_size;
      for (unsigned int i = 0; i < group_size; ++i)
        yg[i] = yg[i] * inv * wg[i];
    }
  }
}

void Mamba2SSMLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                            unsigned int from, unsigned int to,
                                            bool training) {
  NNTR_THROW_IF(training, std::invalid_argument)
    << "[Mamba2SSM] training is not supported";
  NNTR_THROW_IF(to <= from, std::invalid_argument)
    << "[Mamba2SSM] invalid range: from=" << from << ", to=" << to;

  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  NNTR_THROW_IF(input.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[Mamba2SSM] only FP32 activations are supported";

  nntrainer::Tensor &conv_state_t = context.getTensor(tensor_idx[CONV_STATE]);
  nntrainer::Tensor &ssm_state_t = context.getTensor(tensor_idx[SSM_STATE]);

  const float *conv_w = context.getWeight(weight_idx[CONV_W]).getData<float>();
  const float *conv_b = context.getWeight(weight_idx[CONV_B]).getData<float>();
  const float *dt_bias =
    context.getWeight(weight_idx[DT_BIAS]).getData<float>();
  const float *A_log = context.getWeight(weight_idx[A_LOG]).getData<float>();
  const float *D = context.getWeight(weight_idx[D_SKIP]).getData<float>();
  const float *norm_w = context.getWeight(weight_idx[NORM_W]).getData<float>();

  const unsigned int B = input.batch();
  const unsigned int H = input.height();
  const unsigned int len = to - from;
  NNTR_THROW_IF(len > H, std::invalid_argument)
    << "[Mamba2SSM] step of " << len << " rows exceeds the plane height " << H;

  const size_t conv_state_len = conv_state_t.getDim().getFeatureLen();
  const size_t ssm_state_len = ssm_state_t.getDim().getFeatureLen();

  for (unsigned int b = 0; b < B; ++b) {
    float *conv_state = conv_state_t.getData<float>() + b * conv_state_len;
    float *ssm_state = ssm_state_t.getData<float>() + b * ssm_state_len;
    if (from == 0) {
      // a new sequence starts: forget the previous one
      std::fill(conv_state, conv_state + conv_state_len, 0.0f);
      std::fill(ssm_state, ssm_state + ssm_state_len, 0.0f);
    }
    // The rows of this step sit at the start of each batch slice.
    const float *in =
      input.getData<float>() + static_cast<size_t>(b) * H * in_width;
    float *out = output.getData<float>() + static_cast<size_t>(b) * H * d_inner;
    runSegment(in, out, conv_state, ssm_state, len, /*clamp_dt=*/len > 1,
               conv_w, conv_b, dt_bias, A_log, D, norm_w);
  }
}

void Mamba2SSMLayer::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  const unsigned int H = context.getInput(SINGLE_INOUT_IDX).height();
  incremental_forwarding(context, 0, H, training);
}

void Mamba2SSMLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("[Mamba2SSM] calcDerivative is not supported");
}

void Mamba2SSMLayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("[Mamba2SSM] calcGradient is not supported");
}

void Mamba2SSMLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dims) {
  // States are sized by batch and model geometry only; nothing to resize.
}

void Mamba2SSMLayer::exportTo(nntrainer::Exporter &exporter,
                              const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(mamba_props, method, this);
}

void Mamba2SSMLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, mamba_props);
  LayerImpl::setProperty(remain_props);
}

} // namespace causallm

#ifdef PLUGGABLE
extern "C" {
nntrainer::Layer *create_mamba2_ssm_layer() {
  return new causallm::Mamba2SSMLayer();
}
void destroy_mamba2_ssm_layer(nntrainer::Layer *layer) { delete layer; }
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_mamba2_ssm_layer, destroy_mamba2_ssm_layer,
  causallm::Mamba2SSMLayer::type};
}
#endif
