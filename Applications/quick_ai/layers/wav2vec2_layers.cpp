// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   wav2vec2_layers.cpp
 * @date   04 November 2026
 * @brief  wav2vec 2.0 feature encoder and positional convolution layers
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include <nntrainer_error.h>
#include <thread_manager.h>
#include <wav2vec2_layers.h>

namespace quick_ai {

namespace {

using DataType = nntrainer::TensorDim::DataType;

/** rows of the im2col buffer computed at once (bounds its memory) */
constexpr size_t CHUNK_ROWS = 1024;

std::vector<int> parseInts(const std::string &value) {
  std::vector<int> out;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ','))
    if (!item.empty())
      out.push_back(std::stoi(item));
  return out;
}

inline float gelu(float x) {
  return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f));
}

nntrainer::TensorDim fp32Dim(unsigned int b, unsigned int c, unsigned int h,
                             unsigned int w) {
  return nntrainer::TensorDim(
    b, c, h, w,
    nntrainer::TensorDim::TensorType(nntrainer::TensorDim::Format::NCHW,
                                     DataType::FP32));
}

/**
 * @brief out (rows x n) = a (rows x k) . w^T, w is (n x k): sgemm through
 *        nntrainer's tensors
 */
void matmulTransB(const float *a, size_t rows, size_t k, const float *w,
                  size_t n, float *out) {
  nntrainer::Tensor A =
    nntrainer::Tensor::Map(const_cast<float *>(a), rows * k * sizeof(float),
                           fp32Dim(1, 1, static_cast<unsigned int>(rows),
                                   static_cast<unsigned int>(k)),
                           0);
  nntrainer::Tensor W = nntrainer::Tensor::Map(
    const_cast<float *>(w), n * k * sizeof(float),
    fp32Dim(1, 1, static_cast<unsigned int>(n), static_cast<unsigned int>(k)),
    0);
  nntrainer::Tensor O =
    nntrainer::Tensor::Map(out, rows * n * sizeof(float),
                           fp32Dim(1, 1, static_cast<unsigned int>(rows),
                                   static_cast<unsigned int>(n)),
                           0);
  A.dot(W, O, false, true);
}

/**
 * @brief y[t] = sum_j x[t * stride + j] (frames x c_in, row-major) with the
 *        weight (c_out, c_in, k) + bias, for t in [0, t_out)
 */
void conv1d(const float *x, size_t c_in, const float *w, const float *bias,
            size_t c_out, size_t k, size_t stride, size_t t_out, float *y) {
  const size_t cols = c_in * k;
  std::vector<float> col(std::min(CHUNK_ROWS, t_out) * cols);
  for (size_t t0 = 0; t0 < t_out; t0 += CHUNK_ROWS) {
    const size_t rows = std::min(CHUNK_ROWS, t_out - t0);
    for (size_t r = 0; r < rows; ++r) {
      const float *src = x + (t0 + r) * stride * c_in;
      float *dst = col.data() + r * cols;
      // col[ci * k + j] = x[t * stride + j][ci]
      for (size_t j = 0; j < k; ++j)
        for (size_t ci = 0; ci < c_in; ++ci)
          dst[ci * k + j] = src[j * c_in + ci];
    }
    matmulTransB(col.data(), rows, cols, w, c_out, y + t0 * c_out);
  }
  if (bias != nullptr) {
    for (size_t t = 0; t < t_out; ++t)
      for (size_t c = 0; c < c_out; ++c)
        y[t * c_out + c] += bias[c];
  }
}

} // namespace

/* ---------------------------- feature encoder ---------------------------- */

size_t Wav2Vec2FeatureEncoderLayer::framesOf(size_t n,
                                             const std::vector<int> &kernel,
                                             const std::vector<int> &stride) {
  for (size_t i = 0; i < kernel.size(); ++i) {
    if (n < static_cast<size_t>(kernel[i]))
      return 0;
    n = (n - kernel[i]) / stride[i] + 1;
  }
  return n;
}

void Wav2Vec2FeatureEncoderLayer::finalize(
  nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(dim_.empty() || dim_.size() != kernel_.size() ||
                  dim_.size() != stride_.size(),
                std::invalid_argument)
    << "wav2vec2_feature_encoder: conv_dim, conv_kernel and conv_stride need "
       "one value per layer";
  NNTR_THROW_IF(norm_ != "layer" && norm_ != "group", std::invalid_argument)
    << "wav2vec2_feature_encoder: norm is layer or group";

  const auto &in = context.getInputDimensions()[0];
  const size_t frames = framesOf(in.width(), kernel_, stride_);
  NNTR_THROW_IF(frames == 0, std::invalid_argument)
    << "wav2vec2_feature_encoder: input too short";
  context.setOutputDimensions(
    {fp32Dim(1, 1, static_cast<unsigned int>(frames),
             static_cast<unsigned int>(dim_.back()))});

  unsigned int c_in = 1;
  for (size_t i = 0; i < dim_.size(); ++i) {
    const std::string p = "conv" + std::to_string(i);
    const unsigned int c_out = static_cast<unsigned int>(dim_[i]);
    w_idx_.push_back(context.requestWeight(
      fp32Dim(c_out, c_in, 1, static_cast<unsigned int>(kernel_[i])),
      nntrainer::props::InitializerInfo::Enum::ZEROS,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, p + "_weight", false));
    if (bias_)
      b_idx_.push_back(context.requestWeight(
        fp32Dim(1, 1, 1, c_out), nntrainer::props::InitializerInfo::Enum::ZEROS,
        nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, p + "_bias", false));
    if (norm_ == "layer" || i == 0) {
      g_idx_.push_back(context.requestWeight(
        fp32Dim(1, 1, 1, c_out), nntrainer::props::InitializerInfo::Enum::ONES,
        nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, p + "_norm_gamma",
        false));
      beta_idx_.push_back(context.requestWeight(
        fp32Dim(1, 1, 1, c_out), nntrainer::props::InitializerInfo::Enum::ZEROS,
        nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, p + "_norm_beta",
        false));
    }
    c_in = c_out;
  }
}

void Wav2Vec2FeatureEncoderLayer::forwarding(
  nntrainer::RunLayerContext &context, bool training) {
  incremental_forwarding(context, 0, 0, training);
}

void Wav2Vec2FeatureEncoderLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &input = context.getInput(0);
  nntrainer::Tensor &output = context.getOutput(0);
  NNTR_THROW_IF(input.getDataType() != DataType::FP32 ||
                  output.getDataType() != DataType::FP32,
                std::invalid_argument)
    << "wav2vec2_feature_encoder works in FP32";

  const size_t max_samples = input.getDim().width();
  const size_t n =
    num_samples_ > 0 ? std::min(num_samples_, max_samples) : max_samples;
  std::vector<float> x(input.getData<float>(), input.getData<float>() + n);
  size_t frames = n, c_in = 1;

  std::vector<float> y;
  for (size_t i = 0; i < dim_.size(); ++i) {
    const size_t k = kernel_[i], s = stride_[i], c_out = dim_[i];
    const size_t t_out = (frames - k) / s + 1;
    y.assign(t_out * c_out, 0.0f);
    const float *w = context.getWeight(w_idx_[i]).getData<float>();
    const float *b =
      bias_ ? context.getWeight(b_idx_[i]).getData<float>() : nullptr;
    conv1d(x.data(), c_in, w, b, c_out, k, s, t_out, y.data());

    const bool has_norm = norm_ == "layer" || i == 0;
    if (has_norm) {
      const float *gamma = context.getWeight(g_idx_[i]).getData<float>();
      const float *beta = context.getWeight(beta_idx_[i]).getData<float>();
      if (norm_ == "layer") {
        // every frame over its channels
        for (size_t t = 0; t < t_out; ++t) {
          float *row = y.data() + t * c_out;
          double mean = 0.0, var = 0.0;
          for (size_t c = 0; c < c_out; ++c)
            mean += row[c];
          mean /= c_out;
          for (size_t c = 0; c < c_out; ++c)
            var += (row[c] - mean) * (row[c] - mean);
          const float inv =
            1.0f / std::sqrt(static_cast<float>(var / c_out) + eps_);
          for (size_t c = 0; c < c_out; ++c)
            row[c] =
              (row[c] - static_cast<float>(mean)) * inv * gamma[c] + beta[c];
        }
      } else {
        // GroupNorm with one group per channel: every channel over the frames
        for (size_t c = 0; c < c_out; ++c) {
          double mean = 0.0, var = 0.0;
          for (size_t t = 0; t < t_out; ++t)
            mean += y[t * c_out + c];
          mean /= t_out;
          for (size_t t = 0; t < t_out; ++t) {
            const double d = y[t * c_out + c] - mean;
            var += d * d;
          }
          const float inv =
            1.0f / std::sqrt(static_cast<float>(var / t_out) + eps_);
          for (size_t t = 0; t < t_out; ++t) {
            float &v = y[t * c_out + c];
            v = (v - static_cast<float>(mean)) * inv * gamma[c] + beta[c];
          }
        }
      }
    }
    for (float &v : y)
      v = gelu(v);
    x.swap(y);
    frames = t_out;
    c_in = c_out;
  }

  const size_t max_frames = output.getDim().height();
  NNTR_THROW_IF(frames > max_frames, std::runtime_error)
    << "wav2vec2_feature_encoder: " << frames << " frames > " << max_frames;
  float *out = output.getData<float>();
  std::memcpy(out, x.data(), frames * c_in * sizeof(float));
  std::fill(out + frames * c_in, out + max_frames * c_in, 0.0f);
}

void Wav2Vec2FeatureEncoderLayer::setProperty(
  const std::vector<std::string> &values) {
  for (const auto &property : values) {
    std::string key, value;
    if (nntrainer::getKeyValue(property, key, value) != 0)
      throw std::invalid_argument("wav2vec2_feature_encoder: bad property " +
                                  property);
    if (key == "conv_dim")
      dim_ = parseInts(value);
    else if (key == "conv_kernel")
      kernel_ = parseInts(value);
    else if (key == "conv_stride")
      stride_ = parseInts(value);
    else if (key == "conv_bias")
      bias_ = value == "true" || value == "1";
    else if (key == "norm")
      norm_ = value;
    else if (key == "epsilon")
      eps_ = std::stof(value);
    else if (key == "num_samples")
      num_samples_ = static_cast<size_t>(std::stoul(value));
  }
}

/* ---------------------------- positional conv ---------------------------- */

void Wav2Vec2PosConvLayer::finalize(nntrainer::InitLayerContext &context) {
  const auto &in = context.getInputDimensions()[0];
  const unsigned int c = in.width();
  NNTR_THROW_IF(groups_ == 0 || c % groups_ != 0, std::invalid_argument)
    << "wav2vec2_pos_conv: " << c << " channels in " << groups_ << " groups";
  context.setOutputDimensions({in});
  w_idx_ = context.requestWeight(fp32Dim(c, c / groups_, 1, kernel_),
                                 nntrainer::props::InitializerInfo::Enum::ZEROS,
                                 nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f,
                                 "weight", false);
  b_idx_ = context.requestWeight(
    fp32Dim(1, 1, 1, c), nntrainer::props::InitializerInfo::Enum::ZEROS,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "bias", false);
}

void Wav2Vec2PosConvLayer::forwarding(nntrainer::RunLayerContext &context,
                                      bool training) {
  incremental_forwarding(context, 0, context.getInput(0).getDim().height(),
                         training);
}

void Wav2Vec2PosConvLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &input = context.getInput(0);
  nntrainer::Tensor &output = context.getOutput(0);
  NNTR_THROW_IF(input.getDataType() != DataType::FP32 ||
                  output.getDataType() != DataType::FP32,
                std::invalid_argument)
    << "wav2vec2_pos_conv works in FP32";

  const size_t C = input.getDim().width();
  const size_t T = std::min<size_t>(to - from, input.getDim().height());
  const size_t G = groups_, Cg = C / G, K = kernel_;
  const long pad = static_cast<long>(K / 2);
  const float *x = input.getData<float>();
  const float *w = context.getWeight(w_idx_).getData<float>(); // (C, Cg, K)
  const float *bias = context.getWeight(b_idx_).getData<float>();
  float *y = output.getData<float>();

  const size_t cols = Cg * K;
  std::vector<float> col(std::min(CHUNK_ROWS, T) * cols), part;
  for (size_t g = 0; g < G; ++g) {
    const float *wg = w + g * Cg * cols; // output channels of the group
    for (size_t t0 = 0; t0 < T; t0 += CHUNK_ROWS) {
      const size_t rows = std::min(CHUNK_ROWS, T - t0);
      for (size_t r = 0; r < rows; ++r) {
        float *dst = col.data() + r * cols;
        for (size_t ci = 0; ci < Cg; ++ci) {
          for (size_t k = 0; k < K; ++k) {
            const long src =
              static_cast<long>(t0 + r) + static_cast<long>(k) - pad;
            dst[ci * K + k] = (src >= 0 && src < static_cast<long>(T))
                                ? x[src * C + g * Cg + ci]
                                : 0.0f;
          }
        }
      }
      part.resize(rows * Cg);
      matmulTransB(col.data(), rows, cols, wg, Cg, part.data());
      for (size_t r = 0; r < rows; ++r)
        for (size_t co = 0; co < Cg; ++co) {
          const size_t c = g * Cg + co;
          y[(t0 + r) * C + c] = gelu(part[r * Cg + co] + bias[c]);
        }
    }
  }
}

void Wav2Vec2PosConvLayer::setProperty(const std::vector<std::string> &values) {
  for (const auto &property : values) {
    std::string key, value;
    if (nntrainer::getKeyValue(property, key, value) != 0)
      throw std::invalid_argument("wav2vec2_pos_conv: bad property " +
                                  property);
    if (key == "kernel_size")
      kernel_ = static_cast<unsigned int>(std::stoul(value));
    else if (key == "groups")
      groups_ = static_cast<unsigned int>(std::stoul(value));
  }
}

} // namespace quick_ai
