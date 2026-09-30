// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   wespeaker_resnet34.cpp
 * @date   29 September 2026
 * @brief  WeSpeaker ResNet34 Speaker Embedding Model implementation in NNTrainer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "wespeaker_resnet34.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
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

static inline float relu(float x) {
  return std::max(0.0f, x);
}

static inline void im2col(const float *data_im, size_t channels,
                          size_t height, size_t width, size_t kernel_h, size_t kernel_w,
                          size_t pad, size_t stride, float *data_col) {
  size_t height_col = (height + 2 * pad - kernel_h) / stride + 1;
  size_t width_col = (width + 2 * pad - kernel_w) / stride + 1;
  size_t out_spatial = height_col * width_col;
  size_t channels_col = channels * kernel_h * kernel_w;

  #pragma omp parallel for if(channels_col >= 32) schedule(static)
  for (size_t c = 0; c < channels_col; ++c) {
    size_t w_offset = c % kernel_w;
    size_t h_offset = (c / kernel_w) % kernel_h;
    size_t c_im = c / (kernel_h * kernel_w);
    const float *im_c = data_im + c_im * (height * width);
    float *col_row = data_col + c * out_spatial;

    for (size_t h = 0; h < height_col; ++h) {
      int h_pad = static_cast<int>(h * stride + h_offset) - static_cast<int>(pad);
      if (h_pad >= 0 && h_pad < static_cast<int>(height)) {
        const float *im_row = im_c + h_pad * width;
        float *col_target = col_row + h * width_col;
        for (size_t w = 0; w < width_col; ++w) {
          int w_pad = static_cast<int>(w * stride + w_offset) - static_cast<int>(pad);
          if (w_pad >= 0 && w_pad < static_cast<int>(width)) {
            col_target[w] = im_row[w_pad];
          } else {
            col_target[w] = 0.0f;
          }
        }
      } else {
        std::memset(col_row + h * width_col, 0, width_col * sizeof(float));
      }
    }
  }
}

void WeSpeakerResNet34::runConv2d(const float *in, float *out, size_t in_c, size_t out_c,
                                 size_t in_h, size_t in_w, size_t k_h, size_t k_w,
                                 size_t stride, size_t pad, const float *weights) {
  size_t out_h = (in_h + 2 * pad - k_h) / stride + 1;
  size_t out_w = (in_w + 2 * pad - k_w) / stride + 1;
  size_t N = out_h * out_w;
  size_t M = out_c;
  size_t K = in_c * k_h * k_w;

  if (k_h == 1 && k_w == 1 && stride == 1 && pad == 0) {
    RUN_SGEMM(M, N, K, weights, in, out);
    return;
  }

  if (k_h == 1 && k_w == 1 && stride > 1 && pad == 0) {
    thread_local static std::vector<float> subsampled;
    if (subsampled.size() < in_c * N) {
      subsampled.resize(in_c * N);
    }
    for (size_t c = 0; c < in_c; ++c) {
      const float *in_c_ptr = in + c * (in_h * in_w);
      float *sub_c_ptr = subsampled.data() + c * N;
      for (size_t oh = 0; oh < out_h; ++oh) {
        for (size_t ow = 0; ow < out_w; ++ow) {
          sub_c_ptr[oh * out_w + ow] = in_c_ptr[(oh * stride) * in_w + (ow * stride)];
        }
      }
    }
    RUN_SGEMM(M, N, K, weights, subsampled.data(), out);
    return;
  }

  thread_local static std::vector<float> col_data;
  if (col_data.size() < K * N) {
    col_data.resize(K * N);
  }
  im2col(in, in_c, in_h, in_w, k_h, k_w, pad, stride, col_data.data());
  RUN_SGEMM(M, N, K, weights, col_data.data(), out);
}

void WeSpeakerResNet34::runBatchNorm2d(const float *in, float *out, size_t channels,
                                      size_t height, size_t width, const BatchNormParams &bn, bool apply_relu) {
  size_t hw = height * width;
  for (size_t c = 0; c < channels; ++c) {
    float scale = bn.gamma[c] / std::sqrt(bn.var[c] + bn.eps);
    float bias = bn.beta[c] - bn.gamma[c] * bn.mean[c] / std::sqrt(bn.var[c] + bn.eps);

    const float *in_c = in + c * hw;
    float *out_c = out + c * hw;

    for (size_t i = 0; i < hw; ++i) {
      float val = in_c[i] * scale + bias;
      out_c[i] = apply_relu ? relu(val) : val;
    }
  }
}

void WeSpeakerResNet34::runBasicBlock(const float *in, float *out, size_t in_h, size_t in_w,
                                     const BasicBlockWeights &block, size_t &out_h, size_t &out_w) {
  out_h = (in_h + 2 * 1 - 3) / block.stride + 1;
  out_w = (in_w + 2 * 1 - 3) / block.stride + 1;

  // 1. conv1: [out_c, out_h, out_w]
  std::vector<float> conv1_out(block.out_channels * out_h * out_w);
  runConv2d(in, conv1_out.data(), block.in_channels, block.out_channels,
            in_h, in_w, 3, 3, block.stride, 1, block.conv1_w);

  // 2. bn1 + relu
  std::vector<float> bn1_out(block.out_channels * out_h * out_w);
  runBatchNorm2d(conv1_out.data(), bn1_out.data(), block.out_channels, out_h, out_w, block.bn1, true);

  // 3. conv2: stride 1, padding 1
  std::vector<float> conv2_out(block.out_channels * out_h * out_w);
  runConv2d(bn1_out.data(), conv2_out.data(), block.out_channels, block.out_channels,
            out_h, out_w, 3, 3, 1, 1, block.conv2_w);

  // 4. bn2 (no relu)
  std::vector<float> bn2_out(block.out_channels * out_h * out_w);
  runBatchNorm2d(conv2_out.data(), bn2_out.data(), block.out_channels, out_h, out_w, block.bn2, false);

  // 5. Shortcut / Residual
  std::vector<float> residual(block.out_channels * out_h * out_w);
  if (block.downsample) {
    std::vector<float> ds_conv(block.out_channels * out_h * out_w);
    runConv2d(in, ds_conv.data(), block.in_channels, block.out_channels,
              in_h, in_w, 1, 1, block.stride, 0, block.downsample_conv_w);
    runBatchNorm2d(ds_conv.data(), residual.data(), block.out_channels, out_h, out_w, block.downsample_bn, false);
  } else {
    std::memcpy(residual.data(), in, block.out_channels * out_h * out_w * sizeof(float));
  }

  // 6. Addition + ReLU
  size_t total_elems = block.out_channels * out_h * out_w;
  for (size_t i = 0; i < total_elems; ++i) {
    out[i] = relu(bn2_out[i] + residual[i]);
  }
}

static BatchNormParams extract_bn(const WeightLoader &loader, const std::string &prefix) {
  BatchNormParams bn;
  bn.gamma = loader.getTensor(prefix + ".weight");
  bn.beta  = loader.getTensor(prefix + ".bias");
  bn.mean  = loader.getTensor(prefix + ".running_mean");
  bn.var   = loader.getTensor(prefix + ".running_var");
  bn.eps   = 1e-5f;
  return bn;
}

bool WeSpeakerResNet34::init(const WeightLoader &loader) {
  conv1_w_ = loader.getTensor("resnet.conv1.weight");
  bn1_     = extract_bn(loader, "resnet.bn1");

  auto load_stage = [&](const std::string &stage_name, size_t num_blocks,
                        size_t in_c, size_t out_c, size_t init_stride) {
    std::vector<BasicBlockWeights> blocks(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b) {
      std::string pfx = "resnet." + stage_name + "." + std::to_string(b);
      blocks[b].conv1_w = loader.getTensor(pfx + ".conv1.weight");
      blocks[b].bn1     = extract_bn(loader, pfx + ".bn1");
      blocks[b].conv2_w = loader.getTensor(pfx + ".conv2.weight");
      blocks[b].bn2     = extract_bn(loader, pfx + ".bn2");

      if (b == 0) {
        blocks[b].in_channels = in_c;
        blocks[b].out_channels = out_c;
        blocks[b].stride = init_stride;
        blocks[b].downsample = (in_c != out_c || init_stride != 1);
        if (blocks[b].downsample) {
          blocks[b].downsample_conv_w = loader.getTensor(pfx + ".shortcut.0.weight");
          blocks[b].downsample_bn     = extract_bn(loader, pfx + ".shortcut.1");
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
  seg_1_b_ = loader.getTensor("resnet.seg_1.bias");

  return (conv1_w_ && seg_1_w_ && seg_1_b_);
}

void WeSpeakerResNet34::forwardBackbone(const float *fbank, float *out_feat,
                                       size_t chunk_idx, DebugTensorCallback debug_cb) {
  std::string pfx = "emb_chunk" + std::to_string(chunk_idx) + "_";

  // 1. conv1: [1, 1, 80, 998] -> [1, 32, 80, 998]
  std::vector<float> conv1_out(32 * 80 * 998);
  runConv2d(fbank, conv1_out.data(), 1, 32, 80, 998, 3, 3, 1, 1, conv1_w_);
  if (debug_cb) debug_cb(pfx + "resnet_conv1", conv1_out.data(), {1, 32, 80, 998});

  // 2. bn1 (no relu for debug hook parity): [1, 32, 80, 998]
  std::vector<float> bn1_out(32 * 80 * 998);
  runBatchNorm2d(conv1_out.data(), bn1_out.data(), 32, 80, 998, bn1_, false);
  if (debug_cb) debug_cb(pfx + "resnet_bn1", bn1_out.data(), {1, 32, 80, 998});

  for (float &val : bn1_out) val = relu(val);

  // 3. layer1: 3 blocks (32 channels, 80 x 998)
  std::vector<float> curr_feat = std::move(bn1_out);
  size_t curr_h = 80;
  size_t curr_w = 998;

  for (size_t b = 0; b < layer1_.size(); ++b) {
    size_t next_h = 0, next_w = 0;
    std::vector<float> next_feat(layer1_[b].out_channels * curr_h * curr_w);
    runBasicBlock(curr_feat.data(), next_feat.data(), curr_h, curr_w, layer1_[b], next_h, next_w);
    curr_feat = std::move(next_feat);
    curr_h = next_h;
    curr_w = next_w;
  }
  if (debug_cb) debug_cb(pfx + "resnet_layer1", curr_feat.data(), {1, 32, curr_h, curr_w});

  // 4. layer2: 4 blocks (64 channels, 40 x 499)
  for (size_t b = 0; b < layer2_.size(); ++b) {
    size_t next_h = 0, next_w = 0;
    size_t target_h = (b == 0) ? (curr_h + 2 - 3) / 2 + 1 : curr_h;
    size_t target_w = (b == 0) ? (curr_w + 2 - 3) / 2 + 1 : curr_w;
    std::vector<float> next_feat(layer2_[b].out_channels * target_h * target_w);
    runBasicBlock(curr_feat.data(), next_feat.data(), curr_h, curr_w, layer2_[b], next_h, next_w);
    curr_feat = std::move(next_feat);
    curr_h = next_h;
    curr_w = next_w;
  }
  if (debug_cb) debug_cb(pfx + "resnet_layer2", curr_feat.data(), {1, 64, curr_h, curr_w});

  // 5. layer3: 6 blocks (128 channels, 20 x 250)
  for (size_t b = 0; b < layer3_.size(); ++b) {
    size_t next_h = 0, next_w = 0;
    size_t target_h = (b == 0) ? (curr_h + 2 - 3) / 2 + 1 : curr_h;
    size_t target_w = (b == 0) ? (curr_w + 2 - 3) / 2 + 1 : curr_w;
    std::vector<float> next_feat(layer3_[b].out_channels * target_h * target_w);
    runBasicBlock(curr_feat.data(), next_feat.data(), curr_h, curr_w, layer3_[b], next_h, next_w);
    curr_feat = std::move(next_feat);
    curr_h = next_h;
    curr_w = next_w;
  }
  if (debug_cb) debug_cb(pfx + "resnet_layer3", curr_feat.data(), {1, 128, curr_h, curr_w});

  // 6. layer4: 3 blocks (256 channels, 10 x 125)
  for (size_t b = 0; b < layer4_.size(); ++b) {
    size_t next_h = 0, next_w = 0;
    size_t target_h = (b == 0) ? (curr_h + 2 - 3) / 2 + 1 : curr_h;
    size_t target_w = (b == 0) ? (curr_w + 2 - 3) / 2 + 1 : curr_w;
    std::vector<float> next_feat(layer4_[b].out_channels * target_h * target_w);
    runBasicBlock(curr_feat.data(), next_feat.data(), curr_h, curr_w, layer4_[b], next_h, next_w);
    curr_feat = std::move(next_feat);
    curr_h = next_h;
    curr_w = next_w;
  }
  if (debug_cb) debug_cb(pfx + "resnet_layer4", curr_feat.data(), {1, 256, curr_h, curr_w});

  std::memcpy(out_feat, curr_feat.data(), 256 * 10 * 125 * sizeof(float));
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
    const float *feat_d = feat + d * NUM_FRAMES;

    // Weighted mean
    float sum_feat = 0.0f;
    for (size_t t = 0; t < NUM_FRAMES; ++t) {
      sum_feat += feat_d[t] * w[t];
    }
    float mean_val = sum_feat / v1;
    stats[d] = mean_val;

    // Weighted variance & std
    float var_sum = 0.0f;
    for (size_t t = 0; t < NUM_FRAMES; ++t) {
      float diff = feat_d[t] - mean_val;
      var_sum += (diff * diff) * w[t];
    }
    float std_val = std::sqrt(std::max(0.0f, var_sum / var_denom));
    stats[FEAT_DIM + d] = std_val;
  }
  if (debug_cb) debug_cb(pfx + "resnet_pool_tstp", stats.data(), {1, 5120});

  // seg_1: Linear(5120 -> 256)
  std::vector<float> embedding(256, 0.0f);
  for (size_t oc = 0; oc < 256; ++oc) {
    const float *w_row = seg_1_w_ + oc * 5120;
    float acc = seg_1_b_[oc];
    for (size_t ic = 0; ic < 5120; ++ic) {
      acc += w_row[ic] * stats[ic];
    }
    embedding[oc] = acc;
  }
  if (debug_cb) debug_cb(pfx + "resnet_seg_1", embedding.data(), {1, 256});

  return embedding;
}

std::vector<float> WeSpeakerResNet34::forward(const float *fbank, const float *mask,
                                             size_t step_idx, DebugTensorCallback debug_cb) {
  std::vector<float> feat(256 * 10 * 125);
  forwardBackbone(fbank, feat.data(), step_idx, debug_cb);
  return forwardPool(feat.data(), mask, step_idx, 0, debug_cb);
}

} // namespace speaker_diarization
