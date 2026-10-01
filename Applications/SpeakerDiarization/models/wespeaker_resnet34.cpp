// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   wespeaker_resnet34.cpp
 * @date   29 September 2026
 * @brief  WeSpeaker ResNet34 Speaker Embedding Model in NNTrainer (Fused Conv+BN NHWC)
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

// NHWC im2col: gathers input patches into Col[out_h * out_w, k_h * k_w * in_c]
static inline void im2col_nhwc(const float *data_im, size_t in_c,
                              size_t in_h, size_t in_w, size_t k_h, size_t k_w,
                              size_t pad, size_t stride, float *data_col) {
  size_t out_h = (in_h + 2 * pad - k_h) / stride + 1;
  size_t out_w = (in_w + 2 * pad - k_w) / stride + 1;
  size_t K = k_h * k_w * in_c;

  // Fast path for standard 3x3 stride-1 pad-1 convolution: merges 3 horizontal taps into a single contiguous memcpy
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
                                 size_t stride, size_t pad, const float *weights, const float *bias,
                                 bool apply_relu) {
  size_t out_h = (in_h + 2 * pad - k_h) / stride + 1;
  size_t out_w = (in_w + 2 * pad - k_w) / stride + 1;
  size_t M = out_h * out_w;
  size_t N = out_c;
  size_t K = in_c * k_h * k_w;

  if (k_h == 1 && k_w == 1 && stride == 1 && pad == 0) {
    RUN_SGEMM(M, N, K, in, weights, out);
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
    RUN_SGEMM(M, N, K, subsampled.data(), weights, out);
  } else {
    thread_local static std::vector<float> col_data;
    if (col_data.size() < M * K) {
      col_data.resize(M * K);
    }
    im2col_nhwc(in, in_c, in_h, in_w, k_h, k_w, pad, stride, col_data.data());
    RUN_SGEMM(M, N, K, col_data.data(), weights, out);
  }

  // Fused Bias addition and optional ReLU in a single pass in L1 cache
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

void WeSpeakerResNet34::runBasicBlock(const float *in, float *out, size_t in_h, size_t in_w,
                                     const BasicBlockWeights &block, size_t &out_h, size_t &out_w) {
  out_h = (in_h + 2 * 1 - 3) / block.stride + 1;
  out_w = (in_w + 2 * 1 - 3) / block.stride + 1;
  size_t total_out = block.out_channels * out_h * out_w;

  thread_local static std::vector<float> conv1_out;
  thread_local static std::vector<float> ds_out;

  if (conv1_out.size() < total_out) conv1_out.resize(total_out);

  // 1. conv1 + bias1 + relu (Fused in a single pass!)
  runConv2d(in, conv1_out.data(), block.in_channels, block.out_channels,
            in_h, in_w, 3, 3, block.stride, 1, block.conv1_w, block.conv1_b, /*apply_relu=*/true);

  // 2. conv2 + bias2 (Fused, no relu before residual addition)
  runConv2d(conv1_out.data(), out, block.out_channels, block.out_channels,
            out_h, out_w, 3, 3, 1, 1, block.conv2_w, block.conv2_b, /*apply_relu=*/false);

  // 3. Shortcut / Residual addition + final ReLU
  if (block.downsample) {
    if (ds_out.size() < total_out) ds_out.resize(total_out);
    runConv2d(in, ds_out.data(), block.in_channels, block.out_channels,
              in_h, in_w, 1, 1, block.stride, 0, block.downsample_conv_w, block.downsample_conv_b, /*apply_relu=*/false);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total_out; ++i) {
      float val = out[i] + ds_out[i];
      out[i] = (val > 0.0f) ? val : 0.0f;
    }
  } else {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total_out; ++i) {
      float val = out[i] + in[i];
      out[i] = (val > 0.0f) ? val : 0.0f;
    }
  }
}

bool WeSpeakerResNet34::init(const WeightLoader &loader) {
  conv1_w_ = loader.getTensor("resnet.conv1.weight");
  conv1_b_ = loader.getTensor("resnet.conv1.bias");

  auto load_stage = [&](const std::string &stage_name, size_t num_blocks,
                        size_t in_c, size_t out_c, size_t init_stride) {
    std::vector<BasicBlockWeights> blocks(num_blocks);
    for (size_t b = 0; b < num_blocks; ++b) {
      std::string pfx = "resnet." + stage_name + "." + std::to_string(b);
      blocks[b].conv1_w = loader.getTensor(pfx + ".conv1.weight");
      blocks[b].conv1_b = loader.getTensor(pfx + ".conv1.bias");
      blocks[b].conv2_w = loader.getTensor(pfx + ".conv2.weight");
      blocks[b].conv2_b = loader.getTensor(pfx + ".conv2.bias");

      if (b == 0) {
        blocks[b].in_channels = in_c;
        blocks[b].out_channels = out_c;
        blocks[b].stride = init_stride;
        blocks[b].downsample = (in_c != out_c || init_stride != 1);
        if (blocks[b].downsample) {
          blocks[b].downsample_conv_w = loader.getTensor(pfx + ".shortcut.0.weight");
          blocks[b].downsample_conv_b = loader.getTensor(pfx + ".shortcut.0.bias");
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

  return (conv1_w_ && conv1_b_ && seg_1_w_ && seg_1_b_);
}

void WeSpeakerResNet34::forwardBackbone(const float *fbank, float *out_feat,
                                       size_t chunk_idx, DebugTensorCallback debug_cb) {
  std::string pfx = "emb_chunk" + std::to_string(chunk_idx) + "_";

  // 1. Stem Conv1 + Bias1 + ReLU (Fused): [80, 998, 1] -> [80, 998, 32]
  std::vector<float> curr_feat(32 * 80 * 998);
  runConv2d(fbank, curr_feat.data(), 1, 32, 80, 998, 3, 3, 1, 1, conv1_w_, conv1_b_, /*apply_relu=*/true);
  if (debug_cb) debug_cb(pfx + "resnet_conv1", curr_feat.data(), {1, 80, 998, 32});

  size_t curr_h = 80;
  size_t curr_w = 998;

  // 2. layer1: 3 blocks (32 channels, 80 x 998)
  for (size_t b = 0; b < layer1_.size(); ++b) {
    size_t next_h = 0, next_w = 0;
    std::vector<float> next_feat(layer1_[b].out_channels * curr_h * curr_w);
    runBasicBlock(curr_feat.data(), next_feat.data(), curr_h, curr_w, layer1_[b], next_h, next_w);
    curr_feat = std::move(next_feat);
  }
  if (debug_cb) debug_cb(pfx + "resnet_layer1", curr_feat.data(), {1, curr_h, curr_w, 32});

  // 3. layer2: 4 blocks (64 channels, 40 x 499)
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
  if (debug_cb) debug_cb(pfx + "resnet_layer2", curr_feat.data(), {1, curr_h, curr_w, 64});

  // 4. layer3: 6 blocks (128 channels, 20 x 250)
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
  if (debug_cb) debug_cb(pfx + "resnet_layer3", curr_feat.data(), {1, curr_h, curr_w, 128});

  // 5. layer4: 3 blocks (256 channels, 10 x 125)
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
  if (debug_cb) debug_cb(pfx + "resnet_layer4", curr_feat.data(), {1, curr_h, curr_w, 256});

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
  if (debug_cb) debug_cb(pfx + "spk_embedding", embedding.data(), {1, 256});

  return embedding;
}

} // namespace speaker_diarization
