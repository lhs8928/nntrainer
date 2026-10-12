// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   wespeaker_resnet34.h
 * @date   29 September 2026
 * @brief  WeSpeaker ResNet34 Speaker Embedding Model in NNTrainer (Native W8A8 & W4 Quantized)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __WESPEAKER_RESNET34_H__
#define __WESPEAKER_RESNET34_H__

#include <vector>
#include <string>
#include <functional>
#include "weight_loader.h"

namespace speaker_diarization {

using DebugTensorCallback = std::function<void(const std::string &name, const float *data, const std::vector<size_t> &shape)>;

struct BasicBlockWeights {
  const float *conv1_w = nullptr;
  const int8_t *conv1_w_i8 = nullptr;
  std::vector<int8_t> conv1_w_i8_trans;
  const float *conv1_s = nullptr;
  const float *conv1_b = nullptr;
  float conv1_act_scale = 1.0f;

  const float *conv2_w = nullptr;
  const int8_t *conv2_w_i8 = nullptr;
  std::vector<int8_t> conv2_w_i8_trans;
  const float *conv2_s = nullptr;
  const float *conv2_b = nullptr;
  float conv2_act_scale = 1.0f;

  bool downsample = false;
  const float *downsample_conv_w = nullptr;
  const int8_t *downsample_conv_w_i8 = nullptr;
  std::vector<int8_t> downsample_w_i8_trans;
  const float *downsample_conv_s = nullptr;
  const float *downsample_conv_b = nullptr;
  float downsample_act_scale = 1.0f;

  size_t in_channels = 0;
  size_t out_channels = 0;
  size_t stride = 1;
};

class WeSpeakerResNet34 {
public:
  WeSpeakerResNet34() = default;
  ~WeSpeakerResNet34() = default;

  /**
   * @brief Initialize model weights from WeightLoader.
   */
  bool init(const WeightLoader &loader);

  /**
   * @brief Run convolutional backbone on Fbank feature map once per chunk.
   * @param fbank Flat array of size 80 * 998 (mel_bins * frames).
   * @param out_feat Output buffer of size 256 * 10 * 125.
   * @param chunk_idx Optional chunk index for debug output.
   * @param debug_cb Optional callback to inspect intermediate tensors.
   */
  void forwardBackbone(const float *fbank, float *out_feat,
                       size_t chunk_idx = 0, DebugTensorCallback debug_cb = nullptr);

  /**
   * @brief Pool feature map with speaker activity mask and project to 256-dim embedding.
   * @param feat Feature map of size 256 * 10 * 125.
   * @param mask Speaker mask of size 589.
   * @param chunk_idx Optional chunk index.
   * @param spk_idx Optional speaker channel index.
   * @param debug_cb Optional callback.
   * @return 256-dimensional speaker embedding.
   */
  std::vector<float> forwardPool(const float *feat, const float *mask,
                                 size_t chunk_idx = 0, size_t spk_idx = 0,
                                 DebugTensorCallback debug_cb = nullptr);

  /**
   * @brief Forward pass to compute 256-dim speaker embedding from Fbank and speaker mask.
   */
  std::vector<float> forward(const float *fbank, const float *mask,
                             size_t step_idx = 0, DebugTensorCallback debug_cb = nullptr);

private:
  // Stem Conv1: [3, 3, 1, 32]
  const float *conv1_w_ = nullptr;
  const int8_t *conv1_w_i8_ = nullptr;
  std::vector<int8_t> conv1_w_i8_trans_;
  const float *conv1_s_ = nullptr;
  const float *conv1_b_ = nullptr;
  float conv1_act_scale_ = 1.0f;

  // 4 stages of BasicBlocks:
  std::vector<BasicBlockWeights> layer1_;
  std::vector<BasicBlockWeights> layer2_;
  std::vector<BasicBlockWeights> layer3_;
  std::vector<BasicBlockWeights> layer4_;

  // seg_1: Linear(5120 -> 256, W4 quantized)
  const float *seg_1_w_ = nullptr; // [256, 5120]
  const int8_t *seg_1_w_i4_ = nullptr; // [256, 5120] in [-8, 7]
  const float *seg_1_s_ = nullptr; // [256]
  const float *seg_1_b_ = nullptr; // [256]
  float seg_1_act_scale_ = 1.0f;

  void runConv2d(const float *in, float *out, size_t in_c, size_t out_c,
                 size_t in_h, size_t in_w, size_t k_h, size_t k_w,
                 size_t stride, size_t pad,
                 const float *weights_fp32, const int8_t *weights_i8_trans,
                 const float *scales, const float *bias, float act_scale,
                 bool apply_relu = false);

  void runBasicBlock(const float *in, float *out, size_t in_h, size_t in_w,
                     const BasicBlockWeights &block, size_t &out_h, size_t &out_w);
};

} // namespace speaker_diarization

#endif // __WESPEAKER_RESNET34_H__
