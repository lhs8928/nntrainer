// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   wespeaker_resnet34.h
 * @date   29 September 2026
 * @brief  WeSpeaker ResNet34 Speaker Embedding Model in NNTrainer
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

struct BatchNormParams {
  const float *gamma = nullptr;
  const float *beta = nullptr;
  const float *mean = nullptr;
  const float *var = nullptr;
  float eps = 1e-5f;
};

struct BasicBlockWeights {
  const float *conv1_w = nullptr;
  BatchNormParams bn1;
  const float *conv2_w = nullptr;
  BatchNormParams bn2;
  bool downsample = false;
  const float *downsample_conv_w = nullptr;
  BatchNormParams downsample_bn;
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
   * @param step_idx Optional step index for debug output.
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
   * @param fbank Flat array of size 80 * 998 (mel_bins * frames).
   * @param mask Flat array of size 589 (speaker activity weights from segmentation).
   * @param step_idx Optional step index for debug output.
   * @param debug_cb Optional callback to inspect intermediate tensors.
   * @return 256-dimensional speaker embedding vector.
   */
  std::vector<float> forward(const float *fbank, const float *mask,
                             size_t step_idx = 0, DebugTensorCallback debug_cb = nullptr);

private:
  // conv1: [32, 1, 3, 3]
  const float *conv1_w_ = nullptr;
  BatchNormParams bn1_;

  // 4 stages of BasicBlocks:
  // stage 1: 3 blocks (32 channels, stride 1)
  // stage 2: 4 blocks (64 channels, initial stride 2)
  // stage 3: 6 blocks (128 channels, initial stride 2)
  // stage 4: 3 blocks (256 channels, initial stride 2)
  std::vector<BasicBlockWeights> layer1_;
  std::vector<BasicBlockWeights> layer2_;
  std::vector<BasicBlockWeights> layer3_;
  std::vector<BasicBlockWeights> layer4_;

  // seg_1: Linear(5120 -> 256)
  const float *seg_1_w_ = nullptr; // [256, 5120]
  const float *seg_1_b_ = nullptr; // [256]

  void runConv2d(const float *in, float *out, size_t in_c, size_t out_c,
                 size_t in_h, size_t in_w, size_t k_h, size_t k_w,
                 size_t stride, size_t pad, const float *weights);

  void runBatchNorm2d(const float *in, float *out, size_t channels,
                      size_t height, size_t width, const BatchNormParams &bn, bool relu = true);

  void runBasicBlock(const float *in, float *out, size_t in_h, size_t in_w,
                     const BasicBlockWeights &block, size_t &out_h, size_t &out_w);
};

} // namespace speaker_diarization

#endif // __WESPEAKER_RESNET34_H__
