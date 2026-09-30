// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   pyannet.h
 * @date   29 September 2026
 * @brief  PyanNet Segmentation Model for Speaker Diarization in NNTrainer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __PYANNET_H__
#define __PYANNET_H__

#include <vector>
#include <string>
#include <functional>
#include "weight_loader.h"

namespace speaker_diarization {

using DebugTensorCallback = std::function<void(const std::string &name, const float *data, const std::vector<size_t> &shape)>;

class PyanNet {
public:
  PyanNet() = default;
  ~PyanNet() = default;

  /**
   * @brief Initialize model and bind weights.
   * @param loader WeightLoader containing loaded weights.
   * @return true if all weights found and loaded.
   */
  bool init(const WeightLoader &loader);

  /**
   * @brief Forward pass of PyanNet on a 10.0-second audio chunk (160,000 samples).
   * @param chunk 160,000 float samples.
   * @param debug_cb Optional callback to report intermediate layer outputs.
   * @return Multi-label speaker probabilities/activations of shape [589, 3].
   */
  std::vector<float> forwardChunk(const float *chunk, size_t chunk_idx = 0, DebugTensorCallback debug_cb = nullptr);

  /**
   * @brief Run segmentation over all audio chunks and compute speaker counting and masks.
   * @param chunks Vector of 160,000-sample chunks.
   * @param out_segmentations Output multi-label segmentations [num_chunks, 589, 3].
   * @param out_speaker_counting Output frame-level active speaker count [total_frames, 1].
   * @param debug_cb Optional debug callback.
   */
  void runInference(const std::vector<std::vector<float>> &chunks,
                    std::vector<std::vector<float>> &out_segmentations,
                    std::vector<uint8_t> &out_speaker_counting,
                    DebugTensorCallback debug_cb = nullptr);

private:
  // SincNet weights
  const float *wav_norm1d_w_ = nullptr;
  const float *wav_norm1d_b_ = nullptr;
  const float *conv1d_0_w_ = nullptr;   // [80, 1, 251]
  const float *norm1d_0_w_ = nullptr;   // [80]
  const float *norm1d_0_b_ = nullptr;   // [80]

  const float *conv1d_1_w_ = nullptr;   // [60, 80, 5]
  const float *conv1d_1_b_ = nullptr;   // [60]
  const float *norm1d_1_w_ = nullptr;   // [60]
  const float *norm1d_1_b_ = nullptr;   // [60]

  const float *conv1d_2_w_ = nullptr;   // [60, 60, 5]
  const float *conv1d_2_b_ = nullptr;   // [60]
  const float *norm1d_2_w_ = nullptr;   // [60]
  const float *norm1d_2_b_ = nullptr;   // [60]

  // LSTM weights: 4 layers bidirectional
  struct LSTMLayerWeights {
    const float *w_ih;        // [512, in_dim]
    const float *w_hh;        // [512, 128]
    const float *b_ih;        // [512]
    const float *b_hh;        // [512]
    const float *w_ih_rev;    // [512, in_dim]
    const float *w_hh_rev;    // [512, 128]
    const float *b_ih_rev;    // [512]
    const float *b_hh_rev;    // [512]
    size_t in_dim;
  };
  std::vector<LSTMLayerWeights> lstm_layers_;

  // Linear & Classifier weights
  const float *linear_0_w_ = nullptr;   // [128, 256]
  const float *linear_0_b_ = nullptr;   // [128]
  const float *linear_1_w_ = nullptr;   // [128, 128]
  const float *linear_1_b_ = nullptr;   // [128]
  const float *classifier_w_ = nullptr; // [7, 128]
  const float *classifier_b_ = nullptr; // [7]

  void runLSTMLayer(const float *input, size_t seq_len, const LSTMLayerWeights &weights, float *output);
};

} // namespace speaker_diarization

#endif // __PYANNET_H__
