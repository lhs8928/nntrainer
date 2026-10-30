// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   silero_vad.h
 * @date   30 October 2026
 * @brief  Silero VAD v6 (16 kHz) without ONNX Runtime
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The speech probability of each 512-sample block, as the ONNX graph gives it
 * (inputs input[1, 576] = 64 samples of the previous block + the block, h, c):
 * reflect padding of 128, an STFT as 258 fixed filters (kernel 256, stride
 * 128, the first frame dropped), magnitudes of 129 bins, four Conv1d + ReLU
 * (strides 1 2 2 1), one LSTM step of 128 with the state carried to the next
 * block, ReLU, Conv1d 128->1, sigmoid. Weights come from
 * export_silero_vad.py.
 */

#ifndef __SILERO_VAD_H__
#define __SILERO_VAD_H__

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace speaker_diarization {

/** @brief Silero VAD v6 on a stream of 512-sample blocks */
class SileroVad {
public:
  static constexpr size_t BLOCK = 512;  /**< samples per call */
  static constexpr size_t CONTEXT = 64; /**< samples kept from the last block */
  static constexpr size_t HIDDEN = 128; /**< LSTM state */

  /**
   * @brief load the weights written by export_silero_vad.py
   * @param error why it failed (when false is returned)
   */
  bool load(const std::string &path, std::string *error = nullptr);

  /**
   * @brief speech probability of one block of BLOCK samples in [-1, 1];
   *        the context and the LSTM state move on to the next call
   */
  float run(const float *block);

  /** @brief forget the context and the state (a new stream) */
  void reset();

  /** @brief the model manifest JSON stored in the file ("" if none) */
  const std::string &manifest() const { return manifest_; }

private:
  struct Conv {
    std::vector<float> w, b; /**< w[out][in][3] */
    size_t in = 0, out = 0, stride = 1;
  };

  std::vector<float> basis_; /**< [258][256] */
  std::array<Conv, 4> convs_;
  std::vector<float> lstm_w_, lstm_r_; /**< [512][128], gates i o f c */
  std::vector<float> lstm_b_;          /**< [512], both ONNX biases summed */
  std::vector<float> out_w_;           /**< [128] */
  float out_b_ = 0.0f;
  std::string manifest_;

  std::array<float, HIDDEN> h_{}, c_{};
  std::array<float, CONTEXT> context_{};
};

} // namespace speaker_diarization

#endif // __SILERO_VAD_H__
