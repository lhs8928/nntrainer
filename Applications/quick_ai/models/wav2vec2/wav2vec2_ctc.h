// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   wav2vec2_ctc.h
 * @date   04 November 2026
 * @brief  wav2vec 2.0 CTC model (Wav2Vec2ForCTC) for WhisperX alignment
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The model WhisperX aligns transcripts with: waveform -> conv feature
 * encoder -> feature projection -> positional conv -> transformer encoder ->
 * CTC head, log-softmax over the characters of vocab.json per 20 ms frame.
 * Both encoder layouts of HuggingFace's Wav2Vec2 are built from its
 * config.json: do_stable_layer_norm (pre-LN, XLS-R and the large models such
 * as kresnik/wav2vec2-large-xlsr-korean) or post-LN (base models), with
 * feat_extract_norm "layer" or "group".
 *
 * The graph is built for the longest segment (nntr_config.json
 * "max_seconds", 30); a shorter one runs as an incremental step over its own
 * frames only, so neither the attention nor the positional convolution sees
 * padding (the reference gets the segment alone, without a mask).
 *
 * The model directory holds config.json and vocab.json of the checkpoint,
 * nntr_config.json (res/wav2vec2) and the converted weights.
 */

#ifndef __WAV2VEC2_CTC_H__
#define __WAV2VEC2_CTC_H__

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <model.h>
#include <tensor_api.h>

#include "ctc_align.h"

namespace quick_ai {

/**
 * @class Wav2Vec2CTC
 * @brief character emissions of a waveform and forced alignment on them
 */
class Wav2Vec2CTC {
public:
  Wav2Vec2CTC();
  ~Wav2Vec2CTC();

  /** @brief build the graph and load the weights of `model_dir` */
  void load(const std::string &model_dir);

  /**
   * @brief log-softmax emissions of 16 kHz mono samples
   * @param[out] frames the number of 20 ms frames
   * @return frames x vocabSize() log probabilities
   */
  std::vector<float> emissions(const std::vector<float> &pcm, size_t &frames);

  /**
   * @brief WhisperX alignment of `text` to the samples (one segment)
   * @param language code such as "ko" (ja, zh: per character)
   * @return words with times from the start of `pcm`; empty if it fails
   */
  std::vector<CtcWord> align(const std::vector<float> &pcm,
                             const std::string &text,
                             const std::string &language);

  size_t vocabSize() const { return vocab_; }
  size_t maxSamples() const { return max_samples_; }
  const std::map<std::string, int> &dictionary() const { return dict_; }

private:
  std::unique_ptr<ml::train::Model> model_;
  ml::train::Tensor input_, output_;
  std::map<std::string, int> dict_;
  std::vector<int> kernel_, stride_;
  size_t vocab_ = 0;
  size_t max_samples_ = 0;
  size_t max_frames_ = 0;
};

} // namespace quick_ai

#endif // __WAV2VEC2_CTC_H__
