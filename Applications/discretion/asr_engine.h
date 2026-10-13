// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   asr_engine.h
 * @date   12 October 2026
 * @brief  Resident Qwen3-ASR model that transcribes fixed-length PCM segments
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Only asr_lane.cpp and asr_engine.cpp include this header (and with it the
 * quick_ai model headers).
 */

#ifndef __DISCRETION_ASR_ENGINE_H__
#define __DISCRETION_ASR_ENGINE_H__

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "asr_lane.h"

namespace quick_ai {
class Qwen3ASRCausalLM;
}

namespace discretion {

class AsrEngine {
public:
  AsrEngine();
  ~AsrEngine();

  /**
   * @brief read the model configuration, build the graph for one segment of
   *        `opt.segment_sec` seconds and load the weights
   */
  bool init(const AsrOptions &opt, std::string &error);

  /** @brief samples of one segment: the model is built for exactly this length
   */
  size_t segmentSamples() const;

  /**
   * @brief transcribe one segment
   * @param pcm 16 kHz mono float samples, exactly segmentSamples() long
   */
  std::string transcribe(std::vector<float> pcm);

  /** @brief words of the last transcribe(), time from the segment start */
  std::vector<std::pair<std::string, double>> lastWords() const;

private:
  std::unique_ptr<quick_ai::Qwen3ASRCausalLM> model_;
  std::string prompt_;
  size_t segment_samples_ = 0;
};

} // namespace discretion

#endif // __DISCRETION_ASR_ENGINE_H__
