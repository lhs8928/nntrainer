// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   asr_engine.h
 * @date   12 October 2026
 * @brief  Resident ASR model (Qwen3-ASR or Whisper) that transcribes segments
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Only asr_lane.cpp and asr_engine.cpp include this header (and with it the
 * quick_ai model headers).
 *
 * The model directory's config.json picks the model: architectures
 * "WhisperForConditionalGeneration" runs Whisper (docs/whisperx-asr.ko.md),
 * anything else Qwen3-ASR.
 */

#ifndef __DISCRETION_ASR_ENGINE_H__
#define __DISCRETION_ASR_ENGINE_H__

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "asr_lane.h"

#include "wav2vec2_ctc.h"

namespace quick_ai {
class Qwen3ASRCausalLM;
class WhisperCausalLM;
} // namespace quick_ai

namespace discretion {

/** @brief a wav2vec2 CTC model for WhisperX alignment, shared by engines */
struct AsrAligner {
  quick_ai::Wav2Vec2CTC model;
  std::mutex mutex;
};

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
   *         (Whisper: the longest segment, at most 30 s; any length works)
   */
  size_t segmentSamples() const;

  /** @brief true when one engine takes every length (Whisper's 30 s window) */
  bool anyLength() const { return whisper_ != nullptr; }

  /** @brief "nntrainer:qwen3-asr" or "nntrainer:whisper" */
  std::string provider() const;

  /**
   * @brief transcribe one segment
   * @param pcm 16 kHz mono float samples, exactly segmentSamples() long
   */
  std::string transcribe(std::vector<float> pcm);

  /**
   * @brief words of the last transcribe(), time from the segment start: by
   *        WhisperX forced alignment when AsrOptions::align_model_dir is set
   *        (and the segment fits it), else the model's own (Qwen3-ASR's
   *        attention peaks, Whisper's cross-attention DTW)
   */
  std::vector<std::pair<std::string, double>> lastWords() const;

private:
  bool initWhisper(const AsrOptions &opt, std::string &error);

  std::unique_ptr<quick_ai::Qwen3ASRCausalLM> model_;
  std::string prompt_;
  size_t segment_samples_ = 0;

  std::unique_ptr<quick_ai::WhisperCausalLM> whisper_;
  std::string whisper_language_; // code such as "ko", "" = detect
  bool word_timestamps_ = true;

  std::shared_ptr<AsrAligner> aligner_;
  std::string align_language_; // code of the forced language (Qwen3-ASR)
  std::vector<std::pair<std::string, double>> last_words_;
};

} // namespace discretion

#endif // __DISCRETION_ASR_ENGINE_H__
