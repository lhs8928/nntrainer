// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   audio_vad.h
 * @brief  Lightweight zero-dependency adaptive Voice Activity Detection (VAD)
 *         and speech-boundary segmenter for Qwen3-ASR
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 */
#ifndef __QWEN3_ASR_AUDIO_VAD_H__
#define __QWEN3_ASR_AUDIO_VAD_H__

#include <vector>
#include <cstddef>
#include <string>

namespace quick_ai {

/**
 * @brief Speech segment produced by VAD
 */
struct VadSegment {
  size_t start_sample;
  size_t end_sample;
  double start_sec;
  double end_sec;
  std::vector<float> pcm;
};

/**
 * @brief VAD configuration parameters
 */
struct VadConfig {
  unsigned int sample_rate = 16000;
  unsigned int frame_ms = 20;           // 20ms frame = 320 samples @ 16kHz
  float min_silence_rms = 0.005f;       // Minimum absolute silence floor
  float snr_factor = 0.12f;             // Relative energy threshold between noise floor and peak
  float min_speech_ms = 250.0f;         // Minimum speech duration to keep
  float min_silence_gap_ms = 400.0f;    // Silence gap to split utterances
  float pad_onset_ms = 150.0f;          // Padding before speech starts
  float pad_offset_ms = 200.0f;         // Padding after speech ends
  float max_segment_sec = 15.0f;        // Maximum segment length before forcing a cut
  float min_segment_sec = 2.0f;         // Merge short adjacent segments up to this
};

/**
 * @brief Pure C++ Adaptive Energy & Zero-Crossing Rate (ZCR) based Voice Activity Detector
 */
class AudioVAD {
public:
  using Config = VadConfig;

  /**
   * @brief Segment a continuous 16kHz PCM audio stream into natural speech segments.
   *        Silent regions are completely skipped.
   * @param pcm Input 16kHz mono audio samples.
   * @param cfg VAD configuration parameters.
   * @return std::vector<VadSegment> Detected speech segments.
   */
  static std::vector<VadSegment> segment(const std::vector<float> &pcm,
                                         const VadConfig &cfg = VadConfig());
};

} // namespace quick_ai

#endif // __QWEN3_ASR_AUDIO_VAD_H__
