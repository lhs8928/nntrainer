// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   audio_stream.h
 * @brief  Live audio input and segmentation for Qwen3-ASR segment/streaming
 *         mode: PCM sources (real-time WAV replay, microphone through
 *         `arecord`) and a pause-aware segmenter.
 */
#ifndef __QWEN3_ASR_AUDIO_STREAM_H__
#define __QWEN3_ASR_AUDIO_STREAM_H__

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace quick_ai {

/**
 * @brief Blocking source of 16 kHz mono float PCM in [-1, 1].
 */
class PcmSource {
public:
  virtual ~PcmSource() = default;
  /**
   * @brief read up to max_samples samples, blocking until some are available
   * @return number of samples read, 0 at end of stream
   */
  virtual size_t read(float *dst, size_t max_samples) = 0;
};

/**
 * @brief Replays an in-memory recording at real-time speed (1x), so that
 *        timing/latency tests behave like a live microphone.
 */
class WavRealtimeSource : public PcmSource {
public:
  WavRealtimeSource(std::vector<float> pcm, double speed = 1.0);
  size_t read(float *dst, size_t max_samples) override;

private:
  std::vector<float> pcm_;
  size_t pos_ = 0;
  double speed_;
  std::chrono::steady_clock::time_point t0_;
  bool started_ = false;
};

/**
 * @brief Reads raw signed 16-bit little-endian 16 kHz mono PCM from a shell
 *        command's stdout (e.g. "arecord -q -D plughw:2,0 -f S16_LE -r 16000
 *        -c 1 -t raw"), or from stdin when the command is empty.
 */
class PipeSource : public PcmSource {
public:
  explicit PipeSource(const std::string &cmd);
  ~PipeSource() override;
  size_t read(float *dst, size_t max_samples) override;

private:
  FILE *fp_ = nullptr;
  bool is_pipe_ = false;
};

/**
 * @brief Cuts a continuous sample stream into segments of at most max_len
 *        samples. When the buffer reaches max_len the cut is placed at the
 *        quietest 100 ms window inside [min_len, max_len] so that words are
 *        (mostly) not split. Never drops samples.
 */
class Segmenter {
public:
  Segmenter(size_t min_len, size_t max_len, float silence_rms = 0.005f, bool use_vad = false) :
    min_len_(min_len), max_len_(max_len), silence_rms_(silence_rms), use_vad_(use_vad) {}

  /** @brief append samples */
  void push(const float *x, size_t n);

  /**
   * @brief pop the next complete segment if available
   * @param[out] seg segment samples
   * @param[out] start_sample index of seg[0] in the whole stream
   */
  bool pop(std::vector<float> &seg, size_t &start_sample);

  /** @brief pop the remaining tail (any length), for end of stream */
  bool flush(std::vector<float> &seg, size_t &start_sample);

  size_t buffered() const { return buf_.size(); }

private:
  size_t min_len_, max_len_;
  float silence_rms_;
  bool use_vad_;
  std::vector<float> buf_;
  size_t buf_start_ = 0; /**< stream index of buf_[0] */

  // VAD streaming state
  bool in_speech_ = false;
  size_t speech_samples_ = 0;
  size_t silence_samples_ = 0;
  float noise_floor_ = 0.003f;
  float peak_rms_ = 0.01f;
};

/**
 * @brief Keeps the tail of the text recognized so far, to be given to the next
 *        segment as context (system prompt) so that a sentence cut at a
 *        segment boundary continues naturally. Holds at most max_chars UTF-8
 *        characters and starts at a word boundary when possible.
 */
class ContextCarry {
public:
  explicit ContextCarry(size_t max_chars = 0) : max_chars_(max_chars) {}
  bool enabled() const { return max_chars_ > 0; }
  size_t maxChars() const { return max_chars_; }
  /** @brief append the text of a finished segment */
  void add(const std::string &text);
  /** @brief current context string (empty if disabled / nothing yet) */
  const std::string &get() const { return text_; }

private:
  size_t max_chars_;
  std::string text_;
};

/** @brief peak of the 100 ms frame RMS values of a segment (linear, 0..1) */
float segmentPeakRms(const std::vector<float> &x);

} // namespace quick_ai

#endif
