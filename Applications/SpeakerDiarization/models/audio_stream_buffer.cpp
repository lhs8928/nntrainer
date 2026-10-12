// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   audio_stream_buffer.cpp
 * @date   12 October 2026
 * @brief  Rolling FIFO Audio Buffer for Continuous Streaming Diarization
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "audio_stream_buffer.h"
#include <cstring>
#include <algorithm>

namespace speaker_diarization {

AudioStreamBuffer::AudioStreamBuffer(size_t sample_rate,
                                     double window_duration_sec,
                                     double step_duration_sec)
    : sample_rate_(sample_rate),
      window_samples_(static_cast<size_t>(window_duration_sec * sample_rate)),
      step_samples_(static_cast<size_t>(step_duration_sec * sample_rate)),
      min_chunk_samples_(static_cast<size_t>(10.0 * sample_rate)) {
  buffer_.reserve(window_samples_ + step_samples_);
}

void AudioStreamBuffer::pushSamples(const float *samples, size_t count) {
  if (!samples || count == 0) return;
  std::lock_guard<std::mutex> lock(mutex_);
  buffer_.insert(buffer_.end(), samples, samples + count);
  total_samples_ingested_ += count;
}

bool AudioStreamBuffer::hasWindow() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return buffer_.size() >= window_samples_;
}

bool AudioStreamBuffer::getWindow(std::vector<float> &out_window, double &out_start_time_sec) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (buffer_.size() < window_samples_) return false;

  out_window.assign(buffer_.begin(), buffer_.begin() + window_samples_);
  out_start_time_sec = static_cast<double>(current_window_start_sample_) / static_cast<double>(sample_rate_);
  return true;
}

void AudioStreamBuffer::advanceStep() {
  std::lock_guard<std::mutex> lock(mutex_);
  size_t advance = std::min(step_samples_, buffer_.size());
  buffer_.erase(buffer_.begin(), buffer_.begin() + advance);
  current_window_start_sample_ += advance;
}

bool AudioStreamBuffer::flushRemaining(std::vector<float> &out_window, double &out_start_time_sec) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (buffer_.size() < min_chunk_samples_) {
    // Insufficient audio left to form a receptive field
    buffer_.clear();
    return false;
  }

  out_window = buffer_;
  out_start_time_sec = static_cast<double>(current_window_start_sample_) / static_cast<double>(sample_rate_);
  buffer_.clear();
  return true;
}

void AudioStreamBuffer::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  buffer_.clear();
  total_samples_ingested_ = 0;
  current_window_start_sample_ = 0;
}

} // namespace speaker_diarization
