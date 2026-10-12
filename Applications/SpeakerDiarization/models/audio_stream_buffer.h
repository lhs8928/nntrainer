// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   audio_stream_buffer.h
 * @date   12 October 2026
 * @brief  Rolling FIFO Audio Buffer for Continuous Streaming Diarization
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __AUDIO_STREAM_BUFFER_H__
#define __AUDIO_STREAM_BUFFER_H__

#include <vector>
#include <cstddef>
#include <mutex>

namespace speaker_diarization {

/**
 * @brief Thread-safe sliding-window audio buffer maintaining an O(1) bounded memory footprint.
 */
class AudioStreamBuffer {
public:
  /**
   * @brief Constructor.
   * @param sample_rate Audio sample rate in Hz (default 16000).
   * @param window_duration_sec Window duration in seconds (default 20.0s).
   * @param step_duration_sec Slide step duration in seconds (default 10.0s).
   */
  AudioStreamBuffer(size_t sample_rate = 16000,
                    double window_duration_sec = 20.0,
                    double step_duration_sec = 10.0);
  ~AudioStreamBuffer() = default;

  /**
   * @brief Append incoming raw float audio samples from stream or device.
   */
  void pushSamples(const float *samples, size_t count);

  /**
   * @brief Check if sufficient samples are available to extract a full window.
   */
  bool hasWindow() const;

  /**
   * @brief Extract current window audio samples.
   * @param out_window Vector to receive window samples (size = window_samples).
   * @param out_start_time_sec Output timestamp in seconds of the start of this window.
   * @return true if successful, false otherwise.
   */
  bool getWindow(std::vector<float> &out_window, double &out_start_time_sec);

  /**
   * @brief Advance buffer forward by one step, recycling old samples to keep memory flat.
   */
  void advanceStep();

  /**
   * @brief Flush any remaining audio samples at the end of the stream.
   * @param out_window Vector to receive remaining samples.
   * @param out_start_time_sec Output start timestamp in seconds.
   * @return true if there are valid remaining speech samples, false otherwise.
   */
  bool flushRemaining(std::vector<float> &out_window, double &out_start_time_sec);

  /**
   * @brief Get total number of audio samples pushed into this stream.
   */
  size_t totalSamplesIngested() const { return total_samples_ingested_; }

  /**
   * @brief Get configured window size in samples.
   */
  size_t windowSamples() const { return window_samples_; }

  /**
   * @brief Get configured step size in samples.
   */
  size_t stepSamples() const { return step_samples_; }

  /**
   * @brief Reset stream buffer state.
   */
  void reset();

private:
  size_t sample_rate_;
  size_t window_samples_;
  size_t step_samples_;
  size_t min_chunk_samples_;

  std::vector<float> buffer_;
  size_t total_samples_ingested_ = 0;
  size_t current_window_start_sample_ = 0;
  mutable std::mutex mutex_;
};

} // namespace speaker_diarization

#endif // __AUDIO_STREAM_BUFFER_H__
