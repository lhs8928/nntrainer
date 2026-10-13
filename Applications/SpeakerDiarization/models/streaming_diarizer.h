// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   streaming_diarizer.h
 * @date   12 October 2026
 * @brief  Reusable streaming speaker diarization engine (window pipeline)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Wraps the windowed pipeline that used to live inline in the SD application's
 * streaming mode (slice window -> PyanNet -> WeSpeaker embeddings -> clustering
 * -> global speaker tracking -> commit of confirmed turns), so that other
 * programs can feed audio and read confirmed speaker turns without touching
 * stdin/stdout.
 */

#ifndef __STREAMING_DIARIZER_H__
#define __STREAMING_DIARIZER_H__

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "audio_preprocessor.h"
#include "audio_stream_buffer.h"
#include "online_speaker_tracker.h"
#include "plda_vbx.h"
#include "pyannet.h"
#include "weight_loader.h"
#include "wespeaker_resnet34.h"

namespace speaker_diarization {

/**
 * @brief What a push() / finish() call produced.
 */
struct StreamingResult {
  /** Turns confirmed by this call (global speaker names, absolute seconds). */
  std::vector<SpeakerSegment> segments;
  /**
   * Everything that ends before this time (seconds) is final: later calls will
   * not add turns that end earlier. Infinity once finish() has been called.
   */
  double committed_until_sec = 0.0;
  /** Windows processed by this call. */
  size_t windows_processed = 0;
  /** Compute time spent by this call, in milliseconds. */
  double model_ms = 0.0;
};

/**
 * @brief Continuous diarization over an audio stream. Not thread-safe: use one
 *        instance from one thread. push() runs the models in the calling thread
 *        (and in the OpenMP team it creates), so it blocks while a window is
 *        processed.
 */
class StreamingDiarizer {
public:
  /**
   * @param window_sec window length (default 20 s)
   * @param step_sec slide step (default 10 s)
   * @param match_threshold cosine threshold to match an existing global speaker
   * @param ema_alpha centroid smoothing of the global speaker profiles
   */
  explicit StreamingDiarizer(double window_sec = 20.0, double step_sec = 10.0,
                             float match_threshold = 0.65f,
                             float ema_alpha = 0.85f);

  /** @brief load weights and initialize all models; false on failure */
  bool init(const std::string &weights_path);

  /** @brief append 16 kHz mono float samples and process every ready window */
  StreamingResult push(const float *pcm, size_t n);

  /** @brief end of stream: process the remaining audio and commit everything */
  StreamingResult finish();

  size_t numSpeakers() const { return tracker_.numSpeakers(); }
  const OnlineSpeakerTracker &tracker() const { return tracker_; }
  /** @brief total compute time of all windows so far (ms) */
  double totalModelMs() const { return total_model_ms_; }
  /** @brief seconds of audio received so far */
  double totalAudioSec() const {
    return buffer_.totalSamplesIngested() / 16000.0;
  }
  double windowSec() const { return window_sec_; }
  double stepSec() const { return step_sec_; }

private:
  /**
   * @brief run the model pipeline on one block of audio and return its
   *        diarization (absolute times, global speaker names)
   */
  std::vector<SpeakerSegment> processAudio(const std::vector<float> &audio,
                                           double start_time_sec);

  double window_sec_, step_sec_;
  WeightLoader loader_;
  AudioPreprocessor preprocessor_;
  PyanNet seg_model_;
  WeSpeakerResNet34 emb_model_;
  PldaVBx plda_vbx_;
  const float *resample_kernel_ = nullptr;
  const float *mel_banks_ = nullptr;
  const float *fbank_window_ = nullptr;

  AudioStreamBuffer buffer_;
  OnlineSpeakerTracker tracker_;
  std::map<std::string, float>
    last_confirmed_; // end of the last turn per speaker
  double total_model_ms_ = 0.0;
  bool finished_ = false;
};

} // namespace speaker_diarization

#endif // __STREAMING_DIARIZER_H__
