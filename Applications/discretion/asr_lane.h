// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   asr_lane.h
 * @date   12 October 2026
 * @brief  Speech recognition lane of the combined ASR + SD application
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __DISCRETION_ASR_LANE_H__
#define __DISCRETION_ASR_LANE_H__

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "events.h"

namespace discretion {

class AsrEngine;

struct AsrOptions {
  std::string model_dir;        // Qwen3-ASR model directory
  float segment_sec = 8.0f;     // longest segment (the model is built for it)
  float segment_min_sec = 0.0f; // shortest segment, 0 = 0.9 * segment_sec
  float silence_rms = 0.005f;   // segments quieter than this are skipped
  bool vad = false; // cut at pauses (energy VAD), not only at the limit
  std::string language =
    "English";                 // forced language, "" = let the model decide
  unsigned max_backlog = 2;    // drop the oldest waiting segment beyond this
  std::vector<int> cpus;       // CPUs for this lane (empty = no restriction)
  int threads = 0;             // worker threads of the ASR (0 = default)
  std::mutex *gate = nullptr;  // serial mode: held around each model call
  bool word_timestamps = true; // time every word, so a line can change speaker
};

/**
 * @brief Cuts the audio stream into segments (at quiet points) and transcribes
 *        them one after another with a resident model. Two threads: one cuts
 *        segments as audio arrives, one runs the model.
 */
class AsrLane {
public:
  AsrLane();
  ~AsrLane();

  /** @brief build and load the model (call before start()); false on error */
  bool init(const AsrOptions &opt, std::string &error);

  /**
   * @brief use a model that is already loaded (serve mode). The engine is
   *        shared, so the lane can be discarded without unloading the model.
   */
  void adopt(std::shared_ptr<AsrEngine> engine, const AsrOptions &opt);

  /** @brief the loaded model, for lanes that come after this one */
  std::shared_ptr<AsrEngine> engine() const { return engine_; }

  void start(BlockQueue &in, const StreamClock &clock, EventSink &sink);
  void join();

  const AsrStats &stats() const { return stats_; }

private:
  struct Item {
    std::vector<float> pcm;
    size_t index = 0;
    size_t start_sample = 0;
    double complete_sec = 0.0;
  };

  void segmenterLoop(BlockQueue &in, const StreamClock &clock, EventSink &sink);
  void modelLoop(const StreamClock &clock, EventSink &sink);

  AsrOptions opt_;
  std::shared_ptr<AsrEngine> engine_;
  size_t max_len_ = 0; // samples of the longest segment
  size_t min_len_ = 0;

  std::mutex m_;
  std::condition_variable cv_;
  std::deque<Item> queue_;
  bool segmenter_done_ = false;

  std::thread segmenter_, model_;
  AsrStats stats_;
};

} // namespace discretion

#endif // __DISCRETION_ASR_LANE_H__
