// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   sd_lane.h
 * @date   12 October 2026
 * @brief  Speaker diarization lane of the combined ASR + SD application
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __DISCRETION_SD_LANE_H__
#define __DISCRETION_SD_LANE_H__

#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "events.h"

namespace speaker_diarization {
class StreamingDiarizer;
}

namespace discretion {

struct SdOptions {
  std::string weights;      // speaker_diarization.bin
  double window_sec = 20.0; // diarization window
  double step_sec = 20.0;   // slide step (= window: no overlap, half the work)
  std::vector<int> cpus;    // CPUs for this lane (empty = no restriction)
  int threads = 0;          // OpenMP threads of this lane (0 = default)
  std::mutex *gate = nullptr; // serial mode: held around each model call
};

/**
 * @brief Feeds the audio queue to the streaming diarizer in its own thread and
 *        reports confirmed speaker turns.
 */
class SdLane {
public:
  SdLane();
  ~SdLane();

  /** @brief load weights and initialize the models (call before start()) */
  bool init(const SdOptions &opt);

  /** @brief start the lane thread; it ends when `in` is closed and drained */
  void start(BlockQueue &in, const StreamClock &clock, EventSink &sink);
  void join();

  const SdStats &stats() const { return stats_; }

private:
  void run(BlockQueue &in, const StreamClock &clock, EventSink &sink);

  SdOptions opt_;
  std::unique_ptr<speaker_diarization::StreamingDiarizer> diarizer_;
  std::thread thread_;
  SdStats stats_;
};

} // namespace discretion

#endif // __DISCRETION_SD_LANE_H__
