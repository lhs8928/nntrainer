// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   sd_lane.cpp
 * @date   12 October 2026
 * @brief  Speaker diarization lane of the combined ASR + SD application
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "sd_lane.h"

#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../SpeakerDiarization/models/streaming_diarizer.h"
#include "cpu_affinity.h"

namespace discretion {

SdLane::SdLane() = default;
SdLane::~SdLane() { join(); }

bool SdLane::init(const SdOptions &opt) {
  opt_ = opt;
  diarizer_ = std::make_unique<speaker_diarization::StreamingDiarizer>(
    opt.window_sec, opt.step_sec);
  return diarizer_->init(opt.weights);
}

void SdLane::start(BlockQueue &in, const StreamClock &clock, EventSink &sink) {
  thread_ = std::thread([this, &in, &clock, &sink] { run(in, clock, sink); });
}

void SdLane::join() {
  if (thread_.joinable())
    thread_.join();
}

void SdLane::run(BlockQueue &in, const StreamClock &clock, EventSink &sink) {
  // The OpenMP team of this lane is created by this thread on its first
  // parallel region and inherits the CPU set that is set here.
  ScopedAffinity pin(opt_.cpus);
#ifdef _OPENMP
  if (opt_.threads > 0)
    omp_set_num_threads(opt_.threads);
#endif

  auto publish = [&](const speaker_diarization::StreamingResult &r) {
    stats_.windows += r.windows_processed;
    if (r.windows_processed == 0 && r.segments.empty())
      return;
    const double now = clock.now();
    std::vector<SdTurn> turns;
    turns.reserve(r.segments.size());
    for (const auto &s : r.segments)
      turns.push_back({s.speaker, s.start_time, s.end_time, now});
    stats_.turns += turns.size();
    sink.onSdUpdate(turns, SdProgress{r.committed_until_sec, now});
  };

  AudioBlock b;
  while (in.pop(b)) {
    speaker_diarization::StreamingResult r;
    {
      auto hold = holdModelGate(opt_.gate);
      r = diarizer_->push(b.pcm.data(), b.pcm.size());
    }
    publish(r);
  }
  speaker_diarization::StreamingResult last;
  {
    auto hold = holdModelGate(opt_.gate);
    last = diarizer_->finish();
  }
  publish(last);

  stats_.model_ms = diarizer_->totalModelMs();
  stats_.speakers = diarizer_->numSpeakers();
  stats_.audio_sec = diarizer_->totalAudioSec();
  sink.onSdFinished(clock.now());
}

} // namespace discretion
