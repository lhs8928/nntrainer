// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   audio_fanout.h
 * @date   12 October 2026
 * @brief  One audio source shared by several consumers
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __DISCRETION_AUDIO_FANOUT_H__
#define __DISCRETION_AUDIO_FANOUT_H__

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "events.h"

namespace quick_ai {
class PcmSource;
}

namespace discretion {

/**
 * @brief Reads 16 kHz mono PCM from one source (a recording replayed in real
 *        time, a microphone through arecord, or raw PCM on stdin) and hands
 *        every block to all consumer queues, like one microphone shared by
 *        several programs. Starting the capture also starts the stream clock.
 */
class AudioFanout {
public:
  struct Options {
    std::string file;       // replay this wav in real time ...
    double speed = 1.0;     // ... at this speed factor
    bool mic = false;       // capture with arecord
    std::string mic_device; // ALSA device, empty = "default"
    bool stdin_pcm = false; // raw s16le 16 kHz mono on stdin
    double max_seconds = 0; // stop after this much audio (0 = until the end)
  };

  AudioFanout();
  ~AudioFanout();

  /** @brief open the source; false and `error` set when it cannot be opened */
  bool open(const Options &opt, std::string &error);

  /** @brief start capturing in a thread; closes all queues at the end */
  void start(StreamClock &clock, const std::vector<BlockQueue *> &outputs);
  void join();

  /** @brief seconds of audio captured so far */
  double audioSeconds() const { return samples_ / 16000.0; }

  /** @brief ask the capture to stop (async-signal-safe) */
  static void requestStop();

private:
  std::unique_ptr<quick_ai::PcmSource> source_;
  double max_seconds_ = 0.0;
  std::thread thread_;
  size_t samples_ = 0;
};

} // namespace discretion

#endif // __DISCRETION_AUDIO_FANOUT_H__
