// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   audio_fanout.cpp
 * @date   12 October 2026
 * @brief  One audio source shared by several consumers
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "audio_fanout.h"

#include <atomic>
#include <exception>

#include "audio_preprocessor.h"
#include "audio_stream.h"

namespace discretion {

namespace {
std::atomic<bool> g_stop{false};
constexpr size_t BLOCK_SAMPLES = 1600; // 100 ms
} // namespace

AudioFanout::AudioFanout() = default;
AudioFanout::~AudioFanout() { join(); }

void AudioFanout::requestStop() { g_stop.store(true); }

bool AudioFanout::open(const Options &opt, std::string &error) {
  max_seconds_ = opt.max_seconds;
  try {
    if (!opt.file.empty()) {
      quick_ai::AudioPreprocessor pre;
      source_ = std::make_unique<quick_ai::WavRealtimeSource>(
        pre.loadAudio(opt.file), opt.speed);
    } else if (opt.mic) {
      const std::string dev =
        opt.mic_device.empty() ? "default" : opt.mic_device;
      source_ = std::make_unique<quick_ai::PipeSource>(
        "arecord -q -D " + dev + " -f S16_LE -r 16000 -c 1 -t raw");
    } else if (opt.stdin_pcm) {
      source_ = std::make_unique<quick_ai::PipeSource>("");
    } else {
      error = "no audio source: use --stream-file, --mic or --stdin-pcm";
      return false;
    }
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  }
  return true;
}

void AudioFanout::start(StreamClock &clock,
                        const std::vector<BlockQueue *> &outputs) {
  thread_ = std::thread([this, &clock, outputs] {
    clock.start(); // stream time zero
    std::vector<float> buf(BLOCK_SAMPLES);
    const size_t limit =
      max_seconds_ > 0.0 ? static_cast<size_t>(max_seconds_ * 16000.0) : 0;
    size_t n;
    while (!g_stop.load() && (n = source_->read(buf.data(), buf.size())) > 0) {
      const double t = clock.now();
      for (BlockQueue *q : outputs) {
        AudioBlock b;
        b.pcm.assign(buf.begin(), buf.begin() + n);
        b.t_sec = t;
        q->push(std::move(b));
      }
      samples_ += n;
      if (limit > 0 && samples_ >= limit)
        break;
    }
    for (BlockQueue *q : outputs)
      q->close();
  });
}

void AudioFanout::join() {
  if (thread_.joinable())
    thread_.join();
}

bool loadAudioFile(const std::string &path, std::vector<float> &pcm,
                   std::string &error) {
  try {
    quick_ai::AudioPreprocessor pre;
    pcm = pre.loadAudio(path);
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  }
  if (pcm.empty()) {
    error = "no audio in " + path;
    return false;
  }
  return true;
}

} // namespace discretion
