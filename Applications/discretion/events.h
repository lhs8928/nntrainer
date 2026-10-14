// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   events.h
 * @date   12 October 2026
 * @brief  Shared types of the combined ASR + speaker diarization application
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Everything here is free of any model header on purpose: the ASR and the
 * speaker diarization code use headers with the same file names and include
 * guards, so each of them is only ever included from its own source files.
 */

#ifndef __DISCRETION_EVENTS_H__
#define __DISCRETION_EVENTS_H__

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace discretion {

/**
 * @brief Serial mode: holds the gate shared by both lanes for one model call.
 *        Without a gate (parallel mode) it holds nothing.
 */
inline std::unique_lock<std::mutex> holdModelGate(std::mutex *gate) {
  return gate ? std::unique_lock<std::mutex>(*gate)
              : std::unique_lock<std::mutex>();
}

using Clock = std::chrono::steady_clock;

/**
 * @brief Stream time: seconds since the audio feed started. All lanes stamp
 *        their results with it, so a result can be compared with the audio it
 *        belongs to.
 */
class StreamClock {
public:
  void start() { t0_ = Clock::now(); }
  double now() const {
    return std::chrono::duration<double>(Clock::now() - t0_).count();
  }

private:
  Clock::time_point t0_ = Clock::now();
};

/** @brief A block of 16 kHz mono float samples and when it was received. */
struct AudioBlock {
  std::vector<float> pcm;
  double t_sec = 0.0; // stream time at which the block arrived
};

/**
 * @brief Unbounded blocking queue of audio blocks, one per consumer. A slow
 *        consumer only grows its own queue; it never stalls the audio feed or
 *        the other consumer.
 */
class BlockQueue {
public:
  void push(AudioBlock &&b) {
    std::lock_guard<std::mutex> lk(m_);
    q_.push_back(std::move(b));
    cv_.notify_one();
  }
  /** @brief wait for a block; false when closed and drained */
  bool pop(AudioBlock &out) {
    std::unique_lock<std::mutex> lk(m_);
    cv_.wait(lk, [&] { return !q_.empty() || closed_; });
    if (q_.empty())
      return false;
    out = std::move(q_.front());
    q_.pop_front();
    return true;
  }
  void close() {
    std::lock_guard<std::mutex> lk(m_);
    closed_ = true;
    cv_.notify_all();
  }
  size_t size() const {
    std::lock_guard<std::mutex> lk(m_);
    return q_.size();
  }

private:
  mutable std::mutex m_;
  std::condition_variable cv_;
  std::deque<AudioBlock> q_;
  bool closed_ = false;
};

/** @brief What happened to one ASR segment. */
/** @brief One transcribed word and the audio position it was spoken at. */
struct AsrWord {
  std::string text;
  double time_sec = 0.0; // stream time, already shifted by the segment start
};

struct AsrEvent {
  enum class Kind { Text, Silent, Dropped };
  Kind kind = Kind::Text;
  size_t index = 0;       // segments are numbered in stream order
  double start_sec = 0.0; // audio position of the segment
  double end_sec = 0.0;
  std::string text;
  std::vector<AsrWord> words; // empty without word timestamps
  double proc_sec = 0.0;      // model time spent on it
  double complete_sec = 0.0;  // stream time when its audio was complete
  double ready_sec = 0.0;     // stream time when the text was available
};

/** @brief One confirmed speaker turn. */
struct SdTurn {
  std::string speaker;
  double start_sec = 0.0;
  double end_sec = 0.0;
  double emitted_sec = 0.0; // stream time when it was confirmed
};

/** @brief How far the diarization is final, after a window was processed. */
struct SdProgress {
  double committed_until_sec = 0.0; // no later turn will end before this
  double ready_sec = 0.0;           // stream time of this update
};

struct AsrStats {
  size_t segments = 0, silent = 0, dropped = 0;
  double audio_sec = 0.0, proc_sec = 0.0, max_delay_sec = 0.0;
  size_t max_backlog = 0;
};

struct SdStats {
  size_t turns = 0, windows = 0, speakers = 0;
  double model_ms = 0.0, audio_sec = 0.0;
};

/** @brief Receiver of lane results; implemented by the merger. */
class EventSink {
public:
  virtual ~EventSink() = default;
  virtual void onAsrEvent(const AsrEvent &e) = 0;
  virtual void onSdUpdate(const std::vector<SdTurn> &turns,
                          const SdProgress &progress) = 0;
  virtual void onAsrFinished() = 0;
  virtual void onSdFinished(double ready_sec) = 0;
};

} // namespace discretion

#endif // __DISCRETION_EVENTS_H__
