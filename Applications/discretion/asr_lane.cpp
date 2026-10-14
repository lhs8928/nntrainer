// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   asr_lane.cpp
 * @date   12 October 2026
 * @brief  Speech recognition lane of the combined ASR + SD application
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "asr_lane.h"

#include <algorithm>
#include <cstdlib>

#include "asr_engine.h"
#include "audio_stream.h"
#include "cpu_affinity.h"
#include "thread_manager.h"

namespace discretion {

namespace {
constexpr double SAMPLE_RATE = 16000.0;
constexpr size_t MIN_TAIL_SAMPLES = 4800; // ignore stream tails below 0.3 s
} // namespace

namespace {

/**
 * @brief Words come out in speaking order, but the attention peak of a single
 *        word sometimes lands far from its neighbours. Replace the times by the
 *        closest non-decreasing sequence (pool adjacent violators), which
 *        averages such a word with the ones it is out of order with.
 */
void makeNonDecreasing(std::vector<double> &t) {
  struct Block {
    double sum;
    size_t count;
  };
  std::vector<Block> blocks;
  for (double v : t) {
    blocks.push_back({v, 1});
    while (blocks.size() > 1) {
      Block &b = blocks[blocks.size() - 1];
      Block &a = blocks[blocks.size() - 2];
      if (a.sum / a.count <= b.sum / b.count)
        break;
      a.sum += b.sum;
      a.count += b.count;
      blocks.pop_back();
    }
  }
  size_t i = 0;
  for (const Block &b : blocks)
    for (size_t k = 0; k < b.count; ++k)
      t[i++] = b.sum / b.count;
}

} // namespace

AsrLane::AsrLane() = default;
AsrLane::~AsrLane() { join(); }

bool AsrLane::init(const AsrOptions &opt, std::string &error) {
  opt_ = opt;
  // nntrainer's thread pool reads this when it is first used, which happens
  // while the model is built below.
  if (opt_.threads > 0)
    setenv("NNTR_NUM_THREADS", std::to_string(opt_.threads).c_str(), 1);
  if (!opt_.cpus.empty()) {
    // By default the pool pins itself to the fastest cores, which ignores
    // --asr-cpus and lands on the diarization cores. Let the workers inherit
    // the CPU set below instead.
    nntrainer::ThreadManagerConfig cfg;
    cfg.enable_affinity = false;
    nntrainer::ThreadManager::setConfig(cfg);
  }

  engine_ = std::make_unique<AsrEngine>();
  {
    // Worker threads created while the model is built inherit this CPU set.
    ScopedAffinity pin(opt_.cpus);
    if (!engine_->init(opt_, error))
      return false;
  }
  max_len_ = engine_->segmentSamples();
  const float min_sec = opt_.segment_min_sec > 0.0f ? opt_.segment_min_sec
                                                    : 0.9f * opt_.segment_sec;
  min_len_ = static_cast<size_t>(min_sec * SAMPLE_RATE);
  return true;
}

void AsrLane::start(BlockQueue &in, const StreamClock &clock, EventSink &sink) {
  segmenter_ =
    std::thread([this, &in, &clock, &sink] { segmenterLoop(in, clock, sink); });
  model_ = std::thread([this, &clock, &sink] { modelLoop(clock, sink); });
}

void AsrLane::join() {
  if (segmenter_.joinable())
    segmenter_.join();
  if (model_.joinable())
    model_.join();
}

void AsrLane::segmenterLoop(BlockQueue &in, const StreamClock &clock,
                            EventSink &sink) {
  ScopedAffinity pin(opt_.cpus);
  quick_ai::Segmenter segmenter(min_len_, max_len_);
  size_t next_index = 0;

  auto handle = [&](std::vector<float> &&pcm, size_t start_sample,
                    double complete_sec) {
    AsrEvent e;
    e.index = next_index++;
    e.start_sec = start_sample / SAMPLE_RATE;
    e.end_sec = e.start_sec + pcm.size() / SAMPLE_RATE;
    e.complete_sec = complete_sec;
    stats_.audio_sec += pcm.size() / SAMPLE_RATE;

    // Silence never reaches the model, so it cannot take a queue slot.
    if (quick_ai::segmentPeakRms(pcm) < opt_.silence_rms) {
      e.kind = AsrEvent::Kind::Silent;
      e.ready_sec = clock.now();
      ++stats_.silent;
      sink.onAsrEvent(e);
      return;
    }

    std::vector<AsrEvent> dropped;
    {
      std::lock_guard<std::mutex> lk(m_);
      // Slower than real time: drop the oldest waiting segment (bounded delay)
      while (opt_.max_backlog > 0 && queue_.size() >= opt_.max_backlog) {
        const Item &old = queue_.front();
        AsrEvent d;
        d.kind = AsrEvent::Kind::Dropped;
        d.index = old.index;
        d.start_sec = old.start_sample / SAMPLE_RATE;
        d.end_sec = d.start_sec + old.pcm.size() / SAMPLE_RATE;
        d.complete_sec = old.complete_sec;
        d.ready_sec = clock.now();
        dropped.push_back(d);
        queue_.pop_front();
        ++stats_.dropped;
      }
      Item it;
      it.pcm = std::move(pcm);
      it.index = e.index;
      it.start_sample = start_sample;
      it.complete_sec = complete_sec;
      queue_.push_back(std::move(it));
      stats_.max_backlog = std::max(stats_.max_backlog, queue_.size());
    }
    cv_.notify_one();
    for (const auto &d : dropped)
      sink.onAsrEvent(d);
  };

  AudioBlock b;
  std::vector<float> seg;
  size_t start = 0;
  while (in.pop(b)) {
    segmenter.push(b.pcm.data(), b.pcm.size());
    while (segmenter.pop(seg, start))
      handle(std::move(seg), start, b.t_sec);
  }
  if (segmenter.flush(seg, start) && seg.size() >= MIN_TAIL_SAMPLES)
    handle(std::move(seg), start, clock.now());

  {
    std::lock_guard<std::mutex> lk(m_);
    segmenter_done_ = true;
  }
  cv_.notify_all();
}

void AsrLane::modelLoop(const StreamClock &clock, EventSink &sink) {
  ScopedAffinity pin(opt_.cpus);
  while (true) {
    Item it;
    {
      std::unique_lock<std::mutex> lk(m_);
      cv_.wait(lk, [&] { return !queue_.empty() || segmenter_done_; });
      if (queue_.empty())
        break;
      it = std::move(queue_.front());
      queue_.pop_front();
    }

    AsrEvent e;
    e.kind = AsrEvent::Kind::Text;
    e.index = it.index;
    e.start_sec = it.start_sample / SAMPLE_RATE;
    e.end_sec = e.start_sec + it.pcm.size() / SAMPLE_RATE;
    e.complete_sec = it.complete_sec;

    it.pcm.resize(max_len_, 0.0f); // pad to the fixed model length with silence
    const double t0 = clock.now();
    std::vector<std::pair<std::string, double>> words;
    {
      auto hold = holdModelGate(opt_.gate);
      e.text = engine_->transcribe(std::move(it.pcm));
      words = engine_->lastWords();
    }
    // The model pads the segment with silence, so a word cannot be later than
    // the real end of the audio.
    const double seg_len = e.end_sec - e.start_sec;
    std::vector<double> times;
    for (const auto &w : words)
      times.push_back(w.second);
    makeNonDecreasing(times);
    for (size_t i = 0; i < words.size(); ++i)
      e.words.push_back(
        {words[i].first, e.start_sec + std::min(times[i], seg_len)});
    e.ready_sec = clock.now();
    e.proc_sec = e.ready_sec - t0;

    ++stats_.segments;
    stats_.proc_sec += e.proc_sec;
    stats_.max_delay_sec =
      std::max(stats_.max_delay_sec, e.ready_sec - e.complete_sec);
    sink.onAsrEvent(e);
  }
  sink.onAsrFinished();
}

} // namespace discretion
