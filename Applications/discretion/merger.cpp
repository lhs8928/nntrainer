// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   merger.cpp
 * @date   12 October 2026
 * @brief  Merges ASR segments and speaker turns into speaker-labelled lines
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "merger.h"

#include <algorithm>
#include <cmath>
#include <string>

namespace discretion {

Merger::Merger(const StreamClock &clock, std::FILE *out, bool verbose) :
  clock_(clock), out_(out), verbose_(verbose) {}

void Merger::onAsrEvent(const AsrEvent &e) {
  std::lock_guard<std::mutex> lk(m_);
  if (verbose_ && e.kind == AsrEvent::Kind::Text) {
    std::fprintf(out_, "  (asr %6.1fs) [%5.1f-%5.1fs] %s\n", e.ready_sec,
                 e.start_sec, e.end_sec, e.text.c_str());
    std::fflush(out_);
  }
  if (verbose_ && !e.words.empty()) {
    std::fprintf(out_, "  (words)");
    for (const auto &w : e.words)
      std::fprintf(out_, " %.2f:%s", w.time_sec, w.text.c_str());
    std::fprintf(out_, "\n");
    std::fflush(out_);
  }
  pending_[e.index] = e;
  tryEmit();
}

void Merger::onSdUpdate(const std::vector<SdTurn> &turns,
                        const SdProgress &progress) {
  std::lock_guard<std::mutex> lk(m_);
  if (verbose_) {
    for (const auto &t : turns)
      std::fprintf(out_, "  (sd  %6.1fs) %s %.3f-%.3fs\n", t.emitted_sec,
                   t.speaker.c_str(), t.start_sec, t.end_sec);
    std::fflush(out_);
  }
  turns_.insert(turns_.end(), turns.begin(), turns.end());
  committed_ = std::max(committed_, progress.committed_until_sec);
  tryEmit();
}

void Merger::onAsrFinished() {
  std::lock_guard<std::mutex> lk(m_);
  asr_done_ = true;
  tryEmit();
  if (asr_done_ && sd_done_)
    flushAll();
}

void Merger::onSdFinished(double) {
  std::lock_guard<std::mutex> lk(m_);
  sd_done_ = true;
  tryEmit();
  if (asr_done_ && sd_done_)
    flushAll();
}

void Merger::tryEmit() {
  while (true) {
    auto it = pending_.find(next_index_);
    if (it == pending_.end())
      return;
    const AsrEvent &e = it->second;
    if (e.kind == AsrEvent::Kind::Silent) {
      pending_.erase(it);
      ++next_index_;
      continue;
    }
    if (e.kind == AsrEvent::Kind::Dropped) {
      std::fprintf(out_,
                   "[%5.1f-%5.1fs] (dropped: ASR was slower than real time)\n",
                   e.start_sec, e.end_sec);
      std::fflush(out_);
      pending_.erase(it);
      ++next_index_;
      continue;
    }
    // Wait until the diarization has confirmed everything up to the segment
    // end.
    if (!sd_done_ && committed_ < e.end_sec)
      return;
    if (e.words.empty())
      emitText(e);
    else
      emitWords(e);
    pending_.erase(it);
    ++next_index_;
  }
}

void Merger::emitText(const AsrEvent &e) {
  // Time of overlap per speaker
  std::map<std::string, double> overlap;
  double total = 0.0;
  for (const auto &t : turns_) {
    const double o =
      std::min(e.end_sec, t.end_sec) - std::max(e.start_sec, t.start_sec);
    if (o > 0.0) {
      overlap[t.speaker] += o;
      total += o;
    }
  }
  std::vector<std::pair<std::string, double>> who(overlap.begin(),
                                                  overlap.end());
  std::sort(who.begin(), who.end(),
            [](const auto &a, const auto &b) { return a.second > b.second; });
  std::string label;
  if (total <= 0.0) {
    label = "(no speaker)";
  } else {
    for (size_t i = 0; i < who.size(); ++i) {
      if (i)
        label += " + ";
      label +=
        who[i].first + "(" +
        std::to_string(static_cast<int>(100.0 * who[i].second / total + 0.5)) +
        "%)";
    }
  }

  const double now = clock_.now();
  const double lag = now - e.end_sec;
  const double asr_lag = e.ready_sec - e.end_sec;
  lags_.push_back(lag);
  asr_lags_.push_back(asr_lag);
  last_print_sec_ = now;

  std::fprintf(out_, "[%5.1f-%5.1fs] %-34s lag %5.1fs (asr %4.1fs)  %s\n",
               e.start_sec, e.end_sec, label.c_str(), lag, asr_lag,
               e.text.c_str());
  std::fflush(out_);
}

std::string Merger::speakerAt(double t) const {
  // Word times are only accurate to a fraction of a second, so look at the
  // stretch around the word: the speaker whose turns cover most of it. A short
  // turn inside a longer one (an interjection) therefore does not take the
  // word. If no turn is there, the nearest turn within one second.
  constexpr double HALF_WINDOW_SEC = 0.25;
  constexpr double NEAR_SEC = 1.0;
  const double lo = t - HALF_WINDOW_SEC, hi = t + HALF_WINDOW_SEC;
  std::map<std::string, double> covered;
  for (const auto &turn : turns_) {
    const double o = std::min(hi, turn.end_sec) - std::max(lo, turn.start_sec);
    if (o > 0.0)
      covered[turn.speaker] += o;
  }
  std::string best;
  double best_v = 0.0;
  for (const auto &c : covered) {
    if (c.second > best_v) {
      best_v = c.second;
      best = c.first;
    }
  }
  if (!best.empty())
    return best;
  double best_d = NEAR_SEC;
  for (const auto &turn : turns_) {
    const double d = t < turn.start_sec ? turn.start_sec - t : t - turn.end_sec;
    if (d < best_d) {
      best_d = d;
      best = turn.speaker;
    }
  }
  return best;
}

void Merger::emitWords(const AsrEvent &e) {
  struct Run {
    std::string speaker;
    double start;
    std::string text;
  };
  std::vector<Run> runs;
  for (const auto &w : e.words) {
    std::string who = speakerAt(w.time_sec);
    if (who.empty())
      who = runs.empty() ? "(no speaker)" : runs.back().speaker;
    if (runs.empty() || runs.back().speaker != who)
      runs.push_back({who, runs.empty() ? e.start_sec : w.time_sec, ""});
    if (!runs.back().text.empty())
      runs.back().text += ' ';
    runs.back().text += w.text;
  }

  const double now = clock_.now();
  const double lag = now - e.end_sec;
  const double asr_lag = e.ready_sec - e.end_sec;
  lags_.push_back(lag);
  asr_lags_.push_back(asr_lag);
  last_print_sec_ = now;

  for (size_t i = 0; i < runs.size(); ++i) {
    const double end = i + 1 < runs.size() ? runs[i + 1].start : e.end_sec;
    std::fprintf(out_, "[%5.1f-%5.1fs] %-34s lag %5.1fs (asr %4.1fs)  %s\n",
                 runs[i].start, end, runs[i].speaker.c_str(), lag, asr_lag,
                 runs[i].text.c_str());
  }
  std::fflush(out_);
}

void Merger::flushAll() {
  // Both lanes are done: print whatever is left in order, even across gaps.
  while (!pending_.empty()) {
    const size_t idx = pending_.begin()->first; // copy: tryEmit() may erase it
    next_index_ = idx;
    tryEmit();
    if (!pending_.empty() && pending_.begin()->first == idx)
      pending_.erase(pending_.begin()); // could not be emitted: discard
  }
}

Merger::Summary Merger::summary() const {
  std::lock_guard<std::mutex> lk(m_);
  Summary s;
  s.lines = lags_.size();
  s.last_print_sec = last_print_sec_;
  if (!lags_.empty()) {
    double sum = 0.0, asum = 0.0;
    for (double v : lags_)
      sum += v;
    for (double v : asr_lags_)
      asum += v;
    s.mean_lag = sum / lags_.size();
    s.mean_asr_lag = asum / asr_lags_.size();
    s.max_lag = *std::max_element(lags_.begin(), lags_.end());
  }
  return s;
}

} // namespace discretion
