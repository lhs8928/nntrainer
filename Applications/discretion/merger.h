// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   merger.h
 * @date   12 October 2026
 * @brief  Merges ASR segments and speaker turns into speaker-labelled lines
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __DISCRETION_MERGER_H__
#define __DISCRETION_MERGER_H__

#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "events.h"

namespace discretion {

/**
 * @brief Receives the results of both lanes (from their threads) and prints one
 *        line per ASR segment, in stream order, as soon as its text exists and
 *        the diarization has confirmed everything up to the segment's end.
 *
 * With word timestamps every word gets the speaker whose turn covers its time,
 * and a segment is split into one line per run of words of the same speaker.
 * Without them the segment is labelled with the speakers whose turns overlap
 * it, weighted by the overlapping time.
 */
class Merger : public EventSink {
public:
  /**
   * @param clock stream clock, used to stamp the moment a line is printed
   * @param out where merged lines go
   * @param verbose also print every raw lane event
   */
  Merger(const StreamClock &clock, std::FILE *out, bool verbose);

  void onAsrEvent(const AsrEvent &e) override;
  void onSdUpdate(const std::vector<SdTurn> &turns,
                  const SdProgress &progress) override;
  void onAsrFinished() override;
  void onSdFinished(double ready_sec) override;

  struct Summary {
    size_t lines = 0;      // merged text lines printed
    double mean_lag = 0.0; // seconds after the segment end
    double max_lag = 0.0;
    double mean_asr_lag = 0.0;
    double last_print_sec = 0.0;
  };
  Summary summary() const;

  /**
   * @brief Also write every merged line as one JSON object per line (JSON
   * Lines).
   * @param out JSON Lines file, not owned (the caller opens and closes it)
   * @param session identifier written into every object
   */
  void setJsonl(std::FILE *out, const std::string &session);

private:
  void tryEmit();
  void emitText(const AsrEvent &e);
  void emitWords(const AsrEvent &e);
  void emitLine(double start, double end, const std::string &primary,
                const std::string &label, double lag, double asr_lag,
                const std::string &text);
  std::string speakerAt(double t) const;
  void flushAll();

  const StreamClock &clock_;
  std::FILE *out_;
  bool verbose_;

  mutable std::mutex m_;
  std::map<size_t, AsrEvent> pending_; // by segment index
  size_t next_index_ = 0;
  std::vector<SdTurn> turns_;
  double committed_ = 0.0;
  bool asr_done_ = false, sd_done_ = false;

  std::vector<double> lags_, asr_lags_;
  double last_print_sec_ = 0.0;

  std::FILE *jsonl_ = nullptr;
  std::string session_;
  size_t seq_ = 0; // sequence number of the next JSON object
};

} // namespace discretion

#endif // __DISCRETION_MERGER_H__
