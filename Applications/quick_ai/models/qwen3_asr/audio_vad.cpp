// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   audio_vad.cpp
 * @brief  Lightweight zero-dependency adaptive Voice Activity Detection (VAD)
 *         and speech-boundary segmenter for Qwen3-ASR
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 */
#include "audio_vad.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace quick_ai {

static inline float computeFrameRms(const float *data, size_t len) {
  if (len == 0) return 0.0f;
  double sum = 0.0;
  for (size_t i = 0; i < len; ++i) {
    sum += static_cast<double>(data[i]) * data[i];
  }
  return static_cast<float>(std::sqrt(sum / len));
}

std::vector<VadSegment> AudioVAD::segment(const std::vector<float> &pcm,
                                         const VadConfig &cfg) {
  std::vector<VadSegment> segments;
  if (pcm.empty()) return segments;

  const size_t sample_rate = cfg.sample_rate;
  const size_t frame_len = (sample_rate * cfg.frame_ms) / 1000;
  if (frame_len == 0) return segments;

  const size_t num_frames = (pcm.size() + frame_len - 1) / frame_len;
  std::vector<float> frame_rms(num_frames, 0.0f);

  float max_rms = 0.0f;
  float min_rms = 1e9f;

  for (size_t f = 0; f < num_frames; ++f) {
    size_t start = f * frame_len;
    size_t len = std::min(frame_len, pcm.size() - start);
    float rms = computeFrameRms(pcm.data() + start, len);
    frame_rms[f] = rms;
    if (rms > max_rms) max_rms = rms;
    if (rms < min_rms) min_rms = rms;
  }

  // Dynamic speech energy threshold
  float dynamic_thresh = min_rms + (max_rms - min_rms) * cfg.snr_factor;
  float speech_thresh = std::max(cfg.min_silence_rms, dynamic_thresh);

  // 1. Initial frame classification
  std::vector<bool> is_speech(num_frames, false);
  for (size_t f = 0; f < num_frames; ++f) {
    is_speech[f] = (frame_rms[f] >= speech_thresh);
  }

  // 2. Hangover smoothing: close short silence gaps (< min_silence_gap_ms)
  const size_t min_gap_frames = static_cast<size_t>(
    std::round(cfg.min_silence_gap_ms / cfg.frame_ms));
  size_t silence_run = 0;
  for (size_t f = 0; f < num_frames; ++f) {
    if (!is_speech[f]) {
      silence_run++;
    } else {
      if (silence_run > 0 && silence_run < min_gap_frames) {
        // Fill short pause
        for (size_t k = f - silence_run; k < f; ++k) {
          is_speech[k] = true;
        }
      }
      silence_run = 0;
    }
  }

  // 3. Reject short noise bursts (< min_speech_ms)
  const size_t min_speech_frames = static_cast<size_t>(
    std::round(cfg.min_speech_ms / cfg.frame_ms));
  size_t speech_run = 0;
  for (size_t f = 0; f < num_frames; ++f) {
    if (is_speech[f]) {
      speech_run++;
    } else {
      if (speech_run > 0 && speech_run < min_speech_frames) {
        for (size_t k = f - speech_run; k < f; ++k) {
          is_speech[k] = false;
        }
      }
      speech_run = 0;
    }
  }
  if (speech_run > 0 && speech_run < min_speech_frames) {
    for (size_t k = num_frames - speech_run; k < num_frames; ++k) {
      is_speech[k] = false;
    }
  }

  // 4. Extract contiguous speech intervals
  struct Interval {
    size_t start_f;
    size_t end_f;
  };
  std::vector<Interval> intervals;
  bool in_speech = false;
  size_t cur_start = 0;

  for (size_t f = 0; f < num_frames; ++f) {
    if (is_speech[f] && !in_speech) {
      in_speech = true;
      cur_start = f;
    } else if (!is_speech[f] && in_speech) {
      in_speech = false;
      intervals.push_back({cur_start, f});
    }
  }
  if (in_speech) {
    intervals.push_back({cur_start, num_frames});
  }

  if (intervals.empty()) {
    std::cout << "[AudioVAD] No speech detected in audio stream (silence threshold: "
              << speech_thresh << ")" << std::endl;
    return segments;
  }

  // 5. Expand with padding and split intervals exceeding max_segment_sec
  const size_t pad_onset_frames = static_cast<size_t>(
    std::round(cfg.pad_onset_ms / cfg.frame_ms));
  const size_t pad_offset_frames = static_cast<size_t>(
    std::round(cfg.pad_offset_ms / cfg.frame_ms));
  const size_t max_seg_frames = static_cast<size_t>(
    std::round((cfg.max_segment_sec * 1000.0f) / cfg.frame_ms));

  std::vector<Interval> final_intervals;
  for (const auto &inv : intervals) {
    size_t s_f = (inv.start_f >= pad_onset_frames) ? (inv.start_f - pad_onset_frames) : 0;
    size_t e_f = std::min(num_frames, inv.end_f + pad_offset_frames);

    // If interval exceeds max_segment_sec, find natural energy minimum to split
    while ((e_f - s_f) > max_seg_frames) {
      size_t search_start = s_f + static_cast<size_t>(max_seg_frames * 0.65f);
      size_t search_end = s_f + max_seg_frames;

      size_t best_cut = search_end;
      float best_rms = 1e9f;
      for (size_t k = search_start; k < search_end; ++k) {
        if (frame_rms[k] < best_rms) {
          best_rms = frame_rms[k];
          best_cut = k;
        }
      }

      final_intervals.push_back({s_f, best_cut});
      s_f = best_cut;
    }
    if (e_f > s_f) {
      final_intervals.push_back({s_f, e_f});
    }
  }

  // 6. Merge short adjacent intervals if combined length <= max_segment_sec
  std::vector<Interval> merged_intervals;
  for (const auto &inv : final_intervals) {
    if (merged_intervals.empty()) {
      merged_intervals.push_back(inv);
    } else {
      auto &prev = merged_intervals.back();
      if ((inv.end_f - prev.start_f) <= max_seg_frames && (inv.start_f <= prev.end_f + min_gap_frames)) {
        prev.end_f = std::max(prev.end_f, inv.end_f);
      } else {
        merged_intervals.push_back(inv);
      }
    }
  }

  // 7. Materialize VadSegments
  double total_speech_sec = 0.0;
  for (const auto &inv : merged_intervals) {
    size_t start_idx = inv.start_f * frame_len;
    size_t end_idx = std::min(pcm.size(), inv.end_f * frame_len);
    if (end_idx <= start_idx) continue;

    VadSegment seg;
    seg.start_sample = start_idx;
    seg.end_sample = end_idx;
    seg.start_sec = static_cast<double>(start_idx) / sample_rate;
    seg.end_sec = static_cast<double>(end_idx) / sample_rate;
    seg.pcm.assign(pcm.begin() + start_idx, pcm.begin() + end_idx);

    total_speech_sec += (seg.end_sec - seg.start_sec);
    segments.push_back(std::move(seg));
  }

  double total_audio_sec = static_cast<double>(pcm.size()) / sample_rate;
  double silence_skipped_sec = total_audio_sec - total_speech_sec;
  if (silence_skipped_sec < 0) silence_skipped_sec = 0;

  std::cout << "[AudioVAD] Analysis complete: " << total_audio_sec << " s total audio -> "
            << segments.size() << " speech segment(s), " << total_speech_sec << " s voice ("
            << (total_speech_sec * 100.0 / total_audio_sec) << "%), skipped "
            << silence_skipped_sec << " s silence" << std::endl;

  return segments;
}

} // namespace quick_ai
