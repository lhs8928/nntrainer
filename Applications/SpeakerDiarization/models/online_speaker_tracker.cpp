// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   online_speaker_tracker.cpp
 * @date   12 October 2026
 * @brief  Online Global Speaker Registry and Centroid Tracker for Continuous Audio Streaming
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "online_speaker_tracker.h"
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>

namespace speaker_diarization {

static inline float cosine_similarity(const float *a, const float *b, size_t dim) {
  float dot = 0.0f;
  float norm_a = 0.0f;
  float norm_b = 0.0f;
  for (size_t d = 0; d < dim; ++d) {
    dot += a[d] * b[d];
    norm_a += a[d] * a[d];
    norm_b += b[d] * b[d];
  }
  float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
  return (denom > 1e-8f) ? (dot / denom) : 0.0f;
}

static inline void normalize_vector(std::vector<float> &vec) {
  float sum_sq = 0.0f;
  for (float v : vec) sum_sq += v * v;
  float norm = std::sqrt(sum_sq);
  if (norm > 1e-8f) {
    float inv = 1.0f / norm;
    for (float &v : vec) v *= inv;
  }
}

static inline std::string format_speaker_name(int id) {
  std::ostringstream oss;
  oss << "SPEAKER_" << std::setw(2) << std::setfill('0') << id;
  return oss.str();
}

OnlineSpeakerTracker::OnlineSpeakerTracker(float match_threshold, float ema_alpha)
    : match_threshold_(match_threshold), ema_alpha_(ema_alpha) {}

std::vector<int> OnlineSpeakerTracker::matchClusters(
    const std::vector<std::vector<float>> &local_centroids,
    double window_timestamp_sec) {
  size_t num_local = local_centroids.size();
  std::vector<int> mapping(num_local, -1);
  if (num_local == 0) return mapping;

  // Case 1: First window, initialize all global profiles
  if (profiles_.empty()) {
    for (size_t c = 0; c < num_local; ++c) {
      int new_id = static_cast<int>(profiles_.size());
      GlobalSpeakerProfile prof;
      prof.id = new_id;
      prof.name = format_speaker_name(new_id);
      prof.centroid = local_centroids[c];
      normalize_vector(prof.centroid);
      prof.count = 1;
      prof.first_seen_sec = window_timestamp_sec;
      prof.last_seen_sec = window_timestamp_sec;

      profiles_.push_back(prof);
      mapping[c] = new_id;
    }
    return mapping;
  }

  // Case 2: Subsequent windows, bipartite greedy matching
  struct MatchCandidate {
    size_t local_idx;
    size_t global_idx;
    float similarity;
  };

  std::vector<MatchCandidate> candidates;
  candidates.reserve(num_local * profiles_.size());

  for (size_t c = 0; c < num_local; ++c) {
    for (size_t k = 0; k < profiles_.size(); ++k) {
      float sim = cosine_similarity(local_centroids[c].data(),
                                    profiles_[k].centroid.data(), 256);
      candidates.push_back({c, k, sim});
    }
  }

  // Sort by highest similarity first
  std::sort(candidates.begin(), candidates.end(),
            [](const MatchCandidate &a, const MatchCandidate &b) {
              return a.similarity > b.similarity;
            });

  std::vector<bool> local_matched(num_local, false);
  std::vector<bool> global_matched(profiles_.size(), false);

  for (const auto &cand : candidates) {
    if (cand.similarity < match_threshold_) {
      break; // Remaining candidates are below threshold
    }
    if (!local_matched[cand.local_idx] && !global_matched[cand.global_idx]) {
      // Bind local cluster to existing global profile
      local_matched[cand.local_idx] = true;
      global_matched[cand.global_idx] = true;
      int global_id = profiles_[cand.global_idx].id;
      mapping[cand.local_idx] = global_id;

      // Update global centroid with EMA smoothing: c = norm(alpha * c + (1-alpha) * e)
      auto &prof = profiles_[cand.global_idx];
      for (size_t d = 0; d < 256; ++d) {
        prof.centroid[d] = ema_alpha_ * prof.centroid[d] + (1.0f - ema_alpha_) * local_centroids[cand.local_idx][d];
      }
      normalize_vector(prof.centroid);
      prof.count++;
      prof.last_seen_sec = window_timestamp_sec;
    }
  }

  // Any unmatched local clusters register as new global speakers
  for (size_t c = 0; c < num_local; ++c) {
    if (!local_matched[c]) {
      int new_id = static_cast<int>(profiles_.size());
      GlobalSpeakerProfile prof;
      prof.id = new_id;
      prof.name = format_speaker_name(new_id);
      prof.centroid = local_centroids[c];
      normalize_vector(prof.centroid);
      prof.count = 1;
      prof.first_seen_sec = window_timestamp_sec;
      prof.last_seen_sec = window_timestamp_sec;

      profiles_.push_back(prof);
      mapping[c] = new_id;
    }
  }

  return mapping;
}

const GlobalSpeakerProfile *OnlineSpeakerTracker::getProfile(int global_id) const {
  for (const auto &prof : profiles_) {
    if (prof.id == global_id) return &prof;
  }
  return nullptr;
}

void OnlineSpeakerTracker::reset() {
  profiles_.clear();
}

} // namespace speaker_diarization
