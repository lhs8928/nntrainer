// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   online_speaker_tracker.h
 * @date   12 October 2026
 * @brief  Online Global Speaker Registry and Centroid Tracker for Continuous Audio Streaming
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __ONLINE_SPEAKER_TRACKER_H__
#define __ONLINE_SPEAKER_TRACKER_H__

#include <vector>
#include <string>
#include <memory>

namespace speaker_diarization {

/**
 * @brief Persistent profile for a globally tracked speaker.
 */
struct GlobalSpeakerProfile {
  int id;                     /**< Global numeric speaker ID (0, 1, 2, ...) */
  std::string name;           /**< Display name (e.g. "SPEAKER_00") */
  std::vector<float> centroid;/**< Running unit-normalized 256-dim centroid */
  size_t count;               /**< Number of speech chunks contributing to profile */
  double first_seen_sec;      /**< Timestamp of first speech detection */
  double last_seen_sec;       /**< Timestamp of most recent speech detection */
};

/**
 * @brief Online speaker registry tracking speakers across streaming windows without label permutation flips.
 */
class OnlineSpeakerTracker {
public:
  /**
   * @brief Constructor.
   * @param match_threshold Minimum cosine similarity to match an existing global speaker (default 0.65).
   * @param ema_alpha Smoothing factor for centroid update: c = normalize(alpha * c + (1-alpha) * e).
   */
  explicit OnlineSpeakerTracker(float match_threshold = 0.65f, float ema_alpha = 0.85f);
  ~OnlineSpeakerTracker() = default;

  /**
   * @brief Match a set of local cluster centroids in the current window to global speaker profiles.
   * @param local_centroids Local centroids for each cluster found in the current window.
   * @param window_timestamp_sec Current audio window start timestamp in seconds.
   * @return Vector mapping local cluster index [0..num_local-1] to global speaker ID.
   */
  std::vector<int> matchClusters(const std::vector<std::vector<float>> &local_centroids,
                                double window_timestamp_sec = 0.0);

  /**
   * @brief Get global speaker profile by global ID.
   */
  const GlobalSpeakerProfile *getProfile(int global_id) const;

  /**
   * @brief Get all global speaker profiles.
   */
  const std::vector<GlobalSpeakerProfile> &getProfiles() const { return profiles_; }

  /**
   * @brief Get total number of unique global speakers discovered so far.
   */
  size_t numSpeakers() const { return profiles_.size(); }

  /**
   * @brief Reset tracker state.
   */
  void reset();

private:
  float match_threshold_;
  float ema_alpha_;
  std::vector<GlobalSpeakerProfile> profiles_;
};

} // namespace speaker_diarization

#endif // __ONLINE_SPEAKER_TRACKER_H__
