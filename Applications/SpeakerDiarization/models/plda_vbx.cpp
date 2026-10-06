// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   plda_vbx.cpp
 * @date   29 September 2026
 * @brief  PLDA transformation, AHC/VBx clustering, and timeline reconstruction in NNTrainer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "plda_vbx.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <unordered_set>

namespace speaker_diarization {

bool PldaVBx::init(const WeightLoader &loader) {
  mean1_ = loader.getTensor("xvec.mean1");
  lda_   = loader.getTensor("xvec.lda");
  mean2_ = loader.getTensor("xvec.mean2");

  mu_    = loader.getTensor("plda.mu");
  tr_    = loader.getTensor("plda.tr");
  psi_   = loader.getTensor("plda.psi");

  return (mean1_ && lda_ && mean2_ && mu_ && tr_ && psi_);
}

void PldaVBx::transformPLDA(const float *in_embedding, float *out_plda_fea) {
  // 1. Centering with mean1: [256]
  float x1[256];
  for (size_t i = 0; i < 256; ++i) {
    x1[i] = in_embedding[i] - mean1_[i];
  }

  // 2. LDA projection: [256] * [256, 128] -> [128]
  float y[128] = {0.0f};
  for (size_t j = 0; j < 128; ++j) {
    float acc = 0.0f;
    for (size_t i = 0; i < 256; ++i) {
      acc += x1[i] * lda_[i * 128 + j];
    }
    y[j] = acc;
  }

  // 3. Length normalization & Centering with mean2
  float norm2 = 1e-8f;
  for (size_t j = 0; j < 128; ++j) {
    norm2 += y[j] * y[j];
  }
  norm2 = std::sqrt(norm2);

  float z[128];
  float scale_128 = std::sqrt(128.0f);
  for (size_t j = 0; j < 128; ++j) {
    z[j] = scale_128 * (y[j] / norm2) - mean2_[j];
  }

  // 4. PLDA projection: (z - mu) . tr^T
  for (size_t j = 0; j < 128; ++j) {
    float acc = 0.0f;
    for (size_t k = 0; k < 128; ++k) {
      acc += (z[k] - mu_[k]) * tr_[j * 128 + k];
    }
    out_plda_fea[j] = acc;
  }
}

static inline float dot256(const float *a, const float *b) {
  float acc = 0.0f;
  #pragma omp simd reduction(+:acc)
  for (size_t i = 0; i < 256; ++i) {
    acc += a[i] * b[i];
  }
  return acc;
}

static inline void normalize256(float *vec) {
  float norm = 1e-8f;
  #pragma omp simd reduction(+:norm)
  for (size_t i = 0; i < 256; ++i) {
    norm += vec[i] * vec[i];
  }
  norm = std::sqrt(norm);
  float inv_norm = 1.0f / norm;
  #pragma omp simd
  for (size_t i = 0; i < 256; ++i) {
    vec[i] *= inv_norm;
  }
}

std::vector<SpeakerSegment> PldaVBx::diarize(
    const std::vector<float> &embeddings,
    const std::vector<std::vector<float>> &segmentations,
    const std::vector<uint8_t> &speaker_counting,
    size_t total_frames) {

  size_t num_chunks = segmentations.size();
  if (num_chunks == 0) return {};

  constexpr size_t FRAMES_PER_CHUNK = 589;
  constexpr double STEP_SEC = 1.0;
  constexpr double FRAME_STEP_SEC = 0.016875;

  if (total_frames == 0) {
    size_t last_chunk_start = static_cast<size_t>(std::round((num_chunks - 1) * STEP_SEC / FRAME_STEP_SEC));
    total_frames = last_chunk_start + FRAMES_PER_CHUNK;
  }

  // 1. Identify active tracks (speaker channel in chunk with non-zero activity)
  struct ActiveTrack {
    size_t chunk_idx;
    size_t spk_idx;
    std::vector<float> emb; // 256-dim unit vector
  };
  std::vector<ActiveTrack> active_tracks;

  for (size_t c = 0; c < num_chunks; ++c) {
    for (size_t s = 0; s < 3; ++s) {
      float total_active = 0.0f;
      for (size_t f = 0; f < FRAMES_PER_CHUNK; ++f) {
        total_active += segmentations[c][f * 3 + s];
      }
      if (total_active > 1.0f) {
        const float *raw_emb = embeddings.data() + (c * 3 + s) * 256;
        std::vector<float> unit_emb(raw_emb, raw_emb + 256);
        normalize256(unit_emb.data());
        active_tracks.push_back({c, s, std::move(unit_emb)});
      }
    }
  }

  if (active_tracks.empty()) {
    return {};
  }

  size_t num_clusters = 1;
  std::vector<std::vector<float>> centroids;

  if (active_tracks.size() == 1) {
    num_clusters = 1;
    centroids.push_back(active_tracks[0].emb);
  } else {
    // 2. Agglomerative Hierarchical Clustering (AHC) with centroid linkage
    struct Cluster {
      int id;
      std::vector<size_t> track_indices;
      std::vector<float> centroid; // 256-dim unit vector
    };

    std::vector<Cluster> clusters;
    clusters.reserve(active_tracks.size());
    for (size_t i = 0; i < active_tracks.size(); ++i) {
      clusters.push_back({static_cast<int>(i), {i}, active_tracks[i].emb});
    }

    // Distance threshold: 0.78 Euclidean distance on unit-normalized embeddings
    constexpr float DIST_THRESHOLD = 0.78f;

    while (clusters.size() > 1) {
      float min_dist = 1e9f;
      size_t best_i = 0, best_j = 1;

      for (size_t i = 0; i < clusters.size(); ++i) {
        for (size_t j = i + 1; j < clusters.size(); ++j) {
          float cos_sim = dot256(clusters[i].centroid.data(), clusters[j].centroid.data());
          float dist = std::sqrt(std::max(0.0f, 2.0f * (1.0f - cos_sim)));
          if (dist < min_dist) {
            min_dist = dist;
            best_i = i;
            best_j = j;
          }
        }
      }

      if (min_dist > DIST_THRESHOLD) {
        break; // All clusters are farther than threshold
      }

      // Merge cluster best_j into best_i
      clusters[best_i].track_indices.insert(
          clusters[best_i].track_indices.end(),
          clusters[best_j].track_indices.begin(),
          clusters[best_j].track_indices.end());

      // Recompute normalized centroid of merged cluster
      std::fill(clusters[best_i].centroid.begin(), clusters[best_i].centroid.end(), 0.0f);
      for (size_t idx : clusters[best_i].track_indices) {
        for (size_t d = 0; d < 256; ++d) {
          clusters[best_i].centroid[d] += active_tracks[idx].emb[d];
        }
      }
      normalize256(clusters[best_i].centroid.data());

      clusters.erase(clusters.begin() + best_j);
    }

    num_clusters = clusters.size();
    centroids.resize(num_clusters);
    for (size_t k = 0; k < num_clusters; ++k) {
      centroids[k] = clusters[k].centroid;
    }
  }

  // 3. Chunk-level constrained assignment (assign tracks to closest centroid enforcing distinct speakers per chunk)
  std::vector<std::vector<int>> hard_clusters(num_chunks, std::vector<int>(3, -2));

  for (size_t c = 0; c < num_chunks; ++c) {
    std::vector<size_t> active_s;
    for (size_t s = 0; s < 3; ++s) {
      float act = 0.0f;
      for (size_t f = 0; f < FRAMES_PER_CHUNK; ++f) {
        act += segmentations[c][f * 3 + s];
      }
      if (act > 1.0f) {
        active_s.push_back(s);
      }
    }
    if (active_s.empty()) continue;

    // Build similarity pairs (sim, s, k)
    struct PairMatch {
      float sim;
      size_t s;
      size_t k;
    };
    std::vector<PairMatch> matches;
    for (size_t s : active_s) {
      const float *emb = embeddings.data() + (c * 3 + s) * 256;
      std::vector<float> u_emb(emb, emb + 256);
      normalize256(u_emb.data());
      for (size_t k = 0; k < num_clusters; ++k) {
        float sim = dot256(u_emb.data(), centroids[k].data());
        matches.push_back({sim, s, k});
      }
    }

    std::sort(matches.begin(), matches.end(), [](const PairMatch &a, const PairMatch &b) {
      return a.sim > b.sim;
    });

    std::unordered_set<size_t> used_s;
    std::unordered_set<size_t> used_k;
    for (const auto &m : matches) {
      if (used_s.find(m.s) == used_s.end() && used_k.find(m.k) == used_k.end()) {
        hard_clusters[c][m.s] = static_cast<int>(m.k);
        used_s.insert(m.s);
        used_k.insert(m.k);
      }
    }
    // Fallback if clusters exhausted
    for (size_t s : active_s) {
      if (hard_clusters[c][s] == -2) {
        hard_clusters[c][s] = 0;
      }
    }
  }

  // 4. Accumulate cluster activations across overlapping chunks
  std::vector<std::vector<float>> activations(total_frames, std::vector<float>(num_clusters, 0.0f));

  for (size_t c = 0; c < num_chunks; ++c) {
    size_t chunk_start_frame = static_cast<size_t>(std::round(c * STEP_SEC / FRAME_STEP_SEC));
    for (size_t k = 0; k < num_clusters; ++k) {
      std::vector<size_t> matched_s;
      for (size_t s = 0; s < 3; ++s) {
        if (hard_clusters[c][s] == static_cast<int>(k)) {
          matched_s.push_back(s);
        }
      }
      if (matched_s.empty()) continue;

      for (size_t f = 0; f < FRAMES_PER_CHUNK; ++f) {
        size_t gf = chunk_start_frame + f;
        if (gf >= total_frames) break;

        float max_val = 0.0f;
        for (size_t s : matched_s) {
          max_val = std::max(max_val, segmentations[c][f * 3 + s]);
        }
        activations[gf][k] += max_val;
      }
    }
  }

  // 5. Build top-c discrete diarization
  std::vector<std::vector<float>> diarization(total_frames, std::vector<float>(num_clusters, 0.0f));

  for (size_t t = 0; t < total_frames; ++t) {
    uint8_t count = (t < speaker_counting.size()) ? speaker_counting[t] : 0;
    if (count == 0) continue;

    if (num_clusters == 1) {
      diarization[t][0] = 1.0f;
    } else {
      std::vector<std::pair<float, size_t>> scored_clusters(num_clusters);
      for (size_t k = 0; k < num_clusters; ++k) {
        scored_clusters[k] = {activations[t][k], k};
      }
      std::sort(scored_clusters.begin(), scored_clusters.end(),
                [](const auto &a, const auto &b) { return a.first > b.first; });

      size_t pick = std::min(static_cast<size_t>(count), num_clusters);
      for (size_t i = 0; i < pick; ++i) {
        diarization[t][scored_clusters[i].second] = 1.0f;
      }
    }
  }

  // 6. Extract continuous speech segments with sub-second timestamps
  std::vector<SpeakerSegment> segments;

  for (size_t k = 0; k < num_clusters; ++k) {
    bool is_active = false;
    float start_time = 0.0f;
    std::string label = "SPEAKER_" + (k < 10 ? std::string("0") : "") + std::to_string(k);

    for (size_t t = 0; t < total_frames; ++t) {
      float ts = t * static_cast<float>(FRAME_STEP_SEC) + 0.03096875f;
      if (!is_active && diarization[t][k] > 0.5f) {
        is_active = true;
        start_time = ts;
      } else if (is_active && diarization[t][k] < 0.5f) {
        is_active = false;
        float end_time = ts;
        if (end_time - start_time > 0.05f) {
          segments.push_back({label, start_time, end_time, end_time - start_time});
        }
      }
    }
    if (is_active) {
      float end_time = (total_frames - 1) * static_cast<float>(FRAME_STEP_SEC) + 0.03096875f;
      if (end_time - start_time > 0.05f) {
        segments.push_back({label, start_time, end_time, end_time - start_time});
      }
    }
  }

  // Sort segments chronologically
  std::sort(segments.begin(), segments.end(), [](const SpeakerSegment &a, const SpeakerSegment &b) {
    if (std::abs(a.start_time - b.start_time) > 1e-4f) {
      return a.start_time < b.start_time;
    }
    return a.end_time < b.end_time;
  });

  return segments;
}

} // namespace speaker_diarization
