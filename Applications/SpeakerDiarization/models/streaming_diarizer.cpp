// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   streaming_diarizer.cpp
 * @date   12 October 2026
 * @brief  Reusable streaming speaker diarization engine (window pipeline)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "streaming_diarizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

namespace speaker_diarization {

namespace {
constexpr size_t FRAMES_PER_CHUNK = 589;
constexpr double STEP_SEC = 1.0; // chunk hop inside a window
constexpr double FRAME_STEP_SEC = 0.016875;
} // namespace

StreamingDiarizer::StreamingDiarizer(double window_sec, double step_sec,
                                     float match_threshold, float ema_alpha) :
  window_sec_(window_sec),
  step_sec_(step_sec),
  buffer_(16000, window_sec, step_sec),
  tracker_(match_threshold, ema_alpha) {}

bool StreamingDiarizer::init(const std::string &weights_path) {
  if (!loader_.load(weights_path)) {
    std::cerr << "StreamingDiarizer: failed to load weights from "
              << weights_path << "\n";
    return false;
  }
  if (!seg_model_.init(loader_)) {
    std::cerr << "StreamingDiarizer: failed to initialize PyanNet\n";
    return false;
  }
  if (!emb_model_.init(loader_)) {
    std::cerr << "StreamingDiarizer: failed to initialize WeSpeaker ResNet34\n";
    return false;
  }
  if (!plda_vbx_.init(loader_)) {
    std::cerr << "StreamingDiarizer: failed to initialize PLDA & VBx\n";
    return false;
  }
  resample_kernel_ = loader_.getTensor("frontend.resample_kernel");
  mel_banks_ = loader_.getTensor("frontend.fbank_mel_banks");
  fbank_window_ = loader_.getTensor("frontend.fbank_window");
  return true;
}

std::vector<SpeakerSegment>
StreamingDiarizer::processAudio(const std::vector<float> &audio,
                                double start_time_sec) {
  // Slice into 10 s chunks with a 1 s step
  auto chunks = preprocessor_.sliceChunks(audio);
  const size_t num_chunks = chunks.size();
  if (num_chunks == 0)
    return {};

  const size_t last_chunk_start_frame = static_cast<size_t>(
    std::round((num_chunks - 1) * STEP_SEC / FRAME_STEP_SEC));
  const size_t total_frames = last_chunk_start_frame + FRAMES_PER_CHUNK;

  // PyanNet segmentation
  std::vector<std::vector<float>> segmentations(num_chunks);
#pragma omp parallel for schedule(dynamic)
  for (size_t c = 0; c < num_chunks; ++c) {
    segmentations[c] = seg_model_.forwardChunk(chunks[c].data(), c, nullptr);
  }

  std::vector<uint8_t> speaker_counting(total_frames, 0);
  std::vector<float> frame_sums(total_frames, 0.0f);
  std::vector<float> frame_weights(total_frames, 0.0f);
  for (size_t c = 0; c < num_chunks; ++c) {
    const size_t chunk_start_frame =
      static_cast<size_t>(std::round(c * STEP_SEC / FRAME_STEP_SEC));
    for (size_t f = 0; f < FRAMES_PER_CHUNK; ++f) {
      const size_t global_frame = chunk_start_frame + f;
      if (global_frame >= total_frames)
        break;
      const float spk_sum = segmentations[c][f * 3 + 0] +
                            segmentations[c][f * 3 + 1] +
                            segmentations[c][f * 3 + 2];
      frame_sums[global_frame] += spk_sum;
      frame_weights[global_frame] += 1.0f;
    }
  }
  for (size_t f = 0; f < total_frames; ++f) {
    speaker_counting[f] =
      (frame_weights[f] > 0.0f)
        ? static_cast<uint8_t>(std::rint(frame_sums[f] / frame_weights[f]))
        : 0;
  }

  // WeSpeaker ResNet34 embeddings, one per (chunk, local speaker slot)
  std::vector<std::vector<float>> chunk_embeddings(num_chunks,
                                                   std::vector<float>(3 * 256));
#pragma omp parallel for schedule(dynamic)
  for (size_t c = 0; c < num_chunks; ++c) {
    auto fbank = preprocessor_.computeFbank(chunks[c].data(), chunks[c].size(),
                                            mel_banks_, fbank_window_);
    std::vector<float> chunk_feat(256 * 10 * 125);
    emb_model_.forwardBackbone(fbank.data(), chunk_feat.data(), c, nullptr);
    for (size_t s = 0; s < 3; ++s) {
      std::vector<float> spk_mask(FRAMES_PER_CHUNK);
      for (size_t f = 0; f < FRAMES_PER_CHUNK; ++f) {
        spk_mask[f] = segmentations[c][f * 3 + s];
      }
      auto emb = emb_model_.forwardPool(chunk_feat.data(), spk_mask.data(), c,
                                        s, nullptr);
      std::memcpy(chunk_embeddings[c].data() + s * 256, emb.data(),
                  256 * sizeof(float));
    }
  }

  std::vector<float> all_embeddings;
  all_embeddings.reserve(num_chunks * 3 * 256);
  for (size_t c = 0; c < num_chunks; ++c) {
    all_embeddings.insert(all_embeddings.end(), chunk_embeddings[c].begin(),
                          chunk_embeddings[c].end());
  }

  // Cluster inside the window and map to global speaker ids
  return plda_vbx_.diarizeStreaming(all_embeddings, segmentations,
                                    speaker_counting, total_frames,
                                    start_time_sec, tracker_);
}

StreamingResult StreamingDiarizer::push(const float *pcm, size_t n) {
  StreamingResult result;
  if (finished_ || n == 0)
    return result;

  buffer_.pushSamples(pcm, n);

  while (buffer_.hasWindow()) {
    std::vector<float> window_audio;
    double window_start = 0.0;
    if (!buffer_.getWindow(window_audio, window_start))
      break;

    const auto t0 = std::chrono::high_resolution_clock::now();
    auto window_segments = processAudio(window_audio, window_start);
    const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0)
                        .count();
    total_model_ms_ += ms;
    result.model_ms += ms;
    ++result.windows_processed;

    // Commit the turns that end before the next window takes over
    const double commit_horizon = window_start + step_sec_;
    for (const auto &seg : window_segments) {
      const float last_t = last_confirmed_[seg.speaker];
      if (seg.start_time < last_t)
        continue;
      if (seg.end_time <= commit_horizon + 0.1f) {
        result.segments.push_back(seg);
        last_confirmed_[seg.speaker] = seg.end_time;
      } else if (seg.start_time < commit_horizon) {
        // The turn runs past the commit point and the next window starts there,
        // so its first part would be lost. Commit that part now; the next
        // window reports the rest.
        SpeakerSegment head = seg;
        head.end_time = static_cast<float>(commit_horizon);
        head.duration = head.end_time - head.start_time;
        result.segments.push_back(head);
        last_confirmed_[seg.speaker] = head.end_time;
      }
    }
    result.committed_until_sec = commit_horizon;
    buffer_.advanceStep();
  }
  return result;
}

StreamingResult StreamingDiarizer::finish() {
  StreamingResult result;
  if (finished_)
    return result;
  finished_ = true;

  std::vector<float> remaining;
  double remaining_start = 0.0;
  if (buffer_.flushRemaining(remaining, remaining_start)) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    auto final_segments = processAudio(remaining, remaining_start);
    const double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0)
                        .count();
    total_model_ms_ += ms;
    result.model_ms += ms;
    if (!final_segments.empty())
      ++result.windows_processed;

    for (const auto &seg : final_segments) {
      const float last_t = last_confirmed_[seg.speaker];
      if (seg.start_time >= last_t) {
        result.segments.push_back(seg);
        last_confirmed_[seg.speaker] = seg.end_time;
      }
    }
  }
  result.committed_until_sec = std::numeric_limits<double>::infinity();
  return result;
}

} // namespace speaker_diarization
