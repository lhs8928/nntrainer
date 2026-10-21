// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   batch_diarizer.cpp
 * @date   21 October 2026
 * @brief  Whole-recording speaker diarization and voice activity scores
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "batch_diarizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace speaker_diarization {

namespace {
constexpr size_t CHUNK_SAMPLES = 160000; // 10 s
constexpr size_t FRAMES_PER_CHUNK = 589;
constexpr double FRAME_STEP_SEC = 0.016875;

double msSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
           std::chrono::steady_clock::now() - t0)
    .count();
}

/** @brief 10 s chunks every `step` samples; the last one is zero-padded */
std::vector<std::vector<float>> sliceEvery(const std::vector<float> &audio,
                                           size_t step) {
  std::vector<std::vector<float>> chunks;
  if (audio.empty())
    return chunks;
  for (size_t start = 0;; start += step) {
    std::vector<float> chunk(CHUNK_SAMPLES, 0.0f);
    const size_t n = std::min(CHUNK_SAMPLES, audio.size() - start);
    std::memcpy(chunk.data(), audio.data() + start, n * sizeof(float));
    chunks.push_back(std::move(chunk));
    if (start + CHUNK_SAMPLES >= audio.size())
      break;
  }
  return chunks;
}
} // namespace

bool BatchDiarizer::init(const std::string &weights_path) {
  if (!loader_.load(weights_path)) {
    std::cerr << "BatchDiarizer: failed to load weights from " << weights_path
              << "\n";
    return false;
  }
  if (!seg_model_.init(loader_)) {
    std::cerr << "BatchDiarizer: failed to initialize PyanNet\n";
    return false;
  }
  if (!emb_model_.init(loader_)) {
    std::cerr << "BatchDiarizer: failed to initialize WeSpeaker ResNet34\n";
    return false;
  }
  if (!plda_vbx_.init(loader_)) {
    std::cerr << "BatchDiarizer: failed to initialize PLDA & VBx\n";
    return false;
  }
  mel_banks_ = loader_.getTensor("frontend.fbank_mel_banks");
  fbank_window_ = loader_.getTensor("frontend.fbank_window");
  return true;
}

BatchResult BatchDiarizer::run(const std::vector<float> &audio, bool diarize,
                               double vad_step_sec) {
  BatchResult r;
  r.frame_step_sec = FRAME_STEP_SEC;
  // The clustering assumes the 1 s hop of the SD application.
  const double step_sec = diarize ? 1.0 : std::max(0.1, vad_step_sec);
  const size_t step = static_cast<size_t>(step_sec * 16000.0 + 0.5);
  auto chunks =
    diarize ? preprocessor_.sliceChunks(audio) : sliceEvery(audio, step);
  const size_t num_chunks = chunks.size();
  r.chunks = num_chunks;
  if (num_chunks == 0)
    return r;

  const size_t last_chunk_start_frame = static_cast<size_t>(
    std::round((num_chunks - 1) * step_sec / FRAME_STEP_SEC));
  const size_t total_frames = last_chunk_start_frame + FRAMES_PER_CHUNK;

  // PyanNet: speaker activity and voice activity of every chunk
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::vector<float>> segmentations(num_chunks);
  std::vector<std::vector<float>> speech(num_chunks,
                                         std::vector<float>(FRAMES_PER_CHUNK));
#pragma omp parallel for schedule(dynamic)
  for (size_t c = 0; c < num_chunks; ++c) {
    segmentations[c] =
      seg_model_.forwardChunk(chunks[c].data(), c, nullptr, speech[c].data());
  }

  // Overlapping chunks: average each frame over the chunks that cover it
  std::vector<float> speech_sum(total_frames, 0.0f);
  std::vector<float> count_sum(total_frames, 0.0f);
  std::vector<float> weights(total_frames, 0.0f);
  for (size_t c = 0; c < num_chunks; ++c) {
    const size_t first =
      static_cast<size_t>(std::round(c * step_sec / FRAME_STEP_SEC));
    for (size_t f = 0; f < FRAMES_PER_CHUNK && first + f < total_frames; ++f) {
      speech_sum[first + f] += speech[c][f];
      count_sum[first + f] += segmentations[c][f * 3 + 0] +
                              segmentations[c][f * 3 + 1] +
                              segmentations[c][f * 3 + 2];
      weights[first + f] += 1.0f;
    }
  }
  // Frames past the end of the audio only cover zero padding
  const size_t audio_frames = std::min(
    total_frames,
    static_cast<size_t>(std::ceil(audio.size() / 16000.0 / FRAME_STEP_SEC)));
  r.speech.resize(audio_frames);
  for (size_t f = 0; f < audio_frames; ++f)
    r.speech[f] = weights[f] > 0.0f ? speech_sum[f] / weights[f] : 0.0f;
  r.vad_ms = msSince(t0);
  if (!diarize)
    return r;

  std::vector<uint8_t> speaker_counting(total_frames, 0);
  for (size_t f = 0; f < total_frames; ++f) {
    speaker_counting[f] =
      weights[f] > 0.0f
        ? static_cast<uint8_t>(std::rint(count_sum[f] / weights[f]))
        : 0;
  }

  // WeSpeaker ResNet34 embeddings, one per (chunk, local speaker slot)
  t0 = std::chrono::steady_clock::now();
  std::vector<std::vector<float>> chunk_embeddings(num_chunks,
                                                   std::vector<float>(3 * 256));
#pragma omp parallel for schedule(dynamic)
  for (size_t c = 0; c < num_chunks; ++c) {
    auto fbank = preprocessor_.computeFbank(chunks[c].data(), chunks[c].size(),
                                            mel_banks_, fbank_window_);
    std::vector<float> chunk_feat(256 * 10 * 125);
    emb_model_.forwardBackbone(fbank.data(), chunk_feat.data(), c, nullptr);
    for (size_t s = 0; s < 3; ++s) {
      emb_model_.forwardPool(chunk_feat.data(), segmentations[c].data() + s,
                             chunk_embeddings[c].data() + s * 256,
                             /*mask_stride=*/3, c, s, nullptr);
    }
  }
  std::vector<float> all_embeddings;
  all_embeddings.reserve(num_chunks * 3 * 256);
  for (const auto &e : chunk_embeddings)
    all_embeddings.insert(all_embeddings.end(), e.begin(), e.end());

  // Cluster the whole recording at once: speaker labels are global
  r.segments = plda_vbx_.diarize(all_embeddings, segmentations,
                                 speaker_counting, total_frames);
  r.embed_ms = msSince(t0);
  return r;
}

std::vector<float> BatchDiarizer::embed(const std::vector<float> &audio,
                                        double start_sec, double end_sec) {
  // A 10 s window centred on the span (clamped to the recording)
  const double total = audio.size() / 16000.0;
  const double mid = (start_sec + end_sec) / 2;
  const double win_start =
    std::max(0.0, std::min(mid - 5.0, std::max(0.0, total - 10.0)));
  const size_t first = static_cast<size_t>(win_start * 16000.0);
  std::vector<float> chunk(CHUNK_SAMPLES, 0.0f);
  const size_t n = std::min(CHUNK_SAMPLES, audio.size() - first);
  std::memcpy(chunk.data(), audio.data() + first, n * sizeof(float));

  // Pool over the frames that belong to the span
  std::vector<float> mask(FRAMES_PER_CHUNK, 0.0f);
  for (size_t f = 0; f < FRAMES_PER_CHUNK; ++f) {
    const double t = win_start + f * FRAME_STEP_SEC;
    mask[f] = (t >= start_sec && t < end_sec) ? 1.0f : 0.0f;
  }
  auto fbank = preprocessor_.computeFbank(chunk.data(), chunk.size(),
                                          mel_banks_, fbank_window_);
  std::vector<float> feat(256 * 10 * 125);
  emb_model_.forwardBackbone(fbank.data(), feat.data(), 0, nullptr);
  std::vector<float> emb(256);
  emb_model_.forwardPool(feat.data(), mask.data(), emb.data(),
                         /*mask_stride=*/1, 0, 0, nullptr);
  return emb;
}

} // namespace speaker_diarization
