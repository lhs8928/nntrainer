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
#include <stdexcept>

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
  return init(weights_path, weights_path);
}

bool BatchDiarizer::init(const std::string &segmentation_path,
                         const std::string &speaker_path) {
  if (!loader_.load(segmentation_path)) {
    std::cerr << "BatchDiarizer: failed to load weights from "
              << segmentation_path << "\n";
    return false;
  }
  if (!seg_model_.init(loader_)) {
    std::cerr << "BatchDiarizer: failed to initialize PyanNet\n";
    return false;
  }
  if (!speaker_loader_.load(speaker_path)) {
    std::cerr << "BatchDiarizer: failed to load weights from " << speaker_path
              << "\n";
    return false;
  }
  // the voice model is told by its tensors: ERes2NetV2's are named by the
  // 3D-Speaker module path, WeSpeaker's start with "resnet."
  use_eres2_ = speaker_loader_.hasTensor("layer1.0.conv1.weight");
  if (use_eres2_) {
    std::string missing;
    if (!eres2_.init(speaker_loader_, &missing)) {
      std::cerr << "BatchDiarizer: ERes2NetV2 weights lack " << missing << "\n";
      return false;
    }
    has_plda_ = false;
    return true;
  }
  if (!speaker_loader_.hasTensor("resnet.conv1.weight") ||
      !emb_model_.init(speaker_loader_)) {
    std::cerr << "BatchDiarizer: no voice model (WeSpeaker ResNet34 or "
                 "ERes2NetV2) in "
              << speaker_path << "\n";
    return false;
  }
  mel_banks_ = speaker_loader_.getTensor("frontend.fbank_mel_banks");
  fbank_window_ = speaker_loader_.getTensor("frontend.fbank_window");
  has_plda_ = speaker_loader_.hasTensor("plda.mu");
  if (has_plda_ && !plda_vbx_.init(speaker_loader_)) {
    std::cerr << "BatchDiarizer: failed to initialize PLDA & VBx\n";
    return false;
  }
  return true;
}

BatchResult BatchDiarizer::run(const std::vector<float> &audio, bool diarize,
                               double vad_step_sec) {
  if (diarize && !has_plda_)
    throw std::runtime_error("whole-recording diarization needs the WeSpeaker "
                             "voice model and its PLDA");
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

const char *BatchDiarizer::speakerModel() const {
  return use_eres2_ ? "eres2netv2" : "wespeaker";
}

std::string BatchDiarizer::modelManifest() const {
  std::vector<uint32_t> dims;
  if (!speaker_loader_.hasTensor("meta.model_manifest"))
    return "";
  const int8_t *bytes =
    speaker_loader_.getTensorI8("meta.model_manifest", &dims);
  if (!bytes || dims.size() != 1)
    return "";
  return std::string(reinterpret_cast<const char *>(bytes), dims[0]);
}

size_t BatchDiarizer::embeddingDim() const {
  return use_eres2_ ? ERes2NetV2::DIM : 256;
}

std::vector<float> BatchDiarizer::embed(const std::vector<float> &audio,
                                        double start_sec, double end_sec) {
  if (use_eres2_) {
    const size_t a = std::min(
      audio.size(), static_cast<size_t>(std::max(0.0, start_sec) * 16000.0));
    const size_t b = std::min(
      audio.size(),
      std::max(a, static_cast<size_t>(std::max(0.0, end_sec) * 16000.0)));
    return eres2_.embed(audio.data() + a, b - a);
  }
  // The fbank is mean-normalized over the whole 10 s window, so the window must
  // hold the span's speech only: zero padding around a short span drowns its
  // features in silence. A span shorter than the window is repeated to fill it,
  // a longer one gives its middle 10 s.
  const size_t a = std::min(
    audio.size(), static_cast<size_t>(std::max(0.0, start_sec) * 16000.0));
  const size_t b = std::min(
    audio.size(),
    std::max(a, static_cast<size_t>(std::max(0.0, end_sec) * 16000.0)));
  std::vector<float> chunk(CHUNK_SAMPLES, 0.0f);
  const size_t n = b - a;
  if (n >= CHUNK_SAMPLES) {
    std::memcpy(chunk.data(), audio.data() + a + (n - CHUNK_SAMPLES) / 2,
                CHUNK_SAMPLES * sizeof(float));
  } else if (n > 0) {
    for (size_t k = 0; k < CHUNK_SAMPLES; k += n)
      std::memcpy(chunk.data() + k, audio.data() + a,
                  std::min(n, CHUNK_SAMPLES - k) * sizeof(float));
  }
  auto fbank = preprocessor_.computeFbank(chunk.data(), chunk.size(),
                                          mel_banks_, fbank_window_);
  std::vector<float> feat(256 * 10 * 125);
  emb_model_.forwardBackbone(fbank.data(), feat.data(), 0, nullptr);
  std::vector<float> emb(256);
  emb_model_.forwardPool(feat.data(), nullptr, emb.data(), /*mask_stride=*/1, 0,
                         0, nullptr);
  return emb;
}

std::vector<std::vector<uint8_t>>
BatchDiarizer::segment(const std::vector<float> &audio) {
  constexpr size_t MIN_SAMPLES = 1600;
  // PyanNet returns the multi-label form of the winning powerset class;
  // speakers (a, b, c) map back to the class index uniquely.
  auto powerset = [](const float *f) -> uint8_t {
    const int a = f[0] > 0.5f, b = f[1] > 0.5f, c = f[2] > 0.5f;
    if (a + b + c == 0)
      return 0;
    if (a + b + c == 1)
      return a ? 1 : b ? 2 : 3;
    return a && b ? 4 : a && c ? 5 : 6;
  };
  std::vector<std::vector<uint8_t>> out;
  for (size_t start = 0; start < audio.size(); start += CHUNK_SAMPLES) {
    const size_t n = std::min(CHUNK_SAMPLES, audio.size() - start);
    if (n < MIN_SAMPLES)
      break;
    std::vector<float> chunk(CHUNK_SAMPLES, 0.0f);
    std::memcpy(chunk.data(), audio.data() + start, n * sizeof(float));
    const auto multi = seg_model_.forwardChunk(chunk.data(), 0, nullptr);
    std::vector<uint8_t> classes(FRAMES_PER_CHUNK);
    for (size_t f = 0; f < FRAMES_PER_CHUNK; ++f)
      classes[f] = powerset(multi.data() + f * 3);
    out.push_back(std::move(classes));
  }
  return out;
}

} // namespace speaker_diarization
