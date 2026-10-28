// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   batch_diarizer.h
 * @date   21 October 2026
 * @brief  Whole-recording speaker diarization and voice activity scores
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The offline pipeline of the SD application (10 s chunks every 1 s ->
 * PyanNet -> WeSpeaker embeddings -> PLDA + VBx clustering over the whole
 * recording) as a reusable class. Every run also returns the frame-level
 * probability that someone speaks, so the same PyanNet pass doubles as a voice
 * activity detector.
 */

#ifndef __BATCH_DIARIZER_H__
#define __BATCH_DIARIZER_H__

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "audio_preprocessor.h"
#include "eres2netv2.h"
#include "plda_vbx.h"
#include "pyannet.h"
#include "weight_loader.h"
#include "wespeaker_resnet34.h"

namespace speaker_diarization {

/**
 * @brief What BatchDiarizer::run() produced.
 */
struct BatchResult {
  /** Speaker turns of the whole recording; empty without diarization. */
  std::vector<SpeakerSegment> segments;
  /**
   * Probability that anyone speaks, one value per PyanNet frame, averaged over
   * the chunks that cover the frame. Frame i starts at i * frame_step_sec.
   */
  std::vector<float> speech;
  double frame_step_sec = 0.0;
  size_t chunks = 0;     // PyanNet chunks processed
  double vad_ms = 0.0;   // PyanNet time
  double embed_ms = 0.0; // WeSpeaker + clustering time
};

/**
 * @brief Diarizes a complete recording at once. Not thread-safe: use one
 *        instance from one thread. The models run in the calling thread and
 *        in the OpenMP team it creates.
 */
class BatchDiarizer {
public:
  /** @brief load weights and initialize all models; false on failure */
  bool init(const std::string &weights_path);

  /**
   * @brief run the models over 16 kHz mono samples
   * @param diarize also find the speakers; otherwise only the voice activity
   * @param vad_step_sec chunk hop when only the voice activity is needed.
   *        Diarization always uses the 1 s hop the clustering expects.
   */
  BatchResult run(const std::vector<float> &audio, bool diarize,
                  double vad_step_sec = 1.0);

  /**
   * @brief embed() with ERes2NetV2 (the ALICE server's speaker model) instead
   *        of WeSpeaker; run() keeps WeSpeaker, whose embeddings its PLDA was
   *        trained on
   * @param weights_path speaker_eres2netv2.bin (export_eres2netv2.py)
   */
  bool useERes2NetV2(const std::string &weights_path);

  /** @brief "wespeaker" or "eres2netv2": the model embed() uses */
  const char *speakerModel() const;

  /** @brief numbers in one embed() result: 256 (WeSpeaker) or 192 */
  size_t embeddingDim() const;

  /**
   * @brief speaker embedding of the audio in [start_sec, end_sec).
   *        WeSpeaker: over a 10 s window of the span alone (a shorter span is
   *        repeated to fill it, a longer one gives its middle 10 s).
   *        ERes2NetV2: over the span as it is (the server's front end)
   */
  std::vector<float> embed(const std::vector<float> &audio, double start_sec,
                           double end_sec);

  /**
   * @brief PyanNet over consecutive (not overlapping) 10 s windows: for every
   *        window the powerset class of each of its 589 frames (0 = nobody,
   *        1-3 = one local speaker, 4-6 = two). The last window is zero-padded;
   *        one shorter than 0.1 s is left out.
   */
  std::vector<std::vector<uint8_t>> segment(const std::vector<float> &audio);

private:
  WeightLoader loader_;
  AudioPreprocessor preprocessor_;
  PyanNet seg_model_;
  WeSpeakerResNet34 emb_model_;
  PldaVBx plda_vbx_;
  const float *mel_banks_ = nullptr;
  const float *fbank_window_ = nullptr;
  WeightLoader eres2_loader_;
  ERes2NetV2 eres2_;
  bool use_eres2_ = false;
};

} // namespace speaker_diarization

#endif // __BATCH_DIARIZER_H__
