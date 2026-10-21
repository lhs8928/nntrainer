// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   batch_pipeline.h
 * @date   21 October 2026
 * @brief  WhisperX-style transcription of a complete recording
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Follows the steps of WhisperX (m-bain/whisperX) for a whole file:
 *
 *   1. voice activity: PyanNet "someone speaks" probability per frame,
 *      binarized with hysteresis (onset 0.5 / offset 0.363); a region longer
 *      than the chunk size is cut at its lowest score (WhisperX min-cut)
 *   2. merge consecutive speech regions into chunks of at most chunk_sec
 *   3. transcribe every chunk on its own (no text carried between chunks)
 *   4. split each chunk text into sentences, timed by the word timestamps
 *   5. diarize the whole recording once and give every sentence the speaker
 *      whose turns overlap it the longest (no overlap: no speaker)
 *
 * Two variants from another on-device project: --pause-sec ends a chunk at
 * every pause of that length (instead of merging up to the chunk size), and
 * --line-speakers splits sentences where the diarization changes speaker, takes
 * one speaker embedding per line and clusters all lines of the recording.
 */

#ifndef __DISCRETION_BATCH_PIPELINE_H__
#define __DISCRETION_BATCH_PIPELINE_H__

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace speaker_diarization {
class BatchDiarizer;
}

namespace discretion {

class AsrEngine;

struct BatchOptions {
  double chunk_sec = 30.0;     // longest chunk (WhisperX chunk_size)
  float onset = 0.5f;          // WhisperX vad_onset
  float offset = 0.363f;       // WhisperX vad_offset
  double vad_step_sec = 1.0;   // PyanNet hop when only voice activity is needed
  double pause_sec = 0.0;      // > 0: a pause this long always ends a chunk
  bool line_speakers = false;  // speakers by clustering per-line embeddings
  float line_threshold = 0.5f; // cosine similarity that joins two clusters
  bool verbose = false; // print regions, chunks and voice activity to stderr
};

/** @brief One sentence of the transcript. */
struct BatchLine {
  double start = 0.0, end = 0.0;
  std::string speaker; // empty: no speaker (not diarized or no overlap)
  std::string text;
  std::vector<std::pair<std::string, double>> words; // absolute word times
};

struct BatchStats {
  double audio_sec = 0.0;
  size_t regions = 0, chunks = 0;
  size_t retries = 0; // chunks decoded again as two halves
  double vad_sec = 0.0, sd_sec = 0.0, asr_sec = 0.0;
  size_t speakers = 0;
};

/**
 * @brief Runs the steps above with an ASR engine and a diarizer that stay
 *        loaded between recordings.
 */
class BatchPipeline {
public:
  /**
   * @param engine ASR model built for chunks of opt.chunk_sec seconds
   * @param diarizer PyanNet / WeSpeaker models (also used for voice activity)
   */
  BatchPipeline(std::shared_ptr<AsrEngine> engine,
                std::shared_ptr<speaker_diarization::BatchDiarizer> diarizer,
                const BatchOptions &opt);

  /**
   * @brief load the diarization weights and build a pipeline around `engine`
   * @return nullptr (and `error` set) when the weights cannot be loaded
   */
  static std::unique_ptr<BatchPipeline>
  create(std::shared_ptr<AsrEngine> engine, const std::string &sd_weights,
         const BatchOptions &opt, std::string &error);

  /**
   * @brief transcribe 16 kHz mono samples
   * @param diarize label sentences with speakers
   */
  std::vector<BatchLine> run(const std::vector<float> &pcm, bool diarize,
                             BatchStats *stats = nullptr);

private:
  std::shared_ptr<AsrEngine> engine_;
  std::shared_ptr<speaker_diarization::BatchDiarizer> diarizer_;
  BatchOptions opt_;
};

/**
 * @brief print the lines and, when `jsonl` is set, write them as JSON Lines
 *        with the same fields as the streaming merger (speaker null if none)
 */
void writeBatchLines(const std::vector<BatchLine> &lines, std::FILE *out,
                     std::FILE *jsonl, const std::string &session);

} // namespace discretion

#endif // __DISCRETION_BATCH_PIPELINE_H__
