// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   speech_ops.h
 * @date   21 October 2026
 * @brief  Speech model operations for another program's pipeline
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * `--serve --speech` answers three requests on stdin, each on a file of
 * 16 kHz mono PCM16 samples, writing JSON to an output file:
 *
 *   ASR      text and word timestamps of one segment (Qwen3-ASR)
 *   EMBED    one speaker embedding of the whole input (WeSpeaker ResNet34)
 *   SEGMENT  PyanNet powerset classes per frame, per 10 s window
 *
 * A meeting program that keeps its own endpointing and speaker clustering
 * (ALICE) uses them in place of its recognizer, speaker model and
 * segmentation model.
 */

#ifndef __DISCRETION_SPEECH_OPS_H__
#define __DISCRETION_SPEECH_OPS_H__

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "asr_lane.h"

namespace speaker_diarization {
class BatchDiarizer;
}

namespace discretion {

class AsrEngine;

struct SpeechOptions {
  AsrOptions asr;              // model, language, threads
  std::vector<float> asr_secs; // one model graph per length, ascending
  std::string sd_weights;      // speaker_diarization.bin
};

/** @brief one recognized segment: its text and the time of every word */
struct AsrResult {
  std::string text;
  struct Word {
    long start_ms, end_ms; // from the start of the segment
    std::string text;
  };
  std::vector<Word> words;
  size_t pieces = 0; // model calls (a segment longer than the graphs is cut)
  double ms = 0.0;   // model time
};

class SpeechOps {
public:
  SpeechOps();
  ~SpeechOps();

  /** @brief load the ASR graphs and the diarization models */
  bool init(const SpeechOptions &opt, std::string &error);

  /**
   * @brief answer one request
   * @param op ASR, EMBED or SEGMENT
   * @param in PCM16 file, @param out JSON file to write
   */
  bool handle(const std::string &op, const std::string &in,
              const std::string &out, std::string &error);

  /**
   * @brief answer one request on samples already in memory
   * @param pcm16 16 kHz mono little-endian PCM16 bytes
   * @param body the JSON answer
   */
  bool run(const std::string &op, const std::string &pcm16, std::string &body,
           std::string &error);

  /** @brief text and word times of 16 kHz samples (Qwen3-ASR) */
  AsrResult recognize(const std::vector<float> &pcm);

  /** @brief one WeSpeaker ResNet34 embedding of the samples */
  std::vector<float> embedding(const std::vector<float> &pcm);

  /** @brief PyanNet powerset class per frame, per consecutive 10 s window */
  std::vector<std::vector<uint8_t>> segmentation(const std::vector<float> &pcm);

  /** @brief what is loaded, as one JSON object (the HTTP server's /health) */
  std::string describe() const;

private:
  std::string asr(const std::vector<float> &pcm);
  std::string embed(const std::vector<float> &pcm);
  std::string segment(const std::vector<float> &pcm);

  SpeechOptions opt_;
  std::vector<std::shared_ptr<AsrEngine>> engines_; // by graph length
  std::unique_ptr<speaker_diarization::BatchDiarizer> sd_;
};

} // namespace discretion

#endif // __DISCRETION_SPEECH_OPS_H__
