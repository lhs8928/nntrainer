// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   audio_encoder.h
 * @date   09 November 2026
 * @brief  Audio tower of Nemotron 3.5 Transcribe: speech recognition encoder,
 *         speaker diarization model (with its speaker cache), speaker
 *         embedding fusion and the multi-modal projector
 * @see    https://github.com/nntrainer/nntrainer
 * @author Claude <noreply@anthropic.com>
 * @bug    No known bugs except for NYI items
 *
 * The audio tower runs once per request on a variable number of frames
 * (chunks of the recording plus context, plus the diarization speaker cache),
 * so it is written as a small runtime over the NNTrainer CPU backend kernels
 * (nntrainer::sgemm) instead of a fixed-shape layer graph.
 */

#ifndef __NEMOTRON_AUDIO_ENCODER_H__
#define __NEMOTRON_AUDIO_ENCODER_H__

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "json.hpp"

#include "audio_frontend.h"

namespace causallm {
namespace nemotron_transcribe {

using json = nlohmann::json;

/**
 * @class AudioWeights
 * @brief tensors of `nntr_audio.bin` (written by weight_converter.py)
 */
class AudioWeights {
public:
  explicit AudioWeights(const std::string &path);

  bool has(const std::string &name) const;
  const float *get(const std::string &name,
                   const std::vector<int> &expect_shape = {}) const;
  const std::vector<int> &shape(const std::string &name) const;

private:
  struct Entry {
    size_t offset;
    std::vector<int> shape;
  };
  std::map<std::string, Entry> entries_;
  std::vector<float> data_;
};

/** @brief y = x W^T + b with W in torch [out, in] layout */
struct Linear {
  const float *w = nullptr;
  const float *b = nullptr;
  int in = 0, out = 0;
  void load(const AudioWeights &W, const std::string &name, int in_dim,
            int out_dim, bool bias);
  void forward(const float *x, int rows, float *y) const;
};

/** @brief torch.nn.LayerNorm (eps 1e-5) */
struct LayerNorm {
  const float *w = nullptr;
  const float *b = nullptr;
  int dim = 0;
  float eps = 1e-5f;
  void load(const AudioWeights &W, const std::string &name, int d);
  void forward(const float *x, int rows, float *y) const;
};

/** @brief geometry of one bidirectional transformer encoder */
struct EncoderConfig {
  int hidden = 0;
  int intermediate = 0;
  int layers = 0;
  int heads = 0;
  int head_dim = 0;
  int rotary_dim = 0; /**< dims of each head that get RoPE */
  float rope_theta = 10000.0f;
  bool qk_norm = false;
  int subsampling = 8;
  int mel_bins = 128;

  static EncoderConfig fromJson(const json &cfg, bool qk_norm);
};

/**
 * @class TransformerEncoder
 * @brief feature stacking + pre-LN bidirectional transformer with RoPE,
 *        shared by the speech recognition encoder and the diarization encoder
 */
class TransformerEncoder {
public:
  void load(const AudioWeights &W, const std::string &prefix,
            const EncoderConfig &cfg);

  /** @brief stack `subsampling` mel frames and project: [frames/s, hidden] */
  std::vector<float> embed(const float *mel, int frames) const;

  /**
   * @brief run the encoder on input embeddings
   * @param embeds [T, hidden] embedder outputs
   * @param mask   [T] valid frames (keys that may be attended to)
   * @return [T, hidden] after the final LayerNorm
   */
  std::vector<float> forward(const std::vector<float> &embeds, int T,
                             const std::vector<uint8_t> &mask) const;

  const EncoderConfig &config() const { return cfg_; }

private:
  struct Layer {
    LayerNorm ln1, ln2, q_norm, k_norm;
    Linear q, k, v, o, fc1, fc2;
  };
  void attention(const Layer &L, const float *x, int T,
                 const std::vector<uint8_t> &mask, float *out) const;
  void applyRope(float *x, int T) const;

  EncoderConfig cfg_;
  Linear embedder_;
  LayerNorm input_norm_, final_norm_;
  std::vector<Layer> layers_;
  std::vector<float> inv_freq_;
};

/** @brief diarization chunking and speaker cache settings */
struct DiarizationConfig {
  int chunk_length = 500;
  int chunk_left_context = 50;
  int chunk_right_context = 50;
  int fifo_length = 40;
  int speaker_cache_update_period = 300;
  int speaker_cache_length = 188;
  int silence_frames_per_speaker = 1;
  float prediction_score_threshold = 0.25f;
  float latest_frames_score_boost = 0.05f;
  float strong_boost_rate = 0.75f;
  float weak_boost_rate = 1.5f;
  float min_positive_scores_rate = 0.5f;
  int num_speakers = 8;
  int head_hidden = 192;

  static DiarizationConfig fromJson(const json &cfg);
};

/**
 * @class DiarizationModel
 * @brief chunked speaker activity detection with an Arrival-Order Speaker
 *        Cache (offline mode of
 * Nemotron3DiarizationForAudioFrameClassification)
 */
class DiarizationModel {
public:
  void load(const AudioWeights &W, const std::string &prefix, const json &cfg);

  /**
   * @param mel   [frames, mel_bins] log-mel (not normalized)
   * @param mask  [frames] valid frames
   * @return speaker logits [frames, num_speakers] (one row per mel frame)
   */
  std::vector<float> forward(const float *mel, int frames,
                             const std::vector<uint8_t> &mask) const;

  const DiarizationConfig &config() const { return cfg_; }

private:
  /** @brief head on encoder outputs: [T, hidden] -> logits [T*s, S] */
  std::vector<float> head(const std::vector<float> &hidden, int T) const;

  struct Cache;
  void cacheUpdate(Cache &c, const std::vector<float> &input_embeds,
                   const std::vector<float> &logits,
                   const std::vector<uint8_t> &mask, int n_chunk,
                   int n_lookback) const;
  void compress(std::vector<float> &embeds, std::vector<float> &probs,
                int n) const;

  DiarizationConfig cfg_;
  TransformerEncoder encoder_;
  Linear proj_;
  std::vector<float> up_w_[3]; /**< conv taps [out, in] for k = 0, 1, 2 */
  const float *up_b_ = nullptr;
  Linear dense_, out_proj_;
  const float *silence_ = nullptr;
};

/** @brief what the audio tower produces for one recording */
struct AudioOutput {
  int num_frames = 0;       /**< encoder frames (80 ms) incl. padding */
  int num_audio_tokens = 0; /**< valid encoder frames == audio tokens */
  int text_hidden = 0;
  std::vector<float> embeds;         /**< [num_audio_tokens, text_hidden] */
  std::vector<float> asr_hidden;     /**< [num_frames, asr hidden] */
  std::vector<float> speaker_logits; /**< [mel frames, num_speakers] */
};

/**
 * @class AudioTower
 * @brief log-mel -> audio token embeddings for the language model
 */
class AudioTower {
public:
  /**
   * @param weights_path  nntr_audio.bin
   * @param cfg           transcribe_config.json
   */
  AudioTower(const std::string &weights_path, const json &cfg);

  AudioOutput encode(const MelFeatures &mel) const;

  const FrontendConfig &frontendConfig() const { return fe_cfg_; }

private:
  std::unique_ptr<AudioWeights> weights_;
  FrontendConfig fe_cfg_;
  TransformerEncoder asr_;
  DiarizationModel diar_;
  LayerNorm asr_norm_, speaker_norm_;
  Linear projector_;
  std::vector<float> speaker_embedding_; /**< [num_speakers, asr hidden] */
  float activity_threshold_ = 0.5f;
  int text_hidden_ = 0;
};

} // namespace nemotron_transcribe
} // namespace causallm

#endif /* __NEMOTRON_AUDIO_ENCODER_H__ */
