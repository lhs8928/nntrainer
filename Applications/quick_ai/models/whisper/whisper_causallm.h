// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   whisper_causallm.h
 * @date   04 November 2026
 * @brief  Whisper (large-v3) speech recognition: encoder + text decoder
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Two nntrainer graphs:
 *  - WhisperEncoder: 30 s of log-mel frames -> conv stem -> encoder blocks ->
 *    every decoder layer's cross-attention keys and values, which its last
 *    layer stores as named memories (cross_attention.h). Runs once per
 *    segment.
 *  - WhisperCausalLM (the decoder): token embedding + learned positions ->
 *    blocks of causal self-attention (KV cache), cross-attention over the
 *    memories and a GELU MLP -> tied LM head. Greedy decoding.
 *
 * Decoding follows openai/whisper with the options WhisperX uses: no
 * timestamp tokens, no text of the previous segment, temperature 0. Word
 * timestamps come from the alignment heads' cross-attention and DTW
 * (whisper_timing.h).
 *
 * Configuration: config.json and generation_config.json of the HuggingFace
 * checkpoint (alignment_heads, lang_to_id, task_to_id, suppress_tokens, ...)
 * and nntr_config.json (res/whisper/whisper-large-v3).
 */

#ifndef __WHISPER_CAUSAL_LM_H__
#define __WHISPER_CAUSAL_LM_H__

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <causal_lm.h>

#include "whisper_timing.h"

namespace quick_ai {

class Wav2Vec2CTC;

/**
 * @class WhisperEncoder
 * @brief the audio encoder as its own graph, writing the cross memories
 */
class WhisperEncoder {
public:
  struct Dims {
    unsigned int n_mels = 128;
    unsigned int n_frames = 3000; /**< mel frames of the 30 s window */
    unsigned int n_ctx = 1500;    /**< encoder positions */
    unsigned int d_model = 1280;
    unsigned int n_heads = 20;
    unsigned int n_layers = 32;
    unsigned int ffn = 5120;
    unsigned int decoder_layers = 32;
    float eps = 1e-5f;
  };

  /**
   * @param tensor_type model_tensor_type of the encoder graph ("FP32-FP32")
   * @param fc_dtype weight dtype of its fully connected layers
   * @param memory_prefix name of the memories the graph writes
   */
  void construct(const Dims &dims, const std::string &tensor_type,
                 const std::string &fc_dtype, const std::string &memory_prefix);
  void initialize();
  void load_weight(const std::string &path);

  /**
   * @brief encode mel [n_mels][n_frames] (channel-major) into the memories
   * @param positions encoder positions to compute (0 or n_ctx: all). Fewer
   *        runs the blocks over the first positions only (an incremental
   *        step), so the attention sees no position past them and the
   *        memories hold those rows: whisper.cpp's audio_ctx
   */
  void run(const std::vector<float> &mel, unsigned int positions = 0);

private:
  Dims dims_;
  std::string act_dtype_;
  std::unique_ptr<ml::train::Model> model_;
  ml::train::Tensor input_;
  ml::train::Tensor output_;
};

/**
 * @class WhisperCausalLM
 * @brief Whisper decoder and the transcription of one segment
 */
class WhisperCausalLM : public CausalLM {
public:
  static constexpr const char *architectures =
    "WhisperForConditionalGeneration";

  /** @brief result of one segment */
  struct Result {
    std::string text;        /**< transcription, leading space removed */
    std::string language;    /**< language code ("ko"), detected or forced */
    std::vector<int> tokens; /**< generated text tokens */
    std::vector<WhisperWordTiming> words; /**< when word timestamps are on */
    float no_speech_prob = 0.0f;
    float avg_logprob = 0.0f;
    unsigned int encoder_positions = 0; /**< 20 ms positions encoded */
    /** milliseconds of the stages (NNTR_WHISPER_TIMING=1 prints them) */
    double mel_ms = 0, encoder_ms = 0, decoder_ms = 0, words_ms = 0;
  };

  WhisperCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg);
  virtual ~WhisperCausalLM();

  void initialize() override;
  void load_weight(const std::string &path) override;
  void registerCustomLayers() override;

  /**
   * @brief transcribe up to 30 s of 16 kHz mono samples
   * @param language code such as "ko" or "en"; "" detects it
   */
  Result transcribe(const std::vector<float> &pcm, const std::string &language,
                    bool word_timestamps);

  /**
   * @brief run() for the quick_ai CLI: transcribes the audio set with
   *        setAudioPcm()/setAudioPath() (30 s windows one after another)
   */
  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "",
           bool log_output = true) override;

  void setAudioPath(const std::string &path) { audio_path_ = path; }
  void setAudioPcm(std::vector<float> pcm) { audio_pcm_ = std::move(pcm); }
  void setLanguage(const std::string &code) { language_ = code; }
  void setWordTimestamps(bool on) { word_timestamps_ = on; }

  /**
   * @brief encoder context: false runs every segment over the 30 s window
   *        (Whisper as trained); true only over the segment's audio, padded
   *        by pad_sec and at least min_sec (faster for short segments, but
   *        the model never saw a window shorter than 30 s). Also set by
   *        nntr_config.json "encoder_context" ("full" | "audio"),
   *        "encoder_context_pad_sec", "encoder_context_min_sec".
   */
  void setEncoderContext(bool audio, float pad_sec = 0.0f,
                         float min_sec = 0.0f) {
    encoder_audio_ctx_ = audio;
    encoder_pad_sec_ = pad_sec;
    encoder_min_sec_ = min_sec;
  }

  /**
   * @brief run() times the words with this wav2vec2 model (WhisperX forced
   *        alignment) instead of the cross-attention; nullptr: cross-attention
   */
  void setAligner(Wav2Vec2CTC *aligner) { aligner_ = aligner; }

  const Result &getLastResult() const { return last_; }

  /** @brief samples of the encoder window (30 s) */
  size_t windowSamples() const {
    return static_cast<size_t>(enc_dims_.n_frames) * 160;
  }

  /** @brief language codes the model knows (from lang_to_id) */
  std::vector<std::string> languages() const;

protected:
  std::pair<Tensor, Tensor> constructModel() override;

private:
  /** @brief one decoder pass over ids[from, to) at positions from..to */
  std::vector<float> step(std::vector<float> &ids, unsigned int from,
                          unsigned int to);
  void setCapture(bool on);

  WhisperEncoder encoder_;
  WhisperEncoder::Dims enc_dims_;
  std::string memory_prefix_;
  std::string act_dtype_;
  std::string encoder_tensor_type_;
  std::string encoder_fc_dtype_;

  // special tokens
  int sot_ = 50258;
  int eot_ = 50257;
  int transcribe_ = 50360;
  int no_timestamps_ = 50364;
  int no_speech_ = -1;
  std::vector<std::pair<std::string, int>> lang_ids_; /**< "ko" -> id */
  std::vector<int> suppress_;
  std::vector<int> begin_suppress_;
  std::vector<std::pair<int, int>> alignment_heads_; /**< (layer, head) */
  unsigned int sample_len_ = 224;

  std::string audio_path_;
  std::vector<float> audio_pcm_;
  std::string language_;
  bool word_timestamps_ = true;
  bool capture_on_ = false;
  bool encoder_audio_ctx_ = false;
  float encoder_pad_sec_ = 0.0f;
  float encoder_min_sec_ = 0.0f;
  Wav2Vec2CTC *aligner_ = nullptr;
  Result last_;
};

} // namespace quick_ai

#endif // __WHISPER_CAUSAL_LM_H__
