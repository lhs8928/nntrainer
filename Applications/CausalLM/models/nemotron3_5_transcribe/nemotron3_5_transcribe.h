// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nemotron3_5_transcribe.h
 * @date   09 November 2026
 * @brief  Nemotron 3.5 Transcribe pipeline: audio -> audio embeddings ->
 *         prompt with spliced audio -> Nemotron-H generation
 * @see    https://github.com/nntrainer/nntrainer
 * @author Claude <noreply@anthropic.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __NEMOTRON3_5_TRANSCRIBE_H__
#define __NEMOTRON3_5_TRANSCRIBE_H__

#include <memory>
#include <string>
#include <vector>

#include "audio_encoder.h"
#include "audio_frontend.h"
#include "nemotron_h_causallm.h"

namespace causallm {
namespace nemotron_transcribe {

/** @brief text pieces around the prompt and the audio placeholder */
struct PromptTemplate {
  std::string prefix;  /**< before the user prompt */
  std::string between; /**< between the prompt and the audio */
  std::string suffix;  /**< after the audio, incl. the generation prompt */
  std::string default_prompt;
};

/**
 * @class Nemotron35Transcribe
 */
class Nemotron35Transcribe {
public:
  /**
   * @brief load a converted model directory (see weight_converter.py):
   *        config.json, generation_config.json, nntr_config.json,
   *        transcribe_config.json and the weight files
   */
  explicit Nemotron35Transcribe(const std::string &model_dir);

  /** @brief log-mel features of 16 kHz mono audio */
  MelFeatures computeMel(const std::vector<float> &audio) const;

  /** @brief audio tower: mel -> audio token embeddings */
  AudioOutput encodeAudio(const MelFeatures &mel) const;

  /**
   * @brief tokenize the chat prompt around `num_audio_tokens` placeholders
   * @note  needs the tokenizer (nntr_config.tokenizer_file)
   */
  std::vector<int> buildPromptIds(const std::string &prompt,
                                  int num_audio_tokens);

  /**
   * @brief generate from token ids whose audio placeholders are replaced, in
   *        order, by the rows of `audio.embeds`
   */
  void generate(const std::vector<int> &ids, const AudioOutput &audio,
                bool log_output = true);

  /** @brief wav file -> transcript */
  std::string transcribe(const std::string &wav_path,
                         const std::string &prompt = "",
                         bool log_output = true);

  NemotronHCausalLM &llm() { return *llm_; }
  const PromptTemplate &promptTemplate() const { return prompt_; }
  int audioTokenId() const { return audio_token_id_; }

private:
  std::unique_ptr<NemotronHCausalLM> llm_;
  std::unique_ptr<AudioTower> audio_;
  std::unique_ptr<LogMelFrontend> frontend_;
  PromptTemplate prompt_;
  int audio_token_id_ = -1;
};

} // namespace nemotron_transcribe
} // namespace causallm

#endif /* __NEMOTRON3_5_TRANSCRIBE_H__ */
