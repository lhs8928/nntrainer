// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nemotron3_5_transcribe.cpp
 * @date   09 November 2026
 * @brief  Nemotron 3.5 Transcribe pipeline
 * @see    https://github.com/nntrainer/nntrainer
 * @author Claude <noreply@anthropic.com>
 * @bug    No known bugs except for NYI items
 */

#include "nemotron3_5_transcribe.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace causallm {
namespace nemotron_transcribe {

namespace {

std::string resolve(const std::string &dir, const std::string &p) {
  std::filesystem::path path(p);
  if (p.empty() || path.is_absolute())
    return p;
  return (std::filesystem::path(dir) / path).string();
}

void resolveKey(json &cfg, const char *key, const std::string &dir) {
  if (cfg.contains(key) && cfg[key].is_string())
    cfg[key] = resolve(dir, cfg[key].get<std::string>());
}

} // namespace

Nemotron35Transcribe::Nemotron35Transcribe(const std::string &model_dir) {
  json cfg = LoadJsonFile(model_dir + "/config.json");
  json gen_cfg = json::object();
  if (std::filesystem::exists(model_dir + "/generation_config.json"))
    gen_cfg = LoadJsonFile(model_dir + "/generation_config.json");
  json nntr_cfg = LoadJsonFile(model_dir + "/nntr_config.json");
  for (const char *key :
       {"tokenizer_file", "model_file_name", "embedding_file_name",
        "audio_file_name", "transcribe_config_file"})
    resolveKey(nntr_cfg, key, model_dir);

  const std::string tc_path = nntr_cfg.value(
    "transcribe_config_file", model_dir + "/transcribe_config.json");
  json tc = LoadJsonFile(tc_path);
  audio_token_id_ = tc.at("audio_token_id").get<int>();
  if (tc.contains("prompt")) {
    const json &p = tc["prompt"];
    prompt_.prefix = p.value("prefix", std::string());
    prompt_.between = p.value("between", std::string());
    prompt_.suffix = p.value("suffix", std::string());
    prompt_.default_prompt = p.value("default_prompt", std::string());
  }

  llm_ = std::make_unique<NemotronHCausalLM>(cfg, gen_cfg, nntr_cfg);
  llm_->initialize();
  llm_->load_weight(nntr_cfg.at("model_file_name").get<std::string>());

  audio_ = std::make_unique<AudioTower>(
    nntr_cfg.value("audio_file_name", model_dir + "/nntr_audio.bin"), tc);
  frontend_ = std::make_unique<LogMelFrontend>(audio_->frontendConfig());
}

MelFeatures
Nemotron35Transcribe::computeMel(const std::vector<float> &audio) const {
  return frontend_->compute(audio);
}

AudioOutput Nemotron35Transcribe::encodeAudio(const MelFeatures &mel) const {
  return audio_->encode(mel);
}

std::vector<int> Nemotron35Transcribe::buildPromptIds(const std::string &prompt,
                                                      int num_audio_tokens) {
  tokenizers::Tokenizer *tok = llm_->getTokenizer();
  if (tok == nullptr)
    throw std::runtime_error(
      "Nemotron35Transcribe: a tokenizer is needed to build the prompt");
  const std::string text = prompt.empty() ? prompt_.default_prompt : prompt;
  // The audio placeholder is a special token, so tokenizing the pieces on
  // either side of it separately matches tokenizing the whole chat text.
  std::vector<int> ids = tok->Encode(prompt_.prefix + text + prompt_.between,
                                     /*add_special_tokens=*/false);
  ids.insert(ids.end(), static_cast<size_t>(num_audio_tokens), audio_token_id_);
  std::vector<int> tail = tok->Encode(prompt_.suffix, false);
  ids.insert(ids.end(), tail.begin(), tail.end());
  return ids;
}

void Nemotron35Transcribe::generate(const std::vector<int> &ids,
                                    const AudioOutput &audio, bool log_output) {
  const int H = llm_->getHiddenSize();
  if (audio.text_hidden != H)
    throw std::invalid_argument("Nemotron35Transcribe: audio embedding width " +
                                std::to_string(audio.text_hidden) +
                                " != LLM hidden size " + std::to_string(H));

  // The placeholder id is outside the vocabulary: embed it as token 0, then
  // overwrite its rows with the audio frames.
  std::vector<int> text_ids(ids);
  int n_audio = 0;
  for (int &id : text_ids)
    if (id == audio_token_id_) {
      id = 0;
      ++n_audio;
    }
  if (n_audio != audio.num_audio_tokens)
    throw std::invalid_argument(
      "Nemotron35Transcribe: prompt has " + std::to_string(n_audio) +
      " audio tokens, audio has " + std::to_string(audio.num_audio_tokens));

  std::vector<float> embeds = llm_->embedTokens(text_ids);
  int a = 0;
  for (size_t t = 0; t < ids.size(); ++t)
    if (ids[t] == audio_token_id_) {
      std::copy(audio.embeds.begin() + static_cast<size_t>(a) * H,
                audio.embeds.begin() + static_cast<size_t>(a + 1) * H,
                embeds.begin() + t * H);
      ++a;
    }
  llm_->run_with_embeddings(embeds.data(), ids.size(), text_ids,
                            /*do_sample=*/llm_->getDoSample(), log_output);
}

std::string Nemotron35Transcribe::transcribe(const std::string &wav_path,
                                             const std::string &prompt,
                                             bool log_output) {
  int rate = 0;
  std::vector<float> audio = loadWav(wav_path, rate);
  const int want = audio_->frontendConfig().sampling_rate;
  if (rate != want) {
    std::cerr << "[nemotron-transcribe] resampling " << rate << " Hz -> "
              << want << " Hz (linear); resample offline for best quality\n";
    audio = resampleLinear(audio, rate, want);
  }
  MelFeatures mel = computeMel(audio);
  AudioOutput out = encodeAudio(mel);
  std::vector<int> ids = buildPromptIds(prompt, out.num_audio_tokens);
  generate(ids, out, log_output);
  return llm_->getOutput(0);
}

} // namespace nemotron_transcribe
} // namespace causallm
