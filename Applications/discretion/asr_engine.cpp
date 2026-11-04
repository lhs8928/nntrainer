// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   asr_engine.cpp
 * @date   12 October 2026
 * @brief  Resident ASR model (Qwen3-ASR or Whisper) that transcribes segments
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The model setup mirrors the segment mode of the quick_ai application
 * (Applications/quick_ai/main.cpp): the graph is built once for exactly one
 * segment length and then reused for every segment.
 */

#include "asr_engine.h"

#include <algorithm>
#include <cctype>
#include <exception>
#include <filesystem>
#include <map>
#include <mutex>

#include "json.hpp"
#include "qwen3_asr_causallm.h"
#include "wav2vec2_ctc.h"
#include "whisper_causallm.h"

namespace discretion {

namespace {

using json = nlohmann::json;

/** @brief audio tokens the encoder produces for `input_lengths` mel frames */
unsigned int featExtractOutputLengths(unsigned int input_lengths) {
  unsigned int leave = input_lengths % 100;
  unsigned int tail_tokens = 0;
  if (leave > 0) {
    unsigned int feat_lengths = (leave - 1) / 2 + 1;
    tail_tokens = ((feat_lengths - 1) / 2 + 1 - 1) / 2 + 1;
  }
  return tail_tokens + (input_lengths / 100) * 13;
}

/** @brief make a relative path in nntr_config.json relative to the model dir */
void resolvePath(json &cfg, const std::string &key,
                 const std::string &model_dir) {
  if (!cfg.contains(key) || !cfg[key].is_string())
    return;
  std::filesystem::path path = cfg[key].get<std::string>();
  if (path.empty() || path.is_absolute())
    return;
  cfg[key] = (std::filesystem::path(model_dir) / path).string();
}

/**
 * @brief Whisper's language code of a language name ("Korean" -> "ko"); a
 *        code is returned as it is, "" stays "" (detect the language)
 */
std::string whisperLanguage(const std::string &language) {
  static const std::pair<const char *, const char *> names[] = {
    {"english", "en"},    {"korean", "ko"},     {"japanese", "ja"},
    {"chinese", "zh"},    {"german", "de"},     {"spanish", "es"},
    {"french", "fr"},     {"russian", "ru"},    {"portuguese", "pt"},
    {"italian", "it"},    {"vietnamese", "vi"}, {"thai", "th"},
    {"indonesian", "id"}, {"arabic", "ar"},     {"hindi", "hi"},
    {"turkish", "tr"},    {"dutch", "nl"},      {"polish", "pl"},
    {"cantonese", "yue"}};
  std::string lower = language;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  for (const auto &[name, code] : names)
    if (lower == name)
      return code;
  return lower;
}

/** @brief architectures[0] of config.json */
std::string architectureOf(const json &cfg) {
  if (cfg.contains("architectures") && cfg["architectures"].is_array() &&
      !cfg["architectures"].empty())
    return cfg["architectures"][0].get<std::string>();
  return "";
}

// Whisper's rule for a segment without speech: likely silence and a weak
// transcription (openai/whisper no_speech_threshold, logprob_threshold)
constexpr float NO_SPEECH_THRESHOLD = 0.6f;
constexpr float LOGPROB_THRESHOLD = -1.0f;

/**
 * @brief the aligner of a model directory, loaded once and shared by the
 *        engines of a process (SpeechOps keeps one engine per graph length)
 */
std::shared_ptr<AsrAligner> sharedAligner(const std::string &dir) {
  static std::mutex m;
  static std::map<std::string, std::weak_ptr<AsrAligner>> cache;
  std::lock_guard<std::mutex> lock(m);
  if (auto a = cache[dir].lock())
    return a;
  auto a = std::make_shared<AsrAligner>();
  a->model.load(dir);
  cache[dir] = a;
  return a;
}

} // namespace

AsrEngine::AsrEngine() = default;
AsrEngine::~AsrEngine() = default;

std::string AsrEngine::provider() const {
  return whisper_ ? "nntrainer:whisper" : "nntrainer:qwen3-asr";
}

bool AsrEngine::initWhisper(const AsrOptions &opt, std::string &error) {
  json cfg = quick_ai::LoadJsonFile(opt.model_dir + "/config.json");
  json generation_cfg =
    quick_ai::LoadJsonFile(opt.model_dir + "/generation_config.json");
  json nntr_cfg = quick_ai::LoadJsonFile(opt.model_dir + "/nntr_config.json");
  for (const std::string key : {"tokenizer_file", "embedding_file_name"})
    resolvePath(nntr_cfg, key, opt.model_dir);

  // Whisper always encodes a 30 s window; a segment is at most that long
  const float sec = std::min(opt.segment_sec, 30.0f);
  segment_samples_ =
    static_cast<size_t>(static_cast<unsigned int>(sec * 100.0f + 0.5f)) * 160;
  word_timestamps_ = opt.word_timestamps;

  whisper_ =
    std::make_unique<quick_ai::WhisperCausalLM>(cfg, generation_cfg, nntr_cfg);
  whisper_language_ = whisperLanguage(opt.language);
  if (!whisper_language_.empty()) {
    const auto codes = whisper_->languages();
    if (std::find(codes.begin(), codes.end(), whisper_language_) ==
        codes.end()) {
      error = "Whisper does not know the language \"" + opt.language + "\"";
      whisper_.reset();
      return false;
    }
  }
  whisper_->initialize();
  whisper_->load_weight(opt.model_dir + "/" +
                        nntr_cfg["model_file_name"].get<std::string>());
  whisper_->repack_weight();
  return true;
}

bool AsrEngine::init(const AsrOptions &opt, std::string &error) {
  try {
    word_timestamps_ = opt.word_timestamps;
    if (!opt.align_model_dir.empty()) {
      aligner_ = sharedAligner(opt.align_model_dir);
      align_language_ = whisperLanguage(opt.language);
    }
    json cfg = quick_ai::LoadJsonFile(opt.model_dir + "/config.json");
    if (architectureOf(cfg) == quick_ai::WhisperCausalLM::architectures)
      return initWhisper(opt, error);
    json text_cfg = json::object();
    if (cfg.contains("text_config") && cfg["text_config"].is_object()) {
      text_cfg = cfg["text_config"];
    } else if (cfg.contains("thinker_config") &&
               cfg["thinker_config"].is_object() &&
               cfg["thinker_config"].contains("text_config") &&
               cfg["thinker_config"]["text_config"].is_object()) {
      text_cfg = cfg["thinker_config"]["text_config"];
    }
    for (auto &[key, val] : text_cfg.items()) {
      if (key != "architectures" && key != "model_type" && !val.is_null())
        cfg[key] = val;
    }

    json generation_cfg = json::object();
    const std::string generation_path =
      opt.model_dir + "/generation_config.json";
    if (std::filesystem::exists(generation_path))
      generation_cfg = quick_ai::LoadJsonFile(generation_path);
    json nntr_cfg = quick_ai::LoadJsonFile(opt.model_dir + "/nntr_config.json");

    // The graph is built for exactly one segment; shorter ones are padded.
    const unsigned int frames =
      static_cast<unsigned int>(opt.segment_sec * 100.0f + 0.5f);
    segment_samples_ = static_cast<size_t>(frames) * 160;
    nntr_cfg["audio_seq_len"] = frames;
    const unsigned int audio_seq_len = featExtractOutputLengths(frames);
    unsigned int base_seq_len = nntr_cfg.value("init_seq_len", 15);
    if (!opt.language.empty())
      base_seq_len += 8; // room for the "language X<asr_text>" prompt tokens
    // The configured budget suits 8 s segments; longer chunks of fast speech
    // need more tokens, or their transcript stops mid-sentence.
    constexpr float TOKENS_PER_SEC = 10.0f;
    const unsigned int num_to_generate =
      std::max(nntr_cfg.value("num_to_generate", 5u),
               static_cast<unsigned int>(opt.segment_sec * TOKENS_PER_SEC));
    nntr_cfg["init_seq_len"] = base_seq_len + audio_seq_len;
    nntr_cfg["max_seq_len"] = base_seq_len + audio_seq_len + num_to_generate;
    nntr_cfg["sliding_window"] = base_seq_len + audio_seq_len + num_to_generate;

    for (const std::string key :
         {"tokenizer_file", "embedding_file_name", "subsampler_model_file_name",
          "ple_file_name", "sample_input", "yolo_ref_dir"}) {
      resolvePath(nntr_cfg, key, opt.model_dir);
    }
    if (nntr_cfg.contains("subsampler") && nntr_cfg["subsampler"].is_object())
      resolvePath(nntr_cfg["subsampler"], "model_file_name", opt.model_dir);

    const std::string weight_file =
      opt.model_dir + "/" + nntr_cfg["model_file_name"].get<std::string>();
    const std::string user_text = nntr_cfg.value("sample_input", std::string());

    model_ = std::make_unique<quick_ai::Qwen3ASRCausalLM>(cfg, generation_cfg,
                                                          nntr_cfg);
    model_->setWordTimestamps(opt.word_timestamps);

    std::string pad_tokens;
    for (unsigned int i = 0; i < audio_seq_len; ++i)
      pad_tokens += "<|audio_pad|>";
    prompt_ =
      "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>" +
      pad_tokens + "<|audio_end|>" + user_text +
      "<|im_end|>\n<|im_start|>assistant\n";
    if (!opt.language.empty())
      prompt_ += "language " + opt.language + "<asr_text>";

    model_->initialize();
    model_->load_weight(weight_file);
    model_->repack_weight();
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  }
  return true;
}

size_t AsrEngine::segmentSamples() const { return segment_samples_; }

std::string AsrEngine::transcribe(std::vector<float> pcm) {
  // callers pad a segment to segmentSamples() with zeros: drop them, so the
  // word times stay inside the audio
  size_t n = pcm.size();
  while (n > 0 && pcm[n - 1] == 0.0f)
    --n;
  last_words_.clear();
  if (n == 0)
    return "";
  const bool align =
    aligner_ && word_timestamps_ && n <= aligner_->model.maxSamples();
  std::vector<float> audio(pcm.begin(), pcm.begin() + n);

  std::string text, language = align_language_;
  if (whisper_) {
    // the aligner times the words, so no cross-attention timing then
    const auto r = whisper_->transcribe(audio, whisper_language_,
                                        word_timestamps_ && !align);
    if (r.no_speech_prob > NO_SPEECH_THRESHOLD &&
        r.avg_logprob < LOGPROB_THRESHOLD)
      return "";
    for (const auto &w : r.words) {
      const size_t b = w.word.find_first_not_of(' ');
      if (b != std::string::npos)
        last_words_.emplace_back(w.word.substr(b), w.start);
    }
    text = r.text;
    language = r.language;
  } else {
    model_->setAudioPcm(std::move(pcm));
    model_->run(prompt_, /*do_sample=*/false, /*system_prompt=*/"",
                /*tail_prompt=*/"", /*log_output=*/false);
    text = model_->getLastTranscription();
    for (const auto &w : model_->getLastWordTimestamps())
      last_words_.emplace_back(w.text, w.time_sec);
  }

  if (align && !text.empty()) {
    // WhisperX: force-align the transcript with the wav2vec2 CTC model; the
    // model's own word times stay when the alignment fails
    std::vector<quick_ai::CtcWord> aligned;
    {
      std::lock_guard<std::mutex> lock(aligner_->mutex);
      aligned = aligner_->model.align(audio, text, language);
    }
    if (!aligned.empty()) {
      last_words_.clear();
      for (const auto &w : aligned)
        last_words_.emplace_back(w.word, w.start);
    }
  }
  return text;
}

std::vector<std::pair<std::string, double>> AsrEngine::lastWords() const {
  return last_words_;
}

} // namespace discretion
