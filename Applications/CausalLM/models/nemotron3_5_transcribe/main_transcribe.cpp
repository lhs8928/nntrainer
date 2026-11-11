// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   main_transcribe.cpp
 * @date   09 November 2026
 * @brief  nntr_nemotron_transcribe: run Nemotron 3.5 Transcribe on a WAV file,
 *         or check the port against reference tensors of the HuggingFace
 *         implementation (make_tiny_reference.py)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Claude <noreply@anthropic.com>
 * @bug    No known bugs except for NYI items
 *
 * Usage:
 *   nntr_nemotron_transcribe <model_dir> <audio.wav> [prompt]
 *   nntr_nemotron_transcribe --verify <model_dir> <reference_dir>
 *   nntr_nemotron_transcribe --verify-audio <model_dir> <reference_dir>
 *   nntr_nemotron_transcribe --verify-llm <model_dir> <reference_dir>
 *     (language model only: tokenizer, prefill logits and greedy tokens of a
 *      text prompt; works with a checkpoint cut down to a few blocks)
 *     (audio tower only: needs nntr_audio.bin + transcribe_config.json and
 *      the references of make_audio_reference.py, not the language model)
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "nemotron3_5_transcribe.h"

using causallm::nemotron_transcribe::AudioOutput;
using causallm::nemotron_transcribe::json;
using causallm::nemotron_transcribe::MelFeatures;
using causallm::nemotron_transcribe::Nemotron35Transcribe;

namespace {

class Reference {
public:
  explicit Reference(const std::string &dir) : dir_(dir) {
    std::ifstream f(dir + "/ref_index.json");
    if (!f.is_open())
      throw std::runtime_error("cannot open " + dir + "/ref_index.json");
    index_ = json::parse(f);
  }

  template <typename T> std::vector<T> load(const std::string &name) const {
    const auto &shape = index_.at(name).at("shape");
    size_t n = 1;
    for (auto &d : shape)
      n *= d.get<size_t>();
    std::vector<T> v(n);
    std::ifstream f(dir_ + "/ref_" + name + ".bin", std::ios::binary);
    f.read(reinterpret_cast<char *>(v.data()),
           static_cast<std::streamsize>(n * sizeof(T)));
    if (!f)
      throw std::runtime_error("cannot read reference " + name);
    return v;
  }

  const json &meta() const { return index_.at("_meta"); }
  bool has(const std::string &name) const { return index_.contains(name); }

private:
  std::string dir_;
  json index_;
};

struct Checker {
  int failures = 0;

  /** max |a-b| relative to max |b| */
  void close(const std::string &what, const std::vector<float> &got,
             const std::vector<float> &want, double tol) {
    if (got.size() != want.size()) {
      std::cout << "[FAIL] " << what << ": size " << got.size()
                << " != " << want.size() << "\n";
      ++failures;
      return;
    }
    double max_diff = 0.0, max_ref = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      max_diff = std::max(max_diff, std::fabs(double(got[i]) - want[i]));
      max_ref = std::max(max_ref, std::fabs(double(want[i])));
    }
    const double rel = max_diff / std::max(max_ref, 1e-12);
    const bool ok = rel <= tol && std::isfinite(rel);
    std::cout << (ok ? "[ OK ] " : "[FAIL] ") << what << ": max|diff| "
              << max_diff << ", max|ref| " << max_ref << ", rel " << rel
              << " (tol " << tol << ")\n";
    failures += ok ? 0 : 1;
  }

  void same(const std::string &what, const std::vector<int> &got,
            const std::vector<int> &want) {
    const bool ok = got == want;
    std::cout << (ok ? "[ OK ] " : "[FAIL] ") << what << ":";
    for (int v : got)
      std::cout << " " << v;
    if (!ok) {
      std::cout << "\n       expected:";
      for (int v : want)
        std::cout << " " << v;
    }
    std::cout << "\n";
    failures += ok ? 0 : 1;
  }
};

std::vector<int> toInt(const std::vector<unsigned int> &v) {
  return std::vector<int>(v.begin(), v.end());
}

int verify(const std::string &model_dir, const std::string &ref_dir) {
  Reference ref(ref_dir);
  Nemotron35Transcribe pipe(model_dir);
  Checker check;
  auto &llm = pipe.llm();

  // ---- language model alone
  {
    std::vector<int> ids = ref.load<int>("text_ids");
    std::vector<float> emb = llm.embedTokens(ids);
    llm.run_with_embeddings(emb.data(), ids.size(), ids, false, false);
    std::vector<float> logits = ref.load<float>("text_logits");
    const size_t V = llm.getVocabSize();
    std::vector<float> last(logits.end() - V, logits.end());
    check.close("text: last-position logits", llm.getPrefillLogits(), last,
                1e-3);
    check.same("text: greedy tokens", toInt(llm.getGeneratedIds()),
               ref.load<int>("text_generated"));
  }

  // ---- audio + language model
  for (const std::string tag : {"short", "long"}) {
    int rate = 0;
    std::vector<float> audio = causallm::nemotron_transcribe::loadWav(
      ref_dir + "/audio_" + tag + ".wav", rate);
    MelFeatures mel = pipe.computeMel(audio);
    std::vector<int> ref_mask = ref.load<int>(tag + "_mel_mask");
    int ref_valid = 0;
    for (int m : ref_mask)
      ref_valid += m;
    std::cout << "[info] " << tag << ": " << audio.size() << " samples, "
              << mel.num_frames << " mel frames (" << mel.num_valid
              << " valid, reference " << ref_mask.size() << " / " << ref_valid
              << ")\n";
    check.close(tag + ": log-mel", mel.data, ref.load<float>(tag + "_mel"),
                1e-4);

    AudioOutput out = pipe.encodeAudio(mel);
    check.close(tag + ": asr encoder states", out.asr_hidden,
                ref.load<float>(tag + "_asr_hidden"), 1e-3);
    check.close(tag + ": speaker logits", out.speaker_logits,
                ref.load<float>(tag + "_speaker_logits"), 1e-3);
    check.close(tag + ": audio embeddings", out.embeds,
                ref.load<float>(tag + "_audio_embeds"), 1e-3);

    std::vector<int> ids = ref.load<int>(tag + "_input_ids");
    pipe.generate(ids, out, false);
    std::vector<float> logits = ref.load<float>(tag + "_logits");
    const size_t V = llm.getVocabSize();
    std::vector<float> last(logits.end() - V, logits.end());
    check.close(tag + ": last-position logits", llm.getPrefillLogits(), last,
                2e-3);
    check.same(tag + ": greedy tokens", toInt(llm.getGeneratedIds()),
               ref.load<int>(tag + "_generated"));
  }

  // ---- the whole transcribe() path: tokenizer + prompt template
  if (ref.has("transcribe_prompt_ids")) {
    if (llm.getTokenizer() == nullptr) {
      std::cout << "[skip] transcribe: no tokenizer_file configured\n";
    } else {
      const int n_audio = ref.meta().at("short_num_audio_tokens").get<int>();
      check.same("transcribe: prompt ids", pipe.buildPromptIds("", n_audio),
                 ref.load<int>("transcribe_prompt_ids"));
      pipe.transcribe(ref_dir + "/audio_short.wav", "", false);
      check.same("transcribe: greedy tokens", toInt(llm.getGeneratedIds()),
                 ref.load<int>("transcribe_generated"));
      std::cout << "[info] transcribe: text \"" << llm.getOutput(0) << "\"\n";
    }
  }

  std::cout << (check.failures == 0 ? "ALL CHECKS PASSED\n"
                                    : "SOME CHECKS FAILED\n");
  return check.failures == 0 ? 0 : 1;
}

int verifyAudio(const std::string &model_dir, const std::string &ref_dir) {
  using causallm::nemotron_transcribe::AudioTower;
  using causallm::nemotron_transcribe::LogMelFrontend;
  using clock = std::chrono::steady_clock;
  auto ms = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };

  Reference ref(ref_dir);
  std::ifstream cf(model_dir + "/transcribe_config.json");
  if (!cf.is_open())
    throw std::runtime_error("cannot open " + model_dir +
                             "/transcribe_config.json");
  json tc = json::parse(cf);
  auto t0 = clock::now();
  AudioTower tower(model_dir + "/nntr_audio.bin", tc);
  LogMelFrontend frontend(tower.frontendConfig());
  std::cout << "[info] audio tower loaded in " << ms(t0, clock::now())
            << " ms\n";

  Checker check;
  for (const auto &w : ref.meta().at("wavs")) {
    const std::string tag = w.at("tag").get<std::string>();
    const std::string path = w.at("path").get<std::string>();
    int rate = 0;
    std::vector<float> audio =
      causallm::nemotron_transcribe::loadWav(path, rate);
    if (rate != tower.frontendConfig().sampling_rate)
      throw std::runtime_error(path + ": reference wav must be 16 kHz");

    auto t1 = clock::now();
    MelFeatures mel = frontend.compute(audio);
    auto t2 = clock::now();
    AudioOutput out = tower.encode(mel);
    auto t3 = clock::now();
    std::cout << "[info] " << tag << " (" << path
              << "): " << w.at("seconds").get<double>() << " s, "
              << out.num_audio_tokens << " audio tokens, log-mel " << ms(t1, t2)
              << " ms, audio tower " << ms(t2, t3) << " ms\n";

    check.close(tag + ": log-mel", mel.data, ref.load<float>(tag + "_mel"),
                1e-4);
    check.close(tag + ": asr encoder states", out.asr_hidden,
                ref.load<float>(tag + "_asr_hidden"), 1e-3);
    check.close(tag + ": speaker logits", out.speaker_logits,
                ref.load<float>(tag + "_speaker_logits"), 1e-3);
    check.close(tag + ": audio embeddings", out.embeds,
                ref.load<float>(tag + "_audio_embeds"), 1e-3);
  }
  std::cout << (check.failures == 0 ? "ALL CHECKS PASSED\n"
                                    : "SOME CHECKS FAILED\n");
  return check.failures == 0 ? 0 : 1;
}

int verifyLLM(const std::string &model_dir, const std::string &ref_dir) {
  Reference ref(ref_dir);
  causallm::json cfg = causallm::LoadJsonFile(model_dir + "/config.json");
  causallm::json gen =
    causallm::LoadJsonFile(model_dir + "/generation_config.json");
  causallm::json nntr = causallm::LoadJsonFile(model_dir + "/nntr_config.json");
  for (const char *key : {"tokenizer_file", "model_file_name",
                          "embedding_file_name", "moe_expert_file_name"}) {
    if (nntr.contains(key) && nntr[key].is_string()) {
      const std::string v = nntr[key].get<std::string>();
      if (!v.empty() && v[0] != '/')
        nntr[key] = model_dir + "/" + v;
    }
  }
  causallm::NemotronHCausalLM llm(cfg, gen, nntr);
  llm.initialize();
  llm.load_weight(nntr.at("model_file_name").get<std::string>());

  Checker check;
  std::vector<int> ids = ref.load<int>("text_ids");
  if (ref.meta().contains("text_prompt") && llm.getTokenizer() != nullptr) {
    std::vector<int> mine = llm.getTokenizer()->Encode(
      ref.meta()["text_prompt"].get<std::string>(), false);
    check.same("tokenizer: prompt ids", mine, ids);
  }
  std::vector<float> emb = llm.embedTokens(ids);
  llm.run_with_embeddings(emb.data(), ids.size(), ids, false, false);
  std::vector<float> logits = ref.load<float>("text_logits");
  const size_t V = llm.getVocabSize();
  std::vector<float> last(logits.end() - V, logits.end());
  check.close("text: last-position logits", llm.getPrefillLogits(), last, 1e-3);
  check.same("text: greedy tokens", toInt(llm.getGeneratedIds()),
             ref.load<int>("text_generated"));
  std::cout << (check.failures == 0 ? "ALL CHECKS PASSED\n"
                                    : "SOME CHECKS FAILED\n");
  return check.failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char *argv[]) {
  try {
    if (argc >= 4 && std::string(argv[1]) == "--verify")
      return verify(argv[2], argv[3]);
    if (argc >= 4 && std::string(argv[1]) == "--verify-audio")
      return verifyAudio(argv[2], argv[3]);
    if (argc >= 4 && std::string(argv[1]) == "--verify-llm")
      return verifyLLM(argv[2], argv[3]);
    if (argc < 3) {
      std::cerr << "Usage:\n"
                << "  " << argv[0] << " <model_dir> <audio.wav> [prompt]\n"
                << "  " << argv[0] << " --verify <model_dir> <reference_dir>\n"
                << "  " << argv[0]
                << " --verify-audio <model_dir> <reference_dir>\n"
                << "  " << argv[0]
                << " --verify-llm <model_dir> <reference_dir>\n";
      return 1;
    }
    Nemotron35Transcribe pipe(argv[1]);
    const std::string prompt = argc > 3 ? argv[3] : "";
    std::string text = pipe.transcribe(argv[2], prompt, true);
    std::cout << "\n[transcript]\n" << text << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
