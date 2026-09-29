/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	main.cpp
 * @date	23 July 2025
 * @brief	This is a main file for CausalLM application
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <factory.h>

#include "ced/ced_transformer.h"
#include "qwen3_asr_causallm.h"
#include "audio_preprocessor.h"
#if !defined(_WIN32)
#include <sys/resource.h>
#endif

#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>

using json = nlohmann::json;

std::atomic<size_t> peak_rss_kb{0};
std::atomic<bool> tracking_enabled{true};

namespace {

inline unsigned int get_feat_extract_output_lengths(unsigned int input_lengths) {
  unsigned int input_lengths_leave = input_lengths % 100;
  unsigned int tail_tokens = 0;
  if (input_lengths_leave > 0) {
    unsigned int feat_lengths = (input_lengths_leave - 1) / 2 + 1;
    tail_tokens = ((feat_lengths - 1) / 2 + 1 - 1) / 2 + 1;
  }
  return tail_tokens + (input_lengths / 100) * 13;
}

void resolveNntrConfigPath(json &nntr_cfg, const std::string &key,
                           const std::string &model_path) {
  if (!nntr_cfg.contains(key) || !nntr_cfg[key].is_string())
    return;

  std::filesystem::path path = nntr_cfg[key].get<std::string>();
  if (path.empty() || path.is_absolute())
    return;

  nntr_cfg[key] = (std::filesystem::path(model_path) / path).string();
}

} // namespace

/**
 * @brief Print the maximum resident set size for the current process.
 */
void printMemoryUsage() {
#if defined(_WIN32)
  std::cout << "Max Resident Set Size: unavailable on Windows" << std::endl;
#else
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  std::cout << "Max Resident Set Size: " << usage.ru_maxrss << " KB"
            << std::endl;
#endif
}

/**
 * @brief Read the current process resident set size on Linux.
 */
size_t read_vm_rss_kb() {
#if defined(_WIN32)
  return 0;
#else
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.find("VmRSS:") == 0) {
      size_t kb = 0;
      sscanf(line.c_str(), "VmRSS: %zu kB", &kb);
      return kb;
    }
  }
  return 0;
#endif
}

/**
 * @brief Read private resident memory from smaps_rollup on Linux.
 */
size_t read_private_rss_kb() {
#if defined(_WIN32)
  return 0;
#else
  std::ifstream smaps("/proc/self/smaps_rollup");
  std::string line;
  size_t total = 0;
  while (std::getline(smaps, line)) {
    if (line.find("Private_Clean:") == 0 || line.find("Private_Dirty:") == 0) {
      size_t kb;
      sscanf(line.c_str(), "%*s %zu", &kb);
      total += kb;
    }
  }
  return total;
#endif
}

/**
 * @brief Start a background sampler for peak private RSS.
 */
void start_peak_tracker() {
  std::thread([] {
    while (tracking_enabled.load()) {
      size_t current = read_private_rss_kb();
      size_t prev = peak_rss_kb.load();
      if (current > prev) {
        peak_rss_kb.store(current);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }).detach();
}

/**
 * @brief Stop the memory sampler and print the observed peak.
 */
void stop_and_print_peak() {
  tracking_enabled.store(false);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  std::cout << "Peak memory usage (VmRSS): " << peak_rss_kb.load() << " KB"
            << std::endl;
}

/**
 * @brief Resolve config architecture names to registered model factory names.
 */
std::string resolve_architecture(std::string model_type,
                                 const std::string &architecture) {
  std::transform(model_type.begin(), model_type.end(), model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (model_type == "embedding") {
    if (architecture == "Qwen3ForCausalLM") {
      return "Qwen3Embedding";
    } else if (architecture == "Gemma3ForCausalLM" ||
               architecture == "Gemma3TextModel") {
      return "EmbeddingGemma";
    } else if (architecture == "Qwen2Model") {
      return "Qwen2Embedding";
    } else if (architecture == "BertForMaskedLM") {
      return "MultilingualTinyBert";
    } else if (architecture == "XLMRobertaForMaskedLM" ||
               architecture == "XLMRobertaModel") {
      return "XLMRobertaForMaskedLM";
    } else if (architecture == "deberta-v2" ||
               architecture == "DebertaV2Model" ||
               architecture == "DebertaV2ForMaskedLM") {
      return "DebertaV2";
    } else {
      throw std::invalid_argument(
        "Unsupported architecture for embedding model: " + architecture);
    }
  }

  if (architecture == "Gemma4ForConditionalGeneration") {
    return "Gemma4ForCausalLM";
  }

  return architecture;
}

/**
 * @brief Entry point for loading, initializing, and running a CausalLM model.
 */
int main(int argc, char *argv[]) {
  // Parse and consume --audio option if present
  std::string audio_path = "";
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--audio") == 0 && i + 1 < argc) {
      audio_path = argv[i + 1];
      for (int j = i; j < argc - 2; ++j) {
        argv[j] = argv[j + 2];
      }
      argc -= 2;
      break;
    }
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  /** Register CED model */
  quick_ai::Factory::Instance().registerModel(
    "CedForAudioClassification",
    [](json &cfg, json &generation_cfg, json &nntr_cfg) {
      return std::make_unique<quick_ai::CedTransformer>(cfg, generation_cfg,
                                                        nntr_cfg);
    });

  quick_ai::Factory::Instance().registerModel(
    "Qwen3ASRForConditionalGeneration",
    [](json &cfg, json &generation_cfg, json &nntr_cfg) {
      return std::make_unique<quick_ai::Qwen3ASRCausalLM>(cfg, generation_cfg,
                                                          nntr_cfg);
    });

  // Validate arguments
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path> [input_prompt]\n"
              << "  <model_path>   : Path to model directory\n"
              << "  [input_prompt] : Optional input text (uses sample_input or "
                 "chat_input if omitted)\n";
    return EXIT_FAILURE;
  }

  const std::string model_path = argv[1];
  std::string input_text;
  std::string system_head_prompt = "";
  std::string system_tail_prompt = "";

  std::cout << model_path << std::endl;

  try {
    // Load configuration files
    json cfg = quick_ai::LoadJsonFile(model_path + "/config.json");
    json text_cfg = json::object();
    if (cfg.contains("text_config") && cfg["text_config"].is_object()) {
      text_cfg = cfg["text_config"];
    } else if (cfg.contains("thinker_config") && cfg["thinker_config"].is_object() &&
               cfg["thinker_config"].contains("text_config") && cfg["thinker_config"]["text_config"].is_object()) {
      text_cfg = cfg["thinker_config"]["text_config"];
    }

    if (!text_cfg.empty()) {
      for (auto& [key, val] : text_cfg.items()) {
        if (key != "architectures" && key != "model_type" && !val.is_null()) {
          cfg[key] = val;
        }
      }
    }

    json generation_cfg = json::object();
    std::string generation_config_path = model_path + "/generation_config.json";
    if (std::filesystem::exists(generation_config_path)) {
      generation_cfg = quick_ai::LoadJsonFile(generation_config_path);
    }
    json nntr_cfg = quick_ai::LoadJsonFile(model_path + "/nntr_config.json");

    if (!audio_path.empty()) {
      quick_ai::AudioPreprocessor preprocessor;
      auto pcm = preprocessor.loadWav(audio_path);
      unsigned int audio_frames = pcm.size() / 160;
      constexpr unsigned int MAX_AUDIO_FRAMES = 1200 * 100; // 20 minutes (1200s * 100fps)
      if (audio_frames > MAX_AUDIO_FRAMES) {
        audio_frames = MAX_AUDIO_FRAMES;
      }
      nntr_cfg["audio_seq_len"] = audio_frames;
      unsigned int audio_seq_len = get_feat_extract_output_lengths(audio_frames);
      unsigned int base_seq_len = nntr_cfg.value("init_seq_len", 15);
      unsigned int num_to_generate = nntr_cfg.value("num_to_generate", 5);
      nntr_cfg["init_seq_len"] = base_seq_len + audio_seq_len;
      nntr_cfg["max_seq_len"] = base_seq_len + audio_seq_len + num_to_generate;
      nntr_cfg["sliding_window"] = base_seq_len + audio_seq_len + num_to_generate;
    }
    // Resolve relative paths in nntr_config.json against the model directory.
    // Iterate by value (const std::string) rather than by reference: the
    // initializer list holds const char * literals that are converted to
    // temporary std::string objects, and binding those temporaries to a
    // const reference is flagged by -Werror=range-loop-construct.
    for (const std::string key :
         {"tokenizer_file", "embedding_file_name", "subsampler_model_file_name",
          "ple_file_name", "sample_input", "yolo_ref_dir"}) {
      resolveNntrConfigPath(nntr_cfg, key, model_path);
    }

    if (nntr_cfg.contains("subsampler") && nntr_cfg["subsampler"].is_object()) {
      resolveNntrConfigPath(nntr_cfg["subsampler"], "model_file_name", model_path);
    }

    if (nntr_cfg.contains("system_prompt")) {
      system_head_prompt =
        nntr_cfg["system_prompt"]["head_prompt"].get<std::string>();
      system_tail_prompt =
        nntr_cfg["system_prompt"]["tail_prompt"].get<std::string>();
    }

    // Construct weight file path
    const std::string weight_file =
      model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();

    std::cout << weight_file << std::endl;

    // Initialize and run model
    std::string architecture;
    if (cfg.contains("architectures") && cfg["architectures"].is_array() &&
        !cfg["architectures"].empty()) {
      architecture = cfg["architectures"].get<std::vector<std::string>>()[0];
    } else if (cfg.contains("architecture") &&
               cfg["architecture"].is_string()) {
      architecture = cfg["architecture"].get<std::string>();
    } else if (cfg.contains("model_type") && cfg["model_type"].is_string()) {
      architecture = cfg["model_type"].get<std::string>();
    } else {
      throw std::invalid_argument(
        "config.json must contain 'architectures', 'architecture', or "
        "'model_type'.");
    }

    if (nntr_cfg.contains("model_type")) {
      std::string model_type = nntr_cfg["model_type"].get<std::string>();
      architecture = resolve_architecture(model_type, architecture);
    }

    // Determine input text
    if (argc >= 3) {
      input_text = argv[2];
    } else {
      input_text = nntr_cfg["sample_input"].get<std::string>();
    }

    auto model = quick_ai::Factory::Instance().create(architecture, cfg,
                                                      generation_cfg, nntr_cfg);
    if (!model) {
      std::cerr << "Unknown architecture: " << architecture << std::endl;
      std::cerr << "Registered architectures:";
      quick_ai::Factory::Instance().printRegistered(std::cerr);
      std::cerr << std::endl;
      return EXIT_FAILURE;
    }
    if (architecture == "Qwen3ASRForConditionalGeneration" && !audio_path.empty()) {
      auto *asr_model = dynamic_cast<quick_ai::Qwen3ASRCausalLM*>(model.get());
      if (asr_model) {
        asr_model->setAudioPath(audio_path);
        
        quick_ai::AudioPreprocessor preprocessor;
        auto pcm = preprocessor.loadWav(audio_path);
        unsigned int audio_frames = pcm.size() / 160;
        constexpr unsigned int MAX_AUDIO_FRAMES = 1200 * 100; // 20 minutes (1200s * 100fps)
        if (audio_frames > MAX_AUDIO_FRAMES) {
          audio_frames = MAX_AUDIO_FRAMES;
        }
        unsigned int audio_seq_len = get_feat_extract_output_lengths(audio_frames);
        
        std::string pad_tokens = "";
        for (unsigned int idx = 0; idx < audio_seq_len; ++idx) {
          pad_tokens += "<|audio_pad|>";
        }
        input_text = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>" + 
                     pad_tokens + "<|audio_end|>" + input_text + "<|im_end|>\n<|im_start|>assistant\n";
        
        system_head_prompt.clear();
        system_tail_prompt.clear();
      }
    }

    model->initialize();
    model->load_weight(weight_file);
    model->repack_weight();

    bool do_sample = generation_cfg.value("do_sample", false);

#ifdef PROFILE
    start_peak_tracker();
#endif
#if defined(_WIN32)
    model->run(input_text.c_str(), do_sample, system_head_prompt.c_str(),
               system_tail_prompt.c_str());
#else
    model->run(input_text, do_sample, system_head_prompt, system_tail_prompt);
#endif
#ifdef PROFILE
    stop_and_print_peak();
#endif
    auto finish_time = std::chrono::high_resolution_clock::now();
    auto e2e_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      finish_time - start_time);
    std::cout << "[e2e time]: " << e2e_duration.count() << " ms \n";
    printMemoryUsage();

  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
