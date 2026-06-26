// SPDX-License-Identifier: Apache-2.0
/**
 * @file   model_runtime.cpp
 * @brief  Implementation of CausalLM app runtime helpers (see model_runtime.h).
 */
#include "model_runtime.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <thread>
#include <vector>

#include <app_context.h>
#include <factory.h>

#include "causal_lm.h"
#include "chat_template.h"
#include "deberta_v2.h"
#include "embedding_gemma.h"
#include "gemma3_causallm.h"
#include "gemma4_causallm.h"
#if !defined(_WIN32)
#include "gptoss_cached_slim_causallm.h"
#endif
#include "gptoss_causallm.h"
#if !defined(_WIN32) && !defined(__ANDROID__)
#include "multilingual_tinybert_16mb.h"
#endif
#include "qwen2_causallm.h"
#include "qwen2_embedding.h"
#if !defined(_WIN32)
#include "qwen3_cached_slim_moe_causallm.h"
#endif
#include "qwen3_causallm.h"
#include "qwen3_embedding.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"
#include "timm_vit/timm_vit_transformer.h"
#include <models/gemma3/function.h>
#if !defined(_WIN32)
#include <sys/resource.h>
#endif

std::atomic<size_t> peak_rss_kb{0};
std::atomic<bool> tracking_enabled{true};

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
    } else if (architecture == "TimmViT" ||
               architecture == "vit_base_patch16_siglip_224") {
      return "TimmViT";
    } else if (architecture == "deberta-v2" ||
               architecture == "DebertaV2Model" ||
               architecture == "DebertaV2ForMaskedLM") {
      return "DebertaV2";
    } else {
      throw std::invalid_argument(
        "Unsupported architecture for embedding model: " + architecture);
    }
  }

  if (architecture == "TimmViT" ||
      architecture == "vit_base_patch16_siglip_224") {
    return "TimmViT";
  }

  if (architecture == "Gemma4ForConditionalGeneration") {
    return "Gemma4ForCausalLM";
  }

  return architecture;
}

void registerCausalModels() {
  /** Register all runnable causallm models to factory */
  causallm::Factory::Instance().registerModel(
    "LlamaForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::CausalLM>(cfg, generation_cfg,
                                                  nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen2ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen2CausalLM>(cfg, generation_cfg,
                                                       nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen2Embedding", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen2Embedding>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3CausalLM>(cfg, generation_cfg,
                                                       nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3MoeForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3MoECausalLM>(cfg, generation_cfg,
                                                          nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Qwen3SlimMoeForCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3SlimMoECausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
#if !defined(_WIN32)
  causallm::Factory::Instance().registerModel(
    "Qwen3CachedSlimMoeForCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3CachedSlimMoECausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
#endif
  causallm::Factory::Instance().registerModel(
    "Qwen3Embedding", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Qwen3Embedding>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "GptOssForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::GptOssForCausalLM>(cfg, generation_cfg,
                                                           nntr_cfg);
    });
#if !defined(_WIN32)
  causallm::Factory::Instance().registerModel(
    "GptOssCachedSlimCausalLM",
    [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::GptOssCachedSlimCausalLM>(
        cfg, generation_cfg, nntr_cfg);
    });
#endif
  causallm::Factory::Instance().registerModel(
    "Gemma3ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Gemma3CausalLM>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Gemma4ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Gemma4CausalLM>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "EmbeddingGemma", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::EmbeddingGemma>(cfg, generation_cfg,
                                                        nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "DebertaV2", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::DebertaV2>(cfg, generation_cfg,
                                                   nntr_cfg);
    });
#if !defined(_WIN32) && !defined(__ANDROID__)
  causallm::Factory::Instance().registerModel(
    "MultilingualTinyBert", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::MultilingualTinyBert>(
        cfg, generation_cfg, nntr_cfg);
    });
#endif
  causallm::Factory::Instance().registerModel(
    "TimmViT", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::TimmViTTransformer>(cfg, generation_cfg,
                                                            nntr_cfg);
    });
}

Config getConfigs(const std::string model_path) {
  Config config;
  config.cfg = causallm::LoadJsonFile(model_path + "/config.json");
  config.generation_cfg = json::object();
  std::string generation_config_path = model_path + "/generation_config.json";
  if (std::filesystem::exists(generation_config_path)) {
    config.generation_cfg = causallm::LoadJsonFile(generation_config_path);
  }
  config.nntr_cfg = causallm::LoadJsonFile(model_path + "/nntr_config.json");

  return config;
}

std::string getArchitecture(Config &config) {
  json &cfg = config.cfg;
  json &generation_cfg = config.generation_cfg;
  json &nntr_cfg = config.nntr_cfg;

  std::string architecture;
  if (cfg.contains("architectures") && cfg["architectures"].is_array() &&
      !cfg["architectures"].empty()) {
    architecture = cfg["architectures"].get<std::vector<std::string>>()[0];
  } else if (cfg.contains("architecture") && cfg["architecture"].is_string()) {
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

  return architecture;
}

std::shared_ptr<causallm::Transformer>
buildInstance(Config &config, const std::string &architecture,
              const std::string &model_path,
              std::shared_ptr<causallm::Transformer> ref_base) {
  const std::string weight_file =
    model_path + "/" + config.nntr_cfg["model_file_name"].get<std::string>();

  auto model = causallm::Factory::Instance().create(
    architecture, config.cfg, config.generation_cfg, config.nntr_cfg);
  if (!model) {
    std::cerr << "Unknown architecture: " << architecture << std::endl;
    return nullptr;
  }

  // Explicit reference (not the Factory's implicit base). nullptr -> standalone.
  model->initialize(ref_base);
  // When sharing a loaded ref, NeuralNetwork::load() returns early (no reload).
  model->load_weight(weight_file);

  return model;
}

std::string runModel(Config config, std::string architecture,
                     const std::string model_path,
                     std::shared_ptr<causallm::Transformer> model,
                     const std::string &input_prompt) {
  std::string input_text;
  std::string system_head_prompt = "";
  std::string system_tail_prompt = "";

  json &cfg = config.cfg;
  json &generation_cfg = config.generation_cfg;
  json &nntr_cfg = config.nntr_cfg;

  // Load chat template from tokenizer_config.json or jinja (if available)
  std::optional<causallm::ChatTemplate> chat_template;
  if (causallm::ChatTemplate::Exists(model_path)) {
    chat_template.emplace(causallm::ChatTemplate::Load(model_path));
  }

  // Use provided input or fallback to config
  if (!input_prompt.empty()) {
    input_text = input_prompt;
  } else {
    // Determine input text from config
    if (nntr_cfg.contains("chat_input")) {
      if (chat_template.has_value()) {
        input_text = chat_template->apply(nntr_cfg["chat_input"]);
        system_head_prompt.clear();
        system_tail_prompt.clear();
      } else {
        std::cerr << "[Warning] 'chat_input' is set but support for model "
                      "architecture '"
                  << architecture
                  << "' is not implemented. Falling back to 'sample_input'."
                  << std::endl;
        input_text = nntr_cfg["sample_input"].get<std::string>();
      }
    } else {
      input_text = nntr_cfg["sample_input"].get<std::string>();
    }
  }

  if (nntr_cfg.contains("system_prompt")) {
    system_head_prompt =
      nntr_cfg["system_prompt"]["head_prompt"].get<std::string>();
    system_tail_prompt =
      nntr_cfg["system_prompt"]["tail_prompt"].get<std::string>();
  }

  bool do_sample = generation_cfg.value("do_sample", false);

#ifdef PROFILE
  start_peak_tracker();
#endif
#if defined(_WIN32)
  model->run(input_text.c_str(), do_sample, system_head_prompt.c_str(),
             system_tail_prompt.c_str(), false);
#else
  model->run(input_text, do_sample, system_head_prompt, system_tail_prompt,
             false);
#endif
#ifdef PROFILE
  stop_and_print_peak();
#endif

  auto causal_lm_model = dynamic_cast<causallm::CausalLM *>(model.get());
  return causal_lm_model->getOutput(0);
}
