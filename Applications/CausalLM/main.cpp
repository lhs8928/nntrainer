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
 * See the License for the_project_root.
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
#include "lfm2_causallm.h"
#include "qwen3_causallm.h"
#include "qwen3_embedding.h"
#include "qwen3_moe_causallm.h"
#include "qwen3_slim_moe_causallm.h"
#include "timm_vit/timm_vit_transformer.h"
#include "vjepa2_vit/vjepa2_vit.h"
#include "vjepa_lfm2_vl/vjepa_lfm2_vl.h"
#include <models/gemma3/function.h>
#if !defined(_WIN32)
#include <sys/resource.h>
#endif

#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>

#include "orchestrator.h"

using json = nlohmann::json;

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
  causallm::Factory::Instance().registerModel(
    "VJEPA2ViT", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::VJEPA2ViT>(cfg, generation_cfg,
                                                   nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Lfm2ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Lfm2CausalLM>(cfg, generation_cfg,
                                                      nntr_cfg);
    });
  causallm::Factory::Instance().registerModel(
    "Lfm2VLVJepa21BModel", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::VjepaLfm2ForConditionalGeneration>(
        cfg, generation_cfg, nntr_cfg);
    });
}

void displayMenu() {
  std::cout << "\n====== CausalLM Orchestrator ======" << std::endl;
  std::cout << "1 <path> [pool]         - Create service" << std::endl;
  std::cout << "2 <id>                  - Serve service (interactive)" << std::endl;
  std::cout << "3                       - List services" << std::endl;
  std::cout << "4 <id>                  - Remove service" << std::endl;
  std::cout << "5                       - Exit" << std::endl;
  std::cout << "===================================\n" << std::endl;
}

void interactiveShell(Orchestrator &orch, const std::string &init_path = "", int init_pool = 2) {
  if (!init_path.empty()) {
    int id = orch.addService(init_path, init_pool);
    if (id >= 0) {
      orch.serve(id);
    }
    return;
  }

  while (true) {
    displayMenu();
    g_current_prompt = "Enter choice and option: ";
    std::cout << g_current_prompt << std::flush;
    std::string line;
    std::getline(std::cin, line);
    g_current_prompt = "";
    if (line.empty())
      continue;

    std::istringstream iss(line);
    int choice;
    iss >> choice;

    switch (choice) {
      case 1: {
        std::string path;
        if (iss >> path) {
          int pool = 2;
          int pool_val;
          if (iss >> pool_val) {
            pool = pool_val;
          }
          orch.addService(path, pool);
        } else {
          std::cerr << "Usage: 1 <model_path> [pool_size]" << std::endl;
        }
        break;
      }
      case 2: {
        int id;
        if (iss >> id) {
          orch.serve(id);
        } else {
          std::cerr << "Usage: 2 <id>" << std::endl;
        }
        break;
      }
      case 3:
        orch.list();
        break;
      case 4: {
        int id;
        if (iss >> id)
          orch.remove(id);
        else
          std::cerr << "Usage: 4 <id>" << std::endl;
        break;
      }
      case 5:
        std::cout << "Exiting..." << std::endl;
        return;
      default:
        std::cerr << "Invalid choice. Please enter 1, 2, 3, 4, or 5."
                  << std::endl;
    }
  }
}

/**
 * @brief Entry point: register models, then either run the interactive
 *        orchestrator shell or serve a model given on the command line.
 */
int main(int argc, char *argv[]) {
  auto start_time = std::chrono::high_resolution_clock::now();

  registerCausalModels();

  // Sharing policy (orchestration-level). Override at runtime with
  // CAUSALLM_SHARING=0 to disable (non-shared baseline) without rebuilding.
  bool sharing = true;
  if (const char *e = std::getenv("CAUSALLM_SHARING"))
    sharing = !(std::string(e) == "0" || std::string(e) == "off");

  Orchestrator orch(sharing);

  std::string init_path = (argc >= 2) ? argv[1] : "";
  int init_pool = (argc >= 3) ? std::atoi(argv[2]) : 2;

  interactiveShell(orch, init_path, init_pool);

  auto finish_time = std::chrono::high_resolution_clock::now();
  auto e2e = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_time - start_time);
  std::cout << "[e2e time]: " << e2e.count() << " ms \n";
  printMemoryUsage();
  return EXIT_SUCCESS;
}
