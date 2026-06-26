// SPDX-License-Identifier: Apache-2.0
/**
 * @file   model_runtime.h
 * @brief  CausalLM app runtime helpers: config loading, model factory
 *         registration, model build/run, and process memory sampling.
 *         Extracted from main.cpp so the orchestration layer can reuse them.
 */
#ifndef __CAUSALLM_MODEL_RUNTIME_H__
#define __CAUSALLM_MODEL_RUNTIME_H__

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>

#include "json.hpp"
#include <transformer.h>

using json = nlohmann::json;

/**
 * @brief Parsed model configuration
 *        (config.json / generation_config.json / nntr_config.json).
 */
struct Config {
  json cfg;
  json generation_cfg;
  json nntr_cfg;
};

/* ----- process memory sampling ----- */
extern std::atomic<size_t> peak_rss_kb;
extern std::atomic<bool> tracking_enabled;

void printMemoryUsage();
size_t read_vm_rss_kb();
size_t read_private_rss_kb();
void start_peak_tracker();
void stop_and_print_peak();

/* ----- config / factory / build / run ----- */
std::string resolve_architecture(std::string model_type,
                                 const std::string &architecture);
void registerCausalModels();
Config getConfigs(const std::string model_path);
std::string getArchitecture(Config &config);

/**
 * @brief Build an instance with an explicit reference model.
 *        ref_base == nullptr -> standalone instance (loads weights);
 *        ref_base != nullptr -> shares ref_base's weights (load is skipped).
 *        Independent of the Factory's implicit "first instance" bookkeeping.
 */
std::shared_ptr<causallm::Transformer>
buildInstance(Config &config, const std::string &architecture,
              const std::string &model_path,
              std::shared_ptr<causallm::Transformer> ref_base);

std::string runModel(Config config, std::string architecture,
                     const std::string model_path,
                     std::shared_ptr<causallm::Transformer> model,
                     const std::string &input_prompt = "");

#endif // __CAUSALLM_MODEL_RUNTIME_H__
