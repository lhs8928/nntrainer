// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nemotron_h_causallm.cpp
 * @date   09 November 2026
 * @brief  Nemotron-H hybrid Mamba2 / MoE / attention causal language model
 * @see    https://github.com/nntrainer/nntrainer
 * @author Claude <noreply@anthropic.com>
 * @bug    No known bugs except for NYI items
 */

#include "nemotron_h_causallm.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <app_context.h>
#include <engine.h>
#include <llm_util.hpp>
#include <model.h>
#include <neuralnet.h>

#include <mamba2_ssm_layer.h>
#include <nemotron_h_moe_layer.h>

namespace causallm {

namespace {

/** IEEE half -> float (no dependency on ENABLE_FP16) */
inline float half_to_float(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else { // subnormal
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3ffu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

std::vector<char> parseLayerTypes(json &cfg) {
  std::vector<char> types;
  auto from_name = [](const std::string &n) -> char {
    if (n == "linear_attention" || n == "mamba" || n == "M")
      return 'M';
    if (n == "moe" || n == "E")
      return 'E';
    if (n == "full_attention" || n == "attention" || n == "*")
      return '*';
    if (n == "mlp" || n == "-")
      return '-';
    throw std::invalid_argument("NemotronH: unknown block type " + n);
  };
  for (const char *key : {"layers_block_type", "layer_types"}) {
    if (cfg.contains(key) && cfg[key].is_array()) {
      for (auto &v : cfg[key])
        types.push_back(from_name(v.get<std::string>()));
      return types;
    }
  }
  if (cfg.contains("hybrid_override_pattern")) {
    for (char c : cfg["hybrid_override_pattern"].get<std::string>())
      types.push_back(from_name(std::string(1, c)));
    return types;
  }
  throw std::invalid_argument(
    "NemotronH: config needs layers_block_type or hybrid_override_pattern");
}

} // namespace

NemotronHCausalLM::NemotronHCausalLM(json &cfg, json &generation_cfg,
                                     json &nntr_cfg) :
  Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
  CausalLM(cfg, generation_cfg, nntr_cfg) {
  setupParameters(cfg, generation_cfg, nntr_cfg);
}

void NemotronHCausalLM::setupParameters(json &cfg, json &generation_cfg,
                                        json &nntr_cfg) {
  try {
    layer_types_ = parseLayerTypes(cfg);
    NUM_LAYERS = static_cast<int>(layer_types_.size());

    NORM_EPS =
      cfg.value("layer_norm_epsilon", cfg.value("rms_norm_eps", 1e-5f));
    // Nemotron-H attention has no positional encoding: rope off in mha_core.
    ROPE_THETA = 0;

    MAMBA_NUM_HEADS = cfg.value("mamba_num_heads", 128u);
    MAMBA_HEAD_DIM = cfg.value("mamba_head_dim", 64u);
    MAMBA_N_GROUPS = cfg.value("n_groups", cfg.value("mamba_n_groups", 8u));
    SSM_STATE_SIZE = cfg.value("ssm_state_size", 128u);
    CONV_KERNEL = cfg.value("conv_kernel", cfg.value("mamba_d_conv", 4u));
    TIME_STEP_MIN =
      cfg.value("time_step_min", cfg.value("mamba_dt_min", 0.001f));
    MAMBA_PROJ_BIAS = cfg.value("use_bias", false);

    NUM_EXPERTS = cfg.value("n_routed_experts", 0u);
    NUM_EXPERTS_PER_TOK = cfg.value("num_experts_per_tok", 1u);
    MOE_INTERMEDIATE_SIZE = cfg.value("moe_intermediate_size", 0u);
    MOE_SHARED_INTERMEDIATE_SIZE =
      cfg.value("moe_shared_expert_intermediate_size", 0u);
    if (cfg.value("n_shared_experts", 1u) == 0)
      MOE_SHARED_INTERMEDIATE_SIZE = 0;
    ROUTED_SCALING_FACTOR = cfg.value("routed_scaling_factor", 1.0f);
    NORM_TOPK_PROB = cfg.value("norm_topk_prob", true);
    N_GROUP = cfg.value("n_group", 1u);
    TOPK_GROUP = cfg.value("topk_group", 1u);

    if (cfg.contains("moe_latent_size") && !cfg["moe_latent_size"].is_null())
      throw std::invalid_argument(
        "NemotronH: moe_latent_size (latent MoE) is not supported yet");

    EMBEDDING_TABLE_PATH = nntr_cfg.value("embedding_file_name", std::string());
    EMBEDDING_TABLE_DTYPE =
      nntr_cfg.value("embedding_file_dtype", std::string("FP32"));
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("NemotronHCausalLM: config error: ") +
                             e.what());
  }
}

Tensor NemotronHCausalLM::createNorm(const std::string &name, Tensor x) {
  LayerHandle norm(
    createLayer("rms_norm", {withKey("name", name),
                             withKey("epsilon", std::to_string(NORM_EPS)),
                             withKey("packed", "false")}));
  return norm(x);
}

Tensor NemotronHCausalLM::createMambaBlock(int layer_id, Tensor x) {
  const std::string p = "layer" + std::to_string(layer_id);
  const unsigned int d_inner = MAMBA_NUM_HEADS * MAMBA_HEAD_DIM;
  const unsigned int conv_dim = d_inner + 2 * MAMBA_N_GROUPS * SSM_STATE_SIZE;
  const unsigned int proj = d_inner + conv_dim + MAMBA_NUM_HEADS;

  LayerHandle in_proj(
    createLayer("fully_connected",
                {withKey("name", p + "_mamba_in_proj"), withKey("unit", proj),
                 withKey("disable_bias", MAMBA_PROJ_BIAS ? "false" : "true")}));
  Tensor h = in_proj(x);

  LayerHandle ssm(createLayer(
    "mamba2_ssm", {withKey("name", p + "_mamba_ssm"),
                   withKey("mamba_num_heads", MAMBA_NUM_HEADS),
                   withKey("mamba_head_dim", MAMBA_HEAD_DIM),
                   withKey("mamba_n_groups", MAMBA_N_GROUPS),
                   withKey("mamba_state_size", SSM_STATE_SIZE),
                   withKey("mamba_conv_kernel", CONV_KERNEL),
                   withKey("mamba_norm_eps", std::to_string(NORM_EPS)),
                   withKey("mamba_dt_min", std::to_string(TIME_STEP_MIN))}));
  Tensor y = ssm(h);

  LayerHandle out_proj(
    createLayer("fully_connected",
                {withKey("name", p + "_mamba_out_proj"), withKey("unit", DIM),
                 withKey("disable_bias", MAMBA_PROJ_BIAS ? "false" : "true")}));
  return out_proj(y);
}

Tensor NemotronHCausalLM::createMoEBlock(int layer_id, Tensor x) {
  LayerHandle moe(createLayer(
    "nemotron_h_moe",
    {withKey("name", "layer" + std::to_string(layer_id) + "_moe"),
     withKey("num_experts", NUM_EXPERTS),
     withKey("num_experts_per_token", NUM_EXPERTS_PER_TOK),
     withKey("unit", MOE_INTERMEDIATE_SIZE),
     withKey("shared_expert_size", MOE_SHARED_INTERMEDIATE_SIZE),
     withKey("routed_scaling_factor", std::to_string(ROUTED_SCALING_FACTOR)),
     withKey("norm_topk_prob", NORM_TOPK_PROB ? "true" : "false"),
     withKey("n_group", N_GROUP), withKey("topk_group", TOPK_GROUP)}));
  return moe(x);
}

Tensor NemotronHCausalLM::createMlpBlock(int layer_id, Tensor x) {
  // A dense relu^2 MLP is a one-expert MoE whose (zero) router always picks
  // the expert with weight sigmoid(0)/sigmoid(0) == 1; the converter writes a
  // zero router and zero correction bias for these blocks.
  LayerHandle mlp(createLayer(
    "nemotron_h_moe",
    {withKey("name", "layer" + std::to_string(layer_id) + "_mlp"),
     withKey("num_experts", 1), withKey("num_experts_per_token", 1),
     withKey("unit", INTERMEDIATE_SIZE), withKey("shared_expert_size", 0),
     withKey("routed_scaling_factor", "1.0"),
     withKey("norm_topk_prob", "true")}));
  return mlp(x);
}

Tensor NemotronHCausalLM::createAttentionBlock(int layer_id, Tensor x) {
  return createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, HEAD_DIM, x, x, x);
}

std::pair<Tensor, Tensor> NemotronHCausalLM::constructModel() {
  LayerHandle input_layer(createLayer(
    "input", {withKey("name", "input0"),
              withKey("input_shape", "1:" + std::to_string(INIT_SEQ_LEN) + ":" +
                                       std::to_string(DIM))}));
  Tensor input = input_layer(Tensor());
  Tensor x = input;

  for (int i = 0; i < NUM_LAYERS; ++i) {
    Tensor h = createNorm("layer" + std::to_string(i) + "_norm", x);
    switch (layer_types_[i]) {
    case 'M':
      h = createMambaBlock(i, h);
      break;
    case 'E':
      h = createMoEBlock(i, h);
      break;
    case '*':
      h = createAttentionBlock(i, h);
      break;
    case '-':
      h = createMlpBlock(i, h);
      break;
    default:
      throw std::invalid_argument("NemotronH: bad block type");
    }
    x = x.add(h);
  }

  Tensor norm_out = createNorm("output_norm", x);

  LayerHandle lmhead(createLayer(
    "lm_head", {withKey("name", "output_of_causallm"),
                withKey("weight_dtype", LMHEAD_DTYPE),
                withKey("unit", NUM_VOCAB), withKey("disable_bias", "true")}));
  return {input, lmhead(norm_out)};
}

void NemotronHCausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  auto &engine = nntrainer::Engine::Global();
  auto tryRegister = [&](auto factory) {
    try {
      engine.registerLayerFactory("cpu", factory);
    } catch (std::invalid_argument &e) {
      // already registered by another model instance
    }
  };
  tryRegister(nntrainer::createLayer<causallm::Mamba2SSMLayer>);
  tryRegister(nntrainer::createLayer<causallm::NemotronHMoELayer>);
}

void NemotronHCausalLM::load_weight(const std::string &weight_path) {
  Transformer::load_weight(weight_path);
  // A relative table path that does not resolve from the working directory
  // is taken relative to the weight file (tools such as nntr_quantize hand
  // the nntr_config.json values over unresolved).
  std::filesystem::path table(EMBEDDING_TABLE_PATH);
  if (!EMBEDDING_TABLE_PATH.empty() && table.is_relative() &&
      !std::filesystem::exists(table)) {
    auto beside = std::filesystem::path(weight_path).parent_path() / table;
    if (std::filesystem::exists(beside))
      EMBEDDING_TABLE_PATH = beside.string();
  }
  loadEmbeddingTable();
}

void NemotronHCausalLM::loadEmbeddingTable() {
  if (EMBEDDING_TABLE_PATH.empty())
    throw std::runtime_error(
      "NemotronHCausalLM: nntr_config.embedding_file_name is required");

  std::ifstream f(EMBEDDING_TABLE_PATH, std::ios::binary | std::ios::ate);
  if (!f.is_open())
    throw std::runtime_error("NemotronHCausalLM: cannot open embedding file " +
                             EMBEDDING_TABLE_PATH);
  const size_t count = static_cast<size_t>(NUM_VOCAB) * DIM;
  const size_t bytes = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  embedding_table_.resize(count);

  if (EMBEDDING_TABLE_DTYPE == "FP32" || EMBEDDING_TABLE_DTYPE == "fp32") {
    if (bytes != count * sizeof(float))
      throw std::runtime_error("NemotronHCausalLM: FP32 embedding file size " +
                               std::to_string(bytes) + " != expected " +
                               std::to_string(count * sizeof(float)));
    f.read(reinterpret_cast<char *>(embedding_table_.data()),
           static_cast<std::streamsize>(bytes));
  } else if (EMBEDDING_TABLE_DTYPE == "FP16" ||
             EMBEDDING_TABLE_DTYPE == "fp16") {
    if (bytes != count * sizeof(uint16_t))
      throw std::runtime_error("NemotronHCausalLM: FP16 embedding file size " +
                               std::to_string(bytes) + " != expected " +
                               std::to_string(count * sizeof(uint16_t)));
    std::vector<uint16_t> half(count);
    f.read(reinterpret_cast<char *>(half.data()),
           static_cast<std::streamsize>(bytes));
    for (size_t i = 0; i < count; ++i)
      embedding_table_[i] = half_to_float(half[i]);
  } else {
    throw std::runtime_error("NemotronHCausalLM: unsupported embedding dtype " +
                             EMBEDDING_TABLE_DTYPE);
  }
}

const void *NemotronHCausalLM::lookupEmbedding(int token_id) const {
  if (token_id < 0 || token_id >= NUM_VOCAB || embedding_table_.empty())
    return nullptr;
  return embedding_table_.data() + static_cast<size_t>(token_id) * DIM;
}

std::vector<float>
NemotronHCausalLM::embedTokens(const std::vector<int> &ids) const {
  std::vector<float> out(ids.size() * static_cast<size_t>(DIM));
  for (size_t t = 0; t < ids.size(); ++t) {
    const float *row = static_cast<const float *>(lookupEmbedding(ids[t]));
    if (row == nullptr)
      throw std::out_of_range("NemotronHCausalLM: token id " +
                              std::to_string(ids[t]) + " is out of range");
    std::copy(row, row + DIM, out.begin() + t * DIM);
  }
  return out;
}

void NemotronHCausalLM::run(const WSTR prompt, bool do_sample,
                            const WSTR system_prompt, const WSTR tail_prompt,
                            bool log_output) {
  if (!is_initialized)
    throw std::runtime_error("NemotronHCausalLM: call initialize() first");
  ensureTokenizer();
  if (tokenizer == nullptr)
    throw std::runtime_error("NemotronHCausalLM: no tokenizer configured");

  const std::string text = system_prompt + prompt + tail_prompt;
  if (log_output)
    std::cout << text << std::endl;
  std::vector<int> ids = tokenizer->Encode(text);
  std::vector<float> embeds = embedTokens(ids);
  run_with_embeddings(embeds.data(), ids.size(), ids, do_sample, log_output);
}

void NemotronHCausalLM::run_with_embeddings(const void *prefill_embeds,
                                            size_t n_tokens,
                                            std::vector<int> seed_tokens,
                                            bool do_sample, bool log_output) {
  if (!is_initialized)
    throw std::runtime_error("NemotronHCausalLM: call initialize() first");
  if (n_tokens == 0)
    throw std::invalid_argument("NemotronHCausalLM: empty prompt");

  const unsigned int input_len = static_cast<unsigned int>(n_tokens);
  if (input_len >= MAX_SEQ_LEN)
    throw std::invalid_argument(
      "NemotronHCausalLM: prompt of " + std::to_string(input_len) +
      " tokens does not fit max_seq_len " + std::to_string(MAX_SEQ_LEN));

  ensureTokenizer();
  const bool has_tokenizer = tokenizer != nullptr;

  auto start_total = std::chrono::high_resolution_clock::now();
  beginSamplingRun();
  allocateAndBindKVCache();

  // Every request is a new conversation: the Mamba layers reset their states
  // on a step starting at position 0, the attention layers start writing the
  // KV cache at row 0.
  global_token_len = 0;
  setKVCachePosition(0);

  output_list.assign(BATCH_SIZE, "");
  pending_ids_.clear();
  generated_ids_.clear();
  prefill_logits_.clear();
  std::fill(ids_history, ids_history + BATCH_SIZE * MAX_SEQ_LEN, 0u);
  for (unsigned int b = 0; b < BATCH_SIZE; ++b)
    for (size_t i = 0; i < seed_tokens.size() && i < n_tokens; ++i)
      ids_history[b * MAX_SEQ_LEN + i] =
        static_cast<unsigned int>(std::max(0, seed_tokens[i]));

  auto *nn = static_cast<nntrainer::NeuralNetwork *>(model.get());
  const size_t n_inputs = nn->getInputDimension().size();
  const size_t plane = static_cast<size_t>(BATCH_SIZE) * INIT_SEQ_LEN * DIM;
  std::vector<float> plane_buf(plane, 0.0f);
  // KV-cache inputs are resolved by name inside incrementalInference(); every
  // other graph input is the embedding plane.
  std::vector<float *> inputs(n_inputs, plane_buf.data());

  const auto *embeds = static_cast<const float *>(prefill_embeds);

  // ---------------------------------------------------------------- prefill
  auto start_prefill = std::chrono::high_resolution_clock::now();
  std::vector<float *> outputs;
  for (unsigned int o = 0; o < input_len; o += INIT_SEQ_LEN) {
    const unsigned int clen =
      std::min<unsigned int>(INIT_SEQ_LEN, input_len - o);
    std::fill(plane_buf.begin(), plane_buf.end(), 0.0f);
    for (unsigned int b = 0; b < BATCH_SIZE; ++b)
      std::copy(embeds + static_cast<size_t>(o) * DIM,
                embeds + static_cast<size_t>(o + clen) * DIM,
                plane_buf.begin() +
                  static_cast<size_t>(b) * INIT_SEQ_LEN * DIM);
    for (auto *out : outputs)
      releaseLogitsBuf(out);
    outputs = incrementalInference(BATCH_SIZE, inputs, clen, o, o + clen);
    advanceKVCachePosition(clen);
  }

  prefill_logits_.assign(outputs[0], outputs[0] + NUM_VOCAB);
  std::vector<bool> eos_list(BATCH_SIZE, false);
  std::vector<unsigned int> id_list =
    generate(outputs[0], do_sample, REPETITION_PENALTY, ids_history, input_len);
  for (auto *out : outputs)
    releaseLogitsBuf(out);

  auto finish_prefill = std::chrono::high_resolution_clock::now();

  auto accept = [&](const std::vector<unsigned int> &ids, unsigned int pos) {
    generated_ids_.push_back(ids[0]);
    if (has_tokenizer) {
      registerOutputs(tokenizer, ids, pos, eos_list, log_output);
    } else {
      for (unsigned int b = 0; b < BATCH_SIZE; ++b)
        ids_history[b * MAX_SEQ_LEN + pos] = ids[b];
    }
    for (unsigned int b = 0; b < BATCH_SIZE; ++b)
      if (std::find(EOS_TOKEN_ID.begin(), EOS_TOKEN_ID.end(), ids[b]) !=
          EOS_TOKEN_ID.end())
        eos_list[b] = true;
  };
  accept(id_list, input_len);

  // ------------------------------------------------------------- generation
  const unsigned int budget = NUM_TO_GENERATE > 0
                                ? static_cast<unsigned int>(NUM_TO_GENERATE)
                                : MAX_SEQ_LEN;
  unsigned int generation_cnt = 1;
  auto start_generation = std::chrono::high_resolution_clock::now();
  for (unsigned int pos = input_len;
       pos + 1 < MAX_SEQ_LEN && generation_cnt < budget; ++pos) {
    if (std::all_of(eos_list.begin(), eos_list.end(), [](bool v) { return v; }))
      break;
    std::fill(plane_buf.begin(), plane_buf.end(), 0.0f);
    for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
      const float *row = static_cast<const float *>(
        lookupEmbedding(static_cast<int>(id_list[b])));
      if (row == nullptr)
        throw std::out_of_range("NemotronHCausalLM: generated id out of range");
      std::copy(row, row + DIM,
                plane_buf.begin() +
                  static_cast<size_t>(b) * INIT_SEQ_LEN * DIM);
    }
    auto step = incrementalInference(BATCH_SIZE, inputs, 1, pos, pos + 1);
    advanceKVCachePosition(1);
    id_list =
      generate(step[0], do_sample, REPETITION_PENALTY, ids_history, pos + 1);
    for (auto *out : step)
      releaseLogitsBuf(out);
    accept(id_list, pos + 1);
    ++generation_cnt;
  }
  auto finish = std::chrono::high_resolution_clock::now();
  global_token_len = input_len + generation_cnt;

  // flush tokens held back for punctuation / partial UTF-8
  if (has_tokenizer && !pending_ids_.empty()) {
    std::string tail = tokenizer->Decode(pending_ids_);
    if (log_output && streamer_ == nullptr)
      std::cout << tail;
    output_list[0].append(tail);
    pending_ids_.clear();
  }

  using ms = std::chrono::milliseconds;
  const auto prefill_ms =
    std::chrono::duration_cast<ms>(finish_prefill - start_prefill).count();
  const auto gen_ms =
    std::chrono::duration_cast<ms>(finish - start_generation).count();
  const auto total_ms =
    std::chrono::duration_cast<ms>(finish - start_total).count();
  if (log_output) {
    std::cout << "\n\n=================[ Nemotron-H with NNTrainer ]========\n"
              << "prefill: " << input_len << " tokens, " << prefill_ms
              << " ms\n"
              << "generation: " << generation_cnt << " tokens, " << gen_ms
              << " ms\n"
              << "total: " << total_ms << " ms\n"
              << "=====================================================\n";
  }
  performance_metrics.prefill_tokens = input_len;
  performance_metrics.prefill_duration_ms = prefill_ms;
  performance_metrics.generation_tokens = generation_cnt;
  performance_metrics.generation_duration_ms = gen_ms;
  performance_metrics.total_duration_ms = total_ms;
  has_run_ = true;
}

} // namespace causallm
