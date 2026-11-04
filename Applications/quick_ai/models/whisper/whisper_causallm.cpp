// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   whisper_causallm.cpp
 * @date   04 November 2026
 * @brief  Whisper (large-v3) speech recognition: encoder + text decoder
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "whisper_causallm.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>

#include <app_context.h>
#include <engine.h>
#include <layer.h>

#include "audio_preprocessor.h"
#include "wav2vec2_ctc.h"
#include <cross_attention.h>
#include <llm_util.hpp>
#include <mha_core.h>
#include <position_embedding.h>
#include <tie_word_embedding.h>

namespace quick_ai {

namespace {

using ml::train::createLayer;
using ml::train::LayerHandle;

constexpr unsigned int SAMPLE_RATE = 16000;
constexpr unsigned int HOP = 160;

/** @brief "Q4_0-FP16" -> "FP16" */
std::string activationOf(const std::string &tensor_type) {
  const size_t dash = tensor_type.find('-');
  return dash == std::string::npos ? tensor_type : tensor_type.substr(dash + 1);
}

/** @brief Whisper's config keys under the names the Transformer base reads */
json &whisperTextConfig(json &cfg) {
  auto alias = [&cfg](const char *key, const char *whisper_key) {
    if (!cfg.contains(key) && cfg.contains(whisper_key))
      cfg[key] = cfg[whisper_key];
  };
  alias("hidden_size", "d_model");
  alias("num_hidden_layers", "decoder_layers");
  alias("num_attention_heads", "decoder_attention_heads");
  alias("intermediate_size", "decoder_ffn_dim");
  alias("max_position_embeddings", "max_target_positions");
  if (!cfg.contains("tie_word_embeddings"))
    cfg["tie_word_embeddings"] = true; // proj_out is embed_tokens
  return cfg;
}

LayerHandle fc(const std::string &name, unsigned int unit, bool bias,
               const std::string &dtype) {
  return LayerHandle(
    createLayer("fully_connected",
                {nntrainer::withKey("name", name),
                 nntrainer::withKey("unit", std::to_string(unit)),
                 nntrainer::withKey("disable_bias", bias ? "false" : "true"),
                 nntrainer::withKey("weight_dtype", dtype)}));
}

LayerHandle layerNorm(const std::string &name, float eps,
                      const std::string &dtype) {
  return LayerHandle(createLayer(
    "layer_normalization",
    {nntrainer::withKey("name", name), nntrainer::withKey("axis", "3"),
     nntrainer::withKey("epsilon", std::to_string(eps)),
     nntrainer::withKey("weight_dtype", dtype)}));
}

LayerHandle gelu(const std::string &name) {
  return LayerHandle(
    createLayer("activation", {nntrainer::withKey("name", name),
                               nntrainer::withKey("activation", "gelu")}));
}

LayerHandle add(const std::string &name) {
  return LayerHandle(
    createLayer("addition", {nntrainer::withKey("name", name)}));
}

/** @brief log(sum(exp(x))) */
double logSumExp(const std::vector<float> &x) {
  const float mx = *std::max_element(x.begin(), x.end());
  double s = 0.0;
  for (float v : x)
    s += std::exp(static_cast<double>(v - mx));
  return mx + std::log(s);
}

/**
 * @brief debug aid: NNTR_WHISPER_DUMP=<dir> writes intermediate values as
 *        raw float32 files there, to compare with a reference
 */
void dumpFloats(const std::string &name, const float *data, size_t n) {
  const char *dir = std::getenv("NNTR_WHISPER_DUMP");
  if (dir == nullptr || *dir == '\0')
    return;
  std::ofstream f(std::string(dir) + "/" + name + ".f32", std::ios::binary);
  f.write(reinterpret_cast<const char *>(data),
          static_cast<std::streamsize>(n * sizeof(float)));
}

/** @brief the custom layers both graphs use, registered once per process */
void registerWhisperLayers() {
  static std::once_flag once;
  std::call_once(once, []() {
    auto &engine = nntrainer::Engine::Global();
    auto *ctx =
      static_cast<nntrainer::AppContext *>(engine.getRegisteredContext("cpu"));
    try {
      ctx->registerFactory(nntrainer::createLayer<CrossAttentionLayer>);
      ctx->registerFactory(nntrainer::createLayer<CrossMemoryWriterLayer>);
      ctx->registerFactory(nntrainer::createLayer<PositionEmbeddingLayer>);
      ctx->registerFactory(nntrainer::createLayer<MHACoreLayer>);
      ctx->registerFactory(nntrainer::createLayer<TieWordEmbedding>);
    } catch (const std::invalid_argument &) {
      // already registered by another model
    }
  });
}

std::atomic<unsigned int> g_instances{0};

} // namespace

/* -------------------------------- encoder -------------------------------- */

void WhisperEncoder::construct(const Dims &dims, const std::string &tensor_type,
                               const std::string &fc_dtype,
                               const std::string &memory_prefix) {
  registerWhisperLayers();
  dims_ = dims;
  act_dtype_ = activationOf(tensor_type);
  const std::string D = std::to_string(dims.d_model);

  model_ = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model_->setProperty({nntrainer::withKey("batch_size", "1"),
                       nntrainer::withKey("epochs", "1"),
                       nntrainer::withKey("model_tensor_type", tensor_type)});

  // log-mel [n_mels][frames] as (1, n_mels, 1, frames): conv1d channels
  input_ = ml::train::Tensor(
    nntrainer::TensorDim(1, dims.n_mels, 1, dims.n_frames,
                         nntrainer::TensorDim::Format::NCHW,
                         nntrainer::TensorDim::DataType::FP32),
    "whisper_mel");

  auto conv = [&](const std::string &name, unsigned int stride) {
    return LayerHandle(createLayer(
      "conv1d",
      {nntrainer::withKey("name", name), nntrainer::withKey("filters", D),
       nntrainer::withKey("kernel_size", "3"),
       nntrainer::withKey("stride", std::to_string(stride)),
       nntrainer::withKey("padding", "1,1"),
       nntrainer::withKey("weight_dtype", "FP32")}));
  };
  ml::train::Tensor h = conv("encoder_conv1", 1)(input_);
  h = gelu("encoder_conv1_gelu")(h);
  h = conv("encoder_conv2", 2)(h);
  h = gelu("encoder_conv2_gelu")(h);

  // (1, D, 1, n_ctx) -> (1, 1, n_ctx, D)
  h = LayerHandle(
    createLayer("permute", {nntrainer::withKey("name", "encoder_permute"),
                            nntrainer::withKey("direction", "2,3,1")}))(h);
  h = LayerHandle(createLayer(
    "position_embedding",
    {nntrainer::withKey("name", "encoder_pos_embed"),
     nntrainer::withKey("max_position", std::to_string(dims.n_ctx))}))(h);

  for (unsigned int i = 0; i < dims.n_layers; ++i) {
    const std::string p = "enc" + std::to_string(i) + "_";
    ml::train::Tensor x = layerNorm(p + "attn_norm", dims.eps, act_dtype_)(h);
    ml::train::Tensor q = fc(p + "q", dims.d_model, true, fc_dtype)(x);
    ml::train::Tensor k = fc(p + "k", dims.d_model, false, fc_dtype)(x);
    ml::train::Tensor v = fc(p + "v", dims.d_model, true, fc_dtype)(x);
    ml::train::Tensor a = LayerHandle(createLayer(
      "mha_core",
      {nntrainer::withKey("name", p + "attention"),
       nntrainer::withKey("num_heads", std::to_string(dims.n_heads)),
       nntrainer::withKey("num_heads_kv", std::to_string(dims.n_heads)),
       nntrainer::withKey("max_timestep", std::to_string(dims.n_ctx)),
       nntrainer::withKey("use_rope", "false"),
       nntrainer::withKey("is_causal", "false")}))({q, k, v});
    a = fc(p + "attn_out", dims.d_model, true, fc_dtype)(a);
    h = add(p + "attn_residual")({h, a});

    x = layerNorm(p + "ffn_norm", dims.eps, act_dtype_)(h);
    x = fc(p + "ffn_up", dims.ffn, true, fc_dtype)(x);
    x = gelu(p + "ffn_gelu")(x);
    x = fc(p + "ffn_down", dims.d_model, true, fc_dtype)(x);
    h = add(p + "ffn_residual")({h, x});
  }
  h = layerNorm("encoder_norm", dims.eps, act_dtype_)(h);

  // every decoder layer's cross keys and values, stored by the writer
  std::vector<ml::train::Tensor> kv;
  for (unsigned int i = 0; i < dims.decoder_layers; ++i) {
    const std::string p = "dec" + std::to_string(i) + "_";
    kv.push_back(fc(p + "cross_k", dims.d_model, false, fc_dtype)(h));
    kv.push_back(fc(p + "cross_v", dims.d_model, true, fc_dtype)(h));
  }
  output_ = LayerHandle(
    createLayer("cross_memory_writer",
                {nntrainer::withKey("name", "encoder_memory_writer"),
                 nntrainer::withKey("memory_prefix", memory_prefix)}))(kv);
}

void WhisperEncoder::initialize() {
  // compile() with the tensors also initializes the graph
  if (model_->compile(input_, output_, ml::train::ExecutionMode::INFERENCE))
    throw std::runtime_error("WhisperEncoder: compilation failed");
}

void WhisperEncoder::load_weight(const std::string &path) {
  const bool st =
    path.size() > 12 && path.compare(path.size() - 12, 12, ".safetensors") == 0;
  model_->load(path, st ? ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS
                        : ml::train::ModelFormat::MODEL_FORMAT_BIN);
}

void WhisperEncoder::run(const std::vector<float> &mel) {
  if (mel.size() != static_cast<size_t>(dims_.n_mels) * dims_.n_frames)
    throw std::invalid_argument("WhisperEncoder: wrong mel size");
  // mha_core advances its own cache index on every forward: start over
  model_->forEachLayer(
    [](ml::train::Layer &l, nntrainer::RunLayerContext &, void *) {
      if (l.getType() == MHACoreLayer::type)
        l.setProperty({"cache_index=0"});
    },
    nullptr);
  // the output points into the graph's own memory (not to be freed)
  model_->inference(1, {const_cast<float *>(mel.data())});
}

/* -------------------------------- decoder -------------------------------- */

WhisperCausalLM::WhisperCausalLM(json &cfg, json &generation_cfg,
                                 json &nntr_cfg) :
  Transformer(whisperTextConfig(cfg), generation_cfg, nntr_cfg,
              ModelType::CAUSALLM),
  CausalLM(cfg, generation_cfg, nntr_cfg) {
  enc_dims_.n_mels = cfg.value("num_mel_bins", 128u);
  enc_dims_.n_ctx = cfg.value("max_source_positions", 1500u);
  enc_dims_.n_frames = enc_dims_.n_ctx * 2;
  enc_dims_.d_model = cfg.value("d_model", 1280u);
  enc_dims_.n_heads = cfg.value("encoder_attention_heads", 20u);
  enc_dims_.n_layers = cfg.value("encoder_layers", 32u);
  enc_dims_.ffn = cfg.value("encoder_ffn_dim", 5120u);
  enc_dims_.decoder_layers = static_cast<unsigned int>(NUM_LAYERS);

  act_dtype_ = activationOf(MODEL_TENSOR_TYPE);
  const json enc = nntr_cfg.value("encoder", json::object());
  encoder_tensor_type_ = enc.value("model_tensor_type", MODEL_TENSOR_TYPE);
  encoder_fc_dtype_ = enc.value("fc_layer_dtype", FC_LAYER_DTYPE);
  memory_prefix_ = "whisper" + std::to_string(g_instances++);

  sot_ = generation_cfg.value("decoder_start_token_id", 50258);
  eot_ = generation_cfg.contains("eos_token_id") &&
             generation_cfg["eos_token_id"].is_number()
           ? generation_cfg["eos_token_id"].get<int>()
           : 50257;
  if (generation_cfg.contains("task_to_id"))
    transcribe_ = generation_cfg["task_to_id"].value("transcribe", transcribe_);
  no_timestamps_ =
    generation_cfg.value("no_timestamps_token_id", no_timestamps_);
  if (generation_cfg.contains("lang_to_id")) {
    for (auto &[tok, id] : generation_cfg["lang_to_id"].items()) {
      // "<|ko|>" -> "ko"
      std::string code = tok;
      if (code.size() > 4 && code.rfind("<|", 0) == 0)
        code = code.substr(2, code.size() - 4);
      lang_ids_.emplace_back(code, id.get<int>());
    }
  }
  if (generation_cfg.contains("suppress_tokens"))
    suppress_ = generation_cfg["suppress_tokens"].get<std::vector<int>>();
  if (generation_cfg.contains("begin_suppress_tokens"))
    begin_suppress_ =
      generation_cfg["begin_suppress_tokens"].get<std::vector<int>>();

  const json &heads_src = nntr_cfg.contains("alignment_heads")
                            ? nntr_cfg["alignment_heads"]
                            : generation_cfg.value("alignment_heads", json());
  if (heads_src.is_array()) {
    for (const auto &lh : heads_src)
      alignment_heads_.emplace_back(lh[0].get<int>(), lh[1].get<int>());
  }
  if (alignment_heads_.empty()) {
    // openai/whisper's default: every head of the second half of the layers
    for (int l = NUM_LAYERS / 2; l < NUM_LAYERS; ++l)
      for (int h = 0; h < NUM_HEADS; ++h)
        alignment_heads_.emplace_back(l, h);
  }

  // the prompt (4), the generated text and the end-of-text token must fit
  sample_len_ = std::min<unsigned int>(nntr_cfg.value("num_to_generate", 224u),
                                       MAX_SEQ_LEN - 6);

  if (tokenizer) {
    for (const char *t : {"<|nospeech|>", "<|nocaptions|>"}) {
      const int id = tokenizer->TokenToId(t);
      if (id >= 0) {
        no_speech_ = id;
        break;
      }
    }
  }
}

WhisperCausalLM::~WhisperCausalLM() { crossMemoryRelease(memory_prefix_); }

void WhisperCausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  registerWhisperLayers();
}

std::pair<Tensor, Tensor> WhisperCausalLM::constructModel() {
  Tensor input0(nntrainer::TensorDim(1, 1, 1, INIT_SEQ_LEN,
                                     nntrainer::TensorDim::Format::NCHW,
                                     nntrainer::TensorDim::DataType::FP32),
                "input0");

  LayerHandle embedding(createLayer(
    "tie_word_embeddings",
    buildEmbeddingLayerProperties("embedding0", NUM_VOCAB, DIM, EMBEDDING_DTYPE,
                                  EMBEDDING_SCALE, EMBEDDING_FILE_NAME)));
  Tensor h = embedding(input0);
  h = LayerHandle(createLayer(
    "position_embedding",
    {withKey("name", "decoder_pos_embed"),
     withKey("max_position", std::to_string(MAX_POSITION_EMBEDDINGS))}))(h);

  for (int i = 0; i < NUM_LAYERS; ++i) {
    const std::string p = "dec" + std::to_string(i) + "_";

    // causal self-attention with the KV cache
    Tensor x = layerNorm(p + "self_attn_norm", NORM_EPS, act_dtype_)(h);
    Tensor q = fc(p + "self_q", DIM, true, FC_LAYER_DTYPE)(x);
    Tensor k = fc(p + "self_k", DIM, false, FC_LAYER_DTYPE)(x);
    Tensor v = fc(p + "self_v", DIM, true, FC_LAYER_DTYPE)(x);
    auto [cache_k, cache_v] = createKVCachePlaceholders(i, NUM_HEADS);
    Tensor a = LayerHandle(createLayer(
      "mha_core",
      {withKey("name", "layer" + std::to_string(i) + "_attention"),
       withKey("num_heads", NUM_HEADS), withKey("num_heads_kv", NUM_HEADS),
       withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
       withKey("max_position_embeddings", MAX_POSITION_EMBEDDINGS),
       withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
       withKey("use_rope", "false"), withKey("is_causal", "true")}))(
      {q, k, v, cache_k, cache_v});
    a = fc(p + "self_out", DIM, true, FC_LAYER_DTYPE)(a);
    h = add(p + "self_residual")({h, a});

    // cross-attention over the encoder memory of this layer
    x = layerNorm(p + "cross_attn_norm", NORM_EPS, act_dtype_)(h);
    q = fc(p + "cross_q", DIM, true, FC_LAYER_DTYPE)(x);
    a = LayerHandle(createLayer(
      "cross_attention",
      {withKey("name", p + "cross_attention"),
       withKey("num_heads", std::to_string(NUM_HEADS)),
       withKey("layer_index", std::to_string(i)),
       withKey("memory_id", memory_prefix_ + "_l" + std::to_string(i))}))(q);
    a = fc(p + "cross_out", DIM, true, FC_LAYER_DTYPE)(a);
    h = add(p + "cross_residual")({h, a});

    x = layerNorm(p + "ffn_norm", NORM_EPS, act_dtype_)(h);
    x = fc(p + "ffn_up", INTERMEDIATE_SIZE, true, FC_LAYER_DTYPE)(x);
    x = gelu(p + "ffn_gelu")(x);
    x = fc(p + "ffn_down", DIM, true, FC_LAYER_DTYPE)(x);
    h = add(p + "ffn_residual")({h, x});
  }
  h = layerNorm("decoder_norm", NORM_EPS, act_dtype_)(h);

  // Not named output_of_causallm / lm_head: those names make the tied head
  // read rms_norm's cached output_norm instead of its input.
  LayerHandle lm_head(createLayer(
    "tie_word_embeddings",
    {withKey("name", "whisper_lm_head"), withKey("unit", NUM_VOCAB),
     withKey("disable_bias", "true"), withKey("weight_dtype", LMHEAD_DTYPE),
     withKey("shared_from", "embedding0")}));
  return {input0, lm_head(h)};
}

void WhisperCausalLM::initialize() {
  Transformer::initialize();
  encoder_.construct(enc_dims_, encoder_tensor_type_, encoder_fc_dtype_,
                     memory_prefix_);
  encoder_.initialize();
}

void WhisperCausalLM::load_weight(const std::string &path) {
  encoder_.load_weight(path);
  Transformer::load_weight(path);
}

std::vector<std::string> WhisperCausalLM::languages() const {
  std::vector<std::string> out;
  for (const auto &l : lang_ids_)
    out.push_back(l.first);
  return out;
}

std::vector<float> WhisperCausalLM::step(std::vector<float> &ids,
                                         unsigned int from, unsigned int to) {
  setKVCachePosition(from);
  // graph inputs: the token ids, then the KV cache placeholders by name (as
  // Qwen3-ASR feeds them); the caches are bound already, these pointers are
  // the same buffers
  std::vector<std::pair<std::string, float *>> caches;
  for (int i = 0; i < NUM_LAYERS; ++i) {
    caches.emplace_back(
      "cache_k_l" + std::to_string(i),
      reinterpret_cast<float *>(kv_cache.getKeyCache(i).getData()));
    caches.emplace_back(
      "cache_v_l" + std::to_string(i),
      reinterpret_cast<float *>(kv_cache.getValueCache(i).getData()));
  }
  std::sort(caches.begin(), caches.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });
  std::vector<float *> in = {ids.data() + from};
  for (const auto &c : caches)
    in.push_back(c.second);
  std::vector<float *> label;
  auto out = model->incremental_inference(BATCH_SIZE, in, label, INIT_SEQ_LEN,
                                          from, to, false);
  std::vector<float> logits(out[0], out[0] + NUM_VOCAB);
  for (float *p : out)
    delete[] p;
  return logits;
}

void WhisperCausalLM::setCapture(bool on) {
  if (on == capture_on_)
    return;
  std::vector<std::string> per_layer(NUM_LAYERS);
  if (on) {
    for (const auto &[l, h] : alignment_heads_) {
      if (l < 0 || l >= NUM_LAYERS)
        continue;
      if (!per_layer[l].empty())
        per_layer[l] += ",";
      per_layer[l] += std::to_string(h);
    }
  }
  model->forEachLayer(
    [&](ml::train::Layer &layer, nntrainer::RunLayerContext &, void *) {
      if (layer.getType() != CrossAttentionLayer::type)
        return;
      const std::string &name = layer.getName(); // dec<i>_cross_attention
      const int i = std::stoi(name.substr(3, name.find('_') - 3));
      // an empty property value does not parse: "none" records no head
      layer.setProperty(
        {"capture_heads=" + (per_layer[i].empty() ? "none" : per_layer[i])});
    },
    nullptr);
  capture_on_ = on;
}

WhisperCausalLM::Result
WhisperCausalLM::transcribe(const std::vector<float> &pcm,
                            const std::string &language, bool word_timestamps) {
  if (!is_initialized)
    throw std::runtime_error("WhisperCausalLM is not initialized");
  Result r;

  // 1. log-mel of the zero-padded 30 s window, channel-major
  const size_t window = windowSamples();
  const size_t used = std::min(pcm.size(), window);
  std::vector<float> padded(window, 0.0f);
  std::copy(pcm.begin(), pcm.begin() + used, padded.begin());
  AudioPreprocessor pre;
  const std::vector<float> mel_tf =
    pre.computeMelSpectrogram(padded, enc_dims_.n_mels);
  const size_t frames = enc_dims_.n_frames;
  if (mel_tf.size() < frames * enc_dims_.n_mels)
    throw std::runtime_error("WhisperCausalLM: short mel spectrogram");
  std::vector<float> mel(frames * enc_dims_.n_mels);
  for (size_t t = 0; t < frames; ++t)
    for (size_t m = 0; m < enc_dims_.n_mels; ++m)
      mel[m * frames + t] = mel_tf[t * enc_dims_.n_mels + m];

  // 2. encoder: fills the cross memories
  encoder_.run(mel);
  dumpFloats("mel", mel.data(), mel.size());
  if (std::getenv("NNTR_WHISPER_DUMP") != nullptr) {
    const CrossMemory &m0 = crossMemory(memory_prefix_ + "_l0");
    std::vector<float> k0(m0.key.begin(), m0.key.end());
    dumpFloats("cross_k_l0", k0.data(), k0.size());
  }

  // 3. decoder
  allocateAndBindKVCache();
  setCapture(word_timestamps);
  crossAttentionCaptureClear();

  std::vector<float> ids(MAX_SEQ_LEN, 0.0f);
  ids[0] = static_cast<float>(sot_);
  std::vector<float> logits = step(ids, 0, 1);
  dumpFloats("logits_sot", logits.data(), logits.size());

  // the start-of-transcript row: no-speech probability and the language
  const double lse = logSumExp(logits);
  if (no_speech_ >= 0)
    r.no_speech_prob = static_cast<float>(std::exp(logits[no_speech_] - lse));
  int lang_id = -1;
  if (language.empty()) {
    float best = -std::numeric_limits<float>::infinity();
    for (const auto &[code, id] : lang_ids_) {
      if (logits[id] > best) {
        best = logits[id];
        lang_id = id;
        r.language = code;
      }
    }
  } else {
    for (const auto &[code, id] : lang_ids_) {
      if (code == language) {
        lang_id = id;
        r.language = code;
      }
    }
    if (lang_id < 0)
      throw std::invalid_argument("Whisper: unknown language " + language);
  }
  if (lang_id < 0)
    throw std::runtime_error("Whisper: generation_config has no lang_to_id");

  ids[1] = static_cast<float>(lang_id);
  ids[2] = static_cast<float>(transcribe_);
  ids[3] = static_cast<float>(no_timestamps_);
  const unsigned int sot_len = 3; // start of transcript, language, task
  unsigned int pos = 4;
  // the whole prompt from position 0, now with the language
  logits = step(ids, 0, pos);
  dumpFloats("logits_prompt", logits.data(), logits.size());

  // greedy decoding: no special token but the end of text, and neither a
  // space nor the end of text as the first token
  double sum_logprob = 0.0;
  for (;;) {
    for (unsigned int v = static_cast<unsigned int>(eot_) + 1; v < NUM_VOCAB;
         ++v)
      logits[v] = -std::numeric_limits<float>::infinity();
    for (int v : suppress_)
      if (v >= 0 && static_cast<unsigned int>(v) < NUM_VOCAB)
        logits[v] = -std::numeric_limits<float>::infinity();
    if (r.tokens.empty())
      for (int v : begin_suppress_)
        if (v >= 0 && static_cast<unsigned int>(v) < NUM_VOCAB)
          logits[v] = -std::numeric_limits<float>::infinity();

    const int next = static_cast<int>(
      std::max_element(logits.begin(), logits.end()) - logits.begin());
    sum_logprob += logits[next] - logSumExp(logits);
    if (next == eot_)
      break;
    r.tokens.push_back(next);
    ids[pos] = static_cast<float>(next);
    if (r.tokens.size() >= sample_len_) {
      if (word_timestamps) { // its attention row is needed
        step(ids, pos, pos + 1);
        ++pos;
      }
      break;
    }
    logits = step(ids, pos, pos + 1);
    if (r.tokens.size() <= 8) // the first decode steps, for comparison
      dumpFloats("logits_step" + std::to_string(r.tokens.size()), logits.data(),
                 logits.size());
    ++pos;
  }
  r.avg_logprob = static_cast<float>(sum_logprob / (r.tokens.size() + 1));

  if (!r.tokens.empty()) {
    r.text = tokenizer->Decode(r.tokens);
    const size_t b = r.text.find_first_not_of(' ');
    r.text = b == std::string::npos ? "" : r.text.substr(b);
  }

  // 4. word timestamps: the end-of-text row as well, then DTW
  if (word_timestamps && !r.tokens.empty()) {
    ids[pos] = static_cast<float>(eot_);
    step(ids, pos, pos + 1);
    const unsigned int rows = pos + 1;

    const auto &cap = crossAttentionCapture();
    std::vector<whisper_timing::Matrix> heads;
    for (const auto &lh : alignment_heads_) {
      auto it = cap.find(lh);
      if (it == cap.end())
        continue;
      whisper_timing::Matrix m;
      m.rows = rows;
      m.cols = it->second.begin()->second.size();
      m.data.assign(m.rows * m.cols, 0.0f);
      for (unsigned int row = 0; row < rows; ++row) {
        auto rit = it->second.find(row);
        if (rit == it->second.end())
          throw std::runtime_error("Whisper: attention row missing");
        std::copy(rit->second.begin(), rit->second.end(),
                  m.data.begin() + static_cast<size_t>(row) * m.cols);
      }
      heads.push_back(std::move(m));
    }
    auto decode = [this](const std::vector<int> &t) {
      return tokenizer->Decode(t);
    };
    r.words = whisper_timing::findAlignment(heads, sot_len, r.tokens, eot_,
                                            std::min(used / HOP, frames),
                                            r.language, decode);
  }

  last_ = r;
  return r;
}

void WhisperCausalLM::run(const WSTR prompt, bool do_sample,
                          const WSTR system_prompt, const WSTR tail_prompt,
                          bool log_output) {
  std::vector<float> pcm = audio_pcm_;
  if (pcm.empty() && !audio_path_.empty()) {
    AudioPreprocessor pre;
    pcm = pre.loadAudio(audio_path_);
  }
  if (pcm.empty())
    throw std::invalid_argument("Whisper: no audio (use --audio)");

  std::ostream null_log(nullptr);
  std::ostream &out = log_output ? std::cout : null_log;
  const size_t window = windowSamples();
  std::string all;
  for (size_t off = 0; off < pcm.size(); off += window) {
    const size_t n = std::min(window, pcm.size() - off);
    std::vector<float> seg(pcm.begin() + off, pcm.begin() + off + n);
    Result r = transcribe(seg, language_, word_timestamps_ && !aligner_);
    if (aligner_ != nullptr && !r.text.empty()) {
      r.words.clear();
      for (const auto &w : aligner_->align(seg, r.text, r.language))
        r.words.push_back({w.word, {}, w.start, w.end});
    }
    const double t0 = static_cast<double>(off) / SAMPLE_RATE;
    out << "[Whisper] " << std::fixed << std::setprecision(2) << t0 << "s ("
        << r.language << ", no_speech " << std::defaultfloat
        << std::setprecision(4) << r.no_speech_prob << ", avg_logprob "
        << r.avg_logprob << std::fixed << "): " << r.text << std::endl;
    for (const auto &w : r.words)
      out << "  " << std::setprecision(aligner_ ? 3 : 2) << t0 + w.start
          << " - " << t0 + w.end << "  " << w.word << std::endl;
    if (!all.empty() && !r.text.empty())
      all += " ";
    all += r.text;
  }
  if (!output_list.empty())
    output_list[0] = all;
}

} // namespace quick_ai
