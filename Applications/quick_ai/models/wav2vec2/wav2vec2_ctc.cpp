// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   wav2vec2_ctc.cpp
 * @date   04 November 2026
 * @brief  wav2vec 2.0 CTC model (Wav2Vec2ForCTC) for WhisperX alignment
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "wav2vec2_ctc.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include <app_context.h>
#include <engine.h>
#include <layer.h>

#include "json.hpp"
#include <mha_core.h>
#include <wav2vec2_layers.h>

namespace quick_ai {

namespace {

using json = nlohmann::json;
using ml::train::createLayer;
using ml::train::LayerHandle;
using ml::train::Tensor;

constexpr size_t SAMPLE_RATE = 16000;
constexpr size_t MIN_SAMPLES = 400; // shorter segments are zero-padded

json readJson(const std::string &path) {
  std::ifstream f(path);
  if (!f)
    throw std::runtime_error("Wav2Vec2CTC: cannot open " + path);
  return json::parse(f);
}

std::string joinInts(const std::vector<int> &v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i)
    s += (i ? "," : "") + std::to_string(v[i]);
  return s;
}

std::string kv(const std::string &k, const std::string &v) {
  return k + "=" + v;
}

LayerHandle fc(const std::string &name, unsigned int unit) {
  return LayerHandle(
    createLayer("fully_connected",
                {kv("name", name), kv("unit", std::to_string(unit)),
                 kv("disable_bias", "false"), kv("weight_dtype", "FP32")}));
}

LayerHandle layerNorm(const std::string &name, float eps) {
  return LayerHandle(
    createLayer("layer_normalization", {kv("name", name), kv("axis", "3"),
                                        kv("epsilon", std::to_string(eps)),
                                        kv("weight_dtype", "FP32")}));
}

LayerHandle add(const std::string &name) {
  return LayerHandle(createLayer("addition", {kv("name", name)}));
}

void registerLayers() {
  static std::once_flag once;
  std::call_once(once, []() {
    auto &engine = nntrainer::Engine::Global();
    auto *ctx =
      static_cast<nntrainer::AppContext *>(engine.getRegisteredContext("cpu"));
    // each on its own: another model may have registered some already
    auto reg = [](auto &&fn) {
      try {
        fn();
      } catch (const std::invalid_argument &) {
      }
    };
    reg([&] {
      ctx->registerFactory(nntrainer::createLayer<Wav2Vec2FeatureEncoderLayer>);
    });
    reg([&] {
      ctx->registerFactory(nntrainer::createLayer<Wav2Vec2PosConvLayer>);
    });
    reg([&] { ctx->registerFactory(nntrainer::createLayer<MHACoreLayer>); });
  });
}

} // namespace

Wav2Vec2CTC::Wav2Vec2CTC() = default;
Wav2Vec2CTC::~Wav2Vec2CTC() = default;

void Wav2Vec2CTC::load(const std::string &dir) {
  registerLayers();
  const json cfg = readJson(dir + "/config.json");
  const json nntr = readJson(dir + "/nntr_config.json");
  // WhisperX lower-cases the dictionary ("[PAD]" -> "[pad]", the blank)
  const json vocab = readJson(dir + "/vocab.json");
  for (const auto &[ch, id] : vocab.items()) {
    std::string key = ch;
    std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    dict_[key] = id.get<int>();
  }
  normalize_ = nntr.value("normalize_input", false);

  const unsigned int D = cfg.value("hidden_size", 768u);
  const unsigned int L = cfg.value("num_hidden_layers", 12u);
  const unsigned int H = cfg.value("num_attention_heads", 12u);
  const unsigned int F = cfg.value("intermediate_size", 3072u);
  const float eps = cfg.value("layer_norm_eps", 1e-5f);
  const bool stable = cfg.value("do_stable_layer_norm", false);
  const std::vector<int> conv_dim =
    cfg.value("conv_dim", std::vector<int>(7, 512));
  kernel_ = cfg.value("conv_kernel", std::vector<int>{10, 3, 3, 3, 3, 2, 2});
  stride_ = cfg.value("conv_stride", std::vector<int>{5, 2, 2, 2, 2, 2, 2});
  vocab_ = cfg.value("vocab_size", dict_.size());
  if (cfg.value("hidden_act", std::string("gelu")) != "gelu")
    throw std::invalid_argument("Wav2Vec2CTC: only hidden_act gelu");

  max_samples_ =
    static_cast<size_t>(nntr.value("max_seconds", 30.0) * SAMPLE_RATE);
  max_frames_ =
    Wav2Vec2FeatureEncoderLayer::framesOf(max_samples_, kernel_, stride_);

  model_ = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model_->setProperty({kv("batch_size", "1"), kv("epochs", "1"),
                       kv("model_tensor_type", "FP32-FP32")});

  input_ = Tensor(nntrainer::TensorDim(1, 1, 1, max_samples_,
                                       nntrainer::TensorDim::Format::NCHW,
                                       nntrainer::TensorDim::DataType::FP32),
                  "wav2vec2_input");
  Tensor h = LayerHandle(createLayer(
    "wav2vec2_feature_encoder",
    {kv("name", "feature_encoder"), kv("conv_dim", joinInts(conv_dim)),
     kv("conv_kernel", joinInts(kernel_)), kv("conv_stride", joinInts(stride_)),
     kv("conv_bias", cfg.value("conv_bias", false) ? "true" : "false"),
     kv("norm", cfg.value("feat_extract_norm", std::string("group")))}))(
    input_);
  h = layerNorm("feature_projection_norm", eps)(h);
  h = fc("feature_projection", D)(h);
  Tensor pos = LayerHandle(createLayer(
    "wav2vec2_pos_conv",
    {kv("name", "pos_conv"),
     kv("kernel_size",
        std::to_string(cfg.value("num_conv_pos_embeddings", 128u))),
     kv("groups",
        std::to_string(cfg.value("num_conv_pos_embedding_groups", 16u)))}))(h);
  h = add("pos_conv_add")({h, pos});
  if (!stable)
    h = layerNorm("encoder_norm", eps)(h);

  for (unsigned int i = 0; i < L; ++i) {
    const std::string p = "enc" + std::to_string(i) + "_";
    auto attention = [&](Tensor x) {
      Tensor q = fc(p + "q", D)(x), k = fc(p + "k", D)(x),
             v = fc(p + "v", D)(x);
      Tensor a = LayerHandle(createLayer(
        "mha_core",
        {kv("name", p + "attention"), kv("num_heads", std::to_string(H)),
         kv("num_heads_kv", std::to_string(H)),
         kv("max_timestep", std::to_string(max_frames_)),
         kv("use_rope", "false"), kv("is_causal", "false")}))({q, k, v});
      return fc(p + "attn_out", D)(a);
    };
    auto ffn = [&](Tensor x) {
      x = fc(p + "ffn_up", F)(x);
      x = LayerHandle(createLayer("activation", {kv("name", p + "ffn_gelu"),
                                                 kv("activation", "gelu")}))(x);
      return fc(p + "ffn_down", D)(x);
    };
    if (stable) {
      Tensor a = attention(layerNorm(p + "attn_norm", eps)(h));
      h = add(p + "attn_residual")({h, a});
      Tensor f = ffn(layerNorm(p + "ffn_norm", eps)(h));
      h = add(p + "ffn_residual")({h, f});
    } else {
      h = add(p + "attn_residual")({h, attention(h)});
      h = layerNorm(p + "attn_norm", eps)(h);
      h = add(p + "ffn_residual")({h, ffn(h)});
      h = layerNorm(p + "ffn_norm", eps)(h);
    }
  }
  if (stable)
    h = layerNorm("encoder_norm", eps)(h);
  output_ = fc("lm_head", static_cast<unsigned int>(vocab_))(h);

  if (model_->compile(input_, output_, ml::train::ExecutionMode::INFERENCE))
    throw std::runtime_error("Wav2Vec2CTC: compilation failed");
  model_->load(
    dir + "/" +
      nntr.value("model_file_name", std::string("nntr_wav2vec2.safetensors")),
    ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS);
}

std::vector<float> Wav2Vec2CTC::emissions(const std::vector<float> &pcm,
                                          size_t &frames) {
  if (!model_)
    throw std::runtime_error("Wav2Vec2CTC: not loaded");
  if (pcm.size() > max_samples_)
    throw std::invalid_argument("Wav2Vec2CTC: segment longer than max_seconds");
  const size_t n = std::max(pcm.size(), MIN_SAMPLES);
  std::vector<float> in(max_samples_, 0.0f);
  std::copy(pcm.begin(), pcm.end(), in.begin());
  if (normalize_) {
    // Wav2Vec2FeatureExtractor's do_normalize: zero mean, unit variance
    double mean = 0.0, var = 0.0;
    for (float v : pcm)
      mean += v;
    mean /= std::max<size_t>(pcm.size(), 1);
    for (float v : pcm)
      var += (v - mean) * (v - mean);
    var /= std::max<size_t>(pcm.size(), 1);
    const double inv = 1.0 / std::sqrt(var + 1e-7);
    for (size_t i = 0; i < pcm.size(); ++i)
      in[i] = static_cast<float>((pcm[i] - mean) * inv);
  }
  frames = Wav2Vec2FeatureEncoderLayer::framesOf(n, kernel_, stride_);

  model_->forEachLayer(
    [n](ml::train::Layer &l, nntrainer::RunLayerContext &, void *) {
      if (l.getType() == Wav2Vec2FeatureEncoderLayer::type)
        l.setProperty({"num_samples=" + std::to_string(n)});
      else if (l.getType() == MHACoreLayer::type)
        l.setProperty({"cache_index=0"});
    },
    nullptr);

  // an incremental step over the segment's frames: no layer sees padding
  std::vector<float *> label;
  auto out = model_->incremental_inference(
    1, {in.data()}, label, static_cast<unsigned int>(max_frames_), 0,
    static_cast<unsigned int>(frames), false);
  std::vector<float> e(out[0], out[0] + frames * vocab_);
  for (float *p : out)
    delete[] p;

  // debug aid: NNTR_WAV2VEC2_DUMP=<file> writes the logits (float32)
  if (const char *dump = std::getenv("NNTR_WAV2VEC2_DUMP")) {
    std::ofstream f(dump, std::ios::binary);
    f.write(reinterpret_cast<const char *>(e.data()),
            static_cast<std::streamsize>(e.size() * sizeof(float)));
  }

  for (size_t t = 0; t < frames; ++t) { // log_softmax
    float *row = e.data() + t * vocab_;
    const float mx = *std::max_element(row, row + vocab_);
    double s = 0.0;
    for (size_t v = 0; v < vocab_; ++v)
      s += std::exp(static_cast<double>(row[v] - mx));
    const float lse = mx + static_cast<float>(std::log(s));
    for (size_t v = 0; v < vocab_; ++v)
      row[v] -= lse;
  }
  return e;
}

std::vector<CtcWord> Wav2Vec2CTC::align(const std::vector<float> &pcm,
                                        const std::string &text,
                                        const std::string &language) {
  size_t frames = 0;
  const std::vector<float> e = emissions(pcm, frames);
  const double duration = static_cast<double>(pcm.size()) / SAMPLE_RATE;
  return ctc_align::align(e, frames, vocab_, text, 0.0, duration, dict_,
                          language);
}

} // namespace quick_ai
