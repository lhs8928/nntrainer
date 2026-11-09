// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   audio_encoder.cpp
 * @date   09 November 2026
 * @brief  Audio tower of Nemotron 3.5 Transcribe
 * @see    https://github.com/nntrainer/nntrainer
 * @author Claude <noreply@anthropic.com>
 * @bug    No known bugs except for NYI items
 */

#include "audio_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <cpu_backend.h>

namespace causallm {
namespace nemotron_transcribe {

namespace {

constexpr unsigned int kRowMajor = 0;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kPosInf = std::numeric_limits<float>::infinity();

inline float gelu(float x) {
  return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
}

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/** C[M,N] = A[M,K] * B[N,K]^T (+ C if accumulate) */
void gemmABt(const float *A, int lda, const float *B, int ldb, float *C,
             int ldc, int M, int N, int K, float alpha = 1.0f,
             float beta = 0.0f) {
  nntrainer::sgemm(kRowMajor, false, true, M, N, K, alpha, A, lda, B, ldb, beta,
                   C, ldc);
}

/** C[M,N] = A[M,K] * B[K,N] */
void gemmAB(const float *A, int lda, const float *B, int ldb, float *C, int ldc,
            int M, int N, int K) {
  nntrainer::sgemm(kRowMajor, false, false, M, N, K, 1.0f, A, lda, B, ldb, 0.0f,
                   C, ldc);
}

int jsonInt(const json &j, const char *key, int dflt) {
  return j.contains(key) && !j[key].is_null() ? j[key].get<int>() : dflt;
}
float jsonFloat(const json &j, const char *key, float dflt) {
  return j.contains(key) && !j[key].is_null() ? j[key].get<float>() : dflt;
}

} // namespace

// ============================================================ AudioWeights

AudioWeights::AudioWeights(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("AudioWeights: cannot open " + path);
  char magic[8];
  f.read(magic, 8);
  if (!f || std::memcmp(magic, "NNTRAUD1", 8) != 0)
    throw std::runtime_error("AudioWeights: bad magic in " + path);
  uint64_t header_len = 0;
  f.read(reinterpret_cast<char *>(&header_len), sizeof(header_len));
  std::string header(header_len, '\0');
  f.read(&header[0], static_cast<std::streamsize>(header_len));
  if (!f)
    throw std::runtime_error("AudioWeights: truncated header in " + path);

  json index = json::parse(header);
  size_t total = 0;
  for (auto &it : index["tensors"].items()) {
    Entry e;
    e.offset = it.value()["offset"].get<size_t>() / sizeof(float);
    e.shape = it.value()["shape"].get<std::vector<int>>();
    size_t n = 1;
    for (int d : e.shape)
      n *= static_cast<size_t>(d);
    total = std::max(total, e.offset + n);
    entries_[it.key()] = e;
  }
  data_.resize(total);
  f.read(reinterpret_cast<char *>(data_.data()),
         static_cast<std::streamsize>(total * sizeof(float)));
  if (!f)
    throw std::runtime_error("AudioWeights: truncated data in " + path);
}

bool AudioWeights::has(const std::string &name) const {
  return entries_.count(name) != 0;
}

const std::vector<int> &AudioWeights::shape(const std::string &name) const {
  auto it = entries_.find(name);
  if (it == entries_.end())
    throw std::runtime_error("AudioWeights: missing tensor " + name);
  return it->second.shape;
}

const float *AudioWeights::get(const std::string &name,
                               const std::vector<int> &expect_shape) const {
  auto it = entries_.find(name);
  if (it == entries_.end())
    throw std::runtime_error("AudioWeights: missing tensor " + name);
  if (!expect_shape.empty() && it->second.shape != expect_shape) {
    std::string got, want;
    for (int d : it->second.shape)
      got += std::to_string(d) + " ";
    for (int d : expect_shape)
      want += std::to_string(d) + " ";
    throw std::runtime_error("AudioWeights: " + name + " has shape [" + got +
                             "], expected [" + want + "]");
  }
  return data_.data() + it->second.offset;
}

// ============================================================ Linear / LN

void Linear::load(const AudioWeights &W, const std::string &name, int in_dim,
                  int out_dim, bool bias) {
  in = in_dim;
  out = out_dim;
  w = W.get(name + ".weight", {out_dim, in_dim});
  b = bias ? W.get(name + ".bias", {out_dim}) : nullptr;
}

void Linear::forward(const float *x, int rows, float *y) const {
  gemmABt(x, in, w, in, y, out, rows, out, in);
  if (b != nullptr)
    for (int r = 0; r < rows; ++r) {
      float *yr = y + static_cast<size_t>(r) * out;
      for (int o = 0; o < out; ++o)
        yr[o] += b[o];
    }
}

void LayerNorm::load(const AudioWeights &W, const std::string &name, int d) {
  dim = d;
  w = W.get(name + ".weight", {d});
  b = W.get(name + ".bias", {d});
}

void LayerNorm::forward(const float *x, int rows, float *y) const {
  for (int r = 0; r < rows; ++r) {
    const float *xr = x + static_cast<size_t>(r) * dim;
    float *yr = y + static_cast<size_t>(r) * dim;
    double mean = 0.0;
    for (int i = 0; i < dim; ++i)
      mean += xr[i];
    mean /= dim;
    double var = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double d = xr[i] - mean;
      var += d * d;
    }
    var /= dim;
    const float inv = static_cast<float>(1.0 / std::sqrt(var + eps));
    const float m = static_cast<float>(mean);
    for (int i = 0; i < dim; ++i)
      yr[i] = (xr[i] - m) * inv * w[i] + b[i];
  }
}

// ============================================================ Encoder

EncoderConfig EncoderConfig::fromJson(const json &cfg, bool qk_norm) {
  EncoderConfig c;
  c.hidden = cfg.at("hidden_size").get<int>();
  c.intermediate = cfg.at("intermediate_size").get<int>();
  c.layers = cfg.at("num_hidden_layers").get<int>();
  c.heads = cfg.at("num_attention_heads").get<int>();
  c.head_dim = jsonInt(cfg, "head_dim", c.hidden / c.heads);
  float partial = 1.0f;
  if (cfg.contains("rope_parameters") && cfg["rope_parameters"].is_object()) {
    const json &rp = cfg["rope_parameters"];
    partial = jsonFloat(rp, "partial_rotary_factor", 1.0f);
    c.rope_theta = jsonFloat(rp, "rope_theta", 10000.0f);
  }
  c.rotary_dim = static_cast<int>(c.head_dim * partial);
  c.qk_norm = qk_norm;
  c.subsampling = jsonInt(cfg, "subsampling_factor", 8);
  c.mel_bins = jsonInt(cfg, "num_mel_bins", 128);
  if (jsonInt(cfg, "num_key_value_heads", c.heads) != c.heads)
    throw std::invalid_argument("audio encoder: GQA is not supported");
  return c;
}

void TransformerEncoder::load(const AudioWeights &W, const std::string &p,
                              const EncoderConfig &cfg) {
  cfg_ = cfg;
  const int H = cfg.hidden, D = cfg.heads * cfg.head_dim;
  embedder_.load(W, p + "embedder.projection", cfg.subsampling * cfg.mel_bins,
                 H, false);
  input_norm_.load(W, p + "input_layer_norm", H);
  final_norm_.load(W, p + "layer_norm", H);
  layers_.resize(cfg.layers);
  for (int i = 0; i < cfg.layers; ++i) {
    const std::string lp = p + "layers." + std::to_string(i) + ".";
    Layer &L = layers_[i];
    L.ln1.load(W, lp + "layer_norm1", H);
    L.ln2.load(W, lp + "layer_norm2", H);
    L.q.load(W, lp + "self_attn.q_proj", H, D, false);
    L.k.load(W, lp + "self_attn.k_proj", H, D, false);
    L.v.load(W, lp + "self_attn.v_proj", H, D, false);
    L.o.load(W, lp + "self_attn.o_proj", D, H, true);
    if (cfg.qk_norm) {
      L.q_norm.load(W, lp + "self_attn.q_norm", cfg.head_dim);
      L.k_norm.load(W, lp + "self_attn.k_norm", cfg.head_dim);
    }
    L.fc1.load(W, lp + "mlp.fc1", H, cfg.intermediate, true);
    L.fc2.load(W, lp + "mlp.fc2", cfg.intermediate, H, true);
  }
  inv_freq_.resize(cfg.rotary_dim / 2);
  for (int i = 0; i < cfg.rotary_dim / 2; ++i)
    inv_freq_[i] = static_cast<float>(
      1.0 / std::pow(static_cast<double>(cfg.rope_theta),
                     static_cast<double>(2 * i) / cfg.rotary_dim));
}

std::vector<float> TransformerEncoder::embed(const float *mel,
                                             int frames) const {
  const int s = cfg_.subsampling;
  const int T = (frames + s - 1) / s;
  std::vector<float> stacked(static_cast<size_t>(T) * s * cfg_.mel_bins, 0.0f);
  std::memcpy(stacked.data(), mel,
              sizeof(float) * static_cast<size_t>(frames) * cfg_.mel_bins);
  std::vector<float> out(static_cast<size_t>(T) * cfg_.hidden);
  embedder_.forward(stacked.data(), T, out.data());
  return out;
}

void TransformerEncoder::applyRope(float *x, int T) const {
  const int hd = cfg_.head_dim, r = cfg_.rotary_dim, half = r / 2;
  if (r == 0)
    return;
  std::vector<float> cosv(half), sinv(half), tmp(r);
  for (int t = 0; t < T; ++t) {
    for (int i = 0; i < half; ++i) {
      const float angle = static_cast<float>(t) * inv_freq_[i];
      cosv[i] = std::cos(angle);
      sinv[i] = std::sin(angle);
    }
    for (int h = 0; h < cfg_.heads; ++h) {
      float *v = x + (static_cast<size_t>(t) * cfg_.heads + h) * hd;
      for (int i = 0; i < half; ++i) {
        const float a = v[i], b = v[i + half];
        tmp[i] = a * cosv[i] - b * sinv[i];
        tmp[i + half] = b * cosv[i] + a * sinv[i];
      }
      std::memcpy(v, tmp.data(), sizeof(float) * r);
    }
  }
}

void TransformerEncoder::attention(const Layer &L, const float *x, int T,
                                   const std::vector<uint8_t> &mask,
                                   float *out) const {
  const int D = cfg_.heads * cfg_.head_dim, hd = cfg_.head_dim;
  std::vector<float> q(static_cast<size_t>(T) * D), k(q.size()), v(q.size());
  L.q.forward(x, T, q.data());
  L.k.forward(x, T, k.data());
  L.v.forward(x, T, v.data());
  if (cfg_.qk_norm) {
    L.q_norm.forward(q.data(), T * cfg_.heads, q.data());
    L.k_norm.forward(k.data(), T * cfg_.heads, k.data());
  }
  applyRope(q.data(), T);
  applyRope(k.data(), T);

  std::vector<float> scores(static_cast<size_t>(T) * T);
  std::vector<float> ctx(static_cast<size_t>(T) * D);
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  for (int h = 0; h < cfg_.heads; ++h) {
    gemmABt(q.data() + h * hd, D, k.data() + h * hd, D, scores.data(), T, T, T,
            hd, scale);
    for (int i = 0; i < T; ++i) {
      float *row = scores.data() + static_cast<size_t>(i) * T;
      float mx = kNegInf;
      for (int j = 0; j < T; ++j) {
        if (!mask[j])
          row[j] = kNegInf;
        mx = std::max(mx, row[j]);
      }
      double sum = 0.0;
      for (int j = 0; j < T; ++j) {
        row[j] = mask[j] ? std::exp(row[j] - mx) : 0.0f;
        sum += row[j];
      }
      const float inv = sum > 0.0 ? static_cast<float>(1.0 / sum) : 0.0f;
      for (int j = 0; j < T; ++j)
        row[j] *= inv;
    }
    gemmAB(scores.data(), T, v.data() + h * hd, D, ctx.data() + h * hd, D, T,
           hd, T);
  }
  L.o.forward(ctx.data(), T, out);
}

std::vector<float>
TransformerEncoder::forward(const std::vector<float> &embeds, int T,
                            const std::vector<uint8_t> &mask) const {
  const int H = cfg_.hidden;
  const size_t n = static_cast<size_t>(T) * H;
  std::vector<float> x(n), h(n), tmp(n);
  std::vector<float> mid(static_cast<size_t>(T) * cfg_.intermediate);
  input_norm_.forward(embeds.data(), T, x.data());
  for (const Layer &L : layers_) {
    L.ln1.forward(x.data(), T, h.data());
    attention(L, h.data(), T, mask, tmp.data());
    for (size_t i = 0; i < n; ++i)
      x[i] += tmp[i];
    L.ln2.forward(x.data(), T, h.data());
    L.fc1.forward(h.data(), T, mid.data());
    for (float &m : mid)
      m = gelu(m);
    L.fc2.forward(mid.data(), T, tmp.data());
    for (size_t i = 0; i < n; ++i)
      x[i] += tmp[i];
  }
  final_norm_.forward(x.data(), T, h.data());
  return h;
}

// ============================================================ Diarization

DiarizationConfig DiarizationConfig::fromJson(const json &cfg) {
  DiarizationConfig c;
  c.chunk_length = jsonInt(cfg, "chunk_length", c.chunk_length);
  c.chunk_left_context = jsonInt(cfg, "chunk_left_context", 0);
  c.chunk_right_context =
    jsonInt(cfg, "chunk_right_context", c.chunk_right_context);
  c.fifo_length = jsonInt(cfg, "fifo_length", c.fifo_length);
  c.speaker_cache_update_period =
    jsonInt(cfg, "speaker_cache_update_period", c.speaker_cache_update_period);
  const json &s =
    cfg.contains("streaming_config") ? cfg["streaming_config"] : json::object();
  c.speaker_cache_length =
    jsonInt(s, "speaker_cache_length", c.speaker_cache_length);
  c.silence_frames_per_speaker =
    jsonInt(s, "speaker_cache_silence_frames_per_speaker",
            c.silence_frames_per_speaker);
  c.prediction_score_threshold =
    jsonFloat(s, "prediction_score_threshold", c.prediction_score_threshold);
  c.latest_frames_score_boost =
    jsonFloat(s, "latest_frames_score_boost", c.latest_frames_score_boost);
  c.strong_boost_rate = jsonFloat(s, "strong_boost_rate", c.strong_boost_rate);
  c.weak_boost_rate = jsonFloat(s, "weak_boost_rate", c.weak_boost_rate);
  c.min_positive_scores_rate =
    jsonFloat(s, "min_positive_scores_rate", c.min_positive_scores_rate);
  const json &h = cfg.at("head_config");
  c.num_speakers = jsonInt(h, "num_speakers", c.num_speakers);
  c.head_hidden = jsonInt(h, "hidden_size", c.head_hidden);
  return c;
}

void DiarizationModel::load(const AudioWeights &W, const std::string &p,
                            const json &cfg) {
  cfg_ = DiarizationConfig::fromJson(cfg);
  EncoderConfig ec = EncoderConfig::fromJson(cfg.at("audio_config"), false);
  encoder_.load(W, p + "model.audio_tower.", ec);
  const int Hh = cfg_.head_hidden, s = ec.subsampling;
  proj_.load(W, p + "model.proj", ec.hidden, Hh, true);
  const float *cw =
    W.get(p + "model.upsampler.conv.weight", {Hh * s, Hh, 3}); // [o, i, k]
  for (int k = 0; k < 3; ++k) {
    up_w_[k].resize(static_cast<size_t>(Hh) * s * Hh);
    for (int o = 0; o < Hh * s; ++o)
      for (int i = 0; i < Hh; ++i)
        up_w_[k][static_cast<size_t>(o) * Hh + i] =
          cw[(static_cast<size_t>(o) * Hh + i) * 3 + k];
  }
  up_b_ = W.get(p + "model.upsampler.conv.bias", {Hh * s});
  dense_.load(W, p + "classifier.dense", Hh, Hh, true);
  out_proj_.load(W, p + "classifier.out_proj", Hh, cfg_.num_speakers, true);
  silence_ = W.get(p + "silence_embeds", {ec.hidden});
}

std::vector<float> DiarizationModel::head(const std::vector<float> &hidden,
                                          int T) const {
  const int Hh = cfg_.head_hidden, s = encoder_.config().subsampling;
  const int S = cfg_.num_speakers;
  std::vector<float> p(static_cast<size_t>(T) * Hh);
  proj_.forward(hidden.data(), T, p.data());

  // Conv1d(Hh -> Hh*s, k=3, pad=1) over time, as three shifted GEMMs
  std::vector<float> up(static_cast<size_t>(T) * Hh * s);
  for (int t = 0; t < T; ++t)
    std::memcpy(up.data() + static_cast<size_t>(t) * Hh * s, up_b_,
                sizeof(float) * Hh * s);
  // tap k reads x[t + k - 1]
  if (T > 1) {
    gemmABt(p.data(), Hh, up_w_[0].data(), Hh,
            up.data() + static_cast<size_t>(Hh) * s, Hh * s, T - 1, Hh * s, Hh,
            1.0f, 1.0f); // rows 1.. read x[t-1]
    gemmABt(p.data() + Hh, Hh, up_w_[2].data(), Hh, up.data(), Hh * s, T - 1,
            Hh * s, Hh, 1.0f, 1.0f); // rows ..T-2 read x[t+1]
  }
  gemmABt(p.data(), Hh, up_w_[1].data(), Hh, up.data(), Hh * s, T, Hh * s, Hh,
          1.0f, 1.0f);

  // reshape [T, s*Hh] -> [T*s, Hh] is a no-op on row-major data
  const int R = T * s;
  for (float &v : up)
    v = std::max(v, 0.0f);
  std::vector<float> d(static_cast<size_t>(R) * Hh);
  dense_.forward(up.data(), R, d.data());
  for (float &v : d)
    v = std::max(v, 0.0f);
  std::vector<float> logits(static_cast<size_t>(R) * S);
  out_proj_.forward(d.data(), R, logits.data());
  return logits;
}

struct DiarizationModel::Cache {
  std::vector<float> embeds; /**< [num_cache, hidden] */
  std::vector<float> probs;  /**< [num_cache, S] */
  std::vector<float> fifo;   /**< [num_fifo, hidden] */
  int num_cache = 0;
  int num_fifo = 0;
  bool compressed = false;
};

void DiarizationModel::compress(std::vector<float> &embeds,
                                std::vector<float> &probs, int n) const {
  const int S = cfg_.num_speakers, H = encoder_.config().hidden;
  const int L = cfg_.speaker_cache_length;
  const int nsil = cfg_.silence_frames_per_speaker;
  const int budget = L / S - nsil;
  const int min_pos =
    static_cast<int>(std::floor(budget * cfg_.min_positive_scores_rate));
  const int n_strong =
    static_cast<int>(std::floor(budget * cfg_.strong_boost_rate));
  const int n_weak =
    static_cast<int>(std::floor(budget * cfg_.weak_boost_rate));
  const float thr = cfg_.prediction_score_threshold;
  const float log_half = std::log(0.5f);

  // frame scores [n, S]
  std::vector<float> sc(static_cast<size_t>(n) * S);
  for (int f = 0; f < n; ++f) {
    const float *p = probs.data() + static_cast<size_t>(f) * S;
    float sum_lc = 0.0f;
    for (int s = 0; s < S; ++s)
      sum_lc += std::log(std::max(1.0f - p[s], thr));
    for (int s = 0; s < S; ++s) {
      const float lp = std::log(std::max(p[s], thr));
      const float lc = std::log(std::max(1.0f - p[s], thr));
      sc[static_cast<size_t>(f) * S + s] = lp - lc + sum_lc - log_half;
    }
  }
  for (int s = 0; s < S; ++s) {
    int positives = 0;
    for (int f = 0; f < n; ++f) {
      const size_t i = static_cast<size_t>(f) * S + s;
      if (!(probs[i] > 0.5f))
        sc[i] = kNegInf;
      else if (sc[i] > 0.0f)
        ++positives;
    }
    if (positives >= min_pos)
      for (int f = 0; f < n; ++f) {
        const size_t i = static_cast<size_t>(f) * S + s;
        if (probs[i] > 0.5f && !(sc[i] > 0.0f))
          sc[i] = kNegInf;
      }
  }
  for (int f = L; f < n; ++f)
    for (int s = 0; s < S; ++s)
      sc[static_cast<size_t>(f) * S + s] += cfg_.latest_frames_score_boost;

  auto boost = [&](int k, float value) {
    std::vector<int> idx(n);
    for (int s = 0; s < S; ++s) {
      std::iota(idx.begin(), idx.end(), 0);
      const int kk = std::min(k, n);
      std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                        [&](int a, int b) {
                          return sc[static_cast<size_t>(a) * S + s] >
                                 sc[static_cast<size_t>(b) * S + s];
                        });
      for (int j = 0; j < kk; ++j)
        sc[static_cast<size_t>(idx[j]) * S + s] += value;
    }
  };
  boost(n_strong, -2.0f * log_half);
  boost(n_weak, -log_half);

  // flat speaker-major scores, silence rows score +inf
  const int scored = n + nsil;
  const long sentinel = static_cast<long>(scored) * S;
  std::vector<float> flat(static_cast<size_t>(scored) * S);
  for (int s = 0; s < S; ++s)
    for (int f = 0; f < scored; ++f)
      flat[static_cast<size_t>(s) * scored + f] =
        f < n ? sc[static_cast<size_t>(f) * S + s] : kPosInf;
  std::vector<long> order(flat.size());
  std::iota(order.begin(), order.end(), 0L);
  const int keep = std::min<int>(L, static_cast<int>(flat.size()));
  std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                    [&](long a, long b) {
                      if (flat[a] != flat[b])
                        return flat[a] > flat[b];
                      return a < b;
                    });
  std::vector<long> chosen(order.begin(), order.begin() + keep);
  for (long &c : chosen)
    if (flat[c] == kNegInf)
      c = sentinel;
  std::sort(chosen.begin(), chosen.end());

  std::vector<float> new_embeds(static_cast<size_t>(keep) * H);
  std::vector<float> new_probs(static_cast<size_t>(keep) * S, 0.0f);
  for (int i = 0; i < keep; ++i) {
    const int f = chosen[i] == sentinel
                    ? n
                    : std::min<int>(static_cast<int>(chosen[i] % scored), n);
    const float *src =
      f < n ? embeds.data() + static_cast<size_t>(f) * H : silence_;
    std::memcpy(new_embeds.data() + static_cast<size_t>(i) * H, src,
                sizeof(float) * H);
    if (f < n)
      std::memcpy(new_probs.data() + static_cast<size_t>(i) * S,
                  probs.data() + static_cast<size_t>(f) * S, sizeof(float) * S);
  }
  embeds.swap(new_embeds);
  probs.swap(new_probs);
}

void DiarizationModel::cacheUpdate(Cache &c,
                                   const std::vector<float> &input_embeds,
                                   const std::vector<float> &logits,
                                   const std::vector<uint8_t> &mask,
                                   int n_chunk, int n_lookback) const {
  const int S = cfg_.num_speakers, H = encoder_.config().hidden;
  const int sub = encoder_.config().subsampling;
  const int T = static_cast<int>(mask.size());

  // speaker probabilities at the encoder frame rate, zero on padding
  std::vector<float> probs(static_cast<size_t>(T) * S, 0.0f);
  for (int t = 0; t < T; ++t) {
    if (!mask[t])
      continue;
    for (int s = 0; s < S; ++s) {
      float acc = 0.0f;
      for (int j = 0; j < sub; ++j)
        acc += sigmoid(logits[(static_cast<size_t>(t) * sub + j) * S + s]);
      probs[static_cast<size_t>(t) * S + s] = acc / sub;
    }
  }

  const int nc = c.num_cache, nf = c.num_fifo;
  const int chunk_start = nc + nf + n_lookback;
  // fifo + chunk
  std::vector<float> fifo_embeds(static_cast<size_t>(nf + n_chunk) * H);
  std::memcpy(fifo_embeds.data(), c.fifo.data(),
              sizeof(float) * static_cast<size_t>(nf) * H);
  std::memcpy(fifo_embeds.data() + static_cast<size_t>(nf) * H,
              input_embeds.data() + static_cast<size_t>(chunk_start) * H,
              sizeof(float) * static_cast<size_t>(n_chunk) * H);
  const int n_fifo_all = nf + n_chunk;

  int popped = 0;
  if (n_fifo_all > cfg_.fifo_length)
    popped = std::min(
      std::max(cfg_.speaker_cache_update_period, n_fifo_all - cfg_.fifo_length),
      n_fifo_all);
  if (popped > 0) {
    std::vector<float> fifo_probs(static_cast<size_t>(n_fifo_all) * S);
    std::memcpy(fifo_probs.data(), probs.data() + static_cast<size_t>(nc) * S,
                sizeof(float) * static_cast<size_t>(nf) * S);
    std::memcpy(fifo_probs.data() + static_cast<size_t>(nf) * S,
                probs.data() + static_cast<size_t>(chunk_start) * S,
                sizeof(float) * static_cast<size_t>(n_chunk) * S);

    const int n_new = nc + popped;
    std::vector<float> cache_embeds(static_cast<size_t>(n_new) * H);
    std::vector<float> cache_probs(static_cast<size_t>(n_new) * S);
    std::memcpy(cache_embeds.data(), c.embeds.data(),
                sizeof(float) * static_cast<size_t>(nc) * H);
    std::memcpy(cache_embeds.data() + static_cast<size_t>(nc) * H,
                fifo_embeds.data(),
                sizeof(float) * static_cast<size_t>(popped) * H);
    const float *stored = c.compressed ? c.probs.data() : probs.data();
    std::memcpy(cache_probs.data(), stored,
                sizeof(float) * static_cast<size_t>(nc) * S);
    std::memcpy(cache_probs.data() + static_cast<size_t>(nc) * S,
                fifo_probs.data(),
                sizeof(float) * static_cast<size_t>(popped) * S);
    if (n_new > cfg_.speaker_cache_length) {
      compress(cache_embeds, cache_probs, n_new);
      c.compressed = true;
    }
    c.embeds.swap(cache_embeds);
    c.probs.swap(cache_probs);
    c.num_cache = static_cast<int>(c.embeds.size() / H);
  }
  c.num_fifo = n_fifo_all - popped;
  c.fifo.assign(fifo_embeds.begin() + static_cast<size_t>(popped) * H,
                fifo_embeds.end());
}

std::vector<float>
DiarizationModel::forward(const float *mel, int frames,
                          const std::vector<uint8_t> &mel_mask) const {
  const EncoderConfig &ec = encoder_.config();
  const int H = ec.hidden, sub = ec.subsampling, S = cfg_.num_speakers;
  std::vector<float> embeds = encoder_.embed(mel, frames);
  const int N = static_cast<int>(embeds.size() / H);
  std::vector<uint8_t> embed_mask(N);
  for (int t = 0; t < N; ++t)
    embed_mask[t] = mel_mask[static_cast<size_t>(t) * sub];

  Cache cache;
  std::vector<float> logits_all;
  logits_all.reserve(static_cast<size_t>(N) * sub * S);
  for (int start = 0; start < N; start += cfg_.chunk_length) {
    const int end = std::min(start + cfg_.chunk_length, N);
    const int n_chunk = end - start;
    const int cstart = std::max(start - cfg_.chunk_left_context, 0);
    const int cend = std::min(end + cfg_.chunk_right_context, N);
    const int n_lookback = start - cstart;
    const int cached = cache.num_cache + cache.num_fifo;
    const int T = cached + (cend - cstart);

    std::vector<float> input(static_cast<size_t>(T) * H);
    std::memcpy(input.data(), cache.embeds.data(),
                sizeof(float) * static_cast<size_t>(cache.num_cache) * H);
    std::memcpy(input.data() + static_cast<size_t>(cache.num_cache) * H,
                cache.fifo.data(),
                sizeof(float) * static_cast<size_t>(cache.num_fifo) * H);
    std::memcpy(input.data() + static_cast<size_t>(cached) * H,
                embeds.data() + static_cast<size_t>(cstart) * H,
                sizeof(float) * static_cast<size_t>(cend - cstart) * H);
    std::vector<uint8_t> mask(T, 1);
    for (int t = cstart; t < cend; ++t)
      mask[cached + t - cstart] = embed_mask[t];

    std::vector<float> hidden = encoder_.forward(input, T, mask);
    std::vector<float> logits = head(hidden, T);
    cacheUpdate(cache, input, logits, mask, n_chunk, n_lookback);

    const size_t from = static_cast<size_t>(cached + n_lookback) * sub * S;
    const size_t count = static_cast<size_t>(n_chunk) * sub * S;
    logits_all.insert(logits_all.end(), logits.begin() + from,
                      logits.begin() + from + count);
  }
  logits_all.resize(static_cast<size_t>(frames) * S);
  return logits_all;
}

// ============================================================ AudioTower

AudioTower::AudioTower(const std::string &weights_path, const json &cfg) {
  weights_ = std::make_unique<AudioWeights>(weights_path);
  const AudioWeights &W = *weights_;

  const json &fe = cfg.at("feature_extractor");
  fe_cfg_.sampling_rate = jsonInt(fe, "sampling_rate", 16000);
  fe_cfg_.n_fft = jsonInt(fe, "n_fft", 512);
  fe_cfg_.win_length = jsonInt(fe, "win_length", 400);
  fe_cfg_.hop_length = jsonInt(fe, "hop_length", 160);
  fe_cfg_.num_mel_bins = jsonInt(fe, "num_mel_bins", 128);
  fe_cfg_.preemphasis = jsonFloat(fe, "preemphasis", 0.97f);
  fe_cfg_.bf16_constants =
    fe.value("mel_filters_dtype", std::string("float32")) == "bfloat16";

  EncoderConfig asr_cfg = EncoderConfig::fromJson(cfg.at("asr"), true);
  fe_cfg_.pad_to_multiple = asr_cfg.subsampling;
  asr_.load(W, "model.audio_tower.asr_encoder.", asr_cfg);
  diar_.load(W, "model.audio_tower.diarization_model.", cfg.at("diarization"));

  const int H = asr_cfg.hidden, S = diar_.config().num_speakers;
  asr_norm_.load(W, "model.audio_tower.asr_norm", H);
  speaker_norm_.load(W, "model.audio_tower.speaker_norm", S);
  text_hidden_ = cfg.at("text_hidden_size").get<int>();
  projector_.load(W, "model.multi_modal_projector", H, text_hidden_, true);
  activity_threshold_ = jsonFloat(cfg, "speaker_activity_threshold", 0.5f);

  // fixed sinusoidal embedding of each speaker slot
  speaker_embedding_.assign(static_cast<size_t>(S) * H, 0.0f);
  for (int s = 0; s < S; ++s)
    for (int i = 0; i < H / 2; ++i) {
      const float inv = std::exp(static_cast<float>(2 * i) *
                                 -(std::log(10000.0f) / static_cast<float>(H)));
      const float a = static_cast<float>(s) * inv;
      speaker_embedding_[static_cast<size_t>(s) * H + 2 * i] = std::sin(a);
      speaker_embedding_[static_cast<size_t>(s) * H + 2 * i + 1] = std::cos(a);
    }
}

AudioOutput AudioTower::encode(const MelFeatures &mel) const {
  const EncoderConfig &ac = asr_.config();
  const int sub = ac.subsampling, H = ac.hidden, M = mel.num_mel_bins;
  const int F = mel.num_frames;
  if (F % sub != 0)
    throw std::invalid_argument("AudioTower: frame count must be a multiple "
                                "of the subsampling factor");
  if (M != ac.mel_bins)
    throw std::invalid_argument("AudioTower: mel bin mismatch");
  const int N = F / sub;

  std::vector<uint8_t> mel_mask(F, 0);
  for (int t = 0; t < mel.num_valid && t < F; ++t)
    mel_mask[t] = 1;

  // per-feature normalization over the valid frames (unbiased std + 1e-5)
  std::vector<float> norm(static_cast<size_t>(F) * M, 0.0f);
  {
    const int n = mel.num_valid;
    for (int m = 0; m < M; ++m) {
      double sum = 0.0;
      for (int t = 0; t < n; ++t)
        sum += mel.data[static_cast<size_t>(t) * M + m];
      const double mean = sum / std::max(n, 1);
      double ss = 0.0;
      for (int t = 0; t < n; ++t) {
        const double d = mel.data[static_cast<size_t>(t) * M + m] - mean;
        ss += d * d;
      }
      const double stdv = std::sqrt(ss / std::max(n - 1, 1)) + 1e-5;
      for (int t = 0; t < n; ++t)
        norm[static_cast<size_t>(t) * M + m] = static_cast<float>(
          (mel.data[static_cast<size_t>(t) * M + m] - mean) / stdv);
    }
  }

  // speech recognition encoder, chunk by chunk with context
  const DiarizationConfig &dc = diar_.config();
  std::vector<float> asr_hidden(static_cast<size_t>(N) * H);
  for (int start = 0; start < N; start += dc.chunk_length) {
    const int end = std::min(start + dc.chunk_length, N);
    const int cs = std::max(start - dc.chunk_left_context, 0);
    const int ce = std::min(end + dc.chunk_right_context, N);
    const int T = ce - cs;
    std::vector<float> emb =
      asr_.embed(norm.data() + static_cast<size_t>(cs) * sub * M, T * sub);
    std::vector<uint8_t> mask(T);
    for (int t = 0; t < T; ++t)
      mask[t] = mel_mask[static_cast<size_t>(cs + t) * sub];
    std::vector<float> out = asr_.forward(emb, T, mask);
    std::memcpy(asr_hidden.data() + static_cast<size_t>(start) * H,
                out.data() + static_cast<size_t>(start - cs) * H,
                sizeof(float) * static_cast<size_t>(end - start) * H);
  }

  // speaker activity -> speaker embedding, added to the normalized states
  AudioOutput res;
  res.speaker_logits = diar_.forward(mel.data.data(), F, mel_mask);
  const int S = dc.num_speakers;
  std::vector<float> fused(static_cast<size_t>(N) * H);
  asr_norm_.forward(asr_hidden.data(), N, fused.data());
  std::vector<float> act(S), act_n(S);
  for (int t = 0; t < N; ++t) {
    for (int s = 0; s < S; ++s) {
      float acc = 0.0f;
      for (int j = 0; j < sub; ++j)
        acc += sigmoid(
          res.speaker_logits[(static_cast<size_t>(t) * sub + j) * S + s]);
      act[s] = (acc / sub) > activity_threshold_ ? 1.0f : 0.0f;
    }
    speaker_norm_.forward(act.data(), 1, act_n.data());
    float *row = fused.data() + static_cast<size_t>(t) * H;
    for (int s = 0; s < S; ++s) {
      const float *e = speaker_embedding_.data() + static_cast<size_t>(s) * H;
      for (int i = 0; i < H; ++i)
        row[i] += act_n[s] * e[i];
    }
  }

  res.num_frames = N;
  res.num_audio_tokens = (mel.num_valid + sub - 1) / sub;
  res.text_hidden = text_hidden_;
  res.embeds.resize(static_cast<size_t>(res.num_audio_tokens) * text_hidden_);
  projector_.forward(fused.data(), res.num_audio_tokens, res.embeds.data());
  res.asr_hidden = std::move(asr_hidden);
  return res;
}

} // namespace nemotron_transcribe
} // namespace causallm
