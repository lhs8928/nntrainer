// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   eres2netv2.cpp
 * @date   28 October 2026
 * @brief  ERes2NetV2 speaker embedding in NNTrainer (float32)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "eres2netv2.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <complex>
#include <cstring>

#if defined(USE_BLAS)
#include <cblas_interface.h>
#elif defined(__ANDROID__)
// The Android libnntrainer is built with OpenBLAS behind __cblas_sgemm, but
// the app is not compiled with USE_BLAS (PyanNet keeps its own loops), so the
// header refuses to be included: the one function is declared here
namespace nntrainer {
void __cblas_sgemm(const unsigned int TStorageOrder, bool TransA, bool TransB,
                   const unsigned int M, const unsigned int N,
                   const unsigned int K, const float alpha, const float *A,
                   const unsigned int lda, const float *B,
                   const unsigned int ldb, const float beta, float *C,
                   const unsigned int ldc);
} // namespace nntrainer
#endif

#if defined(USE_BLAS) || defined(__ANDROID__)
#define RUN_SGEMM(M, N, K, A, B, C)                                            \
  nntrainer::__cblas_sgemm(0, false, false, M, N, K, 1.0f, A, K, B, N, 0.0f,   \
                           C, N)
#else
static inline void matmul_fallback(size_t M, size_t N, size_t K, const float *A,
                                   const float *B, float *C) {
#pragma omp parallel for schedule(static)
  for (size_t m = 0; m < M; ++m) {
    const float *a_row = A + m * K;
    float *c_row = C + m * N;
    std::memset(c_row, 0, N * sizeof(float));
    for (size_t k = 0; k < K; ++k) {
      const float a_val = a_row[k];
      const float *b_row = B + k * N;
      for (size_t n = 0; n < N; ++n)
        c_row[n] += a_val * b_row[n];
    }
  }
}
#define RUN_SGEMM(M, N, K, A, B, C) matmul_fallback(M, N, K, A, B, C)
#endif

namespace speaker_diarization {

namespace {

constexpr size_t RATE = 16000;
constexpr size_t FRAME_LEN = 400, FRAME_SHIFT = 160, FFT_N = 512;
constexpr size_t BINS = FFT_N / 2; // Kaldi's mel banks leave out Nyquist
constexpr float CLIP = 20.0f;      // the model's ReLU is Hardtanh(0, 20)
constexpr float POOL_EPS = 1e-8f;  // added to the variance before its root
constexpr size_t COLS_BUDGET = 1u << 22; // floats of im2col at once

/** @brief Kaldi MelBanks: 80 triangles in mel space from 20 Hz to 7600 Hz */
const std::vector<float> &melBanks() {
  static const std::vector<float> banks = [] {
    constexpr double LOW = 20.0, HIGH = RATE / 2.0 - 400.0;
    auto mel = [](double f) { return 1127.0 * std::log(1.0 + f / 700.0); };
    const double lo = mel(LOW), hi = mel(HIGH);
    const double delta = (hi - lo) / (ERes2NetV2::MEL + 1);
    std::vector<float> b(ERes2NetV2::MEL * BINS, 0.0f);
    for (size_t m = 0; m < ERes2NetV2::MEL; ++m) {
      const double l = lo + m * delta, c = lo + (m + 1) * delta,
                   r = lo + (m + 2) * delta;
      for (size_t k = 0; k < BINS; ++k) {
        const double x = mel(static_cast<double>(k) * RATE / FFT_N);
        if (x > l && x < r)
          b[m * BINS + k] =
            static_cast<float>(x <= c ? (x - l) / (c - l) : (r - x) / (r - c));
      }
    }
    return b;
  }();
  return banks;
}

/** @brief povey window: hann to the power 0.85 */
const std::vector<double> &poveyWindow() {
  static const std::vector<double> w = [] {
    std::vector<double> v(FRAME_LEN);
    for (size_t i = 0; i < FRAME_LEN; ++i)
      v[i] =
        std::pow(0.5 - 0.5 * std::cos(2.0 * M_PI * i / (FRAME_LEN - 1)), 0.85);
    return v;
  }();
  return w;
}

/** @brief in-place radix-2 FFT of FFT_N points */
void fft(std::complex<double> *x) {
  for (size_t i = 1, j = 0; i < FFT_N; ++i) {
    size_t bit = FFT_N >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j ^= bit;
    if (i < j)
      std::swap(x[i], x[j]);
  }
  for (size_t len = 2; len <= FFT_N; len <<= 1) {
    const std::complex<double> step = std::polar(1.0, -2.0 * M_PI / len);
    for (size_t i = 0; i < FFT_N; i += len) {
      std::complex<double> w = 1.0;
      for (size_t k = 0; k < len / 2; ++k, w *= step) {
        const std::complex<double> u = x[i + k], v = x[i + k + len / 2] * w;
        x[i + k] = u + v;
        x[i + k + len / 2] = u - v;
      }
    }
  }
}

/** @brief a feature map: channels x height (frequency) x width (time) */
struct Map {
  std::vector<float> d;
  size_t c = 0, h = 0, w = 0;
  size_t plane() const { return h * w; }
  float *at(size_t ch) { return d.data() + ch * plane(); }
  const float *at(size_t ch) const { return d.data() + ch * plane(); }
};

/** @brief y = conv(x) with padding k / 2 (every convolution of the model) */
Map conv(const float *w, const float *b, size_t out, size_t k, size_t stride,
         const float *x, size_t c, size_t h, size_t wd) {
  const size_t pad = k / 2;
  Map y;
  y.c = out;
  y.h = (h + 2 * pad - k) / stride + 1;
  y.w = (wd + 2 * pad - k) / stride + 1;
  const size_t n = y.plane(), kk = c * k * k;
  y.d.assign(out * n, 0.0f);
  if (k == 1 && stride == 1) {
    RUN_SGEMM(out, n, kk, w, x, y.d.data());
  } else {
    // im2col over a block of output positions at a time, so long spans do
    // not need the whole column matrix
    const size_t block = std::max<size_t>(1, std::min(n, COLS_BUDGET / kk));
    std::vector<float> cols(kk * block), part(out * block);
    for (size_t n0 = 0; n0 < n; n0 += block) {
      const size_t nb = std::min(block, n - n0);
#pragma omp parallel for schedule(static)
      for (size_t r = 0; r < kk; ++r) {
        const size_t ch = r / (k * k), i = (r / k) % k, j = r % k;
        const float *src = x + ch * h * wd;
        float *dst = cols.data() + r * nb;
        for (size_t q = 0; q < nb; ++q) {
          const size_t o = n0 + q, oh = o / y.w, ow = o % y.w;
          const long ih = static_cast<long>(oh * stride + i) - pad;
          const long iw = static_cast<long>(ow * stride + j) - pad;
          dst[q] = (ih < 0 || iw < 0 || ih >= static_cast<long>(h) ||
                    iw >= static_cast<long>(wd))
                     ? 0.0f
                     : src[ih * wd + iw];
        }
      }
      RUN_SGEMM(out, nb, kk, w, cols.data(), part.data());
      for (size_t o = 0; o < out; ++o)
        std::memcpy(y.d.data() + o * n + n0, part.data() + o * nb,
                    nb * sizeof(float));
    }
  }
  if (b) {
#pragma omp parallel for schedule(static)
    for (size_t o = 0; o < out; ++o) {
      float *row = y.d.data() + o * n;
      for (size_t q = 0; q < n; ++q)
        row[q] += b[o];
    }
  }
  return y;
}

void clip(float *x, size_t n) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < n; ++i)
    x[i] = std::min(std::max(x[i], 0.0f), CLIP);
}

/** @brief channels of a and then b, one map (same height and width) */
Map concat(const float *a, size_t ca, const float *b, size_t cb, size_t h,
           size_t w) {
  Map m;
  m.c = ca + cb;
  m.h = h;
  m.w = w;
  m.d.resize(m.c * h * w);
  std::memcpy(m.d.data(), a, ca * h * w * sizeof(float));
  std::memcpy(m.d.data() + ca * h * w, b, cb * h * w * sizeof(float));
  return m;
}

} // namespace

bool ERes2NetV2::bind(const WeightLoader &loader, const std::string &name,
                      size_t stride, Conv &c, bool required,
                      std::string *missing) const {
  std::vector<uint32_t> dims;
  c.w = loader.hasTensor(name + ".weight")
          ? loader.getTensor(name + ".weight", &dims)
          : nullptr; // an optional one (an identity shortcut) is not an error
  if (!c.w || dims.size() != 4 || dims[2] != dims[3]) {
    c.w = nullptr;
    if (required && missing)
      *missing = name + ".weight";
    return !required;
  }
  c.out = dims[0];
  c.in = dims[1];
  c.k = dims[2];
  c.stride = stride;
  c.b = loader.hasTensor(name + ".bias") ? loader.getTensor(name + ".bias")
                                         : nullptr;
  return true;
}

bool ERes2NetV2::init(const WeightLoader &loader, std::string *missing) {
  if (!bind(loader, "conv1", 1, conv1_, true, missing))
    return false;
  // stage, blocks, stride of its first block, attentional fusion
  const struct {
    const char *name;
    size_t blocks, stride;
    bool fused;
  } stages[] = {{"layer1", 3, 1, false},
                {"layer2", 4, 2, false},
                {"layer3", 6, 2, true},
                {"layer4", 3, 2, true}};
  blocks_.clear();
  for (const auto &s : stages) {
    for (size_t i = 0; i < s.blocks; ++i) {
      const std::string p = std::string(s.name) + "." + std::to_string(i);
      const size_t stride = i == 0 ? s.stride : 1;
      Block b;
      b.fused = s.fused;
      if (!bind(loader, p + ".conv1", stride, b.conv1, true, missing) ||
          !bind(loader, p + ".convs.0", 1, b.convs0, true, missing) ||
          !bind(loader, p + ".convs.1", 1, b.convs1, true, missing) ||
          !bind(loader, p + ".conv3", 1, b.conv3, true, missing) ||
          !bind(loader, p + ".shortcut.shortcut.0", stride, b.shortcut, false,
                missing))
        return false;
      if (s.fused && (!bind(loader, p + ".fuse_models.0.local_att.local_att.0",
                            1, b.fuse.att0, true, missing) ||
                      !bind(loader, p + ".fuse_models.0.local_att.local_att.3",
                            1, b.fuse.att3, true, missing)))
        return false;
      blocks_.push_back(b);
    }
    if (std::string(s.name) == "layer3")
      layer3_last_ = blocks_.size() - 1;
  }
  if (!bind(loader, "layer3_ds", 2, layer3_ds_, true, missing) ||
      !bind(loader, "fuse34.local_att.local_att.0", 1, fuse34_.att0, true,
            missing) ||
      !bind(loader, "fuse34.local_att.local_att.3", 1, fuse34_.att3, true,
            missing))
    return false;
  std::vector<uint32_t> dims;
  seg_w_ = loader.getTensor("seg_1.weight", &dims);
  seg_b_ = loader.getTensor("seg_1.bias");
  if (!seg_w_ || !seg_b_ || dims.size() != 2 || dims[0] != DIM) {
    if (missing)
      *missing = "seg_1.weight";
    return false;
  }
  return true;
}

std::vector<float> ERes2NetV2::fbank(const float *samples, size_t n,
                                     size_t &frames) {
  // frames are centred on i * shift + shift / 2 and reflected at the edges
  // (Kaldi with snip_edges false)
  frames = n == 0 ? 0 : (n + FRAME_SHIFT / 2) / FRAME_SHIFT;
  std::vector<float> out(frames * MEL);
  const auto &banks = melBanks();
  const auto &window = poveyWindow();
  const long len = static_cast<long>(n);
#pragma omp parallel for schedule(static)
  for (size_t f = 0; f < frames; ++f) {
    double frame[FRAME_LEN];
    const long start = static_cast<long>(f * FRAME_SHIFT + FRAME_SHIFT / 2) -
                       static_cast<long>(FRAME_LEN / 2);
    double mean = 0.0;
    for (size_t i = 0; i < FRAME_LEN; ++i) {
      long s = start + static_cast<long>(i);
      while (s < 0 || s >= len)
        s = s < 0 ? -s - 1 : 2 * len - 1 - s;
      frame[i] = samples[s];
      mean += frame[i];
    }
    mean /= FRAME_LEN;
    for (size_t i = 0; i < FRAME_LEN; ++i)
      frame[i] -= mean; // remove the DC offset
    for (size_t i = FRAME_LEN - 1; i > 0; --i)
      frame[i] -= 0.97 * frame[i - 1]; // pre-emphasis
    frame[0] -= 0.97 * frame[0];
    std::complex<double> spec[FFT_N];
    for (size_t i = 0; i < FFT_N; ++i)
      spec[i] = i < FRAME_LEN ? frame[i] * window[i] : 0.0;
    fft(spec);
    double power[BINS];
    for (size_t k = 0; k < BINS; ++k)
      power[k] = std::norm(spec[k]);
    for (size_t m = 0; m < MEL; ++m) {
      double e = 0.0;
      const float *row = banks.data() + m * BINS;
      for (size_t k = 0; k < BINS; ++k)
        e += row[k] * power[k];
      out[f * MEL + m] = static_cast<float>(
        std::log(std::max(e, static_cast<double>(FLT_EPSILON))));
    }
  }
  for (size_t m = 0; m < MEL && frames; ++m) { // global-mean normalization
    double sum = 0.0;
    for (size_t f = 0; f < frames; ++f)
      sum += out[f * MEL + m];
    const float mean = static_cast<float>(sum / frames);
    for (size_t f = 0; f < frames; ++f)
      out[f * MEL + m] -= mean;
  }
  return out;
}

std::vector<float> ERes2NetV2::forward(const float *feats,
                                       size_t frames) const {
  // the pooled time axis is frames / 8: its standard deviation needs two
  if (frames < 9)
    return {};
  auto run = [](const Conv &c, const Map &x) {
    return conv(c.w, c.b, c.out, c.k, c.stride, x.d.data(), x.c, x.h, x.w);
  };
  // attentional feature fusion: x weighted by 1 + tanh(z), ds by 1 - tanh(z)
  auto fuse = [&](const Fuse &f, const Map &x, const Map &ds) {
    Map cat = concat(x.d.data(), x.c, ds.d.data(), ds.c, x.h, x.w);
    Map z = run(f.att0, cat);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < z.d.size(); ++i)
      z.d[i] = z.d[i] / (1.0f + std::exp(-z.d[i])); // SiLU
    Map a = run(f.att3, z);
    Map y = x;
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < y.d.size(); ++i) {
      const float g = 1.0f + std::tanh(a.d[i]);
      y.d[i] = x.d[i] * g + ds.d[i] * (2.0f - g);
    }
    return y;
  };

  Map x;
  x.c = 1;
  x.h = MEL;
  x.w = frames;
  x.d.resize(MEL * frames);
  for (size_t t = 0; t < frames; ++t)
    for (size_t m = 0; m < MEL; ++m)
      x.d[m * frames + t] = feats[t * MEL + m]; // frequency x time
  x = run(conv1_, x);
  for (auto &v : x.d)
    v = std::max(v, 0.0f); // a plain ReLU here

  Map l3;
  for (size_t bi = 0; bi < blocks_.size(); ++bi) {
    const Block &b = blocks_[bi];
    Map out = run(b.conv1, x);
    clip(out.d.data(), out.d.size());
    const size_t width = out.c / 2;
    Map a, rest;
    a.c = rest.c = width;
    a.h = rest.h = out.h;
    a.w = rest.w = out.w;
    a.d.assign(out.d.begin(), out.d.begin() + width * out.plane());
    rest.d.assign(out.d.begin() + width * out.plane(), out.d.end());
    Map y0 = run(b.convs0, a);
    clip(y0.d.data(), y0.d.size());
    Map sp;
    if (b.fused) {
      sp = fuse(b.fuse, y0, rest);
    } else {
      sp = y0;
      for (size_t i = 0; i < sp.d.size(); ++i)
        sp.d[i] += rest.d[i];
    }
    Map y1 = run(b.convs1, sp);
    clip(y1.d.data(), y1.d.size());
    Map cat = concat(y0.d.data(), y0.c, y1.d.data(), y1.c, y0.h, y0.w);
    Map y = run(b.conv3, cat);
    if (b.shortcut.w) {
      const Map s = run(b.shortcut, x);
      for (size_t i = 0; i < y.d.size(); ++i)
        y.d[i] += s.d[i];
    } else {
      for (size_t i = 0; i < y.d.size(); ++i)
        y.d[i] += x.d[i];
    }
    clip(y.d.data(), y.d.size());
    x = std::move(y);
    if (bi == layer3_last_)
      l3 = x;
  }
  const Map ds = run(layer3_ds_, l3);
  x = fuse(fuse34_, x, ds);

  // statistics pooling over time: mean and (unbiased) standard deviation
  const size_t rows = x.c * x.h, t = x.w;
  std::vector<float> stats(2 * rows);
#pragma omp parallel for schedule(static)
  for (size_t r = 0; r < rows; ++r) {
    const float *row = x.d.data() + r * t;
    double sum = 0.0;
    for (size_t i = 0; i < t; ++i)
      sum += row[i];
    const double mean = sum / t;
    double var = 0.0;
    for (size_t i = 0; i < t; ++i)
      var += (row[i] - mean) * (row[i] - mean);
    stats[r] = static_cast<float>(mean);
    stats[rows + r] = static_cast<float>(std::sqrt(var / (t - 1) + POOL_EPS));
  }
  std::vector<float> emb(DIM);
  for (size_t o = 0; o < DIM; ++o) {
    const float *w = seg_w_ + o * 2 * rows;
    double acc = seg_b_[o];
    for (size_t i = 0; i < 2 * rows; ++i)
      acc += static_cast<double>(w[i]) * stats[i];
    emb[o] = static_cast<float>(acc);
  }
  return emb;
}

std::vector<float> ERes2NetV2::embed(const float *samples, size_t n) const {
  size_t frames = 0;
  const auto feats = fbank(samples, n, frames);
  return forward(feats.data(), frames);
}

} // namespace speaker_diarization
