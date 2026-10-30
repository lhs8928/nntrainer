// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   silero_vad.cpp
 * @date   30 October 2026
 * @brief  Silero VAD v6 (16 kHz) without ONNX Runtime
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * One block is about 0.75 M multiply-adds, so plain loops are enough. The
 * sums run in double: with float sums the probability drifted up to 1.7e-5
 * from ONNX Runtime on a 6-minute recording, with double sums 5e-6.
 */

#include "silero_vad.h"

#include "weight_loader.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace speaker_diarization {

namespace {

constexpr size_t PAD = 128, WINDOW = 256, HOP = 128, BINS = 129;
constexpr size_t INPUT = SileroVad::CONTEXT + SileroVad::BLOCK; // 576
constexpr size_t PADDED = INPUT + 2 * PAD;                      // 832
constexpr size_t FRAMES = (PADDED - WINDOW) / HOP;              // 5 - first = 4

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

/** @brief copy a tensor of exactly `size` floats */
bool take(const WeightLoader &w, const std::string &name, size_t size,
          std::vector<float> &out, std::string *error) {
  std::vector<uint32_t> dims;
  if (!w.hasTensor(name)) {
    if (error)
      *error = "missing tensor " + name;
    return false;
  }
  const float *p = w.getTensor(name, &dims);
  size_t n = 1;
  for (uint32_t d : dims)
    n *= d;
  if (!p || n != size) {
    if (error)
      *error = "tensor " + name + " has " + std::to_string(n) +
               " values, expected " + std::to_string(size);
    return false;
  }
  out.assign(p, p + n);
  return true;
}

/** @brief Conv1d (kernel 3, padding 1) + ReLU: x[in][t] -> y[out][t'] */
size_t conv_relu(const float *x, size_t in, size_t t,
                 const std::vector<float> &w, const std::vector<float> &b,
                 size_t out, size_t stride, float *y) {
  const size_t t_out = (t + 2 - 3) / stride + 1;
  for (size_t o = 0; o < out; ++o) {
    for (size_t j = 0; j < t_out; ++j) {
      double acc = b[o];
      const long start = static_cast<long>(j * stride) - 1;
      for (size_t i = 0; i < in; ++i) {
        const float *wk = w.data() + (o * in + i) * 3;
        const float *xi = x + i * t;
        for (long k = 0; k < 3; ++k) {
          const long s = start + k;
          if (s >= 0 && s < static_cast<long>(t))
            acc += wk[k] * xi[s];
        }
      }
      y[o * t_out + j] = std::max(static_cast<float>(acc), 0.0f);
    }
  }
  return t_out;
}

} // namespace

bool SileroVad::load(const std::string &path, std::string *error) {
  WeightLoader w;
  if (!w.load(path)) {
    if (error)
      *error = "cannot read " + path;
    return false;
  }
  if (w.hasTensor("meta.model_manifest")) {
    std::vector<uint32_t> dims;
    const auto *p = reinterpret_cast<const char *>(
      w.getTensorI8("meta.model_manifest", &dims));
    if (p && !dims.empty())
      manifest_.assign(p, dims[0]);
  }
  const size_t shapes[4][3] = {
    {BINS, 128, 1}, {128, 64, 2}, {64, 64, 2}, {64, 128, 1}}; // in, out, stride
  std::vector<float> wb, rb;
  bool ok = take(w, "stft.basis", 2 * BINS * WINDOW, basis_, error);
  for (size_t i = 0; ok && i < 4; ++i) {
    Conv &c = convs_[i];
    c.in = shapes[i][0], c.out = shapes[i][1], c.stride = shapes[i][2];
    const std::string n = "conv" + std::to_string(i);
    ok = take(w, n + ".weight", c.out * c.in * 3, c.w, error) &&
         take(w, n + ".bias", c.out, c.b, error);
  }
  std::vector<float> ob;
  ok = ok && take(w, "lstm.w", 4 * HIDDEN * HIDDEN, lstm_w_, error) &&
       take(w, "lstm.r", 4 * HIDDEN * HIDDEN, lstm_r_, error) &&
       take(w, "lstm.wb", 4 * HIDDEN, wb, error) &&
       take(w, "lstm.rb", 4 * HIDDEN, rb, error) &&
       take(w, "out.weight", HIDDEN, out_w_, error) &&
       take(w, "out.bias", 1, ob, error);
  if (!ok)
    return false;
  lstm_b_.resize(4 * HIDDEN);
  for (size_t i = 0; i < lstm_b_.size(); ++i)
    lstm_b_[i] = wb[i] + rb[i];
  out_b_ = ob[0];
  reset();
  return true;
}

void SileroVad::reset() {
  h_.fill(0.0f);
  c_.fill(0.0f);
  context_.fill(0.0f);
}

float SileroVad::run(const float *block) {
  // input = context + block, reflected by PAD on both sides
  float x[INPUT], padded[PADDED];
  std::memcpy(x, context_.data(), CONTEXT * sizeof(float));
  std::memcpy(x + CONTEXT, block, BLOCK * sizeof(float));
  std::memcpy(padded + PAD, x, INPUT * sizeof(float));
  for (size_t i = 0; i < PAD; ++i) {
    padded[PAD - 1 - i] = x[i + 1];
    padded[PAD + INPUT + i] = x[INPUT - 2 - i];
  }

  // STFT magnitudes of frames 1..4: mag[bin][frame]
  float mag[BINS * FRAMES];
  for (size_t f = 0; f < FRAMES; ++f) {
    const float *frame = padded + (f + 1) * HOP;
    for (size_t k = 0; k < BINS; ++k) {
      const float *re_w = basis_.data() + k * WINDOW;
      const float *im_w = basis_.data() + (k + BINS) * WINDOW;
      double re = 0.0, im = 0.0;
      for (size_t n = 0; n < WINDOW; ++n) {
        re += re_w[n] * frame[n];
        im += im_w[n] * frame[n];
      }
      mag[k * FRAMES + f] = static_cast<float>(std::sqrt(re * re + im * im));
    }
  }

  // four Conv1d + ReLU: 129 x 4 -> 128 x 4 -> 64 x 2 -> 64 x 1 -> 128 x 1
  float a[128 * FRAMES], b[128 * FRAMES];
  size_t t = conv_relu(mag, BINS, FRAMES, convs_[0].w, convs_[0].b,
                       convs_[0].out, convs_[0].stride, a);
  t = conv_relu(a, convs_[1].in, t, convs_[1].w, convs_[1].b, convs_[1].out,
                convs_[1].stride, b);
  t = conv_relu(b, convs_[2].in, t, convs_[2].w, convs_[2].b, convs_[2].out,
                convs_[2].stride, a);
  t = conv_relu(a, convs_[3].in, t, convs_[3].w, convs_[3].b, convs_[3].out,
                convs_[3].stride, b); // t == 1: b is the LSTM input

  // one LSTM step, ONNX gate order i, o, f, c
  float gates[4 * HIDDEN];
  for (size_t g = 0; g < 4 * HIDDEN; ++g) {
    const float *wg = lstm_w_.data() + g * HIDDEN;
    const float *rg = lstm_r_.data() + g * HIDDEN;
    double acc = lstm_b_[g];
    for (size_t i = 0; i < HIDDEN; ++i)
      acc += wg[i] * b[i] + rg[i] * h_[i];
    gates[g] = static_cast<float>(acc);
  }
  float p = out_b_;
  for (size_t i = 0; i < HIDDEN; ++i) {
    const float in_g = sigmoid(gates[i]);
    const float out_g = sigmoid(gates[HIDDEN + i]);
    const float forget = sigmoid(gates[2 * HIDDEN + i]);
    const float cell = std::tanh(gates[3 * HIDDEN + i]);
    c_[i] = forget * c_[i] + in_g * cell;
    h_[i] = out_g * std::tanh(c_[i]);
    p += out_w_[i] * std::max(h_[i], 0.0f);
  }

  std::memcpy(context_.data(), block + BLOCK - CONTEXT,
              CONTEXT * sizeof(float));
  return sigmoid(p);
}

} // namespace speaker_diarization
