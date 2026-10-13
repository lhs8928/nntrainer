// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   pyannet.cpp
 * @date   29 September 2026
 * @brief  PyanNet Segmentation Model implementation in NNTrainer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "pyannet.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cassert>

#if (defined(__ARM_FEATURE_DOTPROD) && __ARM_FEATURE_DOTPROD) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#if defined(USE_BLAS)
#include <cblas_interface.h>
#else
static inline void sgemm_fallback(bool transB, size_t M, size_t N, size_t K,
                                  const float *A, const float *B, float *C) {
  for (size_t m = 0; m < M; ++m) {
    const float *a_row = A + m * K;
    float *c_row = C + m * N;
    for (size_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        float b_val = transB ? B[n * K + k] : B[k * N + n];
        acc += a_row[k] * b_val;
      }
      c_row[n] += acc;
    }
  }
}
static inline void sgemv_fallback(size_t M, size_t N, const float *A, const float *X, float *Y) {
  for (size_t m = 0; m < M; ++m) {
    const float *a_row = A + m * N;
    float acc = 0.0f;
    for (size_t n = 0; n < N; ++n) {
      acc += a_row[n] * X[n];
    }
    Y[m] += acc;
  }
}
#endif

namespace speaker_diarization {

static inline float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

static inline float leaky_relu(float x, float slope = 0.01f) {
  return x >= 0.0f ? x : slope * x;
}

static void instance_norm_1d(const float *src, float *dst, size_t channels, size_t length,
                             const float *gamma, const float *beta, float eps = 1e-5f) {
  for (size_t c = 0; c < channels; ++c) {
    const float *channel_src = src + c * length;
    float *channel_dst = dst + c * length;

    float mean = 0.0f;
    for (size_t i = 0; i < length; ++i) {
      mean += channel_src[i];
    }
    mean /= length;

    float var = 0.0f;
    for (size_t i = 0; i < length; ++i) {
      float diff = channel_src[i] - mean;
      var += diff * diff;
    }
    var /= length;

    float inv_std = 1.0f / std::sqrt(var + eps);
    float g = gamma ? gamma[c] : 1.0f;
    float b = beta ? beta[c] : 0.0f;

    for (size_t i = 0; i < length; ++i) {
      channel_dst[i] = (channel_src[i] - mean) * inv_std * g + b;
    }
  }
}

static void max_pool_1d(const float *src, float *dst, size_t channels, size_t in_len,
                        size_t kernel_size, size_t stride) {
  size_t out_len = (in_len - kernel_size) / stride + 1;
  for (size_t c = 0; c < channels; ++c) {
    const float *c_src = src + c * in_len;
    float *c_dst = dst + c * out_len;
    for (size_t t = 0; t < out_len; ++t) {
      size_t in_start = t * stride;
      float max_val = c_src[in_start];
      for (size_t k = 1; k < kernel_size; ++k) {
        max_val = std::max(max_val, c_src[in_start + k]);
      }
      c_dst[t] = max_val;
    }
  }
}

bool PyanNet::init(const WeightLoader &loader) {
  wav_norm1d_w_ = loader.getTensor("sincnet.wav_norm1d.weight");
  wav_norm1d_b_ = loader.getTensor("sincnet.wav_norm1d.bias");
  conv1d_0_w_   = loader.getTensor("sincnet.conv1d_0.weight");
  norm1d_0_w_   = loader.getTensor("sincnet.norm1d_0.weight");
  norm1d_0_b_   = loader.getTensor("sincnet.norm1d_0.bias");

  conv1d_1_w_   = loader.getTensor("sincnet.conv1d_1.weight");
  conv1d_1_b_   = loader.getTensor("sincnet.conv1d_1.bias");
  norm1d_1_w_   = loader.getTensor("sincnet.norm1d_1.weight");
  norm1d_1_b_   = loader.getTensor("sincnet.norm1d_1.bias");

  conv1d_2_w_   = loader.getTensor("sincnet.conv1d_2.weight");
  conv1d_2_b_   = loader.getTensor("sincnet.conv1d_2.bias");
  norm1d_2_w_   = loader.getTensor("sincnet.norm1d_2.weight");
  norm1d_2_b_   = loader.getTensor("sincnet.norm1d_2.bias");

  lstm_layers_.resize(4);
  for (size_t l = 0; l < 4; ++l) {
    std::string prefix = "lstm.";
    std::string l_str = std::to_string(l);
    lstm_layers_[l].w_ih = loader.getTensor(prefix + "weight_ih_l" + l_str);
    lstm_layers_[l].w_hh = loader.getTensor(prefix + "weight_hh_l" + l_str);
    lstm_layers_[l].b_ih = loader.getTensor(prefix + "bias_ih_l" + l_str);
    lstm_layers_[l].b_hh = loader.getTensor(prefix + "bias_hh_l" + l_str);
    lstm_layers_[l].w_ih_rev = loader.getTensor(prefix + "weight_ih_l" + l_str + "_reverse");
    lstm_layers_[l].w_hh_rev = loader.getTensor(prefix + "weight_hh_l" + l_str + "_reverse");
    lstm_layers_[l].b_ih_rev = loader.getTensor(prefix + "bias_ih_l" + l_str + "_reverse");
    lstm_layers_[l].b_hh_rev = loader.getTensor(prefix + "bias_hh_l" + l_str + "_reverse");
    lstm_layers_[l].in_dim = (l == 0) ? 60 : 256;
  }

  linear_0_w_   = loader.getTensor("linear.0.weight");
  linear_0_b_   = loader.getTensor("linear.0.bias");
  linear_1_w_   = loader.getTensor("linear.1.weight");
  linear_1_b_   = loader.getTensor("linear.1.bias");
  classifier_w_ = loader.getTensor("classifier.weight");
  classifier_b_ = loader.getTensor("classifier.bias");

  return (wav_norm1d_w_ && conv1d_0_w_ && linear_0_w_ && classifier_w_);
}

void PyanNet::runLSTMLayer(const float *input, size_t seq_len, const LSTMLayerWeights &weights, float *output) {
  constexpr size_t HIDDEN_DIM = 128;
  constexpr size_t GATES_DIM = 512;
  size_t in_dim = weights.in_dim;

  std::vector<float> h_fwd(seq_len * HIDDEN_DIM, 0.0f);
  std::vector<float> h_bwd(seq_len * HIDDEN_DIM, 0.0f);

  // 1. Precompute sum of biases: [512]
  float b_fwd_sum[GATES_DIM];
  float b_bwd_sum[GATES_DIM];
  for (size_t g = 0; g < GATES_DIM; ++g) {
    b_fwd_sum[g] = weights.b_ih[g] + weights.b_hh[g];
    b_bwd_sum[g] = weights.b_ih_rev[g] + weights.b_hh_rev[g];
  }

  // 2. Sequence-level GEMM: Gates_ih = X * W_ih^T + Bias_sum
  std::vector<float> gates_fwd(seq_len * GATES_DIM);
  std::vector<float> gates_bwd(seq_len * GATES_DIM);

  for (size_t t = 0; t < seq_len; ++t) {
    std::memcpy(gates_fwd.data() + t * GATES_DIM, b_fwd_sum, GATES_DIM * sizeof(float));
    std::memcpy(gates_bwd.data() + t * GATES_DIM, b_bwd_sum, GATES_DIM * sizeof(float));
  }

#if defined(USE_BLAS)
  nntrainer::__cblas_sgemm(0, false, true, seq_len, GATES_DIM, in_dim,
                           1.0f, input, in_dim, weights.w_ih, in_dim,
                           1.0f, gates_fwd.data(), GATES_DIM);
  nntrainer::__cblas_sgemm(0, false, true, seq_len, GATES_DIM, in_dim,
                           1.0f, input, in_dim, weights.w_ih_rev, in_dim,
                           1.0f, gates_bwd.data(), GATES_DIM);
#else
  sgemm_fallback(true, seq_len, GATES_DIM, in_dim, input, weights.w_ih, gates_fwd.data());
  sgemm_fallback(true, seq_len, GATES_DIM, in_dim, input, weights.w_ih_rev, gates_bwd.data());
#endif

  // 3. Run forward and backward LSTM recurrent loops concurrently across 2 CPU cores
  #pragma omp parallel sections
  {
    #pragma omp section
    {
      float c_fwd[HIDDEN_DIM] = {0.0f};
      float curr_h_fwd[HIDDEN_DIM] = {0.0f};

      for (size_t t = 0; t < seq_len; ++t) {
        float *gates = gates_fwd.data() + t * GATES_DIM;

#if defined(USE_BLAS)
        nntrainer::__cblas_sgemv(0, false, GATES_DIM, HIDDEN_DIM,
                                 1.0f, weights.w_hh, HIDDEN_DIM,
                                 curr_h_fwd, 1, 1.0f, gates, 1);
#else
        sgemv_fallback(GATES_DIM, HIDDEN_DIM, weights.w_hh, curr_h_fwd, gates);
#endif

        for (size_t d = 0; d < HIDDEN_DIM; ++d) {
          float i_gate = sigmoid(gates[d]);
          float f_gate = sigmoid(gates[HIDDEN_DIM + d]);
          float g_gate = std::tanh(gates[2 * HIDDEN_DIM + d]);
          float o_gate = sigmoid(gates[3 * HIDDEN_DIM + d]);

          c_fwd[d] = f_gate * c_fwd[d] + i_gate * g_gate;
          curr_h_fwd[d] = o_gate * std::tanh(c_fwd[d]);
          h_fwd[t * HIDDEN_DIM + d] = curr_h_fwd[d];
        }
      }
    }

    #pragma omp section
    {
      float c_bwd[HIDDEN_DIM] = {0.0f};
      float curr_h_bwd[HIDDEN_DIM] = {0.0f};

      for (int t = static_cast<int>(seq_len) - 1; t >= 0; --t) {
        float *gates = gates_bwd.data() + t * GATES_DIM;

#if defined(USE_BLAS)
        nntrainer::__cblas_sgemv(0, false, GATES_DIM, HIDDEN_DIM,
                                 1.0f, weights.w_hh_rev, HIDDEN_DIM,
                                 curr_h_bwd, 1, 1.0f, gates, 1);
#else
        sgemv_fallback(GATES_DIM, HIDDEN_DIM, weights.w_hh_rev, curr_h_bwd, gates);
#endif

        for (size_t d = 0; d < HIDDEN_DIM; ++d) {
          float i_gate = sigmoid(gates[d]);
          float f_gate = sigmoid(gates[HIDDEN_DIM + d]);
          float g_gate = std::tanh(gates[2 * HIDDEN_DIM + d]);
          float o_gate = sigmoid(gates[3 * HIDDEN_DIM + d]);

          c_bwd[d] = f_gate * c_bwd[d] + i_gate * g_gate;
          curr_h_bwd[d] = o_gate * std::tanh(c_bwd[d]);
          h_bwd[t * HIDDEN_DIM + d] = curr_h_bwd[d];
        }
      }
    }
  }

  // 4. Concatenate forward and backward: [seq_len, 256]
  #pragma omp parallel for schedule(static)
  for (size_t t = 0; t < seq_len; ++t) {
    std::memcpy(output + t * 256, h_fwd.data() + t * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));
    std::memcpy(output + t * 256 + HIDDEN_DIM, h_bwd.data() + t * HIDDEN_DIM, HIDDEN_DIM * sizeof(float));
  }
}

std::vector<float> PyanNet::forwardChunk(const float *chunk, size_t chunk_idx, DebugTensorCallback debug_cb) {
  constexpr size_t IN_LEN = 160000;
  std::string pfx = "seg_chunk" + std::to_string(chunk_idx) + "_";

  // 1. SincNet: wav_norm1d
  std::vector<float> wav_norm(IN_LEN);
  instance_norm_1d(chunk, wav_norm.data(), 1, IN_LEN, wav_norm1d_w_, wav_norm1d_b_);
  if (debug_cb) debug_cb(pfx + "wav_norm1d", wav_norm.data(), {1, 1, IN_LEN});

  // 2. conv1d_0 fused with abs and max_pool1d (k=3, s=3) -> pool0_out: 80 x 5325
  // Eliminates 5.1 MB intermediate buffer (conv0_out) and two redundant memory passes
  constexpr size_t POOL0_LEN = 5325;
  std::vector<float> pool0_out(80 * POOL0_LEN);

  if (debug_cb) {
    constexpr size_t OUT0_LEN = 15975;
    std::vector<float> conv0_out(80 * OUT0_LEN);
    for (size_t oc = 0; oc < 80; ++oc) {
      const float *filt = conv1d_0_w_ + oc * 251;
      float *out_row = conv0_out.data() + oc * OUT0_LEN;
      for (size_t t = 0; t < OUT0_LEN; ++t) {
        size_t in_start = t * 10;
        float acc = 0.0f;
        for (size_t k = 0; k < 251; ++k) acc += wav_norm[in_start + k] * filt[k];
        out_row[t] = acc;
      }
    }
    debug_cb(pfx + "conv1d_0", conv0_out.data(), {1, 80, OUT0_LEN});
    for (float &val : conv0_out) val = std::abs(val);
    max_pool_1d(conv0_out.data(), pool0_out.data(), 80, OUT0_LEN, 3, 3);
  } else {
    #pragma omp parallel for schedule(static)
    for (size_t oc = 0; oc < 80; ++oc) {
      const float *filt = conv1d_0_w_ + oc * 251;
      float *pool_row = pool0_out.data() + oc * POOL0_LEN;

      for (size_t pt = 0; pt < POOL0_LEN; ++pt) {
        float max_val = 0.0f;

        for (size_t step = 0; step < 3; ++step) {
          size_t t = pt * 3 + step;
          size_t in_start = t * 10;
          float acc = 0.0f;
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
          float32x4_t vacc0 = vdupq_n_f32(0.0f);
          float32x4_t vacc1 = vdupq_n_f32(0.0f);
          size_t k = 0;
          for (; k + 8 <= 251; k += 8) {
            vacc0 = vfmaq_f32(vacc0, vld1q_f32(wav_norm.data() + in_start + k + 0), vld1q_f32(filt + k + 0));
            vacc1 = vfmaq_f32(vacc1, vld1q_f32(wav_norm.data() + in_start + k + 4), vld1q_f32(filt + k + 4));
          }
          acc = vaddvq_f32(vaddq_f32(vacc0, vacc1));
          for (; k < 251; ++k) {
            acc += wav_norm[in_start + k] * filt[k];
          }
#else
          for (size_t k = 0; k < 251; ++k) {
            acc += wav_norm[in_start + k] * filt[k];
          }
#endif
          float abs_val = std::abs(acc);
          if (abs_val > max_val) max_val = abs_val;
        }

        pool_row[pt] = max_val;
      }
    }
  }

  // 4. norm1d_0 -> leaky_relu(0.01)
  std::vector<float> norm0_out(80 * POOL0_LEN);
  instance_norm_1d(pool0_out.data(), norm0_out.data(), 80, POOL0_LEN, norm1d_0_w_, norm1d_0_b_);
  if (debug_cb) debug_cb(pfx + "norm1d_0", norm0_out.data(), {1, 80, POOL0_LEN});
  for (float &val : norm0_out) val = leaky_relu(val, 0.01f);

  // 5. conv1d_1: 80 in, 60 out, kernel 5, stride 1 -> out_len = 5321
  constexpr size_t OUT1_LEN = 5321;
  std::vector<float> conv1_out(60 * OUT1_LEN);
  #pragma omp parallel for schedule(static)
  for (size_t oc = 0; oc < 60; ++oc) {
    const float *w_oc = conv1d_1_w_ + oc * (80 * 5);
    float b_oc = conv1d_1_b_[oc];
    float *out_row = conv1_out.data() + oc * OUT1_LEN;
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
    const float32x4_t vb = vdupq_n_f32(b_oc);
    size_t t = 0;
    for (; t + 4 <= OUT1_LEN; t += 4) {
      float32x4_t vacc = vb;
      for (size_t ic = 0; ic < 80; ++ic) {
        const float *in_base = norm0_out.data() + ic * POOL0_LEN + t;
        const float *w_ic = w_oc + ic * 5;

        vacc = vfmaq_f32(vacc, vld1q_f32(in_base + 0), vdupq_n_f32(w_ic[0]));
        vacc = vfmaq_f32(vacc, vld1q_f32(in_base + 1), vdupq_n_f32(w_ic[1]));
        vacc = vfmaq_f32(vacc, vld1q_f32(in_base + 2), vdupq_n_f32(w_ic[2]));
        vacc = vfmaq_f32(vacc, vld1q_f32(in_base + 3), vdupq_n_f32(w_ic[3]));
        vacc = vfmaq_f32(vacc, vld1q_f32(in_base + 4), vdupq_n_f32(w_ic[4]));
      }
      vst1q_f32(out_row + t, vacc);
    }
    for (; t < OUT1_LEN; ++t) {
      float acc = b_oc;
      for (size_t ic = 0; ic < 80; ++ic) {
        const float *in_row = norm0_out.data() + ic * POOL0_LEN + t;
        const float *w_ic = w_oc + ic * 5;
        acc += in_row[0] * w_ic[0] + in_row[1] * w_ic[1] + in_row[2] * w_ic[2]
             + in_row[3] * w_ic[3] + in_row[4] * w_ic[4];
      }
      out_row[t] = acc;
    }
#else
    for (size_t t = 0; t < OUT1_LEN; ++t) {
      float acc = b_oc;
      for (size_t ic = 0; ic < 80; ++ic) {
        const float *in_row = norm0_out.data() + ic * POOL0_LEN + t;
        const float *w_ic = w_oc + ic * 5;
        acc += in_row[0] * w_ic[0] + in_row[1] * w_ic[1] + in_row[2] * w_ic[2]
             + in_row[3] * w_ic[3] + in_row[4] * w_ic[4];
      }
      out_row[t] = acc;
    }
#endif
  }
  if (debug_cb) debug_cb(pfx + "conv1d_1", conv1_out.data(), {1, 60, OUT1_LEN});

  // 6. pool1d_1 (k=3, s=3) -> 1773
  constexpr size_t POOL1_LEN = 1773;
  std::vector<float> pool1_out(60 * POOL1_LEN);
  max_pool_1d(conv1_out.data(), pool1_out.data(), 60, OUT1_LEN, 3, 3);

  // 7. norm1d_1 -> leaky_relu
  std::vector<float> norm1_out(60 * POOL1_LEN);
  instance_norm_1d(pool1_out.data(), norm1_out.data(), 60, POOL1_LEN, norm1d_1_w_, norm1d_1_b_);
  if (debug_cb) debug_cb(pfx + "norm1d_1", norm1_out.data(), {1, 60, POOL1_LEN});
  for (float &val : norm1_out) val = leaky_relu(val, 0.01f);

  // 8. conv1d_2: 60 in, 60 out, kernel 5, stride 1 -> out_len = 1769
  constexpr size_t OUT2_LEN = 1769;
  std::vector<float> conv2_out(60 * OUT2_LEN);
  #pragma omp parallel for schedule(static)
  for (size_t oc = 0; oc < 60; ++oc) {
    const float *w_oc = conv1d_2_w_ + oc * (60 * 5);
    float b_oc = conv1d_2_b_[oc];
    float *out_row = conv2_out.data() + oc * OUT2_LEN;
#if (defined(__ARM_NEON) || defined(__ARM_NEON__))
    const float32x4_t vb = vdupq_n_f32(b_oc);
    size_t t = 0;
    for (; t + 4 <= OUT2_LEN; t += 4) {
      float32x4_t vacc = vb;
      for (size_t ic = 0; ic < 60; ++ic) {
        const float *in_base = norm1_out.data() + ic * POOL1_LEN + t;
        const float *w_ic = w_oc + ic * 5;

        vacc = vfmaq_f32(vacc, vld1q_f32(in_base + 0), vdupq_n_f32(w_ic[0]));
        vacc = vfmaq_f32(vacc, vld1q_f32(in_base + 1), vdupq_n_f32(w_ic[1]));
        vacc = vfmaq_f32(vacc, vld1q_f32(in_base + 2), vdupq_n_f32(w_ic[2]));
        vacc = vfmaq_f32(vacc, vld1q_f32(in_base + 3), vdupq_n_f32(w_ic[3]));
        vacc = vfmaq_f32(vacc, vld1q_f32(in_base + 4), vdupq_n_f32(w_ic[4]));
      }
      vst1q_f32(out_row + t, vacc);
    }
    for (; t < OUT2_LEN; ++t) {
      float acc = b_oc;
      for (size_t ic = 0; ic < 60; ++ic) {
        const float *in_row = norm1_out.data() + ic * POOL1_LEN + t;
        const float *w_ic = w_oc + ic * 5;
        acc += in_row[0] * w_ic[0] + in_row[1] * w_ic[1] + in_row[2] * w_ic[2]
             + in_row[3] * w_ic[3] + in_row[4] * w_ic[4];
      }
      out_row[t] = acc;
    }
#else
    for (size_t t = 0; t < OUT2_LEN; ++t) {
      float acc = b_oc;
      for (size_t ic = 0; ic < 60; ++ic) {
        const float *in_row = norm1_out.data() + ic * POOL1_LEN + t;
        const float *w_ic = w_oc + ic * 5;
        acc += in_row[0] * w_ic[0] + in_row[1] * w_ic[1] + in_row[2] * w_ic[2]
             + in_row[3] * w_ic[3] + in_row[4] * w_ic[4];
      }
      out_row[t] = acc;
    }
#endif
  }
  if (debug_cb) debug_cb(pfx + "conv1d_2", conv2_out.data(), {1, 60, OUT2_LEN});

  // 9. pool1d_2 (k=3, s=3) -> 589
  constexpr size_t POOL2_LEN = 589;
  std::vector<float> pool2_out(60 * POOL2_LEN);
  max_pool_1d(conv2_out.data(), pool2_out.data(), 60, OUT2_LEN, 3, 3);

  // 10. norm1d_2 -> leaky_relu
  std::vector<float> norm2_out(60 * POOL2_LEN);
  instance_norm_1d(pool2_out.data(), norm2_out.data(), 60, POOL2_LEN, norm1d_2_w_, norm1d_2_b_);
  if (debug_cb) debug_cb(pfx + "norm1d_2", norm2_out.data(), {1, 60, POOL2_LEN});
  for (float &val : norm2_out) val = leaky_relu(val, 0.01f);

  // 11. Transpose [60, 589] to [589, 60] for LSTM input
  std::vector<float> lstm_in(589 * 60);
  for (size_t c = 0; c < 60; ++c) {
    for (size_t t = 0; t < 589; ++t) {
      lstm_in[t * 60 + c] = norm2_out[c * 589 + t];
    }
  }

  // 12. 4-Layer Bidirectional LSTM
  std::vector<float> lstm_layer_in = std::move(lstm_in);
  std::vector<float> lstm_layer_out(589 * 256);

  for (size_t l = 0; l < 4; ++l) {
    runLSTMLayer(lstm_layer_in.data(), 589, lstm_layers_[l], lstm_layer_out.data());
    if (l < 3) {
      lstm_layer_in = lstm_layer_out;
    }
  }
  if (debug_cb) debug_cb(pfx + "lstm", lstm_layer_out.data(), {1, 589, 256});

  // 13. linear.0: [589, 256] -> [589, 128] + LeakyReLU(0.01)
  std::vector<float> lin0_out(589 * 128);
  for (size_t t = 0; t < 589; ++t) {
    const float *x_t = lstm_layer_out.data() + t * 256;
    for (size_t oc = 0; oc < 128; ++oc) {
      const float *w_row = linear_0_w_ + oc * 256;
      float acc = linear_0_b_[oc];
      for (size_t ic = 0; ic < 256; ++ic) {
        acc += w_row[ic] * x_t[ic];
      }
      lin0_out[t * 128 + oc] = acc;
    }
  }
  if (debug_cb) debug_cb(pfx + "linear_0", lin0_out.data(), {1, 589, 128});
  for (float &val : lin0_out) val = leaky_relu(val, 0.01f);

  // 14. linear.1: [589, 128] -> [589, 128] + LeakyReLU(0.01)
  std::vector<float> lin1_out(589 * 128);
  for (size_t t = 0; t < 589; ++t) {
    const float *x_t = lin0_out.data() + t * 128;
    for (size_t oc = 0; oc < 128; ++oc) {
      const float *w_row = linear_1_w_ + oc * 128;
      float acc = linear_1_b_[oc];
      for (size_t ic = 0; ic < 128; ++ic) {
        acc += w_row[ic] * x_t[ic];
      }
      lin1_out[t * 128 + oc] = acc;
    }
  }
  if (debug_cb) debug_cb(pfx + "linear_1", lin1_out.data(), {1, 589, 128});
  for (float &val : lin1_out) val = leaky_relu(val, 0.01f);

  // 15. classifier: [589, 128] -> [589, 7]
  std::vector<float> cls_out(589 * 7);
  for (size_t t = 0; t < 589; ++t) {
    const float *x_t = lin1_out.data() + t * 128;
    for (size_t oc = 0; oc < 7; ++oc) {
      const float *w_row = classifier_w_ + oc * 128;
      float acc = classifier_b_[oc];
      for (size_t ic = 0; ic < 128; ++ic) {
        acc += w_row[ic] * x_t[ic];
      }
      cls_out[t * 7 + oc] = acc;
    }
  }
  if (debug_cb) debug_cb(pfx + "classifier", cls_out.data(), {1, 589, 7});

  // 16. LogSoftmax: [589, 7]
  std::vector<float> logsoftmax_out(589 * 7);
  for (size_t t = 0; t < 589; ++t) {
    const float *row = cls_out.data() + t * 7;
    float *log_row = logsoftmax_out.data() + t * 7;
    float max_v = *std::max_element(row, row + 7);
    float sum_exp = 0.0f;
    for (size_t c = 0; c < 7; ++c) {
      sum_exp += std::exp(row[c] - max_v);
    }
    float log_sum = max_v + std::log(sum_exp);
    for (size_t c = 0; c < 7; ++c) {
      log_row[c] = row[c] - log_sum;
    }
  }
  if (debug_cb) debug_cb(pfx + "activation_logsoftmax", logsoftmax_out.data(), {1, 589, 7});

  // 17. Powerset to multi-label: [589, 3]
  // Mapping:
  // 0: [0, 0, 0]
  // 1: [1, 0, 0]
  // 2: [0, 1, 0]
  // 3: [0, 0, 1]
  // 4: [1, 1, 0]
  // 5: [1, 0, 1]
  // 6: [0, 1, 1]
  static const float POWERSET_MAP[7][3] = {
    {0.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 1.0f},
    {1.0f, 1.0f, 0.0f},
    {1.0f, 0.0f, 1.0f},
    {0.0f, 1.0f, 1.0f}
  };

  std::vector<float> multilabel(589 * 3, 0.0f);
  for (size_t t = 0; t < 589; ++t) {
    const float *row = logsoftmax_out.data() + t * 7;
    size_t best_class = 0;
    float max_val = row[0];
    for (size_t c = 1; c < 7; ++c) {
      if (row[c] > max_val) {
        max_val = row[c];
        best_class = c;
      }
    }
    multilabel[t * 3 + 0] = POWERSET_MAP[best_class][0];
    multilabel[t * 3 + 1] = POWERSET_MAP[best_class][1];
    multilabel[t * 3 + 2] = POWERSET_MAP[best_class][2];
  }

  return multilabel;
}

void PyanNet::runInference(const std::vector<std::vector<float>> &chunks,
                          std::vector<std::vector<float>> &out_segmentations,
                          std::vector<uint8_t> &out_speaker_counting,
                          DebugTensorCallback debug_cb) {
  size_t num_chunks = chunks.size();
  out_segmentations.resize(num_chunks);

  for (size_t c = 0; c < num_chunks; ++c) {
    out_segmentations[c] = forwardChunk(chunks[c].data(), c, debug_cb);
  }

  // Frame aggregation across chunks:
  // Step = 0.016875s (approx 1.0s / 589 or 270 samples)
  // Total frames for 15s audio = 949 frames
  constexpr size_t TOTAL_FRAMES = 949;
  constexpr size_t FRAMES_PER_CHUNK = 589;
  constexpr double STEP_SEC = 1.0;
  constexpr double FRAME_STEP_SEC = 0.016875;

  out_speaker_counting.assign(TOTAL_FRAMES, 0);
  std::vector<float> frame_sums(TOTAL_FRAMES, 0.0f);
  std::vector<float> frame_weights(TOTAL_FRAMES, 0.0f);

  for (size_t c = 0; c < num_chunks; ++c) {
    size_t chunk_start_frame = static_cast<size_t>(std::round(c * STEP_SEC / FRAME_STEP_SEC));
    for (size_t f = 0; f < FRAMES_PER_CHUNK; ++f) {
      size_t global_frame = chunk_start_frame + f;
      if (global_frame >= TOTAL_FRAMES) break;

      float spk_sum = out_segmentations[c][f * 3 + 0]
                    + out_segmentations[c][f * 3 + 1]
                    + out_segmentations[c][f * 3 + 2];
      frame_sums[global_frame] += spk_sum;
      frame_weights[global_frame] += 1.0f;
    }
  }

  for (size_t f = 0; f < TOTAL_FRAMES; ++f) {
    if (frame_weights[f] > 0.0f) {
      out_speaker_counting[f] = static_cast<uint8_t>(std::rint(frame_sums[f] / frame_weights[f]));
    } else {
      out_speaker_counting[f] = 0;
    }
  }
}

} // namespace speaker_diarization
