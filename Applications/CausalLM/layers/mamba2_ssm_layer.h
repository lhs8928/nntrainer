// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   mamba2_ssm_layer.h
 * @date   09 November 2026
 * @brief  Mamba2 state-space core (Nemotron-H "M" block, inference only)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Claude <noreply@anthropic.com>
 * @bug    No known bugs except for NYI items
 *
 * The layer takes the output of the Mamba2 `in_proj` (a plain
 * fully_connected layer) and produces the input of `out_proj`:
 *
 *   [z | xBC | dt] = in   (widths d_inner, conv_dim, num_heads)
 *   xBC   = silu(causal_depthwise_conv1d(xBC) + conv_bias)
 *   x,B,C = split(xBC)    (widths d_inner, G*N, G*N)
 *   dt    = softplus(dt + dt_bias)
 *   h_t   = exp(dt * A) * h_{t-1} + dt * B_t x_t^T     (per head)
 *   y_t   = C_t h_t + D * x_t
 *   out   = rmsnorm_grouped(y * silu(z)) * norm_weight
 *
 * The convolution window and the SSM state are kept inside the layer across
 * incremental_forwarding() calls, so a prefill followed by one-token decode
 * steps is equivalent to a single pass over the whole sequence. A call with
 * from == 0 starts a new sequence and resets both states.
 */

#ifndef __MAMBA2_SSM_LAYER_H__
#define __MAMBA2_SSM_LAYER_H__

#include <array>
#include <limits>
#include <string>
#include <vector>

#include <base_properties.h>
#include <layer_context.h>
#include <layer_impl.h>
#include <node_exporter.h>

#ifndef WIN_EXPORT
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif
#endif

namespace causallm {

namespace props {

/** @brief number of Mamba2 heads */
class MambaNumHeads : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "mamba_num_heads";
  using prop_tag = nntrainer::uint_prop_tag;
};

/** @brief Mamba2 head dimension */
class MambaHeadDim : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "mamba_head_dim";
  using prop_tag = nntrainer::uint_prop_tag;
};

/** @brief number of B/C groups */
class MambaNumGroups : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "mamba_n_groups";
  using prop_tag = nntrainer::uint_prop_tag;
};

/** @brief SSM state size (N) */
class MambaStateSize : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "mamba_state_size";
  using prop_tag = nntrainer::uint_prop_tag;
};

/** @brief depthwise causal convolution kernel size */
class MambaConvKernel : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key = "mamba_conv_kernel";
  using prop_tag = nntrainer::uint_prop_tag;
};

/** @brief epsilon of the gated grouped RMSNorm */
class MambaNormEps : public nntrainer::Property<float> {
public:
  MambaNormEps(float value = 1e-5f) { set(value); }
  static constexpr const char *key = "mamba_norm_eps";
  using prop_tag = nntrainer::float_prop_tag;
};

/**
 * @brief lower clamp of dt applied on multi-token (prefill) steps, mirroring
 *        the `dt_limit=(time_step_min, inf)` of the reference chunked scan.
 *        Single-token steps are not clamped (reference selective update).
 */
class MambaDtMin : public nntrainer::Property<float> {
public:
  MambaDtMin(float value = 0.0f) { set(value); }
  static constexpr const char *key = "mamba_dt_min";
  using prop_tag = nntrainer::float_prop_tag;
};

} // namespace props

/**
 * @class   Mamba2SSMLayer
 * @brief   Mamba2 SSM core with persistent conv / ssm state
 */
class WIN_EXPORT Mamba2SSMLayer : public nntrainer::LayerImpl {
public:
  Mamba2SSMLayer();
  ~Mamba2SSMLayer() = default;

  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override;
  void calcGradient(nntrainer::RunLayerContext &context) override;
  void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dims) override;
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;
  void setProperty(const std::vector<std::string> &values) override;

  const std::string getType() const override { return Mamba2SSMLayer::type; }
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "mamba2_ssm";

private:
  /** @brief run one sequence segment of `len` rows for one batch entry */
  void runSegment(const float *in, float *out, float *conv_state,
                  float *ssm_state, unsigned int len, bool clamp_dt,
                  const float *conv_w, const float *conv_b,
                  const float *dt_bias, const float *A_log, const float *D,
                  const float *norm_w);

  std::tuple<props::MambaNumHeads, props::MambaHeadDim, props::MambaNumGroups,
             props::MambaStateSize, props::MambaConvKernel, props::MambaNormEps,
             props::MambaDtMin>
    mamba_props;

  enum WeightIdx {
    CONV_W = 0,
    CONV_B,
    DT_BIAS,
    A_LOG,
    D_SKIP,
    NORM_W,
    NUM_WEIGHTS
  };
  enum TensorIdx { CONV_STATE = 0, SSM_STATE, NUM_TENSORS };

  std::array<unsigned int, NUM_WEIGHTS> weight_idx;
  std::array<unsigned int, NUM_TENSORS> tensor_idx;

  unsigned int n_heads = 0;
  unsigned int head_dim = 0;
  unsigned int n_groups = 0;
  unsigned int d_state = 0;
  unsigned int kernel = 0;
  unsigned int d_inner = 0;
  unsigned int conv_dim = 0;
  unsigned int in_width = 0;
  float norm_eps = 1e-5f;
  float dt_min = 0.0f;
};

} // namespace causallm

#endif /* __MAMBA2_SSM_LAYER_H__ */
