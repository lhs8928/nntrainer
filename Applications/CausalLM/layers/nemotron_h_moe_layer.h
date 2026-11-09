// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nemotron_h_moe_layer.h
 * @date   09 November 2026
 * @brief  Nemotron-H mixture of experts block ("E" block, inference only)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Claude <noreply@anthropic.com>
 * @bug    No known bugs except for NYI items
 *
 *   s       = sigmoid(x W_router)                         (FP32)
 *   chosen  = topk(s + e_score_correction_bias)           (group limited)
 *   w_k     = s[chosen_k] / sum(s[chosen]) * routed_scaling_factor
 *   out     = sum_k w_k * down_k(relu(up_k(x))^2) + down_s(relu(up_s(x))^2)
 *
 * Unlike SwiGLU MoE blocks, the experts are plain two-matrix MLPs with a
 * squared ReLU, and one always-on shared expert is added to the routed sum.
 */

#ifndef __NEMOTRON_H_MOE_LAYER_H__
#define __NEMOTRON_H_MOE_LAYER_H__

#include <limits>
#include <string>
#include <vector>

#include <base_properties.h>
#include <common_properties.h>
#include <layer_context.h>
#include <layer_impl.h>
#include <node_exporter.h>

#include <causallm_common_properties.h>

#ifndef WIN_EXPORT
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif
#endif

namespace causallm {

namespace props {

/** @brief intermediate size of the shared expert (0 = no shared expert) */
class SharedExpertSize : public nntrainer::Property<unsigned int> {
public:
  SharedExpertSize(unsigned int value = 0) { set(value); }
  static constexpr const char *key = "shared_expert_size";
  using prop_tag = nntrainer::uint_prop_tag;
};

/** @brief scale applied to the routed expert weights */
class RoutedScalingFactor : public nntrainer::Property<float> {
public:
  RoutedScalingFactor(float value = 1.0f) { set(value); }
  static constexpr const char *key = "routed_scaling_factor";
  using prop_tag = nntrainer::float_prop_tag;
};

/** @brief whether the chosen routing weights are renormalized */
class NormTopkProb : public nntrainer::Property<bool> {
public:
  NormTopkProb(bool value = true) { set(value); }
  static constexpr const char *key = "norm_topk_prob";
  using prop_tag = nntrainer::bool_prop_tag;
};

/** @brief number of expert groups for group limited routing */
class NumExpertGroups : public nntrainer::Property<unsigned int> {
public:
  NumExpertGroups(unsigned int value = 1) { set(value); }
  static constexpr const char *key = "n_group";
  using prop_tag = nntrainer::uint_prop_tag;
};

/** @brief number of groups kept by group limited routing */
class TopkGroups : public nntrainer::Property<unsigned int> {
public:
  TopkGroups(unsigned int value = 1) { set(value); }
  static constexpr const char *key = "topk_group";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

/**
 * @class   NemotronHMoELayer
 * @brief   sigmoid-routed, squared-ReLU mixture of experts with shared expert
 */
class WIN_EXPORT NemotronHMoELayer : public nntrainer::LayerImpl {
public:
  NemotronHMoELayer();
  ~NemotronHMoELayer() = default;

  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override;
  void calcGradient(nntrainer::RunLayerContext &context) override;
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;
  void setProperty(const std::vector<std::string> &values) override;

  const std::string getType() const override { return NemotronHMoELayer::type; }
  bool supportBackwarding() const override { return false; }

  static constexpr const char *type = "nemotron_h_moe";

private:
  /**
   * @brief route a block of tokens
   * @param x       [n_tokens, hidden] activations
   * @param assign  per expert list of (token, weight)
   */
  void route(const float *x, unsigned int n_tokens, unsigned int hidden,
             const float *router_w, const float *bias,
             std::vector<std::vector<std::pair<unsigned int, float>>> &assign);

  /**
   * @brief out[token] += weight * down(relu(up(x[token]))^2) for every
   *        assigned token, batched in one GEMM per projection
   */
  void runExpert(const nntrainer::Tensor &x_all, nntrainer::Tensor &out_all,
                 const std::vector<std::pair<unsigned int, float>> &tokens,
                 const nntrainer::Tensor &up, const nntrainer::Tensor &down,
                 unsigned int hidden, unsigned int inter);

  std::tuple<props::NumExperts, props::NumExpertsPerToken,
             nntrainer::props::Unit, props::SharedExpertSize,
             props::RoutedScalingFactor, props::NormTopkProb,
             props::NumExpertGroups, props::TopkGroups>
    moe_props;

  unsigned int num_experts = 0;
  unsigned int topk = 0;
  unsigned int inter_size = 0;
  unsigned int shared_size = 0;
  float scaling = 1.0f;
  bool norm_topk = true;
  unsigned int n_group = 1;
  unsigned int topk_group = 1;

  unsigned int router_idx = std::numeric_limits<unsigned int>::max();
  unsigned int bias_idx = std::numeric_limits<unsigned int>::max();
  std::vector<unsigned int> up_idx;
  std::vector<unsigned int> down_idx;
  unsigned int shared_up_idx = std::numeric_limits<unsigned int>::max();
  unsigned int shared_down_idx = std::numeric_limits<unsigned int>::max();
};

} // namespace causallm

#endif /* __NEMOTRON_H_MOE_LAYER_H__ */
