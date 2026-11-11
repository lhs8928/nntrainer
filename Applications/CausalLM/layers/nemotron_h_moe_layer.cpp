// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nemotron_h_moe_layer.cpp
 * @date   09 November 2026
 * @brief  Nemotron-H mixture of experts block ("E" block, inference only)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Claude <noreply@anthropic.com>
 * @bug    No known bugs except for NYI items
 */

#include "nemotron_h_moe_layer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

#include <memory_data.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

NemotronHMoELayer::NemotronHMoELayer() :
  LayerImpl(),
  moe_props(props::NumExperts(), props::NumExpertsPerToken(),
            nntrainer::props::Unit(), props::SharedExpertSize(),
            props::RoutedScalingFactor(), props::NormTopkProb(),
            props::NumExpertGroups(), props::TopkGroups(), props::ExpertFile(),
            props::ExpertOffset()) {}

NemotronHMoELayer::~NemotronHMoELayer() {
  ext_up.clear();
  ext_down.clear();
#if !defined(_WIN32)
  if (map_base != nullptr)
    munmap(map_base, map_length);
#endif
}

size_t
NemotronHMoELayer::projectionBytes(unsigned int rows, unsigned int cols,
                                   ml::train::TensorDim::DataType dtype) {
  const size_t n = static_cast<size_t>(rows) * cols;
  switch (dtype) {
  case ml::train::TensorDim::DataType::FP32:
    return n * sizeof(float);
  case ml::train::TensorDim::DataType::Q4_0:
    return n / 32 * 18; // block_q4_0: fp16 scale + 16 bytes per 32 values
  default:
    throw std::invalid_argument(
      "[NemotronHMoE] external experts support FP32 and Q4_0 only");
  }
}

void NemotronHMoELayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "[NemotronHMoE] requires exactly one input";

  num_experts = std::get<props::NumExperts>(moe_props).get();
  topk = std::get<props::NumExpertsPerToken>(moe_props).get();
  inter_size = std::get<nntrainer::props::Unit>(moe_props).get();
  shared_size = std::get<props::SharedExpertSize>(moe_props).get();
  scaling = std::get<props::RoutedScalingFactor>(moe_props).get();
  norm_topk = std::get<props::NormTopkProb>(moe_props).get();
  n_group = std::get<props::NumExpertGroups>(moe_props).get();
  topk_group = std::get<props::TopkGroups>(moe_props).get();

  NNTR_THROW_IF(topk > num_experts, std::invalid_argument)
    << "[NemotronHMoE] num_experts_per_token exceeds num_experts";
  NNTR_THROW_IF(n_group == 0 || num_experts % n_group != 0,
                std::invalid_argument)
    << "[NemotronHMoE] num_experts must be divisible by n_group";
  NNTR_THROW_IF(topk_group == 0 || topk_group > n_group, std::invalid_argument)
    << "[NemotronHMoE] topk_group must be in [1, n_group]";

  const nntrainer::TensorDim &in_dim = context.getInputDimensions()[0];
  context.setOutputDimensions({in_dim});
  const unsigned int hidden = in_dim.width();
  hidden_size = hidden;
  expert_file = std::get<props::ExpertFile>(moe_props).get();
  expert_offset = static_cast<size_t>(
    std::stoull(std::get<props::ExpertOffset>(moe_props).get()));
  expert_dtype = context.getWeightDataType();
  expert_format = context.getFormat();

  auto fp32_dim = [&](unsigned int h, unsigned int w) {
    return nntrainer::TensorDim(
      {1, 1, h, w},
      {context.getFormat(), ml::train::TensorDim::DataType::FP32});
  };
  auto weight_dim = [&](unsigned int h, unsigned int w) {
    return nntrainer::TensorDim(
      {1, 1, h, w}, {context.getFormat(), context.getWeightDataType()});
  };

  // File order: router, correction bias, (up, down) per expert, shared up,
  // shared down. Projections are stored [in, out] like fully_connected.
  // The router is a [hidden, num_experts] matrix kept as a single FP32 row:
  // Layer::save() leaves one-row weights unquantized, so quantizing the model
  // (nntr_quantize) touches the expert projections only.
  router_idx = context.requestWeight(
    fp32_dim(1, hidden * num_experts), nntrainer::Initializer::NONE,
    nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f, "router", false);
  bias_idx = context.requestWeight(fp32_dim(1, num_experts),
                                   nntrainer::Initializer::ZEROS,
                                   nntrainer::WeightRegularizer::NONE, 0.0f,
                                   0.0f, "e_score_correction_bias", false);

  up_idx.clear();
  down_idx.clear();
  up_idx.reserve(num_experts);
  down_idx.reserve(num_experts);
  // External experts are not weights of the graph: they are mapped from
  // expert_file on the first forward (see mapExternalExperts()).
  for (unsigned int e = 0; e < num_experts && expert_file.empty(); ++e) {
    up_idx.push_back(context.requestWeight(
      weight_dim(hidden, inter_size), nntrainer::Initializer::NONE,
      nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f,
      "expert_up_" + std::to_string(e), false));
    down_idx.push_back(context.requestWeight(
      weight_dim(inter_size, hidden), nntrainer::Initializer::NONE,
      nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f,
      "expert_down_" + std::to_string(e), false));
  }
  if (shared_size > 0) {
    shared_up_idx = context.requestWeight(
      weight_dim(hidden, shared_size), nntrainer::Initializer::NONE,
      nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f, "shared_up", false);
    shared_down_idx = context.requestWeight(
      weight_dim(shared_size, hidden), nntrainer::Initializer::NONE,
      nntrainer::WeightRegularizer::NONE, 0.0f, 0.0f, "shared_down", false);
  }
}

void NemotronHMoELayer::mapExternalExperts() {
#if defined(_WIN32)
  throw std::runtime_error(
    "[NemotronHMoE] expert_file (memory-mapped experts) is not supported on "
    "Windows yet");
#else
  const size_t up_bytes =
    projectionBytes(hidden_size, inter_size, expert_dtype);
  const size_t down_bytes =
    projectionBytes(inter_size, hidden_size, expert_dtype);
  const size_t total = num_experts * (up_bytes + down_bytes);

  const int fd = open(expert_file.c_str(), O_RDONLY);
  NNTR_THROW_IF(fd < 0, std::runtime_error)
    << "[NemotronHMoE] cannot open expert_file " << expert_file;
  struct stat st;
  fstat(fd, &st);
  if (static_cast<size_t>(st.st_size) < expert_offset + total) {
    close(fd);
    NNTR_THROW_IF(true, std::runtime_error)
      << "[NemotronHMoE] " << expert_file << " is too small for " << num_experts
      << " experts at offset " << expert_offset;
  }
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  const size_t aligned = expert_offset / page * page;
  map_length = total + (expert_offset - aligned);
  map_base = mmap(nullptr, map_length, PROT_READ, MAP_SHARED, fd,
                  static_cast<off_t>(aligned));
  close(fd);
  NNTR_THROW_IF(map_base == MAP_FAILED, std::runtime_error)
    << "[NemotronHMoE] mmap of " << expert_file << " failed";
  // the router picks a few experts per token: do not read the others ahead
  madvise(map_base, map_length, MADV_RANDOM);

  char *p = static_cast<char *>(map_base) + (expert_offset - aligned);
  auto wrap = [&](unsigned int rows, unsigned int cols, size_t bytes) {
    nntrainer::Tensor t(
      nntrainer::TensorDim({1, 1, rows, cols}, {expert_format, expert_dtype}),
      false);
    t.setData(std::make_shared<nntrainer::MemoryData>(p), 0, false);
    p += bytes;
    return t;
  };
  ext_up.clear();
  ext_down.clear();
  for (unsigned int e = 0; e < num_experts; ++e) {
    ext_up.push_back(wrap(hidden_size, inter_size, up_bytes));
    ext_down.push_back(wrap(inter_size, hidden_size, down_bytes));
  }
#endif
}

void NemotronHMoELayer::route(
  const float *x, unsigned int n_tokens, unsigned int hidden,
  const float *router_w, const float *bias,
  std::vector<std::vector<std::pair<unsigned int, float>>> &assign) {
  const unsigned int E = num_experts;
  const unsigned int per_group = E / n_group;
  std::vector<float> scores(E), choice(E);
  std::vector<unsigned int> order(E);

  for (unsigned int t = 0; t < n_tokens; ++t) {
    const float *xt = x + static_cast<size_t>(t) * hidden;
    // logits = x . W  (W stored [hidden, E])
    std::fill(scores.begin(), scores.end(), 0.0f);
    for (unsigned int i = 0; i < hidden; ++i) {
      const float xv = xt[i];
      const float *wr = router_w + static_cast<size_t>(i) * E;
      for (unsigned int e = 0; e < E; ++e)
        scores[e] += xv * wr[e];
    }
    for (unsigned int e = 0; e < E; ++e) {
      scores[e] = 1.0f / (1.0f + std::exp(-scores[e]));
      choice[e] = scores[e] + bias[e];
    }

    if (n_group > 1) {
      // keep the topk_group groups with the largest sum of their top-2 scores
      std::vector<float> group_score(n_group);
      for (unsigned int g = 0; g < n_group; ++g) {
        float m1 = -INFINITY, m2 = -INFINITY;
        for (unsigned int j = 0; j < per_group; ++j) {
          const float v = choice[g * per_group + j];
          if (v > m1) {
            m2 = m1;
            m1 = v;
          } else if (v > m2) {
            m2 = v;
          }
        }
        group_score[g] = m1 + (per_group > 1 ? m2 : 0.0f);
      }
      std::vector<unsigned int> gorder(n_group);
      std::iota(gorder.begin(), gorder.end(), 0u);
      std::partial_sort(gorder.begin(), gorder.begin() + topk_group,
                        gorder.end(), [&](unsigned int a, unsigned int b) {
                          return group_score[a] > group_score[b];
                        });
      std::vector<char> keep(n_group, 0);
      for (unsigned int k = 0; k < topk_group; ++k)
        keep[gorder[k]] = 1;
      for (unsigned int e = 0; e < E; ++e)
        if (!keep[e / per_group])
          choice[e] = -INFINITY;
    }

    std::iota(order.begin(), order.end(), 0u);
    std::partial_sort(
      order.begin(), order.begin() + topk, order.end(),
      [&](unsigned int a, unsigned int b) { return choice[a] > choice[b]; });
    float denom = 0.0f;
    for (unsigned int k = 0; k < topk; ++k)
      denom += scores[order[k]];
    denom += 1e-20f;
    for (unsigned int k = 0; k < topk; ++k) {
      float w = scores[order[k]];
      if (norm_topk)
        w /= denom;
      assign[order[k]].emplace_back(t, w * scaling);
    }
  }
}

void NemotronHMoELayer::runExpert(
  const nntrainer::Tensor &x_all, nntrainer::Tensor &out_all,
  const std::vector<std::pair<unsigned int, float>> &tokens,
  const nntrainer::Tensor &up, const nntrainer::Tensor &down,
  unsigned int hidden, unsigned int inter) {
  const unsigned int n = static_cast<unsigned int>(tokens.size());
  if (n == 0)
    return;

  const auto fp32 = nntrainer::TensorDim::TensorType(
    x_all.getFormat(), ml::train::TensorDim::DataType::FP32);
  nntrainer::Tensor xg(nntrainer::TensorDim({1, 1, n, hidden}, fp32));
  nntrainer::Tensor hg(nntrainer::TensorDim({1, 1, n, inter}, fp32));
  nntrainer::Tensor yg(nntrainer::TensorDim({1, 1, n, hidden}, fp32));

  const float *x = x_all.getData<float>();
  float *xd = xg.getData<float>();
  for (unsigned int i = 0; i < n; ++i)
    std::memcpy(xd + static_cast<size_t>(i) * hidden,
                x + static_cast<size_t>(tokens[i].first) * hidden,
                sizeof(float) * hidden);

  xg.dot(up, hg);
  float *h = hg.getData<float>();
  const size_t hn = static_cast<size_t>(n) * inter;
  for (size_t i = 0; i < hn; ++i) {
    const float r = h[i] > 0.0f ? h[i] : 0.0f;
    h[i] = r * r;
  }
  hg.dot(down, yg);

  float *out = out_all.getData<float>();
  const float *y = yg.getData<float>();
  for (unsigned int i = 0; i < n; ++i) {
    float *o = out + static_cast<size_t>(tokens[i].first) * hidden;
    const float *yr = y + static_cast<size_t>(i) * hidden;
    const float w = tokens[i].second;
    for (unsigned int j = 0; j < hidden; ++j)
      o[j] += w * yr[j];
  }
}

void NemotronHMoELayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  NNTR_THROW_IF(training, std::invalid_argument)
    << "[NemotronHMoE] training is not supported";

  nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  NNTR_THROW_IF(input.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[NemotronHMoE] only FP32 activations are supported";

  const unsigned int B = input.batch();
  const unsigned int H = input.height();
  const unsigned int hidden = input.width();
  const unsigned int len = to - from;

  const float *router_w = context.getWeight(router_idx).getData<float>();
  const float *bias = context.getWeight(bias_idx).getData<float>();

  nntrainer::TensorDim step_dim({1, 1, len, hidden}, input.getTensorType());

  for (unsigned int b = 0; b < B; ++b) {
    const size_t offset = static_cast<size_t>(b) * H * hidden;
    nntrainer::Tensor x = input.getSharedDataTensor(step_dim, offset, true);
    nntrainer::Tensor y = output.getSharedDataTensor(step_dim, offset, true);
    y.setZero();

    std::vector<std::vector<std::pair<unsigned int, float>>> assign(
      num_experts);
    route(x.getData<float>(), len, hidden, router_w, bias, assign);

    const bool external = !expert_file.empty();
    if (external && ext_up.empty())
      mapExternalExperts();
    for (unsigned int e = 0; e < num_experts; ++e)
      runExpert(x, y, assign[e],
                external ? ext_up[e] : context.getWeight(up_idx[e]),
                external ? ext_down[e] : context.getWeight(down_idx[e]), hidden,
                inter_size);

    if (shared_size > 0) {
      std::vector<std::pair<unsigned int, float>> all(len);
      for (unsigned int t = 0; t < len; ++t)
        all[t] = {t, 1.0f};
      runExpert(x, y, all, context.getWeight(shared_up_idx),
                context.getWeight(shared_down_idx), hidden, shared_size);
    }
  }
}

void NemotronHMoELayer::forwarding(nntrainer::RunLayerContext &context,
                                   bool training) {
  const unsigned int H = context.getInput(SINGLE_INOUT_IDX).height();
  incremental_forwarding(context, 0, H, training);
}

void NemotronHMoELayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("[NemotronHMoE] calcDerivative is not supported");
}

void NemotronHMoELayer::calcGradient(nntrainer::RunLayerContext &context) {
  throw std::runtime_error("[NemotronHMoE] calcGradient is not supported");
}

void NemotronHMoELayer::exportTo(nntrainer::Exporter &exporter,
                                 const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(moe_props, method, this);
}

void NemotronHMoELayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, moe_props);
  LayerImpl::setProperty(remain_props);
}

} // namespace causallm

#ifdef PLUGGABLE
extern "C" {
nntrainer::Layer *create_nemotron_h_moe_layer() {
  return new causallm::NemotronHMoELayer();
}
void destroy_nemotron_h_moe_layer(nntrainer::Layer *layer) { delete layer; }
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_nemotron_h_moe_layer, destroy_nemotron_h_moe_layer,
  causallm::NemotronHMoELayer::type};
}
#endif
