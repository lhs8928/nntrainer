// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   addition_layer.cpp
 * @date   30 July 2020
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Addition Layer Class for Neural Network
 *
 */

#include <cstring>
#include <fstream>

#include <addition_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void AdditionLayer::finalize(InitLayerContext &context) {
  if (!std::get<props::SkipPrefill>(add_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(add_props).get();
  TensorDim out_dim = context.getInputDimensions()[0];
  // W8A8/PERCH: produce FP32 here instead of requantizing back to QINT8.
  // Root cause (confirmed via on-device bisection 2026-08-11): a QINT8
  // tensor produced by AdditionLayer does not correctly bind to its
  // downstream consumer(s) through nntrainer's graph/tensor-pool -- a
  // conv or concat reading this layer's output sees stale/never-written
  // memory, even though THIS layer's own post-write view of hidden_ is
  // correct (verified byte-exact against golden). Conv2d's per-channel
  // GEMM path and concat's FP32-input branch both already auto-quantize a
  // genuine FP32 input on the fly (the same mechanism the network stem
  // uses), so emitting FP32 here sidesteps the bug entirely.
  if (out_dim.getDataType() == nntrainer::Tdatatype::QINT8)
    out_dim.setDataType(nntrainer::Tdatatype::FP32);
  context.setOutputDimensions({out_dim});
}

void AdditionLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  bool any_qint8_input = false;
  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    if (context.getInput(idx).getDataType() == nntrainer::Tdatatype::QINT8) {
      any_qint8_input = true;
      break;
    }
  }

  if (any_qint8_input &&
      hidden_.getDataType() == nntrainer::Tdatatype::FP32) {
    // See finalize(): dequantize QINT8 inputs and sum in FP32 directly,
    // skipping the QINT8 output requantization that doesn't bind correctly.
    static const bool perch_asym_env =
      std::getenv("NNTR_W8A8_PERCH") != nullptr &&
      std::getenv("NNTR_W8A8_SYM") == nullptr;
    constexpr float kActOff = 0.27846455f;
    float *out = hidden_.getData<float>();
    const size_t n = hidden_.size();
    for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
      const Tensor &input_ = context.getInput(idx);
      if (input_.getDataType() == nntrainer::Tdatatype::QINT8) {
        const int8_t *q = input_.getData<int8_t>();
        const float sc = input_.getScale<float>()[0];
        if (!idx) {
          if (perch_asym_env)
            for (size_t i = 0; i < n; ++i)
              out[i] = ((float)q[i] + 128.f) * sc - kActOff;
          else
            for (size_t i = 0; i < n; ++i)
              out[i] = sc * (float)q[i];
        } else {
          if (perch_asym_env)
            for (size_t i = 0; i < n; ++i)
              out[i] += ((float)q[i] + 128.f) * sc - kActOff;
          else
            for (size_t i = 0; i < n; ++i)
              out[i] += sc * (float)q[i];
        }
      } else {
        const float *d = input_.getData<float>();
        if (!idx)
          std::copy(d, d + n, out);
        else
          for (size_t i = 0; i < n; ++i)
            out[i] += d[i];
      }
    }
  } else {
    /** @todo check possibility for in-place of addition layer */
    for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
      const Tensor &input_ = context.getInput(idx);
      if (!idx) {
        hidden_.copy(input_);
      } else {
        hidden_.add_i(input_);
      }
    }
  }

  if (const char *dl = std::getenv("NNTR_CONCAT_DEBUG");
      dl && std::strstr(dl, context.getName().c_str()) &&
      hidden_.getDataType() == nntrainer::Tdatatype::QINT8) {
    const int8_t *q = hidden_.getData<int8_t>();
    const float sc = hidden_.getScale<float>()[0];
    int8_t qmin = 127, qmax = -128;
    for (size_t i = 0; i < hidden_.size(); ++i) {
      if (q[i] < qmin) qmin = q[i];
      if (q[i] > qmax) qmax = q[i];
    }
    std::cerr << "[CONCATDBG] " << context.getName() << " (producer) scale="
              << sc << " qmin=" << (int)qmin << " qmax=" << (int)qmax
              << " n=" << hidden_.size() << " ptr=" << (const void *)q << "\n"
              << std::flush;
  }

  // Dump one named layer's real output (from inside the untruncated
  // production forward pass), for external bisection.
  if (const char *dl = std::getenv("NNTR_DUMP_LAYER");
      dl && context.getName() == dl) {
    const char *dp = std::getenv("NNTR_DUMP_PATH");
    if (dp) {
      std::vector<float> buf(hidden_.size());
      if (hidden_.getDataType() == nntrainer::Tdatatype::FP16) {
#ifdef ENABLE_FP16
        const _FP16 *d = hidden_.getData<_FP16>();
        for (unsigned int i = 0; i < hidden_.size(); ++i)
          buf[i] = (float)d[i];
#endif
      } else if (hidden_.getDataType() == nntrainer::Tdatatype::QINT8) {
        const int8_t *q = hidden_.getData<int8_t>();
        const float sc = hidden_.getScale<float>()[0];
        static const bool perch_asym_env =
          std::getenv("NNTR_W8A8_PERCH") != nullptr &&
          std::getenv("NNTR_W8A8_SYM") == nullptr;
        constexpr float kActOff = 0.27846455f;
        for (unsigned int i = 0; i < hidden_.size(); ++i)
          buf[i] = perch_asym_env ? ((float)q[i] + 128.f) * sc - kActOff
                                   : sc * (float)q[i];
      } else {
        const float *d = hidden_.getData<float>();
        std::copy(d, d + hidden_.size(), buf.begin());
      }
      std::ofstream of(dp, std::ios::binary);
      of.write(reinterpret_cast<const char *>(buf.data()),
               buf.size() * sizeof(float));
    }
  }
}

void AdditionLayer::incremental_forwarding(RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  TensorDim hidden_dim = hidden_.getDim();
  TensorDim hidden_step_dim = hidden_dim;

  hidden_step_dim.batch(1);
  hidden_step_dim.height(to - from);

  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    /** @todo check possibility for in-place of addition layer */
    for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
      const Tensor &input_ = context.getInput(idx);
      TensorDim input_dim = input_.getDim();

      TensorDim input_step_dim = input_dim;
      input_step_dim.batch(1);
      input_step_dim.height(to - from);

      Tensor input_step = input_.getSharedDataTensor(
        input_step_dim, b * input_dim.getFeatureLen(), true);
      if (!idx) {
        hidden_step.copy(input_step);
      } else {
        hidden_step.add_i(input_step);
      }
    }
  }
}

void AdditionLayer::calcDerivative(RunLayerContext &context) {

  for (unsigned int idx = 0; idx < context.getNumInputs(); ++idx) {
    /**
     * TODO: replace this with tensor assignment during optimization.
     * Tensor assignment needs to make sure that the previous connected layers
     * are not inplace
     */
    context.getOutgoingDerivative(idx).copy(
      context.getIncomingDerivative(SINGLE_INOUT_IDX));
  }
}

void AdditionLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, add_props);
  if (!remain_props.empty()) {
    std::string msg = "[AdditionLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

void AdditionLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  for (size_t i = 0; i < context.getNumInputs(); ++i) {
    context.updateInput(i, input_dimensions[0]);
  }
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);
}

} /* namespace nntrainer */
