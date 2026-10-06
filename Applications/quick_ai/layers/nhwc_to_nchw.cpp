// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nhwc_to_nchw.cpp
 * @date   02 October 2026
 * @brief  Format conversion layer from NHWC to NCHW for Qwen3-ASR Subsampler
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 */

#include "nhwc_to_nchw.h"
#include <app_context.h>
#include <nntrainer_log.h>
#include <nntrainer_log.h>
#include <cstring>

namespace quick_ai {

NHWCToNCHWLayer::NHWCToNCHWLayer() : Layer() {}

void NHWCToNCHWLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "NHWCToNCHWLayer requires exactly 1 input";

  const auto &in_dim = context.getInputDimensions()[0];
  nntrainer::TensorDim out_dim(in_dim.batch(), in_dim.channel(), in_dim.height(), in_dim.width(),
                               nntrainer::TensorDim::Format::NCHW, context.getActivationDataType());
  context.setOutputDimensions({out_dim});
}

void NHWCToNCHWLayer::forwarding(nntrainer::RunLayerContext &context, bool training) {
  const auto &in = context.getInput(0);
  auto &out = context.getOutput(0);

  const size_t H = in.height();
  const size_t W = in.width();
  const size_t C = in.channel();

  if (in.getDataType() == nntrainer::TensorDim::DataType::FP32) {
    const float *s = in.getData<float>();
    float *d = out.getData<float>();

    for (size_t h = 0; h < H; ++h) {
      for (size_t w = 0; w < W; ++w) {
        for (size_t c = 0; c < C; ++c) {
          d[(c * H + h) * W + w] = s[(h * W + w) * C + c];
        }
      }
    }
  } else if (in.getDataType() == nntrainer::TensorDim::DataType::FP16) {
    const _FP16 *s = in.getData<_FP16>();
    _FP16 *d = out.getData<_FP16>();

    for (size_t h = 0; h < H; ++h) {
      for (size_t w = 0; w < W; ++w) {
        for (size_t c = 0; c < C; ++c) {
          d[(c * H + h) * W + w] = s[(h * W + w) * C + c];
        }
      }
    }
  }
}

void NHWCToNCHWLayer::calcDerivative(nntrainer::RunLayerContext &context) {}

void NHWCToNCHWLayer::setProperty(const std::vector<std::string> &values) {
  if (!values.empty()) {
    std::string msg = "[NHWCToNCHWLayer] Unknown properties count " +
                      std::to_string(values.size());
    throw nntrainer::exception::not_supported(msg);
  }
}

} // namespace quick_ai

extern "C" {
__attribute__((constructor)) void init_layer() {
  auto &app_context = nntrainer::AppContext::Global();
  app_context.registerFactory(nntrainer::createLayer<quick_ai::NHWCToNCHWLayer>);
}
void fini_layer() {}
}
