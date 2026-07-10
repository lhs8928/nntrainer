// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   custom_multiply.cpp
 * @date   02 April 2026
 * @brief  Custom multiply layer for CausalLM
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <custom_multiply.h>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <util_func.h>

namespace causallm {

CustomMultiplyLayer::CustomMultiplyLayer() :
  Layer(),
  multiply_props(nntrainer::props::Print(), nntrainer::props::InPlaceProp(),
                 nntrainer::props::InPlaceDirectionProp()),
  support_backwarding(false) {}

void CustomMultiplyLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 2, std::invalid_argument)
    << "CustomMultiplyLayer requires exactly 2 inputs";

  const auto &input_dims = context.getInputDimensions();
  nntrainer::TensorDim out_dim = input_dims[0];
  nntrainer::TensorDim dim1 = input_dims[1];

  // Same broadcasting rule as MultiplyLayer
  for (unsigned int i = 0; i < ml::train::TensorDim::MAXDIM; ++i) {
    if (out_dim[i] != dim1[i]) {
      if (out_dim[i] == 1) {
        out_dim.setTensorDim(i, dim1[i]);
      } else if (dim1[i] != 1) {
        throw std::invalid_argument(
          "CustomMultiplyLayer: incompatible shapes for broadcasting at dim " +
          std::to_string(i) + " (" + std::to_string(out_dim[i]) + " vs " +
          std::to_string(dim1[i]) + ")");
      }
    }
  }

  context.setOutputDimensions({out_dim});
}

void CustomMultiplyLayer::forwarding(nntrainer::RunLayerContext &context,
                                     bool training) {
  nntrainer::Tensor &out = context.getOutput(OUT_IDX);
  const nntrainer::Tensor &in0 = context.getInput(INPUT_IDX_0);
  const nntrainer::Tensor &in1 = context.getInput(INPUT_IDX_1);

  if (!context.getInPlace()) {
    out.copy(in0);
  }
  out.multiply_i(in1);
}

void CustomMultiplyLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {

  NNTR_THROW_IF(to <= from, std::invalid_argument)
    << "CustomMultiplyLayer::incremental_forwarding requires to > from";

  nntrainer::Tensor &out = context.getOutput(OUT_IDX);
  const nntrainer::Tensor &in0 = context.getInput(INPUT_IDX_0);
  nntrainer::Tensor &in1 = context.getInput(INPUT_IDX_1);

  nntrainer::TensorDim out_dim = out.getDim();
  nntrainer::TensorDim in0_dim = in0.getDim();
  nntrainer::TensorDim in1_dim = in1.getDim();

  const unsigned int batch_size = out.batch();
  const unsigned int step_height = to - from;
  const bool in1_broadcast = (in1_dim.height() == 1);

  // Pre-compute dimensions (same pattern as addition_layer)
  const size_t out_feature_len = out_dim.getFeatureLen();
  const size_t in0_feature_len = in0_dim.getFeatureLen();
  const size_t in1_feature_len = in1_dim.getFeatureLen();

  // Output step dimension
  nntrainer::TensorDim out_step_dim = out_dim;
  out_step_dim.batch(1);
  out_step_dim.height(step_height);

  // Input step dimensions
  nntrainer::TensorDim in0_step_dim = in0_dim;
  in0_step_dim.batch(1);
  in0_step_dim.height(step_height);

  nntrainer::TensorDim in1_step_dim = in1_dim;
  in1_step_dim.batch(1);
  in1_step_dim.height(in1_broadcast ? 1 : step_height);

  // Batch loop - same pattern as addition_layer.cpp (no 'from' in offset)
  for (unsigned int b = 0; b < batch_size; ++b) {
    // Follow addition_layer pattern: b * getFeatureLen()
    const size_t out_offset = b * out_feature_len;
    const size_t in0_offset = b * in0_feature_len;
    const size_t in1_offset = b * in1_feature_len;

    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, out_offset, true);
    nntrainer::Tensor in0_step =
      in0.getSharedDataTensor(in0_step_dim, in0_offset, true);
    nntrainer::Tensor in1_step =
      in1.getSharedDataTensor(in1_step_dim, in1_offset, true);

    if (!context.getInPlace()) {
      out_step.copy(in0_step);
    }
    out_step.multiply_i(in1_step);
  }
}

void CustomMultiplyLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // For element-wise multiply: d(in0) = d(out) * in1, d(in1) = d(out) * in0
  context.getOutgoingDerivative(INPUT_IDX_0)
    .copy(context.getIncomingDerivative(OUT_IDX).multiply(
      context.getInput(INPUT_IDX_1)));

  context.getOutgoingDerivative(INPUT_IDX_1)
    .copy(context.getIncomingDerivative(OUT_IDX).multiply(
      context.getInput(INPUT_IDX_0)));
}

void CustomMultiplyLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, multiply_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[CustomMultiplyLayer] Unknown Layer Properties count "
    << std::to_string(values.size());
}

#ifdef PLUGGABLE

nntrainer::Layer *create_custom_multiply_layer() {
  auto layer = new CustomMultiplyLayer();
  return layer;
}

void destroy_custom_multiply_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_custom_multiply_layer, destroy_custom_multiply_layer};
}

#endif

} // namespace causallm
