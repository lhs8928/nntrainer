// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   weight_layer.cpp
 * @date   2 August 2024
 * @brief  This is a layer that simply stores a weight tensor without any
 * operation.
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <common_properties.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>
#include <weight_layer.h>

#include <iostream>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;
WeightLayer::WeightLayer() : LayerImpl(), weight_props({}, {}, {}) {}

void WeightLayer::finalize(InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);

  const auto &weight_dim = std::get<props::TensorDimension>(weight_props).get();
  const auto &weight_dtype = std::get<props::TensorDataType>(weight_props);
  const auto &weight_name = std::get<props::WeightName>(weight_props);

  std::vector<TensorDim> output_dims(1);

  if (context.getNumInputs() > 0) {
    output_dims[SINGLE_INOUT_IDX] = context.getInputDimensions()[0];
  } else {
    output_dims[SINGLE_INOUT_IDX] = weight_dim;
  }
  output_dims[SINGLE_INOUT_IDX].setTensorType(
    {context.getFormat(), weight_dtype});

  context.setOutputDimensions(output_dims);

  TensorDim typed_weight_dim = weight_dim;
  typed_weight_dim.setTensorType({context.getFormat(), weight_dtype});

  weight_idx = context.requestWeight(
    typed_weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, weight_name, true);
}

void WeightLayer::exportTo(Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(weight_props, method, this);
}

void WeightLayer::setProperty(const std::vector<std::string> &values) {
  std::vector<std::string> mapped_values = values;
  for (auto &v : mapped_values) {
    if (v.rfind("weight_dtype=", 0) == 0) {
      v = "tensor_dtype=" + v.substr(13);
    }
  }
  auto remain_props = loadProperties(mapped_values, weight_props);
  LayerImpl::setProperty(remain_props);
}

void WeightLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &weight = context.getWeight(weight_idx);
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  if (output.getDim() == weight.getDim()) {
    output.copy(weight);
  } else {
    size_t copy_bytes = output.bytes();
    NNTR_THROW_IF(copy_bytes > weight.bytes(), std::invalid_argument)
      << "Output dimension exceeds precomputed weight table size in WeightLayer";
    std::memcpy(output.getData(), weight.getData(), copy_bytes);
  }
}

void WeightLayer::calcDerivative(RunLayerContext &context) {
  throw exception::not_supported(
    "calcDerivative for weight layer is not supported");
}

void WeightLayer::calcGradient(RunLayerContext &context) {
  Tensor &djdw = context.getWeightGrad(weight_idx);
  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  djdw.copy(derivative_);
}

} /* namespace nntrainer */
