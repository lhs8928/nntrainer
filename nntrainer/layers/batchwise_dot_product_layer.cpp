// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   batchwise_dot_product_layer.cpp
 * @date   29 November 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Batchwise dot product Layer Class for Neural Network
 *
 */

#include <cmath>

#include <batchwise_dot_product_layer.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

BatchwiseDotproductLayer::BatchwiseDotproductLayer() :
  batchwise_dot_product_props(props::ScaledDotProduct(),
                              props::TransposeQuery(), props::TransposeKey()) {}

BatchwiseDotproductLayer::~BatchwiseDotproductLayer() {}

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum AttentionParams { query, key };

void BatchwiseDotproductLayer::finalizeCommon(InitLayerContext &context) {
  if (context.getNumInputs() != 2)
    throw std::runtime_error("Batchwise Dot product layer needs 2 inputs.");
}

void BatchwiseDotproductLayer::finalize(InitLayerContext &context) {
  finalizeCommon(context);

  sm.setActiFunc(ActivationType::ACT_SOFTMAX);

  auto const &all_dims = context.getInputDimensions();
  auto const &query_dim = all_dims[AttentionParams::query];
  auto const &key_dim = all_dims[AttentionParams::key];

  const bool transpose_query =
    std::get<props::TransposeQuery>(batchwise_dot_product_props).get();
  const bool transpose_key =
    std::get<props::TransposeKey>(batchwise_dot_product_props).get();

  TensorDim output_dim(query_dim.batch(), 1,
                       transpose_query ? query_dim.width() : query_dim.height(),
                       transpose_key ? key_dim.height() : key_dim.width());

  context.setOutputDimensions({output_dim});
}

void BatchwiseDotproductLayer::forwarding(RunLayerContext &context,
                                          bool training) {
  Tensor &query = context.getInput(AttentionParams::query);
  Tensor &key = context.getInput(AttentionParams::key);

  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  const bool transpose_query =
    std::get<props::TransposeQuery>(batchwise_dot_product_props).get();
  const bool transpose_key =
    std::get<props::TransposeKey>(batchwise_dot_product_props).get();

  query.dotBatched(key, output, transpose_query, transpose_key); /** dot 1 */

  if (std::get<props::ScaledDotProduct>(batchwise_dot_product_props).get()) {
    output.multiply_i(1 / sqrt((float)key.getDim().width()));
  }
}

void BatchwiseDotproductLayer::calcDerivative(RunLayerContext &context) {
  const bool transpose_query =
    std::get<props::TransposeQuery>(batchwise_dot_product_props).get();
  const bool transpose_key =
    std::get<props::TransposeKey>(batchwise_dot_product_props).get();

  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);

  Tensor &query = context.getInput(AttentionParams::query);
  Tensor &key = context.getInput(AttentionParams::key);

  Tensor &dquery = context.getOutgoingDerivative(AttentionParams::query);
  Tensor &dkey = context.getOutgoingDerivative(AttentionParams::key);

  Tensor derivative_ = derivative;

  if (std::get<props::ScaledDotProduct>(batchwise_dot_product_props).get()) {
    derivative_.multiply_i(1 / sqrt((float)key.getDim().width()));
  }

  /** derivative for dot 1 */
  dquery.dot_batched_deriv_wrt_1(key, derivative_, transpose_query,
                                 transpose_key);
  query.dot_batched_deriv_wrt_2(dkey, derivative_, transpose_query,
                                transpose_key);
}

void BatchwiseDotproductLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, batchwise_dot_product_props);
  if (!remain_props.empty()) {
    std::string msg =
      "[BatchwiseDotproductLayer] Unknown Layer Properties count " +
      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */
