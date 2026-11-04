// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   position_embedding.cpp
 * @date   04 November 2026
 * @brief  Adds a table of position embeddings by the rows' absolute position
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <stdexcept>

#include <nntrainer_error.h>
#include <position_embedding.h>

namespace quick_ai {

void PositionEmbeddingLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "position_embedding takes one input";
  NNTR_THROW_IF(max_position == 0, std::invalid_argument)
    << "position_embedding needs max_position";

  const auto &in_dim = context.getInputDimensions()[0];
  context.setOutputDimensions({in_dim});

  nntrainer::TensorDim table_dim(
    1, 1, max_position, in_dim.width(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32));
  weight_idx = context.requestWeight(
    table_dim, nntrainer::props::InitializerInfo::Enum::ZEROS,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "weight", false);
}

void PositionEmbeddingLayer::forwarding(nntrainer::RunLayerContext &context,
                                        bool training) {
  const unsigned int rows = context.getInput(0).getDim().height();
  incremental_forwarding(context, 0, rows, training);
}

void PositionEmbeddingLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &in = context.getInput(0);
  nntrainer::Tensor &out = context.getOutput(0);
  const nntrainer::Tensor &table = context.getWeight(weight_idx);

  const auto &dim = in.getDim();
  const size_t width = dim.width();
  const size_t steps = std::min<size_t>(to - from, dim.height());
  NNTR_THROW_IF(from + steps > max_position, std::out_of_range)
    << "position_embedding: position " << from + steps << " beyond "
    << max_position;

  const float *pos = table.getData<float>() + static_cast<size_t>(from) * width;
  for (unsigned int b = 0; b < dim.batch(); ++b) {
    const size_t off = b * dim.getFeatureLen();
    if (in.getDataType() == nntrainer::TensorDim::DataType::FP32) {
      const float *x = in.getData<float>() + off;
      float *y = out.getData<float>() + off;
      for (size_t i = 0; i < steps * width; ++i)
        y[i] = x[i] + pos[i];
#ifdef ENABLE_FP16
    } else if (in.getDataType() == nntrainer::TensorDim::DataType::FP16) {
      const _FP16 *x = in.getData<_FP16>() + off;
      _FP16 *y = out.getData<_FP16>() + off;
      for (size_t i = 0; i < steps * width; ++i)
        y[i] = static_cast<_FP16>(static_cast<float>(x[i]) + pos[i]);
#endif
    } else {
      throw std::invalid_argument("position_embedding: unsupported data type");
    }
  }
}

void PositionEmbeddingLayer::setProperty(
  const std::vector<std::string> &values) {
  for (const auto &property : values) {
    std::string key, value;
    if (nntrainer::getKeyValue(property, key, value) != 0)
      throw std::invalid_argument("position_embedding: bad property " +
                                  property);
    if (key == "max_position")
      max_position = static_cast<unsigned int>(std::stoul(value));
  }
}

#ifdef PLUGGABLE

nntrainer::Layer *create_position_embedding_layer() {
  return new PositionEmbeddingLayer();
}

void destroy_position_embedding_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{
  create_position_embedding_layer, destroy_position_embedding_layer};
}

#endif

} // namespace quick_ai
