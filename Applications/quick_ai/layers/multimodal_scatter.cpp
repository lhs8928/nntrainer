// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   multimodal_scatter.cpp
 * @date   14 September 2026
 * @brief  Multimodal scatter/fusion layer for Qwen3-ASR
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 */

#include <cstring>
#include <cmath>
#include <iostream>

#include <multimodal_scatter.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace quick_ai {

MultimodalScatterLayer::MultimodalScatterLayer() :
  audio_token_id(151676) {}

void MultimodalScatterLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 3, std::invalid_argument)
    << "MultimodalScatter layer requires exactly 3 inputs: text_embed, audio_features, input_ids";

  const nntrainer::TensorDim &text_dim = context.getInputDimensions()[0];
  const nntrainer::TensorDim &audio_dim = context.getInputDimensions()[1];
  const nntrainer::TensorDim &ids_dim = context.getInputDimensions()[2];

  NNTR_THROW_IF(text_dim.width() != audio_dim.width(), std::invalid_argument)
    << "Hidden size of text embeddings and audio features must match. Got "
    << text_dim.width() << " vs " << audio_dim.width();

  // Output shape is exactly the same as text embeddings input
  context.setOutputDimensions({text_dim});
}

void MultimodalScatterLayer::forwarding(nntrainer::RunLayerContext &context,
                                        bool training) {
  nntrainer::Tensor &text_embed = context.getInput(0);
  nntrainer::Tensor &audio_features = context.getInput(1);
  nntrainer::Tensor &input_ids = context.getInput(2);
  nntrainer::Tensor &fused_embed = context.getOutput(0);

  if (text_embed.getData() == nullptr || fused_embed.getData() == nullptr || audio_features.getData() == nullptr) {
    std::cerr << "[MultimodalScatter] Error: NULL buffer detected! "
              << "text_embed: " << text_embed.getData()
              << ", fused_embed: " << fused_embed.getData()
              << ", audio_features: " << audio_features.getData() << std::endl;
    return;
  }

  // Copy all text embeddings to output first
  fused_embed.copyData(text_embed);

  unsigned int batch = text_embed.batch();
  unsigned int seq_len = text_embed.height();
  unsigned int hidden_size = text_embed.width();
  unsigned int audio_seq_len = audio_features.height();

  bool is_fp16 = (text_embed.getDataType() == ml::train::TensorDim::DataType::FP16);
  size_t element_bytes = is_fp16 ? sizeof(uint16_t) : sizeof(float);
  size_t vector_bytes = hidden_size * element_bytes;

  for (unsigned int b = 0; b < batch; ++b) {
    unsigned int audio_idx = 0;
    
    // Find matching special token positions in input_ids
    for (unsigned int i = 0; i < seq_len; ++i) {
      float token_val = ((float *)input_ids.getData())[b * seq_len + i];

      if (std::round(token_val) == audio_token_id) {
        if (audio_idx < audio_seq_len) {
          // Copy audio vector to fused embedding vector
          char *dest_ptr = (char *)fused_embed.getData() + 
                           (b * seq_len + i) * vector_bytes;
          char *src_ptr = (char *)audio_features.getData() + 
                          (b * audio_seq_len + audio_idx) * vector_bytes;
          std::memcpy(dest_ptr, src_ptr, vector_bytes);
          audio_idx++;
        }
      }
    }
  }
}

void MultimodalScatterLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                                   unsigned int from, unsigned int to,
                                                   bool training) {
  if (from == 0) {
    forwarding(context, training);
  } else {
    nntrainer::Tensor &text_embed = context.getInput(0);
    nntrainer::Tensor &fused_embed = context.getOutput(0);
    fused_embed.copyData(text_embed);
  }
}

void MultimodalScatterLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  // MultimodalScatter does not support backwarding/training gradients for now
}

void MultimodalScatterLayer::setProperty(const std::vector<std::string> &values) {
  for (const auto &property : values) {
    std::string key;
    std::string value;
    if (nntrainer::getKeyValue(property, key, value) != 0) {
      throw std::invalid_argument("Failed to parse property: " + property);
    }
    if (key == "audio_token_id") {
      audio_token_id = std::stoul(value);
    }
  }
}

} // namespace quick_ai
