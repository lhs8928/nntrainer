// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen3_asr_subsampler.cpp
 * @date   14 September 2026
 * @brief  Qwen3-ASR Audio Subsampler independent sub-model implementation
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 */

#include "qwen3_asr_subsampler.h"
#include <app_context.h>
#include <engine.h>
#include <layer.h>
#include <util_func.h>
#include <stdexcept>
#include <iostream>

namespace quick_ai {

void Qwen3ASRSubsampler::constructModel(const std::string &conv_dtype,
                                        const std::string &subsampler_tensor_type) {
  if (model_constructed) return;

  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model->setProperty({
    nntrainer::withKey("batch_size", "1"),
    nntrainer::withKey("epochs", "1"),
    nntrainer::withKey("model_tensor_type", subsampler_tensor_type),
    nntrainer::withKey("tensor_format", "NHWC")
  });

  // Extract weight and act dtype from subsampler_tensor_type
  size_t dash = subsampler_tensor_type.find('-');
  std::string subsampler_weight_dtype = (dash != std::string::npos) ? subsampler_tensor_type.substr(0, dash) : subsampler_tensor_type;
  std::string subsampler_act_dtype = (dash != std::string::npos) ? subsampler_tensor_type.substr(dash + 1) : subsampler_tensor_type;

  // Input: [1, 1, 128, 100] in NHWC
  ml::train::TensorDim::DataType in_dtype = (subsampler_act_dtype == "FP16")
    ? ml::train::TensorDim::DataType::FP16
    : ml::train::TensorDim::DataType::FP32;
  input_tensor = ml::train::Tensor(
    nntrainer::TensorDim(1, 1, 128, 100, nntrainer::TensorDim::Format::NHWC, in_dtype),
    "subsampler_input");

  // 1. conv2d1: stride 2, padding 1, filters 480, kernel size 3
  // Stem layer (in_ch = 1, CRS = 9) is not divisible by 32, so it stays FP32
  std::string conv1_dtype = (conv_dtype == "Q8_0" || conv_dtype == "Q4_0" || conv_dtype == "QINT8") ? "FP32" : conv_dtype;
  ml::train::LayerHandle conv1(ml::train::createLayer("conv2d", {
    nntrainer::withKey("name", "audio_tower_conv1"),
    nntrainer::withKey("filters", "480"),
    nntrainer::withKey("kernel_size", "3,3"),
    nntrainer::withKey("stride", "2,2"),
    nntrainer::withKey("padding", "1,1"),
    nntrainer::withKey("disable_bias", "false"),
    nntrainer::withKey("weight_dtype", conv1_dtype)
  }));
  ml::train::Tensor h = conv1(input_tensor);

  ml::train::LayerHandle gelu1(ml::train::createLayer("activation", {
    nntrainer::withKey("name", "audio_tower_gelu1"),
    nntrainer::withKey("activation", "gelu")
  }));
  h = gelu1(h);

  // 2. conv2d2: stride 2, padding 1, filters 480, kernel size 3
  ml::train::LayerHandle conv2(ml::train::createLayer("conv2d", {
    nntrainer::withKey("name", "audio_tower_conv2"),
    nntrainer::withKey("filters", "480"),
    nntrainer::withKey("kernel_size", "3,3"),
    nntrainer::withKey("stride", "2,2"),
    nntrainer::withKey("padding", "1,1"),
    nntrainer::withKey("disable_bias", "false"),
    nntrainer::withKey("weight_dtype", conv_dtype)
  }));
  h = conv2(h);

  ml::train::LayerHandle gelu2(ml::train::createLayer("activation", {
    nntrainer::withKey("name", "audio_tower_gelu2"),
    nntrainer::withKey("activation", "gelu")
  }));
  h = gelu2(h);

  // 3. conv2d3: stride 2, padding 1, filters 480, kernel size 3
  ml::train::LayerHandle conv3(ml::train::createLayer("conv2d", {
    nntrainer::withKey("name", "audio_tower_conv3"),
    nntrainer::withKey("filters", "480"),
    nntrainer::withKey("kernel_size", "3,3"),
    nntrainer::withKey("stride", "2,2"),
    nntrainer::withKey("padding", "1,1"),
    nntrainer::withKey("disable_bias", "false"),
    nntrainer::withKey("weight_dtype", conv_dtype)
  }));
  h = conv3(h);

  ml::train::LayerHandle gelu3(ml::train::createLayer("activation", {
    nntrainer::withKey("name", "audio_tower_gelu3"),
    nntrainer::withKey("activation", "gelu")
  }));
  h = gelu3(h);

  // Bridge format from NHWC [16, 13, 480] to NCHW [480, 16, 13] for post-conv Qwen3 decoder alignment
  ml::train::LayerHandle bridge(ml::train::createLayer("nhwc_to_nchw", {
    nntrainer::withKey("name", "audio_tower_nhwc_to_nchw")
  }));
  h = bridge(h);

  // 4. Permute from [480, 16, 13] to [13, 480, 16] in FP32
  ml::train::LayerHandle permute(ml::train::createLayer("permute", {
    nntrainer::withKey("name", "audio_tower_permute"),
    nntrainer::withKey("direction", "3,1,2")
  }));
  h = permute(h);

  // 5. Reshape to [1, 1, 13, 7680] in FP32
  ml::train::LayerHandle reshape(ml::train::createLayer("reshape", {
    nntrainer::withKey("name", "audio_tower_reshape"),
    nntrainer::withKey("target_shape", "1:1:13:7680")
  }));
  h = reshape(h);

  // 6. conv_out: nn.Linear(7680, 1024, bias=False)
  ml::train::LayerHandle conv_out(ml::train::createLayer("fully_connected", {
    nntrainer::withKey("name", "audio_tower_conv_out"),
    nntrainer::withKey("unit", "1024"),
    nntrainer::withKey("disable_bias", "true"),
    nntrainer::withKey("weight_dtype", subsampler_weight_dtype)
  }));
  h = conv_out(h);

  // 7. Positional Embedding: 13 rows [1, 1, 13, 1024]
  ml::train::LayerHandle pos_embed(ml::train::createLayer("weight", {
    nntrainer::withKey("name", "audio_tower_pos_embed"),
    nntrainer::withKey("dim", "1:1:13:1024"),
    nntrainer::withKey("weight_name", "weights"),
    nntrainer::withKey("weight_dtype", subsampler_weight_dtype),
    nntrainer::withKey("tensor_dtype", subsampler_act_dtype)
  }));
  ml::train::Tensor pos = pos_embed(h);

  ml::train::LayerHandle pos_add(ml::train::createLayer("addition", {
    nntrainer::withKey("name", "audio_tower_pos_add")
  }));
  
  output_tensor = pos_add({h, pos});
  model_constructed = true;
}

void Qwen3ASRSubsampler::initialize(const std::string &model_tensor_type) {
  if (!model_constructed) {
    constructModel();
  }

  if (model->compile(input_tensor, output_tensor, ml::train::ExecutionMode::INFERENCE)) {
    throw std::runtime_error("Qwen3ASRSubsampler compilation failed!");
  }
  model->initialize();
}

void Qwen3ASRSubsampler::load_weight(const std::string &path) {
  auto dot = path.find_last_of('.');
  auto fmt = (dot != std::string::npos && path.substr(dot + 1) == "safetensors")
    ? ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS
    : ml::train::ModelFormat::MODEL_FORMAT_BIN;
  std::cout << "[Subsampler] Loading weights from: " << path << std::endl;
  model->load(path, fmt);
}

std::vector<float *> Qwen3ASRSubsampler::inference(float *input_data) {
  return model->inference(1, {input_data});
}

} // namespace quick_ai
