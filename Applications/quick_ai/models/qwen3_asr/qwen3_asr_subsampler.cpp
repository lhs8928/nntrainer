// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen3_asr_subsampler.cpp
 * @brief  Qwen3-ASR Audio Subsampler independent sub-model implementation
 */

#include "qwen3_asr_subsampler.h"
#include <app_context.h>
#include <engine.h>
#include <layer.h>
#include <util_func.h>
#include <stdexcept>
#include <iostream>

namespace quick_ai {

void Qwen3ASRSubsampler::initialize(const std::string &model_tensor_type) {
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model->setProperty({
    nntrainer::withKey("batch_size", "1"),
    nntrainer::withKey("epochs", "1"),
    nntrainer::withKey("model_tensor_type", "FP32-FP32")
  });

  // Input: [1, 1, 128, 100] in FP32
  ml::train::Tensor x({1, 1, 128, 100}, "subsampler_input");

  // 1. conv2d1: stride 2, padding 1, filters 480, kernel size 3
  ml::train::LayerHandle conv1(ml::train::createLayer("conv2d", {
    nntrainer::withKey("name", "audio_tower_conv1"),
    nntrainer::withKey("filters", "480"),
    nntrainer::withKey("kernel_size", "3,3"),
    nntrainer::withKey("stride", "2,2"),
    nntrainer::withKey("padding", "1,1"),
    nntrainer::withKey("disable_bias", "false"),
    nntrainer::withKey("weight_dtype", "FP32")
  }));
  ml::train::Tensor h = conv1(x);

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
    nntrainer::withKey("weight_dtype", "FP32")
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
    nntrainer::withKey("weight_dtype", "FP32")
  }));
  h = conv3(h);

  ml::train::LayerHandle gelu3(ml::train::createLayer("activation", {
    nntrainer::withKey("name", "audio_tower_gelu3"),
    nntrainer::withKey("activation", "gelu")
  }));
  h = gelu3(h);

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

  // 6. conv_out: nn.Linear(7680, 1024, bias=False) in FP32
  ml::train::LayerHandle conv_out(ml::train::createLayer("fully_connected", {
    nntrainer::withKey("name", "audio_tower_conv_out"),
    nntrainer::withKey("unit", "1024"),
    nntrainer::withKey("disable_bias", "true"),
    nntrainer::withKey("weight_dtype", "FP32")
  }));
  h = conv_out(h);

  // 7. Positional Embedding: 13 rows [1, 1, 13, 1024] in FP32
  ml::train::LayerHandle pos_embed(ml::train::createLayer("weight", {
    nntrainer::withKey("name", "audio_tower_pos_embed"),
    nntrainer::withKey("dim", "1:1:13:1024"),
    nntrainer::withKey("weight_name", "weights"),
    nntrainer::withKey("weight_dtype", "FP32"),
    nntrainer::withKey("tensor_dtype", "FP32")
  }));
  ml::train::Tensor pos = pos_embed(h);

  ml::train::LayerHandle pos_add(ml::train::createLayer("addition", {
    nntrainer::withKey("name", "audio_tower_pos_add")
  }));
  h = pos_add({h, pos});

  if (model->compile(x, h, ml::train::ExecutionMode::INFERENCE)) {
    throw std::runtime_error("Qwen3ASRSubsampler compilation failed!");
  }
  model->initialize();
}

void Qwen3ASRSubsampler::load_weight(const std::string &path) {
  auto dot = path.find_last_of('.');
  auto fmt = (dot != std::string::npos && path.substr(dot + 1) == "safetensors")
    ? ml::train::ModelFormat::MODEL_FORMAT_SAFETENSORS
    : ml::train::ModelFormat::MODEL_FORMAT_BIN;
  model->load(path, fmt);
}

std::vector<float *> Qwen3ASRSubsampler::inference(float *input_data) {
  return model->inference(1, {input_data});
}

} // namespace quick_ai
