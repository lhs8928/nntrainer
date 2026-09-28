// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen3_asr_subsampler.h
 * @date   14 September 2026
 * @brief  Qwen3-ASR Audio Subsampler independent sub-model
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 */

#ifndef __QWEN3_ASR_SUBSAMPLER_H__
#define __QWEN3_ASR_SUBSAMPLER_H__

#include <model.h>
#include <tensor_api.h>
#include <string>
#include <memory>
#include <vector>

namespace quick_ai {

/**
 * @class   Qwen3ASRSubsampler
 * @brief   Independent NNTrainer Model for Qwen3-ASR 100-frame audio chunk subsampling.
 *          Maps [1, 1, 128, 100] (FP32) to [1, 1, 13, 1024] (FP16) using standard native layers:
 *          Conv2D -> GELU -> Conv2D -> GELU -> Conv2D -> GELU -> Permute -> Cast -> Reshape -> FullyConnected -> PosEmbed + Add.
 */
class Qwen3ASRSubsampler {
public:
  Qwen3ASRSubsampler() = default;
  ~Qwen3ASRSubsampler() = default;

  void initialize(const std::string &model_tensor_type = "FP16-FP16");
  void constructModel(const std::string &conv_dtype = "FP32",
                      const std::string &subsampler_tensor_type = "FP32-FP32");
  void load_weight(const std::string &weight_path);
  std::vector<float *> inference(float *input_data);

  ml::train::Model *getModel() { return model.get(); }

private:
  std::unique_ptr<ml::train::Model> model;
  ml::train::Tensor input_tensor;
  ml::train::Tensor output_tensor;
  bool model_constructed = false;
};

} // namespace quick_ai

#endif // __QWEN3_ASR_SUBSAMPLER_H__
