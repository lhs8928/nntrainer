// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen3_asr_causallm.h
 * @brief  Qwen3 ASR multimodal conditional generation model implementation.
 * @date   14 September 2026
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 */

#ifndef __QWEN3_ASR_CAUSAL_LM_H__
#define __QWEN3_ASR_CAUSAL_LM_H__

#include <causal_lm.h>
#include "qwen3_asr_subsampler.h"

namespace quick_ai {

/**
 * @brief Qwen3ASRTransformer class
 */
class Qwen3ASRTransformer : virtual public Transformer {
public:
  static constexpr const char *architectures = "Qwen3ASRTransformer";

  Qwen3ASRTransformer(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg) {
    audio_seq_len = 1500;
    if (nntr_cfg.contains("audio_seq_len")) {
      audio_seq_len = nntr_cfg["audio_seq_len"].get<unsigned int>();
    }

    // Dynamic data type configuration with fallbacks
    audio_tower_weight_dtype = nntr_cfg.value("audio_tower_weight_dtype", FC_LAYER_DTYPE);

    if (nntr_cfg.contains("subsampler") && nntr_cfg["subsampler"].is_object()) {
      const auto &sub_cfg = nntr_cfg["subsampler"];
      conv_layer_dtype = sub_cfg.value("conv_layer_dtype", "FP32");
      subsampler_model_tensor_type = sub_cfg.value("model_tensor_type", "FP32-FP32");
      subsampler_fc_layer_dtype = sub_cfg.value("fc_layer_dtype", "FP32");
      subsampler_model_file_name = sub_cfg.value("model_file_name", "");
    } else {
      conv_layer_dtype = nntr_cfg.value("conv_layer_dtype", "FP32");
      subsampler_model_tensor_type = nntr_cfg.value("subsampler_model_tensor_type", "FP32-FP32");
      subsampler_fc_layer_dtype = nntr_cfg.value("subsampler_fc_layer_dtype", "FP32");
      subsampler_model_file_name = nntr_cfg.value("subsampler_model_file_name", "");
    }

    size_t dash = MODEL_TENSOR_TYPE.find('-');
    act_dtype = (dash != std::string::npos) ? MODEL_TENSOR_TYPE.substr(dash + 1) : "FP16";
  }

  virtual ~Qwen3ASRTransformer() = default;

  std::pair<Tensor, Tensor> constructModel() override;

protected:
  Tensor createAudioEncoder(Tensor audio_input);
  Tensor createAudioAttentionBlock(const int layer_id, Tensor input, unsigned int max_timestep);

  unsigned int audio_seq_len;
  std::string audio_tower_weight_dtype;
  std::string conv_layer_dtype;
  std::string subsampler_fc_layer_dtype;
  std::string subsampler_model_tensor_type;
  std::string subsampler_model_file_name;
  std::string act_dtype;

public:
  Qwen3ASRSubsampler* getSubsampler() { return &subsampler; }

protected:
  Qwen3ASRSubsampler subsampler;
};

/**
 * @brief Qwen3ASRCausalLM class
 */
class Qwen3ASRCausalLM : public CausalLM, public Qwen3ASRTransformer {
public:
  static constexpr const char *architectures = "Qwen3ASRForConditionalGeneration";

  Qwen3ASRCausalLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    CausalLM(cfg, generation_cfg, nntr_cfg),
    Qwen3ASRTransformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Qwen3ASRCausalLM() = default;

  std::pair<Tensor, Tensor> constructModel() override;

  void initialize() override;
  void load_weight(const std::string &path) override;

  void registerCustomLayers() override;

  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "",
           bool log_output = true) override;

  void setAudioPath(const std::string &path) { audio_path = path; }

  ml::train::Model* getModel() { return model.get(); }


private:
  std::string audio_path;

};

} // namespace quick_ai

#endif /* __QWEN3_ASR_CAUSAL_LM_H__ */
