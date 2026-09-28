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
  }

  virtual ~Qwen3ASRTransformer() = default;

  std::pair<Tensor, Tensor> constructModel() override;

protected:
  Tensor createAudioEncoder(Tensor audio_input);
  Tensor createAudioAttentionBlock(const int layer_id, Tensor input, unsigned int max_timestep);

  unsigned int audio_seq_len;
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
