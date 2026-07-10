// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   xlm_roberta.h
 * @brief  XLM-RoBERTa encoder-only embedding model.
 * @date   18 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @note   This file constructs a class for XLM-RoBERTa encoder-only embedding
 *         model built on top of the quick_ai BertTransformer base class.
 * @note   Please refer to the following code :
 *  https://github.com/huggingface/transformers/blob/v4.52.3/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py
 */

#ifndef __XLM_ROBERTA_H__
#define __XLM_ROBERTA_H__

#include "../bert/bert_transformer.h"

namespace quick_ai {

/**
 * @brief XLMRobertaForMaskedLM class
 * @note  Concrete runnable model for XLM-RoBERTa (XLMRobertaForMaskedLM).
 *        It inherits BertTransformer and provides the encode / run
 *        methods that feed three inputs (input_ids, position_ids,
 *        token_type_ids) into the underlying nntrainer model.
 *
 *        XLM-RoBERTa differs from BERT in:
 *          - type_vocab_size = 1 (single token type, all zeros)
 *          - layer_norm_eps = 1e-5 (vs BERT's 1e-12)
 *          - architectures name: XLMRobertaForMaskedLM
 */
class XLMRobertaForMaskedLM : public BertTransformer {

public:
  static constexpr const char *architectures = "XLMRobertaForMaskedLM";

  XLMRobertaForMaskedLM(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(sanitizeConfig(cfg), generation_cfg, nntr_cfg,
                ModelType::EMBEDDING),
    BertTransformer(cfg, generation_cfg, nntr_cfg) {
    // BertTransformer ctor calls setupParameters() via BertTransformer's vtable
    // (virtual dispatch in constructors uses the currently-constructing class),
    // so the override below is never reached from BertTransformer's
    // constructor. Override TYPE_VOCAB_SIZE here, before initialize() /
    // constructBertGraph().
    TYPE_VOCAB_SIZE = cfg.value("type_vocab_size", 1u);
  }

  virtual ~XLMRobertaForMaskedLM() = default;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override {
    BertTransformer::setupParameters(cfg, generation_cfg, nntr_cfg);
    // XLM-RoBERTa uses type_vocab_size=1 (only token type 0 is used)
    TYPE_VOCAB_SIZE = cfg.value("type_vocab_size", 1u);
  }

  std::pair<Tensor, Tensor> constructModel() override {
    return BertTransformer::constructModel();
  }

  void registerCustomLayers() override {
    BertTransformer::registerCustomLayers();
  }

  /**
   * @brief Encode the prompt and return the raw encoder hidden state
   */
  std::vector<float *> encode(const WSTR prompt, const WSTR system_prompt = "",
                              const WSTR tail_prompt = "");
};

} // namespace quick_ai

#endif /* __XLM_ROBERTA_H__ */
