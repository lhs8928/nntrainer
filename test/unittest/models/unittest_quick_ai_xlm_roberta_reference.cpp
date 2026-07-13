// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_quick_ai_xlm_roberta_reference.cpp
 * @date   18 June 2026
 * @brief  Differential test for the tiny XLM-RoBERTa encoder.
 *
 * Compares nntrainer's XLM-RoBERTa encoder output (raw, unpooled last hidden
 * state) against a golden fixture generated from HuggingFace XLMRobertaModel.
 * The reference is the full [seq_len, hidden] hidden state flattened row-major,
 * since the nntrainer path performs no pooling.
 *
 * XLMRobertaForMaskedLM sanitizes its config in the constructor, so a thin
 * subclass owns the (virtual) Transformer mem-initializer.
 *
 * Guarded for Linux/non-Android only, matching the factory registration in
 * Applications/quick_ai/main.cpp. The test skips when the fixture is absent.
 *
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#if !defined(_WIN32) && !defined(__ANDROID__)

#include <quick_ai_test_utils.h>

#include <xlm_roberta.h>

#include <memory>

namespace {

/**
 * @brief Tiny XLM-RoBERTa differential adapter
 *
 * XLMRobertaForMaskedLM sanitizes its config before initializing the virtual
 * Transformer base, so this thin subclass owns that mem-initializer.
 */
class TinyXLMRobertaRefAdapter final
  : public quick_ai_test::EmbeddingTestAdapter<
      quick_ai::XLMRobertaForMaskedLM> {
public:
  TinyXLMRobertaRefAdapter(quick_ai::json &cfg, quick_ai::json &generation_cfg,
                           quick_ai::json &nntr_cfg) :
    quick_ai::Transformer(sanitizeConfig(cfg), generation_cfg, nntr_cfg,
                          quick_ai::ModelType::EMBEDDING),
    quick_ai_test::EmbeddingTestAdapter<quick_ai::XLMRobertaForMaskedLM>(
      cfg, generation_cfg, nntr_cfg) {}
};

/**
 * @brief Differential model descriptor for the tiny XLM-RoBERTa fixture
 */
quick_ai_test::DifferentialModel xlmRobertaModel() {
  return {
    "xlm_roberta_tiny",
    [](quick_ai::json &cfg, quick_ai::json &gen_cfg, quick_ai::json &nntr_cfg) {
      return std::make_unique<TinyXLMRobertaRefAdapter>(cfg, gen_cfg, nntr_cfg);
    },
  };
}

/**
 * @brief FP32 raw encoder output matches the HF XLMRobertaModel reference
 */
TEST(TinyXLMRobertaDifferentialTest, FP32MatchesHFReference) {
  quick_ai_test::runFp32EmbeddingDifferentialChecks(xlmRobertaModel());
}

/**
 * @brief Q4_0 encoder output is close to the HF FP32 reference
 *        (skips automatically if nntr_quantize does not support
 * XLMRobertaForMaskedLM)
 */
TEST(TinyXLMRobertaDifferentialTest, Q40CloseToFP32Reference) {
  quick_ai_test::runQ40EmbeddingDifferentialChecks(xlmRobertaModel());
}

} // namespace

#endif // !_WIN32 && !__ANDROID__
