// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_quick_ai_deberta_v2_reference.cpp
 * @date   16 June 2026
 * @brief  Differential test for the tiny DebertaV2 embedding model.
 *
 * Compares nntrainer's DebertaV2 output (DeBERTa V2 encoder + mean pooling
 * + L2 normalize) against a golden fixture from HuggingFace transformers
 * (generate_deberta_v2_reference.py). DebertaV2 sanitizes its configs in the
 * constructor, so a thin subclass initializes the (virtual) Transformer base
 * with the processed configs. The test skips when the fixture is absent.
 *
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <quick_ai_test_utils.h>

#include <gtest/gtest.h>

#include <deberta_v2.h>

#include <memory>

namespace {

/**
 * @brief Tiny DebertaV2 differential adapter
 *
 * DebertaV2 sanitizes its configs before initializing the virtual Transformer
 * base, so this thin subclass owns that mem-initializer (the one in
 * EmbeddingTestAdapter is ignored per virtual-base rules).
 */
class DebertaV2RefAdapter final
  : public quick_ai_test::EmbeddingTestAdapter<quick_ai::DebertaV2> {
public:
  DebertaV2RefAdapter(quick_ai::json &cfg, quick_ai::json &generation_cfg,
                      quick_ai::json &nntr_cfg) :
    quick_ai::Transformer(quick_ai::DebertaV2::sanitizeConfig(cfg),
                          generation_cfg, nntr_cfg,
                          quick_ai::ModelType::EMBEDDING),
    quick_ai_test::EmbeddingTestAdapter<quick_ai::DebertaV2>(
      cfg, generation_cfg, nntr_cfg) {}
};

/**
 * @brief Differential model descriptor for the tiny DebertaV2 fixture
 */
quick_ai_test::DifferentialModel debertaV2Model() {
  return {
    "deberta_v2_tiny",
    [](quick_ai::json &cfg, quick_ai::json &gen_cfg, quick_ai::json &nntr_cfg) {
      return std::make_unique<DebertaV2RefAdapter>(cfg, gen_cfg, nntr_cfg);
    },
  };
}

/**
 * @brief FP32 output embedding matches the HF reference (atol + cosine)
 */
TEST(DebertaV2DifferentialTest, FP32MatchesHFReference) {
  quick_ai_test::runFp32EmbeddingDifferentialChecks(debertaV2Model());
}

/**
 * @brief Q4_0 embedding is close to the HF FP32 reference
 *        (skips automatically if nntr_quantize does not support DebertaV2Model)
 */
TEST(DebertaV2DifferentialTest, Q40CloseToFP32Reference) {
  quick_ai_test::runQ40EmbeddingDifferentialChecks(debertaV2Model());
}

} // namespace
