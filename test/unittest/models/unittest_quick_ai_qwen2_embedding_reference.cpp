// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_quick_ai_qwen2_embedding_reference.cpp
 * @date   16 June 2026
 * @brief  Differential tests for the tiny Qwen2Embedding / KaLM-Embedding
 * models.
 *
 * The nntrainer Qwen2Embedding class backs both the plain Qwen2 embedding model
 * and KaLM-Embedding; the two fixtures differ only in their pooling mode
 * (last-token vs mean). Each compares nntrainer's pooled + L2-normalized output
 * against a golden fixture from HuggingFace transformers
 * (generate_qwen2_embedding_reference.py). Tests skip when fixtures are absent.
 *
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <quick_ai_test_utils.h>

#include <gtest/gtest.h>

#include <qwen2_embedding.h>

#include <memory>
#include <string>

namespace {

/**
 * @brief Differential model descriptor for a Qwen2Embedding-backed fixture
 */
quick_ai_test::DifferentialModel
qwen2EmbeddingModel(const std::string &fixture) {
  return {
    fixture,
    [](quick_ai::json &cfg, quick_ai::json &gen_cfg, quick_ai::json &nntr_cfg) {
      return std::make_unique<
        quick_ai_test::EmbeddingTestAdapter<quick_ai::Qwen2Embedding>>(
        cfg, gen_cfg, nntr_cfg);
    },
  };
}

/**
 * @brief FP32 Qwen2Embedding (last-token pooling) matches the HF reference
 */
TEST(Qwen2EmbeddingDifferentialTest, FP32MatchesHFReference) {
  quick_ai_test::runFp32EmbeddingDifferentialChecks(
    qwen2EmbeddingModel("qwen2_embedding_tiny"));
}

/**
 * @brief Q4_0 quantized Qwen2Embedding is close to the HF FP32 reference
 */
TEST(Qwen2EmbeddingDifferentialTest, Q40CloseToFP32Reference) {
  quick_ai_test::runQ40EmbeddingDifferentialChecks(
    qwen2EmbeddingModel("qwen2_embedding_tiny"));
}

/**
 * @brief FP32 KaLM-Embedding (mean pooling) matches the HF reference
 */
TEST(KalmEmbeddingDifferentialTest, FP32MatchesHFReference) {
  quick_ai_test::runFp32EmbeddingDifferentialChecks(
    qwen2EmbeddingModel("kalm_embedding_tiny"));
}

/**
 * @brief Q4_0 quantized KaLM-Embedding is close to the HF FP32 reference
 */
TEST(KalmEmbeddingDifferentialTest, Q40CloseToFP32Reference) {
  quick_ai_test::runQ40EmbeddingDifferentialChecks(
    qwen2EmbeddingModel("kalm_embedding_tiny"));
}

} // namespace
