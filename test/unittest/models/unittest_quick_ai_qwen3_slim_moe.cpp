// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   unittest_quick_ai_qwen3_slim_moe.cpp
 * @date   15 June 2026
 * @brief  Tiny Qwen3SlimMoE CausalLM model unit tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jungwon Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <quick_ai_test_utils.h>

#include <gtest/gtest.h>

#include <layer.h>
#include <layer_context.h>
#include <qwen3_slim_moe_causallm.h>

#include <map>

namespace {

using TinyQwen3SlimMoECausalLM =
  quick_ai_test::CausalLMTestAdapter<quick_ai::Qwen3SlimMoECausalLM>;

/**
 * @brief Make the tiny Qwen3SlimMoE model config
 */
quick_ai::json makeTinyQwen3SlimMoEConfig() {
  return {
    {"architectures", {"Qwen3SlimMoeForCausalLM"}},
    {"bos_token_id", 0},
    {"eos_token_id", {31}},
    {"head_dim", 8},
    {"hidden_size", 64},
    {"intermediate_size", 64},
    {"moe_intermediate_size", 64},
    {"is_causal", true},
    {"max_position_embeddings", 8},
    {"num_attention_heads", 8},
    {"num_hidden_layers", 1},
    {"num_key_value_heads", 4},
    {"num_experts", 4},
    {"num_experts_per_tok", 2},
    {"rms_norm_eps", 1e-5},
    {"rope_theta", 10000},
    {"tie_word_embeddings", true},
    {"vocab_size", 32},
  };
}

/**
 * @brief Verify that Qwen3SlimMoE can be instantiated and that greedy
 *        generation selects the argmax token from supplied logits.
 *
 * WeightRoundTrip and PromptProducesExpectedLogits are not included here
 * because Qwen3SlimMoECausalLM::forwarding() calls Tensor::activate() for
 * lazy mmap-based expert weight loading.  In the tiny-test environment
 * weights live in normal in-memory tensors (not mmap'd files), so activate()
 * hangs waiting for storage that is not available via the binary-file round-
 * trip used by the shared test helpers.
 */
TEST(Qwen3SlimMoETinyModelTest, GreedyGenerationSelectsArgmaxLogit) {
  auto tokenizer_path =
    quick_ai_test::makeTinyCausalLMFiles("Qwen3SlimMoETinyModelTest",
                                         "GreedyGenerationSelectsArgmaxLogit",
                                         "Qwen3SlimMoE_FP32")
      .tokenizer_path;

  auto model_cfg = makeTinyQwen3SlimMoEConfig();
  auto gen_cfg = quick_ai_test::makeTinyGenerationConfig();
  auto nntr_cfg = quick_ai_test::makeTinyNntrainerConfig(
    tokenizer_path, quick_ai_test::makeTinyFp32DataType());

  auto model =
    std::make_unique<TinyQwen3SlimMoECausalLM>(model_cfg, gen_cfg, nntr_cfg);
  quick_ai_test::expectGreedyGenerationSelectsArgmax(*model);
}

} // namespace
