// SPDX-License-Identifier: Apache-2.0
/**
 *
 * @file   llm_util.cpp
 * @brief  util functions for llm (refactored from main.cpp)
 * @date   21 August 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <llm_util.hpp>
#include <tensor_dim.h>

std::vector<unsigned int>
generate_multi_tokens(ml::train::TensorDim::IO_TensorType logits,
                      unsigned int NUM_VOCAB, unsigned int NUM_TARGET_TOKENS,
                      float repetition_penalty, unsigned int *input_ids,
                      unsigned int NUM_INPUT_IDS, unsigned int *bad_words_ids,
                      unsigned int NUM_BAD_WORDS_IDS) {

  std::vector<unsigned int> ret;
  float *logits_fp32;

  if (std::holds_alternative<float *>(logits)) {
    logits_fp32 = std::get<float *>(logits);
#ifdef ENABLE_FP16
  } else if (std::holds_alternative<_FP16 *>(logits)) {
    logits_fp32 = new float[NUM_VOCAB];
    _FP16 *logits_fp16 = std::get<_FP16 *>(logits);
    for (size_t i = 0; i < NUM_VOCAB; ++i) {
      logits_fp32[i] = (float)logits_fp16[i];
    }
#endif
  } else {
    throw std::invalid_argument("logit data type is not supported");
  }

  ret = generate_multi_tokens(logits_fp32, NUM_VOCAB, NUM_TARGET_TOKENS,
                              repetition_penalty, input_ids, NUM_INPUT_IDS,
                              bad_words_ids, NUM_BAD_WORDS_IDS);

#ifdef ENABLE_FP16
  if (std::holds_alternative<_FP16 *>(logits)) {
    delete[] logits_fp32;
  }
#endif

  return ret;
}

std::vector<unsigned int> generate_multi_tokens(
  float *logits, unsigned int NUM_VOCAB, unsigned int NUM_TARGET_TOKENS,
  float repetition_penalty, unsigned int *input_ids, unsigned int NUM_INPUT_IDS,
  unsigned int *bad_words_ids, unsigned int NUM_BAD_WORDS_IDS) {

  std::vector<unsigned int> outputs;

  // apply repetition penalty
  if (repetition_penalty != 1 && input_ids != nullptr && NUM_INPUT_IDS != 0) {
    applyRepetitionPenalty(logits, input_ids, NUM_INPUT_IDS,
                           repetition_penalty);
  }

  // apply bad words penalty
  if (bad_words_ids != nullptr && NUM_BAD_WORDS_IDS != 0)
    applyBadWordsPenalty(logits, bad_words_ids, NUM_BAD_WORDS_IDS);

  // Sort and generate multiple tokens
  std::vector<std::pair<unsigned int, float>> top_indices_and_logits;
  for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
    top_indices_and_logits.push_back({i, logits[i]});
  }
  std::partial_sort(top_indices_and_logits.begin(),
                    top_indices_and_logits.begin() + NUM_TARGET_TOKENS,
                    top_indices_and_logits.end(),
                    [](auto &a, auto &b) { return a.second > b.second; });

  // add sampled words
  for (unsigned int i = 0; i < NUM_TARGET_TOKENS; ++i) {
    outputs.push_back(top_indices_and_logits[i].first);
  }

  return outputs;
}

void applyRepetitionPenalty(float *logits, unsigned int *input_ids,
                            unsigned int NUM_INPUT_IDS,
                            float repetition_penalty) {
  for (unsigned int i = 0; i < NUM_INPUT_IDS; ++i) {
    if (logits[input_ids[i]] < 0) {
      logits[input_ids[i]] *= repetition_penalty;
    } else {
      logits[input_ids[i]] /= repetition_penalty;
    }
  }
}

void applyBadWordsPenalty(float *logits, unsigned int *bad_words_ids,
                          unsigned int NUM_BAD_WORDS_IDS) {
  for (unsigned int i = 0; i < NUM_BAD_WORDS_IDS; ++i) {
    logits[bad_words_ids[i]] = -INFINITY;
  }
}

/**
 * @brief Apply temperature & top-k & top-p to logits
 * @return Max logit for softmax
 */
float applyTKP(float *logits, int len, float temperature, unsigned int top_k,
               float top_p) {

  // Apply temperature & Sort logits
  std::vector<std::pair<int, float>> top_indices_and_logits;
  for (int i = 0; i < len; ++i) {
    if (temperature > 1e-5)
      logits[i] = logits[i] / temperature;
    top_indices_and_logits.push_back({i, logits[i]});
  }
  std::partial_sort(top_indices_and_logits.begin(),
                    top_indices_and_logits.begin() + 1,
                    top_indices_and_logits.end(),
                    [](auto &a, auto &b) { return a.second > b.second; });

  return top_indices_and_logits[0].second;
}
