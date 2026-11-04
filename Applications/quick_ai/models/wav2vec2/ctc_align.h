// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   ctc_align.h
 * @date   04 November 2026
 * @brief  WhisperX forced alignment of a transcript to CTC emissions
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The steps of whisperx/alignment.py (WhisperX 3.8.6, align() for one
 * segment), which follow torchaudio's forced alignment tutorial:
 *
 *   1. the transcript's characters, lower-cased, spaces as "|" (except for
 *      ja and zh), leading and trailing spaces left out; a character the
 *      model's dictionary lacks (digit, symbol, other script) is kept as a
 *      wildcard whose emission is the best non-blank score of the frame
 *   2. trellis of the characters over the frames, backtrack, merge repeats:
 *      every character gets its first and last frame
 *   3. frame k is at t1 + k * (t2 - t1) / frames, rounded to 1 ms
 *   4. words split on spaces (every character a word for ja and zh); a word
 *      starts at its first character and ends at its last
 *
 * Plain C++: the caller hands in log-softmax emissions.
 */

#ifndef __CTC_ALIGN_H__
#define __CTC_ALIGN_H__

#include <map>
#include <string>
#include <vector>

namespace quick_ai {

/** @brief one aligned word, times in seconds */
struct CtcWord {
  std::string word;
  double start = 0.0;
  double end = 0.0;
  double score = 0.0; /**< mean frame probability of its characters */
  bool timed = false; /**< false: no character could be aligned */
};

namespace ctc_align {

/**
 * @brief align `text` to the emissions of the segment [t1, t2]
 * @param emission frames x vocab log probabilities (row-major)
 * @param dictionary the model's vocab.json: character -> id
 * @param language code ("ko"); ja and zh are aligned per character
 * @return the words; empty when no character fits or the alignment fails
 */
std::vector<CtcWord> align(const std::vector<float> &emission, size_t frames,
                           size_t vocab, const std::string &text, double t1,
                           double t2,
                           const std::map<std::string, int> &dictionary,
                           const std::string &language);

/** @brief the blank id: "[pad]" or "<pad>" of the dictionary, else 0 */
int blankId(const std::map<std::string, int> &dictionary);

} // namespace ctc_align
} // namespace quick_ai

#endif // __CTC_ALIGN_H__
