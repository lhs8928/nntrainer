// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   whisper_timing.h
 * @date   04 November 2026
 * @brief  Word timestamps from Whisper's cross-attention (DTW)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The same steps as openai/whisper whisper/timing.py (find_alignment and the
 * word part of add_word_timestamps):
 *
 *   1. the alignment heads' attention logits of every token row, cut to the
 *      audio frames of the segment, softmax over those frames
 *   2. per head and frame: normalize over the token rows (mean / std)
 *   3. median filter of width 7 along the frames, mean over the heads
 *   4. DTW of the text rows against the frames, the frame where each token's
 *      row starts is the token's time (50 frames per second)
 *   5. tokens into words (on spaces, or on characters for languages without
 *      spaces), a word starts at its first token and ends at the next word
 *   6. long words at sentence ends are shortened, punctuation joins the
 *      neighbouring word
 *
 * Only plain C++: the model hands in the captured logits and a decoder.
 */

#ifndef __WHISPER_TIMING_H__
#define __WHISPER_TIMING_H__

#include <functional>
#include <string>
#include <vector>

namespace quick_ai {

/** @brief one word of a segment, times in seconds from the segment start */
struct WhisperWordTiming {
  std::string word;
  std::vector<int> tokens;
  double start = 0.0;
  double end = 0.0;
};

namespace whisper_timing {

/** @brief rows x cols matrix in row-major order */
struct Matrix {
  size_t rows = 0;
  size_t cols = 0;
  std::vector<float> data;
  float &at(size_t r, size_t c) { return data[r * cols + c]; }
  float at(size_t r, size_t c) const { return data[r * cols + c]; }
};

/**
 * @brief steps 1-3: heads[h] holds the logits of every token row over at
 *        least `frames` frames; returns the (rows x frames) alignment matrix
 */
Matrix alignmentMatrix(const std::vector<Matrix> &heads, size_t frames,
                       int medfilt_width = 7);

/**
 * @brief DTW over a cost matrix (lower is better), as whisper's dtw_cpu;
 *        returns the path as (row, col) index pairs from (0, 0) to the end
 */
void dtw(const Matrix &cost, std::vector<int> &row_idx,
         std::vector<int> &col_idx);

/** @brief decodes token ids into text (special tokens included) */
using Decoder = std::function<std::string(const std::vector<int> &)>;

/**
 * @brief whisper's split_to_word_tokens: on spaces, or on every character
 *        for the languages written without them (zh, ja, th, lo, my, yue)
 * @param eot the end-of-text id; ids >= eot are special
 */
void splitToWordTokens(const std::vector<int> &tokens, int eot,
                       const std::string &language, const Decoder &decode,
                       std::vector<std::string> &words,
                       std::vector<std::vector<int>> &word_tokens);

/**
 * @brief the whole alignment of one segment
 * @param heads     per alignment head: logits of the token rows, rows from
 *                  position 0 (the start-of-transcript token) to the
 *                  end-of-text token fed after the text
 * @param sot_len   tokens before the no-timestamps token (start of transcript,
 *                  language, task)
 * @param text_tokens the generated tokens (without the end-of-text token)
 * @param num_frames mel frames of the segment's audio (100 per second)
 */
std::vector<WhisperWordTiming>
findAlignment(const std::vector<Matrix> &heads, size_t sot_len,
              const std::vector<int> &text_tokens, int eot, size_t num_frames,
              const std::string &language, const Decoder &decode);

} // namespace whisper_timing
} // namespace quick_ai

#endif // __WHISPER_TIMING_H__
