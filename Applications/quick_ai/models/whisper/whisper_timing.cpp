// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   whisper_timing.cpp
 * @date   04 November 2026
 * @brief  Word timestamps from Whisper's cross-attention (DTW)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "whisper_timing.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace quick_ai {
namespace whisper_timing {

namespace {

constexpr double TOKENS_PER_SECOND = 50.0; // encoder frames per second

/** @brief UTF-8 to code points (invalid bytes become U+FFFD) */
std::u32string toCodePoints(const std::string &s) {
  std::u32string out;
  size_t i = 0;
  while (i < s.size()) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    char32_t cp = 0xFFFD;
    size_t len = 1;
    if (c < 0x80) {
      cp = c;
    } else if ((c >> 5) == 0x6 && i + 1 < s.size()) {
      cp = ((c & 0x1F) << 6) | (s[i + 1] & 0x3F);
      len = 2;
    } else if ((c >> 4) == 0xE && i + 2 < s.size()) {
      cp = ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
      len = 3;
    } else if ((c >> 3) == 0x1E && i + 3 < s.size()) {
      cp = ((c & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) |
           ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
      len = 4;
    }
    out.push_back(cp);
    i += len;
  }
  return out;
}

/** @brief Python's str.strip() for the ASCII whitespace whisper meets */
std::string strip(const std::string &s) {
  const char *ws = " \t\n\r\f\v";
  const size_t b = s.find_first_not_of(ws);
  if (b == std::string::npos)
    return "";
  const size_t e = s.find_last_not_of(ws);
  return s.substr(b, e - b + 1);
}

/** @brief Python's `a in b` for strings */
bool contains(const std::string &haystack, const std::string &needle) {
  return haystack.find(needle) != std::string::npos;
}

bool startsWith(const std::string &s, const std::string &p) {
  return s.rfind(p, 0) == 0;
}

bool endsWith(const std::string &s, const std::string &p) {
  return s.size() >= p.size() &&
         s.compare(s.size() - p.size(), p.size(), p) == 0;
}

void splitOnUnicode(const std::vector<int> &tokens, const Decoder &decode,
                    std::vector<std::string> &words,
                    std::vector<std::vector<int>> &word_tokens) {
  const std::u32string full = toCodePoints(decode(tokens));
  std::vector<int> current;
  size_t offset = 0;
  for (int t : tokens) {
    current.push_back(t);
    const std::string decoded = decode(current);
    const std::u32string cps = toCodePoints(decoded);
    const size_t pos = cps.find(U'�');
    if (pos == std::u32string::npos ||
        (offset + pos < full.size() && full[offset + pos] == U'�')) {
      words.push_back(decoded);
      word_tokens.push_back(current);
      current.clear();
      offset += cps.size();
    }
  }
}

void splitOnSpaces(const std::vector<int> &tokens, int eot,
                   const Decoder &decode, std::vector<std::string> &words,
                   std::vector<std::vector<int>> &word_tokens) {
  static const std::string punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
  std::vector<std::string> subwords;
  std::vector<std::vector<int>> subword_tokens;
  splitOnUnicode(tokens, decode, subwords, subword_tokens);
  for (size_t i = 0; i < subwords.size(); ++i) {
    const bool special = subword_tokens[i][0] >= eot;
    const bool with_space = startsWith(subwords[i], " ");
    const bool punct = contains(punctuation, strip(subwords[i]));
    if (special || with_space || punct || words.empty()) {
      words.push_back(subwords[i]);
      word_tokens.push_back(subword_tokens[i]);
    } else {
      words.back() += subwords[i];
      word_tokens.back().insert(word_tokens.back().end(),
                                subword_tokens[i].begin(),
                                subword_tokens[i].end());
    }
  }
}

/** @brief whisper's median_filter along the last axis (reflect padding) */
void medianFilterRows(Matrix &m, int width) {
  const int pad = width / 2;
  if (static_cast<int>(m.cols) <= pad)
    return;
  std::vector<float> padded(m.cols + 2 * pad), window(width);
  for (size_t r = 0; r < m.rows; ++r) {
    float *row = &m.data[r * m.cols];
    const int n = static_cast<int>(m.cols);
    for (int i = 0; i < n + 2 * pad; ++i) {
      int src = i - pad;
      if (src < 0)
        src = -src;
      if (src >= n)
        src = 2 * (n - 1) - src;
      padded[i] = row[src];
    }
    for (int i = 0; i < n; ++i) {
      std::copy(padded.begin() + i, padded.begin() + i + width, window.begin());
      std::nth_element(window.begin(), window.begin() + pad, window.end());
      row[i] = window[pad];
    }
  }
}

} // namespace

Matrix alignmentMatrix(const std::vector<Matrix> &heads, size_t frames,
                       int medfilt_width) {
  if (heads.empty())
    throw std::invalid_argument("alignmentMatrix: no heads");
  const size_t rows = heads[0].rows;
  Matrix out;
  out.rows = rows;
  out.cols = frames;
  out.data.assign(rows * frames, 0.0f);

  Matrix w;
  w.rows = rows;
  w.cols = frames;
  w.data.resize(rows * frames);
  for (const Matrix &h : heads) {
    if (h.rows != rows || h.cols < frames)
      throw std::invalid_argument("alignmentMatrix: head shapes differ");
    // softmax over the segment's frames
    for (size_t r = 0; r < rows; ++r) {
      const float *src = &h.data[r * h.cols];
      float *dst = &w.data[r * frames];
      const float mx = *std::max_element(src, src + frames);
      float sum = 0.0f;
      for (size_t c = 0; c < frames; ++c) {
        dst[c] = std::exp(src[c] - mx);
        sum += dst[c];
      }
      for (size_t c = 0; c < frames; ++c)
        dst[c] /= sum;
    }
    // normalize every frame over the token rows (population std)
    for (size_t c = 0; c < frames; ++c) {
      double mean = 0.0;
      for (size_t r = 0; r < rows; ++r)
        mean += w.at(r, c);
      mean /= static_cast<double>(rows);
      double var = 0.0;
      for (size_t r = 0; r < rows; ++r) {
        const double d = w.at(r, c) - mean;
        var += d * d;
      }
      const float sd = static_cast<float>(std::sqrt(var / rows));
      for (size_t r = 0; r < rows; ++r)
        w.at(r, c) = (w.at(r, c) - static_cast<float>(mean)) / sd;
    }
    medianFilterRows(w, medfilt_width);
    for (size_t i = 0; i < out.data.size(); ++i)
      out.data[i] += w.data[i];
  }
  for (float &v : out.data)
    v /= static_cast<float>(heads.size());
  return out;
}

void dtw(const Matrix &x, std::vector<int> &row_idx,
         std::vector<int> &col_idx) {
  const size_t N = x.rows, M = x.cols;
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<float> cost((N + 1) * (M + 1), inf);
  std::vector<int> trace((N + 1) * (M + 1), -1);
  auto C = [&](size_t i, size_t j) -> float & { return cost[i * (M + 1) + j]; };
  auto T = [&](size_t i, size_t j) -> int & { return trace[i * (M + 1) + j]; };
  C(0, 0) = 0.0f;
  for (size_t j = 1; j <= M; ++j) {
    for (size_t i = 1; i <= N; ++i) {
      const float c0 = C(i - 1, j - 1), c1 = C(i - 1, j), c2 = C(i, j - 1);
      float c;
      int t;
      if (c0 < c1 && c0 < c2) {
        c = c0;
        t = 0;
      } else if (c1 < c0 && c1 < c2) {
        c = c1;
        t = 1;
      } else {
        c = c2;
        t = 2;
      }
      C(i, j) = x.at(i - 1, j - 1) + c;
      T(i, j) = t;
    }
  }
  // backtrace
  for (size_t j = 0; j <= M; ++j)
    T(0, j) = 2;
  for (size_t i = 0; i <= N; ++i)
    T(i, 0) = 1;
  size_t i = N, j = M;
  std::vector<std::pair<int, int>> path;
  while (i > 0 || j > 0) {
    path.emplace_back(static_cast<int>(i) - 1, static_cast<int>(j) - 1);
    const int t = T(i, j);
    if (t == 0) {
      --i;
      --j;
    } else if (t == 1) {
      --i;
    } else if (t == 2) {
      --j;
    } else {
      throw std::runtime_error("dtw: unexpected trace");
    }
  }
  std::reverse(path.begin(), path.end());
  row_idx.clear();
  col_idx.clear();
  for (const auto &p : path) {
    row_idx.push_back(p.first);
    col_idx.push_back(p.second);
  }
}

void splitToWordTokens(const std::vector<int> &tokens, int eot,
                       const std::string &language, const Decoder &decode,
                       std::vector<std::string> &words,
                       std::vector<std::vector<int>> &word_tokens) {
  words.clear();
  word_tokens.clear();
  static const std::vector<std::string> no_space = {"zh", "ja", "th",
                                                    "lo", "my", "yue"};
  if (std::find(no_space.begin(), no_space.end(), language) != no_space.end())
    splitOnUnicode(tokens, decode, words, word_tokens);
  else
    splitOnSpaces(tokens, eot, decode, words, word_tokens);
}

std::vector<WhisperWordTiming>
findAlignment(const std::vector<Matrix> &heads, size_t sot_len,
              const std::vector<int> &text_tokens, int eot, size_t num_frames,
              const std::string &language, const Decoder &decode) {
  std::vector<WhisperWordTiming> out;
  if (heads.empty() || text_tokens.empty())
    return out;
  const size_t frames = std::max<size_t>(1, num_frames / 2);
  const size_t total_rows = heads[0].rows; // sot_len + 1 + text + eot
  if (total_rows != sot_len + 2 + text_tokens.size())
    throw std::invalid_argument("findAlignment: unexpected number of rows");

  Matrix full = alignmentMatrix(heads, std::min(frames, heads[0].cols));
  // rows from the no-timestamps token to the last text token, negated
  Matrix cost;
  cost.rows = text_tokens.size() + 1;
  cost.cols = full.cols;
  cost.data.resize(cost.rows * cost.cols);
  for (size_t r = 0; r < cost.rows; ++r)
    for (size_t c = 0; c < cost.cols; ++c)
      cost.at(r, c) = -full.at(sot_len + r, c);

  std::vector<int> text_idx, time_idx;
  dtw(cost, text_idx, time_idx);

  std::vector<int> with_eot = text_tokens;
  with_eot.push_back(eot);
  std::vector<std::string> words;
  std::vector<std::vector<int>> word_tokens;
  splitToWordTokens(with_eot, eot, language, decode, words, word_tokens);
  if (word_tokens.size() <= 1)
    return out;

  std::vector<double> jump_times;
  for (size_t k = 0; k < text_idx.size(); ++k)
    if (k == 0 || text_idx[k] != text_idx[k - 1])
      jump_times.push_back(time_idx[k] / TOKENS_PER_SECOND);

  std::vector<size_t> boundaries(1, 0);
  for (size_t w = 0; w + 1 < word_tokens.size(); ++w)
    boundaries.push_back(boundaries.back() + word_tokens[w].size());

  for (size_t w = 0; w + 1 < boundaries.size(); ++w) {
    WhisperWordTiming t;
    t.word = words[w];
    t.tokens = word_tokens[w];
    t.start = jump_times.at(boundaries[w]);
    t.end = jump_times.at(boundaries[w + 1]);
    out.push_back(std::move(t));
  }

  // add_word_timestamps: shorten long words at sentence boundaries
  std::vector<double> durations;
  for (const auto &t : out)
    if (t.end - t.start != 0.0)
      durations.push_back(t.end - t.start);
  if (!durations.empty()) {
    std::vector<double> d = durations;
    std::sort(d.begin(), d.end());
    const size_t n = d.size();
    double median = n % 2 ? d[n / 2] : (d[n / 2 - 1] + d[n / 2]) / 2.0;
    median = std::min(0.7, median);
    const double max_duration = median * 2.0;
    static const std::string sentence_end = ".。!！?？";
    for (size_t i = 1; i < out.size(); ++i) {
      if (out[i].end - out[i].start > max_duration) {
        if (contains(sentence_end, out[i].word))
          out[i].end = out[i].start + max_duration;
        else if (contains(sentence_end, out[i - 1].word))
          out[i].start = out[i].end - max_duration;
      }
    }
  }

  // merge_punctuations
  static const std::string prepended = "\"'“¿([{-";
  static const std::string appended = "\"'.。,，!！?？:：”)]}、";
  if (out.size() >= 2) {
    int i = static_cast<int>(out.size()) - 2;
    int j = static_cast<int>(out.size()) - 1;
    while (i >= 0) {
      auto &prev = out[i];
      auto &next = out[j];
      if (startsWith(prev.word, " ") && contains(prepended, strip(prev.word))) {
        next.word = prev.word + next.word;
        next.tokens.insert(next.tokens.begin(), prev.tokens.begin(),
                           prev.tokens.end());
        prev.word.clear();
        prev.tokens.clear();
      } else {
        j = i;
      }
      --i;
    }
    size_t a = 0, b = 1;
    while (b < out.size()) {
      auto &prev = out[a];
      auto &next = out[b];
      if (!endsWith(prev.word, " ") && contains(appended, next.word)) {
        prev.word += next.word;
        prev.tokens.insert(prev.tokens.end(), next.tokens.begin(),
                           next.tokens.end());
        next.word.clear();
        next.tokens.clear();
      } else {
        a = b;
      }
      ++b;
    }
  }

  std::vector<WhisperWordTiming> kept;
  for (auto &t : out)
    if (!t.word.empty())
      kept.push_back(std::move(t));
  return kept;
}

} // namespace whisper_timing
} // namespace quick_ai
