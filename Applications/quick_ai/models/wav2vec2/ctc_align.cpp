// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   ctc_align.cpp
 * @date   04 November 2026
 * @brief  WhisperX forced alignment of a transcript to CTC emissions
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "ctc_align.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace quick_ai {
namespace ctc_align {

namespace {

/** @brief UTF-8 to code points (invalid bytes become U+FFFD) */
std::u32string decodeUtf8(const std::string &s) {
  std::u32string out;
  for (size_t i = 0; i < s.size();) {
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

std::string encodeUtf8(char32_t cp) {
  std::string out;
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return out;
}

/** @brief Python's str.lower() for ASCII and Latin-1 letters */
char32_t lower(char32_t c) {
  if ((c >= U'A' && c <= U'Z') || (c >= 0xC0 && c <= 0xDE && c != 0xD7))
    return c + 32;
  return c;
}

bool isSpace(char32_t c) {
  return c == U' ' || c == U'\t' || c == U'\n' || c == U'\r' || c == U'\f' ||
         c == U'\v';
}

/** @brief Python's round(x, 3) (to even on a tie of the scaled value) */
double round3(double x) { return std::nearbyint(x * 1000.0) / 1000.0; }

struct Point {
  size_t token, time;
  float prob;
};

struct CharSegment {
  size_t start = 0, end = 0;
  double score = 0.0;
};

} // namespace

int blankId(const std::map<std::string, int> &dictionary) {
  int blank = 0;
  for (const auto &[ch, id] : dictionary)
    if (ch == "[pad]" || ch == "<pad>")
      blank = id;
  return blank;
}

std::vector<CtcWord> align(const std::vector<float> &emission_in, size_t frames,
                           size_t vocab, const std::string &text, double t1,
                           double t2,
                           const std::map<std::string, int> &dictionary,
                           const std::string &language) {
  std::vector<CtcWord> words;
  if (frames == 0 || emission_in.size() < frames * vocab)
    return words;
  const bool no_space = language == "ja" || language == "zh";
  const std::u32string chars = decodeUtf8(text);
  const size_t n = chars.size();

  // 1. characters to align
  size_t leading = 0, trailing = 0;
  while (leading < n && isSpace(chars[leading]))
    ++leading;
  while (trailing < n - leading && isSpace(chars[n - 1 - trailing]))
    ++trailing;
  std::vector<std::string> clean; // dictionary keys (or wildcard chars)
  std::vector<size_t> clean_cdx;  // their index in the text
  for (size_t cdx = 0; cdx < n; ++cdx) {
    char32_t c = lower(chars[cdx]);
    if (!no_space && c == U' ')
      c = U'|';
    if (cdx < leading || cdx + trailing >= n)
      continue;
    const std::string key = encodeUtf8(c);
    if (dictionary.count(key) || (c != U' ' && c != U'|')) {
      clean.push_back(key);
      clean_cdx.push_back(cdx);
    }
  }
  if (clean.empty())
    return words;

  // the wildcard column: best non-blank score of every frame
  const int blank = blankId(dictionary);
  bool wildcard = false;
  for (const auto &c : clean)
    wildcard |= dictionary.count(c) == 0;
  const size_t V = vocab + (wildcard ? 1 : 0);
  std::vector<float> emission(frames * V);
  for (size_t t = 0; t < frames; ++t) {
    const float *src = emission_in.data() + t * vocab;
    std::copy(src, src + vocab, emission.begin() + t * V);
    if (wildcard) {
      float best = -std::numeric_limits<float>::infinity();
      for (size_t v = 0; v < vocab; ++v)
        if (static_cast<int>(v) != blank)
          best = std::max(best, src[v]);
      emission[t * V + vocab] = best;
    }
  }
  std::vector<size_t> tokens;
  for (const auto &c : clean) {
    auto it = dictionary.find(c);
    tokens.push_back(it != dictionary.end() ? static_cast<size_t>(it->second)
                                            : vocab);
  }
  auto E = [&](size_t t, size_t v) { return emission[t * V + v]; };

  // 2. trellis (frames + 1) x (tokens + 1)
  const size_t N = tokens.size(), R = frames + 1, Cn = N + 1;
  const float inf = std::numeric_limits<float>::infinity();
  std::vector<float> trellis(R * Cn, 0.0f);
  auto TR = [&](size_t t, size_t j) -> float & { return trellis[t * Cn + j]; };
  TR(0, 0) = 0.0f;
  float acc = 0.0f;
  for (size_t t = 1; t < R; ++t) {
    acc += E(t - 1, blank);
    TR(t, 0) = acc;
  }
  for (size_t j = 1; j < Cn; ++j)
    TR(0, j) = -inf;
  for (size_t t = (R > N ? R - N : 0); t < R; ++t)
    TR(t, 0) = inf;
  for (size_t t = 0; t < frames; ++t)
    for (size_t j = 1; j < Cn; ++j)
      TR(t + 1, j) =
        std::max(TR(t, j) + E(t, blank), TR(t, j - 1) + E(t, tokens[j - 1]));

  // backtrack from the frame where the last token scores best
  size_t j = N, t_start = 0;
  for (size_t t = 1; t < R; ++t)
    if (TR(t, j) > TR(t_start, j))
      t_start = t;
  std::vector<Point> path;
  bool done = false;
  for (size_t t = t_start; t > 0; --t) {
    const float stayed = TR(t - 1, j) + E(t - 1, blank);
    const float changed = TR(t - 1, j - 1) + E(t - 1, tokens[j - 1]);
    const bool change = changed > stayed;
    path.push_back(
      {j - 1, t - 1, std::exp(E(t - 1, change ? tokens[j - 1] : blank))});
    if (change && --j == 0) {
      done = true;
      break;
    }
  }
  if (!done)
    return words;
  std::reverse(path.begin(), path.end());

  // merge repeats: one segment per token, in order
  std::vector<CharSegment> segs;
  for (size_t i1 = 0, i2 = 0; i1 < path.size(); i1 = i2) {
    double sum = 0.0;
    while (i2 < path.size() && path[i2].token == path[i1].token)
      sum += path[i2++].prob;
    segs.push_back({path[i1].time, path[i2 - 1].time + 1, sum / (i2 - i1)});
  }
  if (segs.size() != N)
    return words;

  // 3. character times, 4. words
  const double ratio = (t2 - t1) / static_cast<double>(R - 1);
  struct CharTime {
    bool timed = false;
    double start = 0, end = 0, score = 0;
  };
  std::vector<CharTime> ct(n);
  for (size_t k = 0; k < N; ++k) {
    CharTime &c = ct[clean_cdx[k]];
    c.timed = true;
    c.start = round3(segs[k].start * ratio + t1);
    c.end = round3(segs[k].end * ratio + t1);
    c.score = std::nearbyint(segs[k].score * 1000.0) / 1000.0;
  }
  std::vector<size_t> word_of(n);
  for (size_t cdx = 0, w = 0; cdx < n; ++cdx) {
    word_of[cdx] = w;
    if (no_space || cdx + 1 == n || chars[cdx + 1] == U' ')
      ++w;
  }
  for (size_t cdx = 0; cdx < n;) {
    const size_t w = word_of[cdx];
    size_t e = cdx;
    while (e < n && word_of[e] == w)
      ++e;
    std::u32string wtext(chars.begin() + cdx, chars.begin() + e);
    size_t b0 = 0, b1 = wtext.size();
    while (b0 < b1 && isSpace(wtext[b0]))
      ++b0;
    while (b1 > b0 && isSpace(wtext[b1 - 1]))
      --b1;
    if (b0 < b1) {
      CtcWord word;
      for (size_t i = b0; i < b1; ++i)
        word.word += encodeUtf8(wtext[i]);
      double score_sum = 0.0;
      size_t scored = 0;
      for (size_t i = cdx; i < e; ++i) {
        if (chars[i] == U' ' || !ct[i].timed)
          continue;
        word.start =
          word.timed ? std::min(word.start, ct[i].start) : ct[i].start;
        word.end = word.timed ? std::max(word.end, ct[i].end) : ct[i].end;
        word.timed = true;
        score_sum += ct[i].score;
        ++scored;
      }
      if (scored > 0)
        word.score = std::nearbyint(score_sum / scored * 1000.0) / 1000.0;
      words.push_back(std::move(word));
    }
    cdx = e;
  }

  // a word without an aligned character takes its nearest timed neighbour's
  // times (WhisperX's interpolate_method "nearest")
  for (size_t i = 0; i < words.size(); ++i) {
    if (words[i].timed)
      continue;
    for (size_t d = 1; d < words.size(); ++d) {
      if (i >= d && words[i - d].timed) {
        words[i].start = words[i - d].start;
        words[i].end = words[i - d].end;
        break;
      }
      if (i + d < words.size() && words[i + d].timed) {
        words[i].start = words[i + d].start;
        words[i].end = words[i + d].end;
        break;
      }
    }
  }
  return words;
}

} // namespace ctc_align
} // namespace quick_ai
