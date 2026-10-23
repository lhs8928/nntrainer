// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   room_pipeline.cpp
 * @date   22 October 2026
 * @brief  Meeting lines from one room microphone (ALICE's AudioPipeline)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Function by function as ambient/audio.py, ambient/align.py and
 * Segmenter.detail in ambient/diarize.py; comments name the Python function.
 */

#include "room_pipeline.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <stdexcept>
#include <tuple>

#include "json.hpp"

namespace discretion {

namespace {

using json = nlohmann::json;

constexpr long RATE = RoomPipeline::RATE;
constexpr long GAP_MS = 2000, PREROLL_MS = 300, PAUSE_MS = 800;
constexpr long BRIDGE_MS = 8000, SOFT_CAP_MS = 2000;
constexpr double FLOOR_MIN = .002;
constexpr size_t FRAME = 320; // 20 ms
constexpr long MAX_SECONDS = 12;
constexpr double DOUBT_SCORE = .6, DOUBT_MARGIN = .1;
constexpr long DOUBT_OVERLAP_MS = 1000;
constexpr double MAX_CHARS_PER_SECOND = 25;
constexpr double FRAME_MS = 10000.0 / 589.0; // PyanNet frames
constexpr size_t WINDOW = 160000;            // 10 s

/** @brief Python's // (floor division) */
long long fdiv(long long a, long long b) {
  long long q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0)))
    --q;
  return q;
}

/** @brief Python's round(): half to even */
long pyround(double x) { return static_cast<long>(std::nearbyint(x)); }

std::u32string utf32(const std::string &s) {
  std::u32string out;
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = s[i];
    char32_t cp;
    size_t n;
    if (c < 0x80) {
      cp = c;
      n = 1;
    } else if ((c >> 5) == 6) {
      cp = c & 0x1f;
      n = 2;
    } else if ((c >> 4) == 14) {
      cp = c & 0x0f;
      n = 3;
    } else {
      cp = c & 0x07;
      n = 4;
    }
    if (i + n > s.size())
      break;
    for (size_t k = 1; k < n; ++k)
      cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3f);
    out.push_back(cp);
    i += n;
  }
  return out;
}

/**
 * @brief str.isalnum for the scripts a meeting has (Hangul, CJK, Latin, ...):
 *        letters and digits, not punctuation, symbols or marks
 */
bool isAlnum(char32_t c) {
  if (c < 0x80)
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z');
  if (c < 0xC0)
    return c == 0xAA || c == 0xB2 || c == 0xB3 || c == 0xB5 || c == 0xB9 ||
           c == 0xBA || (c >= 0xBC && c <= 0xBE);
  if (c == 0xD7 || c == 0xF7)
    return false;
  if (c >= 0x0300 && c <= 0x036F) // combining marks
    return false;
  if (c >= 0x2000 && c <= 0x2BFF) // punctuation, symbols, arrows, shapes
    return false;
  if (c >= 0x2E00 && c <= 0x2E7F)
    return false;
  if (c >= 0x3000 && c <= 0x303F) // CJK symbols and punctuation
    return c == 0x3005 || c == 0x3006 || c == 0x3007 ||
           (c >= 0x3021 && c <= 0x3029) || (c >= 0x3031 && c <= 0x3035) ||
           (c >= 0x3038 && c <= 0x303C);
  if (c >= 0xFE00 && c <= 0xFE6F) // variation selectors, small forms
    return false;
  if (c >= 0xFF00 && c <= 0xFF0F)
    return false;
  if ((c >= 0xFF1A && c <= 0xFF20) || (c >= 0xFF3B && c <= 0xFF40) ||
      (c >= 0xFF5B && c <= 0xFF65))
    return false;
  if (c >= 0x1F000) // emoji and symbols
    return false;
  return true;
}

char32_t lower(char32_t c) {
  if ((c >= 'A' && c <= 'Z') || (c >= 0xC0 && c <= 0xDE && c != 0xD7))
    return c + 32;
  if (c >= 0xFF21 && c <= 0xFF3A) // fullwidth Latin
    return c + 32;
  return c;
}

/** @brief align._chars / normalize_text */
std::u32string chars(const std::string &text) {
  std::u32string out;
  for (char32_t c : utf32(text))
    if (isAlnum(c))
      out.push_back(lower(c));
  return out;
}

bool isSpace(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' ||
         c == '\f';
}

/** @brief str.split() */
std::vector<std::string> split(const std::string &text) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : text) {
    if (isSpace(c)) {
      if (!cur.empty())
        out.push_back(cur);
      cur.clear();
    } else {
      cur += c;
    }
  }
  if (!cur.empty())
    out.push_back(cur);
  return out;
}

std::string join(const std::vector<std::string> &words, size_t a, size_t b) {
  std::string out;
  for (size_t i = a; i < b; ++i) {
    if (i > a)
      out += ' ';
    out += words[i];
  }
  return out;
}

std::string strip(const std::string &s) {
  size_t a = 0, b = s.size();
  while (a < b && isSpace(s[a]))
    ++a;
  while (b > a && isSpace(s[b - 1]))
    --b;
  return s.substr(a, b - a);
}

/** @brief difflib.SequenceMatcher(None, a, b, autojunk=False) */
class Matcher {
public:
  Matcher(const std::u32string &a, const std::u32string &b) : a_(a), b_(b) {
    for (size_t j = 0; j < b.size(); ++j)
      b2j_[b[j]].push_back(j);
  }
  struct Block {
    size_t a, b, size;
  };
  std::vector<Block> blocks() const {
    std::vector<std::array<size_t, 4>> queue{{0, a_.size(), 0, b_.size()}};
    std::vector<Block> out;
    while (!queue.empty()) {
      const auto [alo, ahi, blo, bhi] = queue.back();
      queue.pop_back();
      const Block m = longest(alo, ahi, blo, bhi);
      if (m.size) {
        out.push_back(m);
        if (alo < m.a && blo < m.b)
          queue.push_back({alo, m.a, blo, m.b});
        if (m.a + m.size < ahi && m.b + m.size < bhi)
          queue.push_back({m.a + m.size, ahi, m.b + m.size, bhi});
      }
    }
    std::sort(out.begin(), out.end(), [](const Block &x, const Block &y) {
      return std::tie(x.a, x.b, x.size) < std::tie(y.a, y.b, y.size);
    });
    return out;
  }

private:
  Block longest(size_t alo, size_t ahi, size_t blo, size_t bhi) const {
    size_t besti = alo, bestj = blo, bestsize = 0;
    std::map<size_t, size_t> j2len;
    for (size_t i = alo; i < ahi; ++i) {
      std::map<size_t, size_t> next;
      auto it = b2j_.find(a_[i]);
      if (it != b2j_.end()) {
        for (size_t j : it->second) {
          if (j < blo)
            continue;
          if (j >= bhi)
            break;
          auto p = j > 0 ? j2len.find(j - 1) : j2len.end();
          const size_t k = (p != j2len.end() ? p->second : 0) + 1;
          next[j] = k;
          if (k > bestsize) {
            besti = i - k + 1;
            bestj = j - k + 1;
            bestsize = k;
          }
        }
      }
      j2len.swap(next);
    }
    while (besti > alo && bestj > blo && a_[besti - 1] == b_[bestj - 1]) {
      --besti;
      --bestj;
      ++bestsize;
    }
    while (besti + bestsize < ahi && bestj + bestsize < bhi &&
           a_[besti + bestsize] == b_[bestj + bestsize])
      ++bestsize;
    return {besti, bestj, bestsize};
  }

  const std::u32string &a_, &b_;
  std::map<char32_t, std::vector<size_t>> b2j_;
};

/** @brief align.word_times */
std::optional<std::vector<std::pair<std::string, double>>>
wordTimes(const std::string &text, const std::vector<SpeechModels::Word> &w) {
  const auto api = split(text);
  std::u32string flat, local;
  std::vector<size_t> owner;
  std::vector<double> when;
  for (size_t k = 0; k < api.size(); ++k)
    for (char32_t c : chars(api[k])) {
      flat.push_back(c);
      owner.push_back(k);
    }
  for (const auto &x : w) {
    const auto cs = chars(x.text);
    for (size_t j = 0; j < cs.size(); ++j) {
      local.push_back(cs[j]);
      when.push_back(x.start_ms + (x.end_ms - x.start_ms) * (j + .5) /
                                    static_cast<double>(cs.size()));
    }
  }
  if (flat.empty() || local.empty())
    return std::nullopt;
  std::vector<std::optional<double>> at(flat.size());
  for (const auto &b : Matcher(flat, local).blocks())
    for (size_t n = 0; n < b.size; ++n)
      at[b.a + n] = when[b.b + n];
  size_t matched = 0;
  for (const auto &t : at)
    matched += t.has_value();
  if (matched < .5 * flat.size())
    return std::nullopt;
  std::vector<size_t> known;
  for (size_t i = 0; i < at.size(); ++i)
    if (at[i])
      known.push_back(i);
  for (size_t i = 0; i < at.size(); ++i) {
    if (at[i])
      continue;
    auto up = std::upper_bound(known.begin(), known.end(), i);
    const bool has_after = up != known.end(), has_before = up != known.begin();
    if (!has_before)
      at[i] = at[*up];
    else if (!has_after)
      at[i] = at[*(up - 1)];
    else {
      const size_t before = *(up - 1), after = *up;
      at[i] = *at[before] + (*at[after] - *at[before]) *
                              static_cast<double>(i - before) /
                              static_cast<double>(after - before);
    }
  }
  std::vector<std::pair<std::string, std::optional<double>>> out;
  for (size_t k = 0; k < api.size(); ++k) {
    std::vector<double> times;
    for (size_t i = 0; i < flat.size(); ++i)
      if (owner[i] == k)
        times.push_back(*at[i]);
    std::sort(times.begin(), times.end());
    out.emplace_back(
      api[k], times.empty() ? std::nullopt
                            : std::optional<double>(times[times.size() / 2]));
  }
  // a word of punctuation only takes its neighbour's time
  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i].second)
      continue;
    std::optional<double> t;
    for (size_t j = i; j-- > 0;)
      if (out[j].second) {
        t = out[j].second;
        break;
      }
    if (!t || *t == 0.0) {
      t.reset();
      for (size_t j = i + 1; j < out.size(); ++j)
        if (out[j].second) {
          t = out[j].second;
          break;
        }
    }
    out[i].second = t && *t != 0.0 ? *t : 0.0;
  }
  std::vector<std::pair<std::string, double>> res;
  for (auto &[word, t] : out)
    res.emplace_back(word, *t);
  return res;
}

/** @brief audio.speaker_doubt */
std::vector<std::string> doubtOf(const TurnInfo *info, long overlap_ms) {
  std::vector<std::string> out;
  if (info && info->basis == "pinned")
    return out;
  const std::string basis = info ? info->basis : "";
  if (basis == "short" || basis == "new" || basis == "undecided")
    out.push_back(basis);
  else if (info &&
           ((info->speaker_score && *info->speaker_score < DOUBT_SCORE) ||
            (info->speaker_margin && *info->speaker_margin < DOUBT_MARGIN)))
    out.push_back("weak");
  if (info && info->voices == 1)
    out.push_back("alone");
  if (info && (info->absorbed == "not_distinct" || info->absorbed == "cap"))
    out.push_back("merged");
  if (overlap_ms >= DOUBT_OVERLAP_MS)
    out.push_back("overlap");
  return out;
}

std::vector<float> floats(const int16_t *x, size_t n) {
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i)
    out[i] = x[i] / 32768.0f;
  return out;
}

std::vector<float> unitOf(const std::vector<float> &v) {
  double s = 0.0;
  for (float x : v)
    s += static_cast<double>(x) * x;
  const double n = std::sqrt(s);
  if (v.empty() || !std::isfinite(n) || n < 1e-8)
    throw std::invalid_argument("유효한 목소리 특징이 없습니다.");
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    out[i] = static_cast<float>(v[i] / n);
  return out;
}

double frameRms(const int16_t *x, size_t n) {
  long long s = 0;
  for (size_t i = 0; i < n; ++i)
    s += static_cast<long long>(x[i]) * x[i];
  return std::sqrt(static_cast<double>(s) / n) / 32768;
}

} // namespace

std::u32string normalizeText(const std::string &text) { return chars(text); }

bool plausible(const std::string &text, long ms) {
  return normalizeText(text).size() <=
         MAX_CHARS_PER_SECOND * std::max(1.0, ms / 1000.0) + 10;
}

std::vector<LocalTurn>
detailTurns(const std::vector<std::vector<uint8_t>> &windows, size_t samples,
            long min_ms) {
  static const std::vector<std::vector<int>> mapping = {
    {}, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}};
  std::vector<LocalTurn> out;
  if (samples < 1600)
    return out;
  for (size_t w = 0; w < windows.size(); ++w) {
    const size_t offset = w * WINDOW;
    if (offset >= samples)
      break;
    const size_t frames = static_cast<size_t>(
      std::min<double>(WINDOW, samples - offset) / 16 / FRAME_MS);
    struct Run {
      size_t a, b;
      int who;
      long overlap, folded;
    };
    std::vector<Run> runs;
    std::optional<int> current;
    for (size_t i = 0; i < std::min(frames, windows[w].size()); ++i) {
      const auto &active = mapping.at(windows[w][i]);
      if (active.empty())
        continue;
      const int who = current && std::find(active.begin(), active.end(),
                                           *current) != active.end()
                        ? *current
                        : active[0];
      const long both = active.size() > 1;
      if (!runs.empty() && runs.back().who == who && i - runs.back().b <= 3) {
        runs.back().b = i + 1;
        runs.back().overlap += both;
      } else {
        runs.push_back({i, i + 1, who, both, 0});
      }
      current = who;
    }
    std::vector<Run> merged;
    for (const auto &r : runs) {
      if (!merged.empty() && (r.b - r.a) * FRAME_MS < min_ms) {
        Run &m = merged.back();
        m.b = r.b;
        m.overlap += r.overlap;
        m.folded +=
          r.folded + (r.who != m.who ? static_cast<long>(r.b - r.a) : 0);
        continue;
      }
      if (!merged.empty() && merged.back().who == r.who) {
        Run &m = merged.back();
        m.b = r.b;
        m.overlap += r.overlap;
        m.folded += r.folded;
        continue;
      }
      if (!merged.empty() &&
          (merged.back().b - merged.back().a) * FRAME_MS < min_ms) {
        Run &m = merged.back();
        m.folded += static_cast<long>(m.b - m.a);
        m.b = r.b;
        m.who = r.who;
        m.overlap += r.overlap;
        m.folded += r.folded;
        continue;
      }
      merged.push_back(r);
    }
    const double base = offset / 16.0;
    for (const auto &m : merged)
      out.push_back(
        {pyround(base + m.a * FRAME_MS), pyround(base + m.b * FRAME_MS),
         std::to_string(offset) + ":" + std::to_string(m.who),
         pyround(m.overlap * FRAME_MS), pyround(m.folded * FRAME_MS)});
  }
  return out;
}

std::vector<std::string> shareText(const std::string &text,
                                   const std::vector<long> &durations) {
  const auto words = split(text);
  if (durations.size() < 2 || words.size() < 2) {
    std::vector<std::string> out{text};
    out.resize(std::max<size_t>(1, durations.size()));
    return out;
  }
  double total = 0;
  for (long d : durations)
    total += std::max(1L, d);
  std::vector<std::u32string> w32;
  double chars_n = 0;
  std::vector<double> ends;
  for (const auto &w : words) {
    w32.push_back(utf32(w));
    chars_n += w32.back().size();
    ends.push_back(chars_n);
  }
  std::vector<std::string> out;
  size_t start = 0;
  double run = 0;
  for (size_t di = 0; di + 1 < durations.size(); ++di) {
    const long d = durations[di];
    run += std::max(1L, d);
    const double goal = chars_n * run / total;
    if (start >= words.size() - 1) {
      out.push_back("");
      continue;
    }
    auto near = [&](size_t k) { return std::abs(ends[k] - goal); };
    size_t best = start;
    for (size_t k = start; k < words.size() - 1; ++k)
      if (near(k) < near(best))
        best = k;
    const double reach =
      .4 * chars_n *
      std::min<double>(std::max(1L, d),
                       std::max(1L, durations[out.size() + 1])) /
      total;
    std::optional<size_t> end_cut, comma_cut;
    for (size_t k = start; k < words.size() - 1; ++k) {
      const char32_t last = w32[k].back();
      if ((last == '.' || last == '?' || last == '!') &&
          near(k) <= std::max(4.0, reach) &&
          (!end_cut || near(k) < near(*end_cut)))
        end_cut = k;
      if (last == ',' && near(k) <= std::max(3.0, reach / 2) &&
          (!comma_cut || near(k) < near(*comma_cut)))
        comma_cut = k;
    }
    const size_t cut = end_cut ? *end_cut : comma_cut ? *comma_cut : best;
    out.push_back(join(words, start, cut + 1));
    start = cut + 1;
  }
  out.push_back(join(words, start, words.size()));
  return out;
}

std::optional<std::vector<std::string>>
splitByTime(const std::string &text, const std::vector<SpeechModels::Word> &w,
            const std::vector<std::pair<long, long>> &turns) {
  const auto timed = wordTimes(text, w);
  if (!timed)
    return std::nullopt;
  std::vector<std::vector<std::string>> parts(turns.size());
  for (const auto &timed_word : *timed) {
    const double t = timed_word.second;
    std::optional<size_t> k;
    for (size_t i = 0; i < turns.size(); ++i)
      if (turns[i].first <= t && t < turns[i].second) {
        k = i;
        break;
      }
    if (!k) {
      auto dist = [&](size_t i) {
        return std::min(std::abs(t - turns[i].first),
                        std::abs(t - turns[i].second));
      };
      size_t best = 0;
      for (size_t i = 1; i < turns.size(); ++i)
        if (dist(i) < dist(best))
          best = i;
      k = best;
    }
    parts[*k].push_back(timed_word.first);
  }
  std::vector<std::string> out;
  for (const auto &p : parts)
    out.push_back(join(p, 0, p.size()));
  return out;
}

RoomPipeline::RoomPipeline(SpeechModels &models, RoomDiarizer &registry,
                           std::unique_ptr<VoiceGate> gate, Emit emit,
                           Moved moved) :
  models_(models),
  registry_(registry),
  vad_(gate ? std::move(gate) : std::make_unique<EnergyGate>()),
  floor_(FLOOR_MIN),
  emit_(std::move(emit)),
  moved_(std::move(moved)) {
  silence_frames_ = std::max<long>(1, PAUSE_MS / 20);
  max_frames_ = MAX_SECONDS * 50;
  soft_frames_ =
    max_frames_ - std::min<size_t>(SOFT_CAP_MS / 20, max_frames_ / 4);
  preroll_ = PREROLL_MS / 20;
  bridge_ = BRIDGE_MS / 20;
}

// align
void RoomPipeline::align(double at_ms, bool exact) {
  const long long at = fdiv(std::llround(at_ms) * RATE, 1000);
  if (at - cursor_ <= (exact && active_.empty() ? 0 : GAP_MS * RATE / 1000))
    return;
  if (!active_.empty())
    finish();
  pending_.clear();
  silent_ = 0;
  cursor_ = at;
  before_.clear();
  closed_ = false;
  heard_sound_ = false;
  first_sound_.reset();
}

// feed
void RoomPipeline::feed(const int16_t *pcm, size_t n, double at_ms,
                        bool exact) {
  if (at_ms >= 0)
    align(at_ms, exact);
  pending_.insert(pending_.end(), pcm, pcm + n);
  while (pending_.size() >= FRAME) {
    // off the stream before it is heard: a segment that fails to recognize
    // never brings its frames back
    Frame frame(pending_.begin(), pending_.begin() + FRAME);
    pending_.erase(pending_.begin(), pending_.begin() + FRAME);
    step(frame);
  }
}

void RoomPipeline::step(const Frame &frame) {
  const double rms = frameRms(frame.data(), frame.size());
  const bool voiced = vad_->voiced(rms, frame.data(), frame.size());
  const bool sound = floor_.voiced(rms, frame.data(), frame.size());
  if (voiced && active_.empty()) {
    // since the last segment ended less than BRIDGE_MS ago: everything from
    // the first sound above the noise floor since then, with the pre-roll
    size_t n = preroll_;
    if (bridge_ && closed_ && before_.size() < bridge_ && heard_sound_)
      n = static_cast<size_t>(std::min<long long>(
        before_.size(), fdiv(cursor_ - *first_sound_, FRAME) + preroll_));
    n = std::min(n, before_.size());
    active_.assign(before_.end() - n, before_.end());
    before_.clear();
    start_ = cursor_ - static_cast<long long>(active_.size() * FRAME);
    onset_ = cursor_;
    heard_sound_ = false;
    first_sound_.reset();
  } else if (active_.empty() && (preroll_ || bridge_)) {
    before_.push_back(frame);
    const size_t keep = std::max(preroll_, bridge_);
    if (before_.size() > keep)
      before_.erase(before_.begin(), before_.end() - keep);
    if (sound && !first_sound_)
      first_sound_ = cursor_;
    heard_sound_ = heard_sound_ || sound;
  }
  if (voiced || !active_.empty()) {
    active_.push_back(frame);
    if (voiced) {
      last_voice_ = cursor_ + FRAME;
      silent_ = 0;
    } else {
      ++silent_;
    }
  }
  cursor_ += FRAME;
  // frames since the onset: the bridge does not use up the cap
  const long long spoken = static_cast<long long>(active_.size()) -
                           fdiv(std::max(0LL, onset_ - start_), FRAME);
  if (!active_.empty() && (silent_ >= silence_frames_ ||
                           (soft_frames_ && !voiced &&
                            spoken >= static_cast<long long>(soft_frames_))))
    finish();
  else if (!active_.empty() && spoken >= static_cast<long long>(max_frames_))
    cap();
  else if (active_.empty() && bridge_ && before_.size() >= bridge_)
    quietWindow();
}

// _quiet_cut
size_t RoomPipeline::quietCut(const std::vector<Frame> &frames) const {
  const size_t tail = SOFT_CAP_MS / 20;
  if (frames.size() <= tail + 10)
    return frames.size();
  std::vector<double> rms;
  for (const auto &f : frames)
    rms.push_back(frameRms(f.data(), f.size()));
  std::vector<double> smooth;
  for (size_t i = 0; i + 9 < rms.size(); ++i) {
    double s = 0.0;
    for (size_t j = i; j < i + 10; ++j)
      s += rms[j];
    smooth.push_back(s / 10);
  }
  size_t k = frames.size() - tail;
  for (size_t i = k; i < smooth.size(); ++i)
    if (smooth[i] < smooth[k])
      k = i;
  std::vector<double> sorted(smooth);
  std::sort(sorted.begin(), sorted.end());
  const double typical = sorted[sorted.size() / 2];
  return smooth[k] < typical * .5 ? k + 5 : frames.size();
}

// _cap
void RoomPipeline::cap() {
  const size_t keep = quietCut(active_);
  std::vector<Frame> rest(active_.begin() + keep, active_.end());
  active_.resize(keep);
  if (!rest.empty())
    last_voice_ = start_ + static_cast<long long>(keep * FRAME);
  finish();
  if (!rest.empty()) {
    active_ = std::move(rest);
    start_ = cursor_ - static_cast<long long>(active_.size() * FRAME);
    onset_ = start_;
    last_voice_ = cursor_;
    silent_ = 0;
  }
}

// _quiet_window
void RoomPipeline::quietWindow(bool final) {
  std::vector<Frame> frames;
  frames.swap(before_);
  if (!frames.empty() && heard_sound_) {
    const size_t keep = final ? frames.size() : quietCut(frames);
    std::vector<Frame> rest(frames.begin() + keep, frames.end());
    const long long start =
      cursor_ - static_cast<long long>(frames.size() * FRAME);
    frames.resize(keep);
    const size_t k = first_sound_
                       ? static_cast<size_t>(std::max<long long>(
                           0, fdiv(*first_sound_ - start, FRAME) - preroll_))
                       : 0;
    active_.assign(frames.begin() + std::min(k, frames.size()), frames.end());
    start_ = start + static_cast<long long>(k * FRAME);
    onset_ = std::max(start_, first_sound_ ? *first_sound_ : start_);
    last_voice_ = start + static_cast<long long>(keep * FRAME);
    silent_ = 0;
    finish();
    if (!rest.empty()) {
      before_ = std::move(rest);
      heard_sound_ = true;
      first_sound_ = cursor_ - static_cast<long long>(before_.size() * FRAME);
      return;
    }
  }
  heard_sound_ = false;
  first_sound_.reset();
  closed_ = true;
}

// _finish (a local model: recognized here, before the next audio)
void RoomPipeline::finish() {
  struct Closer {
    bool &closed;
    ~Closer() { closed = true; }
  } closer{closed_};
  auto seg = take();
  if (seg)
    complete(*seg);
}

// _take
std::optional<RoomPipeline::Segment> RoomPipeline::take() {
  Segment seg;
  for (const auto &f : active_)
    seg.data.insert(seg.data.end(), f.begin(), f.end());
  seg.start = start_;
  seg.end = last_voice_;
  seg.lead = std::max(0LL, onset_ - start_);
  active_.clear();
  silent_ = 0;
  if (seg.end - seg.start - seg.lead < 3200)
    return std::nullopt;
  seg.group = ++segments_;
  return seg;
}

// _confirmed
std::vector<LocalTurn>
RoomPipeline::confirmed(std::vector<LocalTurn> local,
                        const std::vector<float> &speech) {
  const double same = registry_.same();
  auto vec = [&](long a, long b) {
    const size_t x = std::min<size_t>(speech.size(), a * RATE / 1000);
    const size_t y = std::min<size_t>(speech.size(), b * RATE / 1000);
    return unitOf(models_.embed(
      std::vector<float>(speech.begin() + x, speech.begin() + std::max(x, y))));
  };
  std::vector<LocalTurn> out;
  for (const auto &t : local) {
    LocalTurn *p = out.empty() ? nullptr : &out.back();
    if (p && (t.end_ms - t.start_ms < 1000 || p->end_ms - p->start_ms < 1000 ||
              p->who == t.who)) {
      if (p->who != t.who)
        p->folded_ms +=
          std::min(t.end_ms - t.start_ms, p->end_ms - p->start_ms);
    } else {
      bool alike = false;
      if (p) {
        const auto u = vec(p->start_ms, p->end_ms),
                   v = vec(t.start_ms, t.end_ms);
        double s = 0.0;
        for (size_t i = 0; i < u.size() && i < v.size(); ++i)
          s += static_cast<double>(u[i]) * v[i];
        alike = s >= same;
      }
      if (!alike) {
        out.push_back(t);
        continue;
      }
    }
    p->end_ms = t.end_ms;
    p->overlap_ms += t.overlap_ms;
    p->folded_ms += t.folded_ms;
  }
  return out;
}

// _hear + _complete
void RoomPipeline::complete(const Segment &seg) {
  const size_t n_speech = static_cast<size_t>(
    std::min<long long>(seg.data.size(), std::max(0LL, seg.end - seg.start)));
  const std::vector<float> speech = floats(seg.data.data(), n_speech);
  const std::vector<float> data = floats(seg.data.data(), seg.data.size());

  // _hear: where the speaker changes inside it, and its text
  auto local =
    confirmed(detailTurns(models_.segment(speech), speech.size()), speech);
  const auto row = models_.recognize(data);
  std::set<std::string> whos;
  for (const auto &t : local)
    whos.insert(t.who);
  std::vector<SpeechModels::Word> words;
  if (whos.size() > 1)
    for (const auto &w : row.words)
      if (!strip(w.text).empty())
        words.push_back({w.start_ms, w.end_ms, strip(w.text)});

  // _complete
  const long ahead = static_cast<long>(seg.lead * 1000 / RATE);
  const long spoken = static_cast<long>((seg.end - seg.start) * 1000 / RATE);
  const std::string text = strip(row.text);
  if (text.empty())
    return;
  struct Hidden {
    long overlap_ms = 0, folded_ms = 0;
    std::string shared; // "" = False
  };
  struct Part {
    long a, b;
    std::string who, text;
    Hidden hidden;
  };
  std::vector<Part> turns;
  if (whos.size() > 1) {
    // the speaker changes inside this segment: its text divided between the
    // parts by where each word was said, else by speaking time
    std::vector<std::pair<long, long>> spans;
    std::vector<long> durations;
    for (const auto &t : local) {
      spans.emplace_back(t.start_ms, t.end_ms);
      durations.push_back(t.end_ms - t.start_ms);
    }
    std::optional<std::vector<std::string>> parts;
    if (!words.empty())
      parts = splitByTime(text, words, spans);
    const std::string how = parts ? "words" : "time";
    const auto texts = parts ? *parts : shareText(text, durations);
    for (size_t i = 0; i < local.size() && i < texts.size(); ++i)
      if (!texts[i].empty())
        turns.push_back({local[i].start_ms,
                         local[i].end_ms,
                         local[i].who,
                         texts[i],
                         {local[i].overlap_ms, local[i].folded_ms, how}});
  } else {
    // one voice: from where the segmentation model first heard it, at the
    // latest the onset
    Hidden hidden;
    for (const auto &t : local) {
      hidden.overlap_ms += t.overlap_ms;
      hidden.folded_ms += t.folded_ms;
    }
    const long a = local.empty() ? ahead : std::min(local[0].start_ms, ahead);
    turns.push_back({a, spoken, "one", text, hidden});
  }
  std::vector<Part> kept;
  for (const auto &t : turns)
    if (plausible(t.text, t.b - t.a))
      kept.push_back(t);
  looped_ += turns.size() - kept.size(); // recognition looped: not speech
  if (kept.empty())
    return;

  const long base = static_cast<long>(seg.start * 1000 / RATE);
  struct Line {
    std::string id;
    std::vector<std::string> parts;
    long start, end;
    std::string text, local;
    std::vector<float> vector;
    Hidden hidden;
  };
  std::vector<Line> lines;
  for (const auto &t : kept) {
    Line l;
    l.id = "audio-" + std::to_string(++counter_);
    l.parts = {l.id};
    l.start = base + t.a;
    l.end = std::max(base + t.a, base + t.b);
    l.text = t.text;
    l.local = t.who;
    l.hidden = t.hidden;
    const size_t x =
      std::min<size_t>(speech.size(), std::max(0L, t.a) * RATE / 1000);
    const size_t y =
      std::min<size_t>(speech.size(), std::max(0L, t.b) * RATE / 1000);
    if (y > x && y - x >= static_cast<size_t>(RATE))
      l.vector = models_.embed(
        std::vector<float>(speech.begin() + x, speech.begin() + y));
    lines.push_back(std::move(l));
  }
  std::vector<RoomDiarizer::NewTurn> add;
  for (const auto &l : lines)
    add.push_back({l.id, l.vector, static_cast<double>(l.end - l.start),
                   std::to_string(seg.group), l.local});
  const auto r = registry_.add(add);
  if (!r.moved.empty() && moved_)
    moved_(r.moved);
  auto label = [&](const std::string &id) {
    auto it = r.labels.find(id);
    return it == r.labels.end() ? std::string() : it->second;
  };
  // parts of one segment the clustering gave one voice are one line again
  std::vector<Line> joined;
  for (auto &l : lines) {
    if (!joined.empty() && !label(l.id).empty() &&
        label(joined.back().id) == label(l.id)) {
      Line &j = joined.back();
      j.parts.push_back(l.id);
      j.end = l.end;
      j.text += " " + l.text;
      j.hidden.overlap_ms += l.hidden.overlap_ms;
      j.hidden.folded_ms += l.hidden.folded_ms;
    } else {
      joined.push_back(std::move(l));
    }
  }
  if (joined.size() == 1)
    joined[0].hidden.shared.clear();
  for (const auto &l : joined) {
    auto it = r.info.find(l.id);
    const TurnInfo *ti = it == r.info.end() ? nullptr : &it->second;
    json info;
    info["asr_provider"] = "nntrainer:qwen3-asr";
    info["asr_ms"] = std::lround(row.ms);
    info["separation"] = "diarize";
    info["split"] = joined.size() > 1;
    if (ti) {
      info["speaker_score"] =
        ti->speaker_score ? json(*ti->speaker_score) : json(nullptr);
      info["speaker_margin"] =
        ti->speaker_margin ? json(*ti->speaker_margin) : json(nullptr);
      info["basis"] = ti->basis;
      info["cluster_size"] =
        ti->cluster_size ? json(*ti->cluster_size) : json(nullptr);
      if (!ti->absorbed.empty())
        info["absorbed"] = ti->absorbed;
      info["voices"] = ti->voices;
    }
    if (l.hidden.overlap_ms)
      info["overlap_ms"] = l.hidden.overlap_ms;
    if (l.hidden.folded_ms)
      info["folded_ms"] = l.hidden.folded_ms;
    if (!l.hidden.shared.empty())
      info["shared"] = l.hidden.shared;
    RoomLine out;
    out.id = l.id;
    out.parts = l.parts;
    out.speaker = label(l.id);
    out.start_ms = l.start;
    out.end_ms = l.end;
    out.text = l.text;
    out.doubt = doubtOf(ti, l.hidden.overlap_ms);
    info["doubt"] = out.doubt;
    out.info = info.dump();
    emit_(out);
  }
}

// flush
void RoomPipeline::flush() {
  if (!pending_.empty()) {
    std::vector<int16_t> pad(FRAME - pending_.size(), 0);
    feed(pad.data(), pad.size());
  }
  if (!active_.empty())
    finish();
  else if (!before_.empty() && bridge_)
    quietWindow(true);
}

} // namespace discretion
