// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   batch_pipeline.cpp
 * @date   21 October 2026
 * @brief  WhisperX-style transcription of a complete recording
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "batch_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>

#include "../SpeakerDiarization/models/batch_diarizer.h"
#include "asr_engine.h"
#include "merger.h"

namespace discretion {

namespace {

constexpr double SAMPLE_RATE = 16000.0;

struct Region {
  double start = 0.0, end = 0.0;
};

struct Chunk {
  double start = 0.0, end = 0.0;
  std::vector<Region> parts; // the speech regions merged into this chunk
};

double secondsSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
    .count();
}

/**
 * @brief WhisperX Binarize for one class: hysteresis thresholding, and an
 *        active region longer than max_duration is cut at the lowest score of
 *        its second half. Ported as is, including the state it keeps.
 */
std::vector<Region> binarize(const std::vector<float> &scores, double step,
                             float onset, float offset, double max_duration) {
  std::vector<Region> active;
  if (scores.empty())
    return active;
  double start = 0.0, t = 0.0;
  bool is_active = scores[0] > onset;
  std::vector<float> curr_scores{scores[0]};
  std::vector<double> curr_times{start};
  for (size_t i = 1; i < scores.size(); ++i) {
    t = i * step;
    const float y = scores[i];
    if (is_active) {
      if (t - start > max_duration) {
        const size_t after = curr_scores.size() / 2;
        const size_t idx =
          after +
          (std::min_element(curr_scores.begin() + after, curr_scores.end()) -
           (curr_scores.begin() + after));
        active.push_back({start, curr_times[idx]});
        start = curr_times[idx];
        curr_scores.erase(curr_scores.begin(), curr_scores.begin() + idx + 1);
        curr_times.erase(curr_times.begin(), curr_times.begin() + idx + 1);
      } else if (y < offset) {
        active.push_back({start, t});
        start = t;
        is_active = false;
        curr_scores.clear();
        curr_times.clear();
      }
      curr_scores.push_back(y);
      curr_times.push_back(t);
    } else if (y > onset) {
      start = t;
      is_active = true;
    }
  }
  if (is_active)
    active.push_back({start, t});
  return active;
}

/** @brief WhisperX merge_chunks: join regions while a chunk fits chunk_size */
std::vector<Chunk> mergeChunks(const std::vector<Region> &regions,
                               double chunk_size) {
  std::vector<Chunk> chunks;
  if (regions.empty())
    return chunks;
  Chunk cur;
  cur.start = regions[0].start;
  for (const auto &r : regions) {
    if (r.end - cur.start > chunk_size && cur.end - cur.start > 0) {
      chunks.push_back(cur);
      cur = Chunk();
      cur.start = r.start;
    }
    cur.end = r.end;
    cur.parts.push_back(r);
  }
  chunks.push_back(cur);
  return chunks;
}

bool endsWith(const std::string &s, size_t n, const char *suffix) {
  const size_t k = std::strlen(suffix);
  return n >= k && s.compare(n - k, k, suffix) == 0;
}

/** @brief the word closes a sentence: . ? ! (also full-width), before quotes */
bool endsSentence(const std::string &word) {
  size_t n = word.size();
  while (n > 0 && std::strchr("\"')]}", word[n - 1]))
    --n;
  if (n == 0)
    return false;
  const char c = word[n - 1];
  return c == '.' || c == '?' || c == '!' ||
         endsWith(word, n, "\xE3\x80\x82") ||
         endsWith(word, n, "\xEF\xBC\x9F") ||
         endsWith(word, n, "\xEF\xBC\x81") || endsWith(word, n, "\xE2\x80\xA6");
}

/**
 * @brief split one chunk transcript into sentences. A sentence starts at its
 *        first word and ends at the end of the speech region of its last word,
 *        or where the next sentence starts if that is earlier.
 */
std::vector<BatchLine>
splitSentences(const Chunk &chunk, const std::string &text,
               std::vector<std::pair<std::string, double>> words) {
  std::vector<BatchLine> out;
  if (words.empty()) {
    if (!text.empty())
      out.push_back({chunk.start, chunk.end, "", text});
    return out;
  }
  // Attention-based word times can step back a little; keep them in order and
  // inside the chunk.
  double prev = 0.0;
  for (auto &w : words) {
    w.second = std::clamp(w.second, prev, chunk.end - chunk.start);
    prev = w.second;
  }

  struct Span {
    size_t first, last;
  };
  std::vector<Span> spans;
  size_t first = 0;
  for (size_t i = 0; i < words.size(); ++i) {
    if (endsSentence(words[i].first) || i + 1 == words.size()) {
      spans.push_back({first, i});
      first = i + 1;
    }
  }

  for (size_t k = 0; k < spans.size(); ++k) {
    BatchLine line;
    line.start = chunk.start + words[spans[k].first].second;
    const double next = k + 1 < spans.size()
                          ? chunk.start + words[spans[k + 1].first].second
                          : chunk.end;
    const double last_word = chunk.start + words[spans[k].last].second;
    double region_end = chunk.end;
    for (const auto &p : chunk.parts) {
      if (p.end >= last_word) {
        region_end = p.end;
        break;
      }
    }
    line.end = std::min(region_end, next);
    for (size_t i = spans[k].first; i <= spans[k].last; ++i) {
      if (!line.text.empty())
        line.text += ' ';
      line.text += words[i].first;
    }
    out.push_back(std::move(line));
  }

  // A sentence without duration (its words share one time) joins the next.
  std::vector<BatchLine> merged;
  for (size_t k = 0; k < out.size(); ++k) {
    if (out[k].end <= out[k].start && k + 1 < out.size()) {
      out[k + 1].start = out[k].start;
      out[k + 1].text = out[k].text + ' ' + out[k + 1].text;
      continue;
    }
    if (out[k].end <= out[k].start && !merged.empty()) {
      merged.back().end = std::max(merged.back().end, out[k].end);
      merged.back().text += ' ' + out[k].text;
      continue;
    }
    merged.push_back(std::move(out[k]));
  }
  return merged;
}

/** @brief WhisperX assign_word_speakers for a segment: longest total overlap */
std::string
dominantSpeaker(const std::vector<speaker_diarization::SpeakerSegment> &turns,
                double start, double end) {
  std::map<std::string, double> overlap;
  for (const auto &t : turns) {
    const double o =
      std::min<double>(end, t.end_time) - std::max<double>(start, t.start_time);
    if (o > 0.0)
      overlap[t.speaker] += o;
  }
  std::string best;
  double best_v = 0.0;
  for (const auto &[speaker, v] : overlap) {
    if (v > best_v) {
      best_v = v;
      best = speaker;
    }
  }
  return best;
}

using Words = std::vector<std::pair<std::string, double>>;

/** @brief words [i, i + k) equal words [j, j + k) */
bool sameRun(const Words &w, size_t i, size_t j, size_t k) {
  for (size_t x = 0; x < k; ++x)
    if (w[i + x].first != w[j + x].first)
      return false;
  return true;
}

/**
 * @brief the decoder looped: one word five times in a row, or a phrase of up
 *        to 15 words three times in a row (stands in for Whisper's compression
 *        ratio test)
 */
bool repeats(const Words &w) {
  for (size_t k = 1; k <= 15; ++k) {
    const size_t need = k == 1 ? 5 : 3;
    for (size_t i = 0; i + k * need <= w.size(); ++i) {
      size_t r = 1;
      while (i + (r + 1) * k <= w.size() && sameRun(w, i, i + r * k, k))
        ++r;
      if (r >= need)
        return true;
    }
  }
  return false;
}

/** @brief keep one copy of every phrase that repeats back to back */
void collapseRepeats(Words &w) {
  for (size_t k = 15; k >= 1; --k) {
    for (size_t i = 0; i + 2 * k <= w.size(); ++i) {
      while (i + 2 * k <= w.size() && sameRun(w, i, i + k, k))
        w.erase(w.begin() + i + k, w.begin() + i + 2 * k);
    }
  }
}

/** @brief seconds of detected speech in the chunk after time t */
double speechAfter(const Chunk &c, double t) {
  double sum = 0.0;
  for (const auto &p : c.parts)
    sum += std::max(0.0, p.end - std::max(p.start, t));
  return sum;
}

/**
 * @brief cut a chunk in two: at the pause between merged regions closest to
 *        its middle, or else at the lowest voice activity score of its middle
 *        third
 */
std::pair<Chunk, Chunk>
splitChunk(const Chunk &c, const std::vector<float> &speech, double step) {
  const double len = c.end - c.start, mid = c.start + len / 2;
  size_t gap = 0;
  double gap_dist = len / 4; // only pauses in the middle half
  for (size_t i = 1; i < c.parts.size(); ++i) {
    const double g = (c.parts[i - 1].end + c.parts[i].start) / 2;
    if (std::fabs(g - mid) <= gap_dist) {
      gap_dist = std::fabs(g - mid);
      gap = i;
    }
  }
  Chunk a, b;
  if (gap > 0) {
    a.parts.assign(c.parts.begin(), c.parts.begin() + gap);
    b.parts.assign(c.parts.begin() + gap, c.parts.end());
  } else {
    const size_t lo =
      static_cast<size_t>(std::ceil((c.start + len / 3) / step));
    const size_t hi =
      std::min(speech.empty() ? 0 : speech.size() - 1,
               static_cast<size_t>(std::floor((c.end - len / 3) / step)));
    double cut = mid;
    if (lo <= hi)
      cut = (std::min_element(speech.begin() + lo, speech.begin() + hi + 1) -
             speech.begin()) *
            step;
    for (const auto &p : c.parts) {
      if (p.end <= cut)
        a.parts.push_back(p);
      else if (p.start >= cut)
        b.parts.push_back(p);
      else {
        a.parts.push_back({p.start, cut});
        b.parts.push_back({cut, p.end});
      }
    }
  }
  a.start = a.parts.front().start;
  a.end = a.parts.back().end;
  b.start = b.parts.front().start;
  b.end = b.parts.back().end;
  return {a, b};
}

} // namespace

BatchPipeline::BatchPipeline(
  std::shared_ptr<AsrEngine> engine,
  std::shared_ptr<speaker_diarization::BatchDiarizer> diarizer,
  const BatchOptions &opt) :
  engine_(std::move(engine)), diarizer_(std::move(diarizer)), opt_(opt) {}

std::unique_ptr<BatchPipeline>
BatchPipeline::create(std::shared_ptr<AsrEngine> engine,
                      const std::string &sd_weights, const BatchOptions &opt,
                      std::string &error) {
  auto diarizer = std::make_shared<speaker_diarization::BatchDiarizer>();
  if (!diarizer->init(sd_weights)) {
    error = "cannot initialize the diarization models (" + sd_weights + ")";
    return nullptr;
  }
  return std::make_unique<BatchPipeline>(std::move(engine), std::move(diarizer),
                                         opt);
}

std::vector<BatchLine> BatchPipeline::run(const std::vector<float> &pcm,
                                          bool diarize, BatchStats *stats) {
  BatchStats st;
  st.audio_sec = pcm.size() / SAMPLE_RATE;

  // 1. voice activity (and the speakers, from the same PyanNet pass)
  auto t0 = std::chrono::steady_clock::now();
  const auto sd = diarizer_->run(pcm, diarize, opt_.vad_step_sec);
  st.vad_sec = sd.vad_ms / 1000.0;
  st.sd_sec = diarize ? secondsSince(t0) - st.vad_sec : 0.0;
  std::map<std::string, int> speakers;
  for (const auto &s : sd.segments)
    speakers[s.speaker] = 1;
  st.speakers = speakers.size();

  // 2. speech regions -> chunks
  const auto regions = binarize(sd.speech, sd.frame_step_sec, opt_.onset,
                                opt_.offset, opt_.chunk_sec);
  const auto chunks = mergeChunks(regions, opt_.chunk_sec);
  st.regions = regions.size();
  st.chunks = chunks.size();
  if (opt_.verbose) {
    const size_t per_sec = static_cast<size_t>(1.0 / sd.frame_step_sec);
    for (size_t f = 0; f + per_sec <= sd.speech.size(); f += per_sec) {
      double sum = 0.0;
      for (size_t k = f; k < f + per_sec; ++k)
        sum += sd.speech[k];
      std::fprintf(stderr, "  (vad %4.0fs) %.2f\n", f * sd.frame_step_sec,
                   sum / per_sec);
    }
    for (const auto &r : regions)
      std::fprintf(stderr, "  (region) %.2f-%.2f\n", r.start, r.end);
    for (const auto &c : chunks)
      std::fprintf(stderr, "  (chunk) %.2f-%.2f (%zu regions)\n", c.start,
                   c.end, c.parts.size());
  }

  // 3. + 4. transcribe every chunk and split it into sentences. Like Whisper's
  // fallback, a chunk whose transcript loops or stops while speech goes on is
  // decoded again, here as two shorter chunks.
  constexpr double MIN_SPLIT_SEC = 4.0;  // halves are at least this long
  constexpr double EARLY_STOP_SEC = 2.0; // untranscribed speech that is a stop
  std::vector<BatchLine> lines;
  const size_t max_len = engine_->segmentSamples();
  std::function<void(const Chunk &)> transcribe = [&](const Chunk &chunk) {
    const size_t s = std::min(
      pcm.size(), static_cast<size_t>(std::lround(chunk.start * SAMPLE_RATE)));
    size_t e = std::min(
      pcm.size(), static_cast<size_t>(std::lround(chunk.end * SAMPLE_RATE)));
    e = std::min(e, s + max_len);
    if (e <= s)
      return;
    std::vector<float> seg(pcm.begin() + s, pcm.begin() + e);
    seg.resize(max_len, 0.0f); // the model is built for one chunk length
    const std::string text = engine_->transcribe(std::move(seg));
    Words words = engine_->lastWords();
    if (opt_.verbose)
      std::fprintf(stderr, "  (asr %.2f-%.2f) %s\n", chunk.start, chunk.end,
                   text.c_str());

    const double heard_until =
      words.empty() ? chunk.start : chunk.start + words.back().second + 1.0;
    const bool stopped = (!words.empty() || text.empty()) &&
                         speechAfter(chunk, heard_until) > EARLY_STOP_SEC;
    if ((repeats(words) || stopped) &&
        chunk.end - chunk.start >= 2 * MIN_SPLIT_SEC) {
      ++st.retries;
      const auto [a, b] = splitChunk(chunk, sd.speech, sd.frame_step_sec);
      transcribe(a);
      transcribe(b);
      return;
    }
    collapseRepeats(words);
    for (auto &line : splitSentences(chunk, text, std::move(words)))
      lines.push_back(std::move(line));
  };
  t0 = std::chrono::steady_clock::now();
  for (const auto &chunk : chunks)
    transcribe(chunk);
  st.asr_sec = secondsSince(t0);

  // 5. speakers
  if (diarize) {
    for (auto &line : lines)
      line.speaker = dominantSpeaker(sd.segments, line.start, line.end);
  }
  if (stats)
    *stats = st;
  return lines;
}

void writeBatchLines(const std::vector<BatchLine> &lines, std::FILE *out,
                     std::FILE *jsonl, const std::string &session) {
  size_t seq = 0;
  for (const auto &l : lines) {
    std::fprintf(out, "[%6.1f-%6.1fs] %-12s %s\n", l.start, l.end,
                 l.speaker.empty() ? "(no speaker)" : l.speaker.c_str(),
                 l.text.c_str());
    if (!jsonl)
      continue;
    char nums[128];
    std::string j = "{\"session\":";
    appendJsonString(j, session);
    j += ",\"seq\":" + std::to_string(seq++);
    std::snprintf(nums, sizeof(nums),
                  ",\"start\":%.3f,\"end\":%.3f,\"speaker\":", l.start, l.end);
    j += nums;
    if (l.speaker.empty())
      j += "null";
    else
      appendJsonString(j, l.speaker);
    j += ",\"lag\":0.00,\"asr_lag\":0.00,\"text\":";
    appendJsonString(j, l.text);
    j += "}\n";
    std::fputs(j.c_str(), jsonl);
  }
  std::fflush(out);
  if (jsonl)
    std::fflush(jsonl);
}

} // namespace discretion
