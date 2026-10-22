// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   speech_ops.cpp
 * @date   21 October 2026
 * @brief  Speech model operations for another program's pipeline
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "speech_ops.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>

#include "../SpeakerDiarization/models/batch_diarizer.h"
#include "asr_engine.h"
#include "json.hpp"

namespace discretion {

namespace {

using json = nlohmann::json;
constexpr double RATE = 16000.0;
constexpr size_t WINDOW = 1600; // 100 ms

bool decodePcm(const std::string &raw, std::vector<float> &pcm,
               std::string &error) {
  if (raw.size() % 2) {
    error = "PCM16 file with an odd byte count";
    return false;
  }
  pcm.resize(raw.size() / 2);
  for (size_t i = 0; i < pcm.size(); ++i) {
    const int16_t v = static_cast<int16_t>(
      static_cast<uint8_t>(raw[2 * i]) |
      (static_cast<uint16_t>(static_cast<uint8_t>(raw[2 * i + 1])) << 8));
    pcm[i] = v / 32768.0f;
  }
  return true;
}

double windowRms(const float *x, size_t n) {
  double s = 0.0;
  for (size_t i = 0; i < n; ++i)
    s += x[i] * x[i];
  return std::sqrt(s / std::max<size_t>(1, n));
}

/**
 * @brief cut audio longer than the longest model graph into pieces that fit:
 *        each cut in the middle of the quietest 100 ms of the last 15% of the
 *        piece, so it falls between words
 */
std::vector<std::pair<size_t, size_t>> pieces(size_t total, size_t max_len,
                                              const std::vector<float> &pcm) {
  std::vector<std::pair<size_t, size_t>> out;
  size_t start = 0;
  while (total - start > max_len) {
    const size_t lo = start + max_len * 85 / 100;
    size_t best_end = start + max_len;
    double best = 1e30;
    for (size_t end = start + max_len; end >= lo + WINDOW; end -= WINDOW / 2) {
      const double r = windowRms(pcm.data() + end - WINDOW, WINDOW);
      if (r < best) {
        best = r;
        best_end = end;
      }
    }
    const size_t cut = best_end - WINDOW / 2;
    out.emplace_back(start, cut);
    start = cut;
  }
  out.emplace_back(start, total);
  return out;
}

double msSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(
           std::chrono::steady_clock::now() - t0)
    .count();
}

} // namespace

SpeechOps::SpeechOps() = default;
SpeechOps::~SpeechOps() = default;

bool SpeechOps::init(const SpeechOptions &opt, std::string &error) {
  opt_ = opt;
  std::sort(opt_.asr_secs.begin(), opt_.asr_secs.end());
  if (opt_.asr_secs.empty())
    opt_.asr_secs.push_back(opt_.asr.segment_sec);
  if (opt_.asr.threads > 0)
    setenv("NNTR_NUM_THREADS", std::to_string(opt_.asr.threads).c_str(), 1);
  for (float sec : opt_.asr_secs) {
    AsrOptions a = opt_.asr;
    a.segment_sec = sec;
    a.word_timestamps = true; // the caller divides a segment by word times
    auto engine = std::make_shared<AsrEngine>();
    if (!engine->init(a, error))
      return false;
    engines_.push_back(std::move(engine));
  }
  sd_ = std::make_unique<speaker_diarization::BatchDiarizer>();
  if (!sd_->init(opt_.sd_weights)) {
    error =
      "cannot initialize the diarization models (" + opt_.sd_weights + ")";
    return false;
  }
  return true;
}

std::string SpeechOps::describe() const {
  json j;
  j["asr_secs"] = opt_.asr_secs;
  j["language"] = opt_.asr.language;
  j["model"] = opt_.asr.model_dir;
  j["sd_weights"] = opt_.sd_weights;
  j["embedding_dim"] = 256;
  j["frame_ms"] = 10000.0 / 589.0;
  return j.dump();
}

AsrResult SpeechOps::recognize(const std::vector<float> &pcm) {
  const auto t0 = std::chrono::steady_clock::now();
  AsrResult out;
  const size_t max_len = engines_.back()->segmentSamples();
  const auto parts = pieces(pcm.size(), max_len, pcm);
  for (const auto &[a, b] : parts) {
    // the shortest graph the piece fits: shorter graphs decode faster
    std::shared_ptr<AsrEngine> engine = engines_.back();
    for (const auto &e : engines_) {
      if (e->segmentSamples() >= b - a) {
        engine = e;
        break;
      }
    }
    std::vector<float> seg(pcm.begin() + a, pcm.begin() + b);
    seg.resize(engine->segmentSamples(), 0.0f);
    const std::string t = engine->transcribe(std::move(seg));
    if (!t.empty()) {
      if (!out.text.empty())
        out.text += ' ';
      out.text += t;
    }
    const auto w = engine->lastWords();
    const double len_ms = (b - a) * 1000.0 / RATE, base = a * 1000.0 / RATE;
    double prev = 0.0;
    for (size_t i = 0; i < w.size(); ++i) {
      // attention times can step back a little: keep them in order and inside
      // the piece (the model pads it with silence)
      const double s = std::clamp(w[i].second * 1000.0, prev, len_ms);
      prev = s;
      const double e = i + 1 < w.size()
                         ? std::clamp(w[i + 1].second * 1000.0, s, len_ms)
                         : len_ms;
      out.words.push_back(
        {std::lround(base + s), std::lround(base + e), w[i].first});
    }
  }
  out.pieces = parts.size();
  out.ms = msSince(t0);
  return out;
}

std::vector<float> SpeechOps::embedding(const std::vector<float> &pcm) {
  return sd_->embed(pcm, 0.0, pcm.size() / RATE);
}

std::vector<std::vector<uint8_t>>
SpeechOps::segmentation(const std::vector<float> &pcm) {
  return sd_->segment(pcm);
}

std::string SpeechOps::asr(const std::vector<float> &pcm) {
  const AsrResult r = recognize(pcm);
  json words = json::array();
  for (const auto &w : r.words)
    words.push_back({w.start_ms, w.end_ms, w.text});
  json j;
  j["text"] = r.text;
  j["words"] = words;
  j["pieces"] = r.pieces;
  j["ms"] = std::lround(r.ms);
  return j.dump();
}

std::string SpeechOps::embed(const std::vector<float> &pcm) {
  const auto t0 = std::chrono::steady_clock::now();
  const auto v = embedding(pcm);
  json j;
  j["vector"] = v;
  j["ms"] = std::lround(msSince(t0));
  return j.dump();
}

std::string SpeechOps::segment(const std::vector<float> &pcm) {
  const auto t0 = std::chrono::steady_clock::now();
  json windows = json::array();
  size_t offset = 0;
  for (const auto &classes : segmentation(pcm)) {
    std::string digits(classes.size(), '0');
    for (size_t f = 0; f < classes.size(); ++f)
      digits[f] = static_cast<char>('0' + classes[f]);
    windows.push_back({{"offset", offset}, {"classes", digits}});
    offset += 160000;
  }
  json j;
  j["frame_ms"] = 10000.0 / 589.0;
  j["windows"] = windows;
  j["ms"] = std::lround(msSince(t0));
  return j.dump();
}

bool SpeechOps::run(const std::string &op, const std::string &pcm16,
                    std::string &body, std::string &error) {
  std::vector<float> pcm;
  if (!decodePcm(pcm16, pcm, error))
    return false;
  if (pcm.empty()) {
    error = "no audio";
    return false;
  }
  try {
    if (op == "ASR")
      body = asr(pcm);
    else if (op == "EMBED")
      body = embed(pcm);
    else if (op == "SEGMENT")
      body = segment(pcm);
    else {
      error = "unknown request " + op;
      return false;
    }
  } catch (const std::exception &e) {
    error = e.what();
    return false;
  }
  return true;
}

bool SpeechOps::handle(const std::string &op, const std::string &in,
                       const std::string &out, std::string &error) {
  std::ifstream f(in, std::ios::binary);
  if (!f) {
    error = "cannot open " + in;
    return false;
  }
  const std::string raw((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
  std::string body;
  if (!run(op, raw, body, error))
    return false;
  std::ofstream o(out, std::ios::binary | std::ios::trunc);
  if (!o) {
    error = "cannot open " + out;
    return false;
  }
  o << body;
  return static_cast<bool>(o);
}

} // namespace discretion
