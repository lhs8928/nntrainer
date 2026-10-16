// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   audio_stream.cpp
 * @brief  Live audio input and segmentation for Qwen3-ASR segment mode
 */
#include "audio_stream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <thread>

#if defined(_WIN32)
#include <fcntl.h>
#include <io.h>
#define NNTR_POPEN _popen
#define NNTR_PCLOSE _pclose
#define NNTR_POPEN_MODE "rb"
#else
#define NNTR_POPEN popen
#define NNTR_PCLOSE pclose
#define NNTR_POPEN_MODE "r"
#endif

namespace quick_ai {

static constexpr size_t kSampleRate = 16000;
static constexpr size_t kFrame = kSampleRate / 10; // 100 ms

WavRealtimeSource::WavRealtimeSource(std::vector<float> pcm, double speed) :
  pcm_(std::move(pcm)), speed_(speed > 0 ? speed : 1.0) {}

size_t WavRealtimeSource::read(float *dst, size_t max_samples) {
  if (pos_ >= pcm_.size())
    return 0;
  if (!started_) {
    t0_ = std::chrono::steady_clock::now();
    started_ = true;
  }
  const size_t n = std::min({max_samples, pcm_.size() - pos_, kFrame});
  // the n samples "arrive" once the stream clock has reached pos_ + n
  const double due_s = static_cast<double>(pos_ + n) / kSampleRate / speed_;
  std::this_thread::sleep_until(
    t0_ + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(due_s)));
  std::copy_n(pcm_.begin() + pos_, n, dst);
  pos_ += n;
  return n;
}

PipeSource::PipeSource(const std::string &cmd) {
  if (cmd.empty()) {
    fp_ = stdin;
    is_pipe_ = false;
#if defined(_WIN32)
    _setmode(_fileno(stdin), _O_BINARY); // raw PCM: no CRLF translation
#endif
  } else {
    fp_ = NNTR_POPEN(cmd.c_str(), NNTR_POPEN_MODE);
    if (!fp_)
      throw std::runtime_error("PipeSource: cannot start: " + cmd);
    is_pipe_ = true;
  }
}

PipeSource::~PipeSource() {
  if (fp_ && is_pipe_)
    NNTR_PCLOSE(fp_);
}

size_t PipeSource::read(float *dst, size_t max_samples) {
  std::vector<int16_t> tmp(std::min(max_samples, kFrame));
  const size_t n = fread(tmp.data(), sizeof(int16_t), tmp.size(), fp_);
  for (size_t i = 0; i < n; ++i)
    dst[i] = static_cast<float>(tmp[i]) / 32768.0f;
  return n;
}

static float windowRms(const float *x, size_t n) {
  double s = 0;
  for (size_t i = 0; i < n; ++i)
    s += static_cast<double>(x[i]) * x[i];
  return static_cast<float>(std::sqrt(s / std::max<size_t>(n, 1)));
}

void Segmenter::push(const float *x, size_t n) {
  if (!use_vad_) {
    buf_.insert(buf_.end(), x, x + n);
    return;
  }

  // VAD streaming mode: evaluate frame energy and track speech state
  float r = windowRms(x, n);
  if (r < noise_floor_) noise_floor_ = 0.95f * noise_floor_ + 0.05f * r;
  else noise_floor_ = 0.999f * noise_floor_ + 0.001f * r;
  if (r > peak_rms_) peak_rms_ = r;

  float thresh = std::max(silence_rms_, noise_floor_ + (peak_rms_ - noise_floor_) * 0.12f);
  bool frame_is_speech = (r >= thresh);

  if (frame_is_speech) {
    in_speech_ = true;
    speech_samples_ += n;
    silence_samples_ = 0;
    buf_.insert(buf_.end(), x, x + n);
  } else {
    if (in_speech_) {
      silence_samples_ += n;
      buf_.insert(buf_.end(), x, x + n);
    } else {
      // Pure silence before speech: keep a rolling pre-pad of 200ms (3200 samples)
      buf_.insert(buf_.end(), x, x + n);
      constexpr size_t kPrePad = 3200;
      if (buf_.size() > kPrePad) {
        size_t drop = buf_.size() - kPrePad;
        buf_.erase(buf_.begin(), buf_.begin() + drop);
        buf_start_ += drop;
      }
    }
  }
}

bool Segmenter::pop(std::vector<float> &seg, size_t &start_sample) {
  if (!use_vad_) {
    if (buf_.size() < max_len_)
      return false;
    // quietest 100 ms window whose end is within [min_len, max_len]
    size_t best_end = max_len_;
    float best = 1e30f;
    const size_t lo = std::max(min_len_, kFrame);
    for (size_t end = max_len_; end >= lo; end -= kFrame / 2) {
      const float r = windowRms(buf_.data() + end - kFrame, kFrame);
      if (r < best) {
        best = r;
        best_end = end;
      }
      if (end < kFrame / 2)
        break;
    }
    // cut in the middle of the quietest window
    const size_t cut = best_end - kFrame / 2;
    seg.assign(buf_.begin(), buf_.begin() + cut);
    start_sample = buf_start_;
    buf_.erase(buf_.begin(), buf_.begin() + cut);
    buf_start_ += cut;
    return true;
  }

  // --- VAD-based streaming segmentation ---
  // Trigger A: Forced cut if maximum window length reached
  if (buf_.size() >= max_len_) {
    size_t best_end = max_len_;
    float best = 1e30f;
    const size_t lo = std::max(min_len_, kFrame);
    for (size_t end = max_len_; end >= lo; end -= kFrame / 2) {
      const float r = windowRms(buf_.data() + end - kFrame, kFrame);
      if (r < best) {
        best = r;
        best_end = end;
      }
      if (end < kFrame / 2)
        break;
    }
    const size_t cut = best_end - kFrame / 2;
    seg.assign(buf_.begin(), buf_.begin() + cut);
    start_sample = buf_start_;
    buf_.erase(buf_.begin(), buf_.begin() + cut);
    buf_start_ += cut;

    in_speech_ = false;
    speech_samples_ = 0;
    silence_samples_ = 0;
    return true;
  }

  // Trigger B: Silence gap detected after meaningful speech! (min_silence_gap = 400ms = 6400 samples)
  constexpr size_t kMinGap = 6400;
  if (in_speech_ && speech_samples_ >= min_len_ && silence_samples_ >= kMinGap) {
    constexpr size_t kPostPad = 2400; // 150 ms trailing padding
    size_t cut = buf_.size();
    if (silence_samples_ > kPostPad) {
      cut -= (silence_samples_ - kPostPad);
    }
    seg.assign(buf_.begin(), buf_.begin() + cut);
    start_sample = buf_start_;
    buf_.erase(buf_.begin(), buf_.begin() + cut);
    buf_start_ += cut;

    in_speech_ = false;
    speech_samples_ = 0;
    silence_samples_ = 0;
    return true;
  }

  return false;
}

bool Segmenter::flush(std::vector<float> &seg, size_t &start_sample) {
  if (buf_.empty())
    return false;
  if (use_vad_ && !in_speech_ && speech_samples_ == 0) {
    buf_.clear();
    return false;
  }
  seg.swap(buf_);
  buf_.clear();
  start_sample = buf_start_;
  buf_start_ += seg.size();

  in_speech_ = false;
  speech_samples_ = 0;
  silence_samples_ = 0;
  return true;
}

void ContextCarry::add(const std::string &text) {
  if (max_chars_ == 0)
    return;
  // trim and drop special-token look-alikes that would confuse the prompt
  std::string t = text;
  size_t pos;
  while ((pos = t.find("<|")) != std::string::npos) {
    const size_t end = t.find("|>", pos);
    t.erase(pos, end == std::string::npos ? std::string::npos : end + 2 - pos);
  }
  const size_t b = t.find_first_not_of(" \t\r\n");
  if (b == std::string::npos)
    return;
  t = t.substr(b, t.find_last_not_of(" \t\r\n") - b + 1);
  if (!text_.empty())
    text_ += ' ';
  text_ += t;
  // keep the last max_chars_ code points
  size_t n_chars = 0;
  for (unsigned char c : text_)
    if ((c & 0xC0) != 0x80)
      ++n_chars;
  if (n_chars <= max_chars_)
    return;
  size_t skip = n_chars - max_chars_, i = 0, seen = 0;
  while (i < text_.size()) {
    if (((unsigned char)text_[i] & 0xC0) != 0x80) {
      if (seen == skip)
        break;
      ++seen;
    }
    ++i;
  }
  text_.erase(0, i);
  // prefer to start at a word boundary (within the first quarter)
  const size_t sp = text_.find(' ');
  if (sp != std::string::npos && sp < text_.size() / 4)
    text_.erase(0, sp + 1);
}

float segmentPeakRms(const std::vector<float> &x) {
  float peak = 0.f;
  for (size_t i = 0; i + kFrame <= x.size(); i += kFrame)
    peak = std::max(peak, windowRms(x.data() + i, kFrame));
  if (x.size() < kFrame)
    peak = windowRms(x.data(), x.size());
  return peak;
}

} // namespace quick_ai
