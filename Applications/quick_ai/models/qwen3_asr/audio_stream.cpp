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
  } else {
    fp_ = popen(cmd.c_str(), "r");
    if (!fp_)
      throw std::runtime_error("PipeSource: cannot start: " + cmd);
    is_pipe_ = true;
  }
}

PipeSource::~PipeSource() {
  if (fp_ && is_pipe_)
    pclose(fp_);
}

size_t PipeSource::read(float *dst, size_t max_samples) {
  std::vector<int16_t> tmp(std::min(max_samples, kFrame));
  const size_t n = fread(tmp.data(), sizeof(int16_t), tmp.size(), fp_);
  for (size_t i = 0; i < n; ++i)
    dst[i] = static_cast<float>(tmp[i]) / 32768.0f;
  return n;
}

void Segmenter::push(const float *x, size_t n) {
  buf_.insert(buf_.end(), x, x + n);
}

static float windowRms(const float *x, size_t n) {
  double s = 0;
  for (size_t i = 0; i < n; ++i)
    s += static_cast<double>(x[i]) * x[i];
  return static_cast<float>(std::sqrt(s / std::max<size_t>(n, 1)));
}

bool Segmenter::pop(std::vector<float> &seg, size_t &start_sample) {
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

bool Segmenter::flush(std::vector<float> &seg, size_t &start_sample) {
  if (buf_.empty())
    return false;
  seg.swap(buf_);
  buf_.clear();
  start_sample = buf_start_;
  buf_start_ += seg.size();
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
