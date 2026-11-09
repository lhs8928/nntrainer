// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   audio_frontend.cpp
 * @date   09 November 2026
 * @brief  WAV loading and the log-mel front-end of Nemotron 3.5 Transcribe
 * @see    https://github.com/nntrainer/nntrainer
 * @author Claude <noreply@anthropic.com>
 * @bug    No known bugs except for NYI items
 */

#include "audio_frontend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace causallm {
namespace nemotron_transcribe {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr float kLogGuard = 5.9604644775390625e-08f; // 2^-24

uint32_t readU32(const unsigned char *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}
uint16_t readU16(const unsigned char *p) {
  return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

/** round a float to the nearest bfloat16 (ties to even), back to float */
float roundBF16(float v) {
  uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  if ((bits & 0x7f800000u) == 0x7f800000u) // inf / nan
    return v;
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  bits &= 0xffff0000u;
  std::memcpy(&v, &bits, sizeof(v));
  return v;
}

/** librosa Slaney mel scale */
double hzToMel(double f) {
  const double f_sp = 200.0 / 3.0;
  const double min_log_hz = 1000.0;
  const double min_log_mel = min_log_hz / f_sp;
  const double logstep = std::log(6.4) / 27.0;
  if (f >= min_log_hz)
    return min_log_mel + std::log(f / min_log_hz) / logstep;
  return f / f_sp;
}

double melToHz(double m) {
  const double f_sp = 200.0 / 3.0;
  const double min_log_hz = 1000.0;
  const double min_log_mel = min_log_hz / f_sp;
  const double logstep = std::log(6.4) / 27.0;
  if (m >= min_log_mel)
    return min_log_hz * std::exp(logstep * (m - min_log_mel));
  return f_sp * m;
}

} // namespace

std::vector<float> loadWav(const std::string &path, int &sample_rate) {
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open())
    throw std::runtime_error("loadWav: cannot open " + path);
  std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)),
                                 std::istreambuf_iterator<char>());
  if (buf.size() < 12 || std::memcmp(buf.data(), "RIFF", 4) != 0 ||
      std::memcmp(buf.data() + 8, "WAVE", 4) != 0)
    throw std::runtime_error("loadWav: not a RIFF/WAVE file: " + path);

  uint16_t format = 0, channels = 0, bits = 0;
  uint32_t rate = 0;
  const unsigned char *data = nullptr;
  size_t data_size = 0;
  size_t pos = 12;
  while (pos + 8 <= buf.size()) {
    const unsigned char *chunk = buf.data() + pos;
    const uint32_t size = readU32(chunk + 4);
    const size_t body = pos + 8;
    const size_t avail = std::min<size_t>(size, buf.size() - body);
    if (std::memcmp(chunk, "fmt ", 4) == 0 && avail >= 16) {
      format = readU16(buf.data() + body);
      channels = readU16(buf.data() + body + 2);
      rate = readU32(buf.data() + body + 4);
      bits = readU16(buf.data() + body + 14);
      if (format == 0xFFFE && avail >= 26) // WAVE_FORMAT_EXTENSIBLE
        format = readU16(buf.data() + body + 24);
    } else if (std::memcmp(chunk, "data", 4) == 0) {
      data = buf.data() + body;
      data_size = avail;
    }
    pos = body + size + (size & 1u);
  }
  if (data == nullptr || channels == 0 || bits == 0)
    throw std::runtime_error("loadWav: missing fmt or data chunk: " + path);

  const size_t bytes_per_sample = bits / 8;
  const size_t frames = data_size / (bytes_per_sample * channels);
  std::vector<float> out(frames, 0.0f);
  for (size_t i = 0; i < frames; ++i) {
    double acc = 0.0;
    for (uint16_t c = 0; c < channels; ++c) {
      const unsigned char *s = data + (i * channels + c) * bytes_per_sample;
      double v = 0.0;
      if (format == 1) { // PCM
        switch (bits) {
        case 8:
          v = (static_cast<int>(s[0]) - 128) / 128.0;
          break;
        case 16:
          v = static_cast<int16_t>(readU16(s)) / 32768.0;
          break;
        case 24: {
          int32_t x = s[0] | (s[1] << 8) | (s[2] << 16);
          if (x & 0x800000)
            x |= ~0xffffff;
          v = x / 8388608.0;
          break;
        }
        case 32:
          v = static_cast<int32_t>(readU32(s)) / 2147483648.0;
          break;
        default:
          throw std::runtime_error("loadWav: unsupported PCM bit depth");
        }
      } else if (format == 3) { // IEEE float
        if (bits == 32) {
          float x;
          std::memcpy(&x, s, 4);
          v = x;
        } else if (bits == 64) {
          double x;
          std::memcpy(&x, s, 8);
          v = x;
        } else {
          throw std::runtime_error("loadWav: unsupported float bit depth");
        }
      } else {
        throw std::runtime_error("loadWav: unsupported WAVE format " +
                                 std::to_string(format));
      }
      acc += v;
    }
    out[i] = static_cast<float>(acc / channels);
  }
  sample_rate = static_cast<int>(rate);
  return out;
}

std::vector<float> resampleLinear(const std::vector<float> &x, int from_rate,
                                  int to_rate) {
  if (from_rate == to_rate || x.empty())
    return x;
  const double ratio = static_cast<double>(from_rate) / to_rate;
  const size_t n = static_cast<size_t>(std::floor(x.size() / ratio));
  std::vector<float> y(n);
  for (size_t i = 0; i < n; ++i) {
    const double p = i * ratio;
    const size_t i0 = static_cast<size_t>(p);
    const size_t i1 = std::min(i0 + 1, x.size() - 1);
    const double fr = p - i0;
    y[i] = static_cast<float>(x[i0] * (1.0 - fr) + x[i1] * fr);
  }
  return y;
}

LogMelFrontend::LogMelFrontend(const FrontendConfig &cfg) : cfg_(cfg) {
  const int n_fft = cfg_.n_fft;
  if (n_fft <= 0 || (n_fft & (n_fft - 1)) != 0)
    throw std::invalid_argument("LogMelFrontend: n_fft must be a power of 2");
  if (cfg_.win_length > n_fft)
    throw std::invalid_argument("LogMelFrontend: win_length > n_fft");

  // symmetric Hann window (torch.hann_window(periodic=False)), zero padded on
  // both sides to n_fft as torch.stft does
  window_.assign(n_fft, 0.0f);
  const int left = (n_fft - cfg_.win_length) / 2;
  for (int n = 0; n < cfg_.win_length; ++n) {
    float w = cfg_.win_length > 1
                ? static_cast<float>(
                    0.5 - 0.5 * std::cos(2.0 * kPi * n / (cfg_.win_length - 1)))
                : 1.0f;
    window_[left + n] = cfg_.bf16_constants ? roundBF16(w) : w;
  }

  // librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, norm="slaney")
  const int n_bins = n_fft / 2 + 1;
  const int n_mels = cfg_.num_mel_bins;
  const double fmax = cfg_.sampling_rate / 2.0;
  std::vector<double> mel_f(n_mels + 2);
  const double mmin = hzToMel(0.0), mmax = hzToMel(fmax);
  for (int i = 0; i < n_mels + 2; ++i)
    mel_f[i] = melToHz(mmin + (mmax - mmin) * i / (n_mels + 1));
  mel_filters_.assign(static_cast<size_t>(n_mels) * n_bins, 0.0f);
  for (int m = 0; m < n_mels; ++m) {
    const double fdiff_lo = mel_f[m + 1] - mel_f[m];
    const double fdiff_hi = mel_f[m + 2] - mel_f[m + 1];
    const double enorm = 2.0 / (mel_f[m + 2] - mel_f[m]);
    for (int k = 0; k < n_bins; ++k) {
      const double fk = static_cast<double>(k) * cfg_.sampling_rate / n_fft;
      const double lower = -(mel_f[m] - fk) / fdiff_lo;
      const double upper = (mel_f[m + 2] - fk) / fdiff_hi;
      const float w = static_cast<float>(std::max(0.0, std::min(lower, upper)));
      float v = static_cast<float>(w * enorm);
      mel_filters_[static_cast<size_t>(m) * n_bins + k] =
        cfg_.bf16_constants ? roundBF16(v) : v;
    }
  }

  // radix-2 FFT tables
  int log2n = 0;
  while ((1 << log2n) < n_fft)
    ++log2n;
  bitrev_.resize(n_fft);
  for (int i = 0; i < n_fft; ++i) {
    unsigned int r = 0;
    for (int b = 0; b < log2n; ++b)
      if (i & (1 << b))
        r |= 1u << (log2n - 1 - b);
    bitrev_[i] = r;
  }
  twiddle_re_.resize(n_fft / 2);
  twiddle_im_.resize(n_fft / 2);
  for (int k = 0; k < n_fft / 2; ++k) {
    twiddle_re_[k] = std::cos(-2.0 * kPi * k / n_fft);
    twiddle_im_[k] = std::sin(-2.0 * kPi * k / n_fft);
  }
}

void LogMelFrontend::fft(std::vector<double> &re,
                         std::vector<double> &im) const {
  const unsigned int n = static_cast<unsigned int>(re.size());
  for (unsigned int i = 0; i < n; ++i)
    if (bitrev_[i] > i) {
      std::swap(re[i], re[bitrev_[i]]);
      std::swap(im[i], im[bitrev_[i]]);
    }
  for (unsigned int len = 2; len <= n; len <<= 1) {
    const unsigned int half = len / 2;
    const unsigned int step = n / len;
    for (unsigned int i = 0; i < n; i += len)
      for (unsigned int j = 0; j < half; ++j) {
        const double wr = twiddle_re_[j * step], wi = twiddle_im_[j * step];
        const double xr = re[i + j + half] * wr - im[i + j + half] * wi;
        const double xi = re[i + j + half] * wi + im[i + j + half] * wr;
        re[i + j + half] = re[i + j] - xr;
        im[i + j + half] = im[i + j] - xi;
        re[i + j] += xr;
        im[i + j] += xi;
      }
  }
}

MelFeatures LogMelFrontend::compute(const std::vector<float> &audio) const {
  const int n_fft = cfg_.n_fft;
  const int hop = cfg_.hop_length;
  const int n_bins = n_fft / 2 + 1;
  const int n_mels = cfg_.num_mel_bins;
  const long len = static_cast<long>(audio.size());

  // pre-emphasis (first sample kept)
  std::vector<float> x(audio.size());
  if (!x.empty()) {
    x[0] = audio[0];
    for (long i = 1; i < len; ++i)
      x[i] = audio[i] - cfg_.preemphasis * audio[i - 1];
  }

  // centered STFT with zero padding of n_fft / 2 on both sides
  const long pad = n_fft / 2;
  const int total_frames = static_cast<int>(1 + len / hop);
  const int valid = static_cast<int>((len + 2 * pad - n_fft) / hop);
  const int m = std::max(1, cfg_.pad_to_multiple);
  const int padded = ((total_frames + m - 1) / m) * m;

  MelFeatures out;
  out.num_mel_bins = n_mels;
  out.num_frames = padded;
  out.num_valid = std::max(0, std::min(valid, total_frames));
  out.data.assign(static_cast<size_t>(padded) * n_mels, 0.0f);

  std::vector<double> re(n_fft), im(n_fft);
  std::vector<float> power(n_bins);
  for (int t = 0; t < out.num_valid; ++t) {
    const long start = static_cast<long>(t) * hop - pad;
    for (int n = 0; n < n_fft; ++n) {
      const long idx = start + n;
      const float s = (idx >= 0 && idx < len) ? x[idx] : 0.0f;
      re[n] = static_cast<double>(s) * window_[n];
      im[n] = 0.0;
    }
    fft(re, im);
    for (int k = 0; k < n_bins; ++k)
      power[k] = static_cast<float>(re[k] * re[k] + im[k] * im[k]);
    float *row = out.data.data() + static_cast<size_t>(t) * n_mels;
    for (int mm = 0; mm < n_mels; ++mm) {
      const float *filt =
        mel_filters_.data() + static_cast<size_t>(mm) * n_bins;
      double acc = 0.0;
      for (int k = 0; k < n_bins; ++k)
        acc += static_cast<double>(filt[k]) * power[k];
      row[mm] = std::log(static_cast<float>(acc) + kLogGuard);
    }
  }
  return out;
}

} // namespace nemotron_transcribe
} // namespace causallm
