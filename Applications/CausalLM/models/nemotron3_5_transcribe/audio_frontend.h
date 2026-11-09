// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   audio_frontend.h
 * @date   09 November 2026
 * @brief  WAV loading and the log-mel front-end of Nemotron 3.5 Transcribe
 * @see    https://github.com/nntrainer/nntrainer
 * @author Claude <noreply@anthropic.com>
 * @bug    No known bugs except for NYI items
 *
 * Mirrors `NemotronAsrStreamingFeatureExtractor` (transformers): pre-emphasis,
 * centered STFT with a symmetric Hann window, power spectrum, Slaney mel
 * filter bank, natural log with a 2^-24 guard, frames past the audio zeroed.
 */

#ifndef __NEMOTRON_AUDIO_FRONTEND_H__
#define __NEMOTRON_AUDIO_FRONTEND_H__

#include <string>
#include <vector>

namespace causallm {
namespace nemotron_transcribe {

/**
 * @brief read a RIFF/WAVE file (PCM 8/16/24/32 bit or IEEE float 32/64),
 *        downmixed to mono, values in [-1, 1)
 * @param[out] sample_rate sampling rate of the file
 */
std::vector<float> loadWav(const std::string &path, int &sample_rate);

/** @brief linear-interpolation resampler (use only when the file is not
 *         already at the model rate) */
std::vector<float> resampleLinear(const std::vector<float> &x, int from_rate,
                                  int to_rate);

/** @brief log-mel settings */
struct FrontendConfig {
  int sampling_rate = 16000;
  int n_fft = 512;
  int win_length = 400;
  int hop_length = 160;
  int num_mel_bins = 128;
  float preemphasis = 0.97f;
  bool bf16_constants = true; /**< round window / filter bank to bfloat16 */
  int pad_to_multiple = 8;    /**< pad the frame count (subsampling factor) */
};

/** @brief log-mel spectrogram */
struct MelFeatures {
  std::vector<float> data; /**< [num_frames, num_mel_bins] row major */
  int num_frames = 0;      /**< padded to FrontendConfig::pad_to_multiple */
  int num_valid = 0;       /**< frames covered by the audio */
  int num_mel_bins = 0;
};

/**
 * @class LogMelFrontend
 */
class LogMelFrontend {
public:
  explicit LogMelFrontend(const FrontendConfig &cfg);

  MelFeatures compute(const std::vector<float> &audio) const;

  const FrontendConfig &config() const { return cfg_; }

private:
  void fft(std::vector<double> &re, std::vector<double> &im) const;

  FrontendConfig cfg_;
  std::vector<float> window_;      /**< [n_fft], win_length centered */
  std::vector<float> mel_filters_; /**< [num_mel_bins, n_fft / 2 + 1] */
  std::vector<double> twiddle_re_, twiddle_im_;
  std::vector<unsigned int> bitrev_;
};

} // namespace nemotron_transcribe
} // namespace causallm

#endif /* __NEMOTRON_AUDIO_FRONTEND_H__ */
