// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   audio_preprocessor.h
 * @date   14 September 2026
 * @brief  Lightweight C++ WAV decoder and log-Mel spectrogram feature extractor for Qwen3-ASR
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 */

#ifndef __AUDIO_PREPROCESSOR_H__
#define __AUDIO_PREPROCESSOR_H__
#ifdef __cplusplus

#pragma once
#include <string>
#include <vector>

namespace quick_ai {

/**
 * @class AudioPreprocessor
 * @brief Handles WAV file loading (16kHz mono conversion) and Mel-spectrogram extraction.
 */
class AudioPreprocessor {
public:
  AudioPreprocessor() = default;
  ~AudioPreprocessor() = default;

  /**
   * @brief Load and decode a WAV file, resampling 48kHz down to 16kHz mono if necessary.
   * @param file_path Local path to WAV file.
   * @return std::vector<float> Normalized audio samples [-1.0, 1.0] at 16kHz.
   */
  std::vector<float> loadWav(const std::string &file_path);

  /**
   * @brief Compute the Log-Mel Spectrogram matching Whisper/Qwen feature extractor.
   * @param waveform Input 16kHz audio samples.
   * @param num_mel_bins Number of Mel filters (e.g. 128 for Qwen3-ASR).
   * @param n_fft FFT size (e.g. 400).
   * @param hop_length Hop length in samples (e.g. 160).
   * @return std::vector<float> Log-mel spectrogram flattened in row-major order (num_mel_bins x seq_len).
   */
  std::vector<float> computeMelSpectrogram(const std::vector<float> &waveform,
                                          unsigned int num_mel_bins = 128,
                                          unsigned int n_fft = 400,
                                          unsigned int hop_length = 160);

private:
  /**
   * @brief Apply Hann window to input buffer of size N.
   */
  void applyHannWindow(const float *input, float *output, unsigned int N);

  /**
   * @brief Lightweight Radix-2 Cooley-Tukey FFT implementation.
   * @param real Input/Output real part. Must be size N (N is power of 2, e.g. 512).
   * @param imag Input/Output imaginary part. Must be size N.
   */
  void radix2FFT(std::vector<float> &real, std::vector<float> &imag);

  /**
   * @brief Generate Slaney-style Mel filter bank matrix.
   * @param num_mel_bins Number of Mel filters (e.g., 128).
   * @param num_fft_bins Number of FFT bins (1 + n_fft/2, e.g., 201).
   * @param sampling_rate Audio sampling rate (16000 Hz).
   * @return std::vector<float> Mel filter bank matrix of size (num_mel_bins * num_fft_bins) flat.
   */
  std::vector<float> generateMelFilterBank(unsigned int num_mel_bins,
                                          unsigned int num_fft_bins,
                                          float sampling_rate);

  /**
   * @brief Helper to convert Hz frequency to Slaney Mel scale.
   */
  float hzToSlaneyMel(float freq);

  /**
   * @brief Helper to convert Slaney Mel back to Hz frequency.
   */
  float slaneyMelToHz(float mel);
};

} // namespace quick_ai

#endif // __cplusplus
#endif // __AUDIO_PREPROCESSOR_H__
