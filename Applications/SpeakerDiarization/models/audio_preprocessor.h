// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   audio_preprocessor.h
 * @date   29 September 2026
 * @brief  WAV loader and Kaldi Fbank feature extractor for Speaker Diarization
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __AUDIO_PREPROCESSOR_H__
#define __AUDIO_PREPROCESSOR_H__

#include <string>
#include <vector>
#include <cstdint>

namespace speaker_diarization {

class AudioPreprocessor {
public:
  AudioPreprocessor() = default;
  ~AudioPreprocessor() = default;

  /**
   * @brief Load WAV file and convert to 16kHz mono float samples [-1.0, 1.0].
   * @param file_path Path to WAV file.
   * @param resample_kernel Optional 41-tap sinc/hann resample filter.
   * @return 16kHz normalized audio samples.
   */
  std::vector<float> loadWav(const std::string &file_path, const float *resample_kernel = nullptr);

  /**
   * @brief Slices 16kHz audio into 10.0s chunks (160,000 samples) with 1.0s step (16,000 samples).
   * @param waveform 16kHz mono audio samples.
   * @return Vector of chunks, each chunk of size 160,000 (last chunk zero-padded).
   */
  std::vector<std::vector<float>> sliceChunks(const std::vector<float> &waveform);

  /**
   * @brief Computes Kaldi-compliant 80-bin Fbank for a 160,000-sample audio chunk.
   * @param chunk 160,000 samples at 16kHz.
   * @param mel_banks Pointer to 80x256 mel filter bank matrix.
   * @param window Pointer to 400-point Hamming window.
   * @return Flat vector of size 998 * 80 (frames * mel_bins), mean-centered across frames.
   */
  std::vector<float> computeFbank(const float *chunk, size_t num_samples,
                                  const float *mel_banks, const float *window);

private:
  /**
   * @brief Apply 41-tap sinc/hann filter to resample 48kHz audio to 16kHz.
   */
  std::vector<float> resample48kTo16k(const std::vector<float> &input_48k, const float *kernel);

  /**
   * @brief In-place Radix-2 Cooley-Tukey FFT for N=512.
   */
  void fft512(float *real, float *imag);
};

} // namespace speaker_diarization

#endif // __AUDIO_PREPROCESSOR_H__
