// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   audio_preprocessor.cpp
 * @date   29 September 2026
 * @brief  WAV loader and Kaldi Fbank implementation for Speaker Diarization
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "audio_preprocessor.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace speaker_diarization {

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Fallback 41-tap torchaudio sinc_interp_hann filter for 48kHz -> 16kHz
static const float DEFAULT_RESAMPLE_KERNEL[41] = {
  1.5955707744987805e-24f, -8.175343282346148e-07f, -0.0001830169785534963f,
  -0.0005382237141020596f, 0.00024459161795675755f, 0.002641299506649375f,
  0.0036252611316740513f, -0.0008614701800979674f, -0.008977459743618965f,
  -0.010861670598387718f, 0.00169034069404006f, 0.02137400023639202f,
  0.025451896712183952f, -0.0025134084280580282f, -0.046781014651060104f,
  -0.05947994068264961f, 0.0031138660851866007f, 0.13534589111804962f,
  0.27194276452064514f, 0.33000001311302185f, 0.27194276452064514f,
  0.13534589111804962f, 0.0031138660851866007f, -0.05947994068264961f,
  -0.046781014651060104f, -0.0025134084280580282f, 0.025451896712183952f,
  0.02137400023639202f, 0.00169034069404006f, -0.010861670598387718f,
  -0.008977459743618965f, -0.0008614701800979674f, 0.0036252611316740513f,
  0.002641299506649375f, 0.00024459161795675755f, -0.0005382237141020596f,
  -0.0001830169785534963f, -8.175343282346148e-07f, 1.5955707744987805e-24f,
  1.5955707744987805e-24f, 1.5955707744987805e-24f
};

std::vector<float> AudioPreprocessor::loadWav(const std::string &file_path, const float *resample_kernel) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open WAV file: " + file_path);
  }

  char riff[4];
  uint32_t chunk_size = 0;
  char wave[4];

  if (!file.read(riff, 4) || !file.read(reinterpret_cast<char*>(&chunk_size), 4) || !file.read(wave, 4)) {
    throw std::runtime_error("Failed to read RIFF chunk: " + file_path);
  }

  if (std::strncmp(riff, "RIFF", 4) != 0 || std::strncmp(wave, "WAVE", 4) != 0) {
    throw std::runtime_error("Invalid RIFF WAVE file: " + file_path);
  }

  uint16_t num_channels = 1;
  uint32_t sample_rate = 16000;
  uint16_t bits_per_sample = 16;
  uint32_t data_size = 0;
  std::vector<char> raw_data;

  char sub_chunk_id[4];
  uint32_t sub_chunk_size = 0;

  while (file.read(sub_chunk_id, 4) && file.read(reinterpret_cast<char*>(&sub_chunk_size), 4)) {
    if (std::strncmp(sub_chunk_id, "fmt ", 4) == 0) {
      uint16_t audio_format;
      file.read(reinterpret_cast<char*>(&audio_format), 2);
      file.read(reinterpret_cast<char*>(&num_channels), 2);
      file.read(reinterpret_cast<char*>(&sample_rate), 4);
      file.seekg(6, std::ios::cur); // skip byte_rate + block_align
      file.read(reinterpret_cast<char*>(&bits_per_sample), 2);

      if (audio_format != 1 && audio_format != 3) {
        throw std::runtime_error("Unsupported WAV audio format: " + std::to_string(audio_format));
      }
      if (sub_chunk_size > 16) {
        file.seekg(sub_chunk_size - 16, std::ios::cur);
      }
    } else if (std::strncmp(sub_chunk_id, "data", 4) == 0) {
      data_size = sub_chunk_size;
      raw_data.resize(data_size);
      file.read(raw_data.data(), data_size);
      break;
    } else {
      file.seekg(sub_chunk_size, std::ios::cur);
    }
  }

  if (raw_data.empty()) {
    throw std::runtime_error("No 'data' chunk found in: " + file_path);
  }

  size_t bytes_per_sample = bits_per_sample / 8;
  size_t total_samples = data_size / (num_channels * bytes_per_sample);
  std::vector<float> mono_pcm(total_samples);

  for (size_t i = 0; i < total_samples; ++i) {
    float sum_ch = 0.0f;
    for (size_t c = 0; c < num_channels; ++c) {
      size_t idx = (i * num_channels + c) * bytes_per_sample;
      if (bits_per_sample == 16) {
        int16_t val;
        std::memcpy(&val, &raw_data[idx], 2);
        sum_ch += val / 32768.0f;
      } else if (bits_per_sample == 24) {
        int32_t val = 0;
        uint8_t b0 = static_cast<uint8_t>(raw_data[idx]);
        uint8_t b1 = static_cast<uint8_t>(raw_data[idx + 1]);
        uint8_t b2 = static_cast<uint8_t>(raw_data[idx + 2]);
        val = b0 | (b1 << 8) | (b2 << 16);
        if (val & 0x800000) val |= 0xFF000000;
        sum_ch += val / 8388608.0f;
      } else if (bits_per_sample == 32) {
        float val;
        std::memcpy(&val, &raw_data[idx], 4);
        sum_ch += val;
      }
    }
    mono_pcm[i] = sum_ch / num_channels;
  }

  if (sample_rate == 48000) {
    const float *k = resample_kernel ? resample_kernel : DEFAULT_RESAMPLE_KERNEL;
    return resample48kTo16k(mono_pcm, k);
  } else if (sample_rate == 16000) {
    return mono_pcm;
  } else {
    throw std::runtime_error("Unsupported sample rate: " + std::to_string(sample_rate) + " Hz. Only 16kHz or 48kHz supported.");
  }
}

std::vector<float> AudioPreprocessor::resample48kTo16k(const std::vector<float> &input_48k, const float *kernel) {
  size_t in_len = input_48k.size();
  if (in_len == 0) return {};
  size_t target_len = static_cast<size_t>(std::ceil(static_cast<double>(in_len) / 3.0));
  std::vector<float> out(target_len, 0.0f);

  // torchaudio padding: left = 19, right = 19 + 3 = 22
  size_t pad_left = 19;
  size_t pad_right = 22;
  size_t padded_len = in_len + pad_left + pad_right;
  std::vector<float> padded(padded_len, 0.0f);
  std::copy(input_48k.begin(), input_48k.end(), padded.begin() + pad_left);

  // 1D conv with kernel size 41, stride 3
  for (size_t out_idx = 0; out_idx < target_len; ++out_idx) {
    size_t in_start = out_idx * 3;
    float acc = 0.0f;
    for (size_t k = 0; k < 41; ++k) {
      acc += padded[in_start + k] * kernel[k];
    }
    out[out_idx] = acc;
  }

  return out;
}

std::vector<std::vector<float>> AudioPreprocessor::sliceChunks(const std::vector<float> &waveform) {
  const size_t WINDOW_SIZE = 160000; // 10.0s @ 16kHz
  const size_t STEP_SIZE = 16000;    // 1.0s @ 16kHz
  size_t num_samples = waveform.size();

  std::vector<std::vector<float>> chunks;
  if (num_samples == 0) return chunks;

  size_t num_complete_chunks = 0;
  if (num_samples >= WINDOW_SIZE) {
    num_complete_chunks = (num_samples - WINDOW_SIZE) / STEP_SIZE + 1;
  }

  for (size_t c = 0; c < num_complete_chunks; ++c) {
    size_t start = c * STEP_SIZE;
    std::vector<float> chunk(waveform.begin() + start, waveform.begin() + start + WINDOW_SIZE);
    chunks.push_back(std::move(chunk));
  }

  // Last incomplete chunk
  bool has_last_chunk = (num_samples < WINDOW_SIZE) || ((num_samples - WINDOW_SIZE) % STEP_SIZE > 0);
  if (has_last_chunk) {
    size_t start = num_complete_chunks * STEP_SIZE;
    std::vector<float> last_chunk(WINDOW_SIZE, 0.0f);
    if (start < num_samples) {
      size_t remaining = num_samples - start;
      std::memcpy(last_chunk.data(), waveform.data() + start, remaining * sizeof(float));
    }
    chunks.push_back(std::move(last_chunk));
  }

  return chunks;
}

void AudioPreprocessor::fft512(float *real, float *imag) {
  constexpr int N = 512;
  // Bit reversal
  int j = 0;
  for (int i = 0; i < N - 1; ++i) {
    if (i < j) {
      std::swap(real[i], real[j]);
      std::swap(imag[i], imag[j]);
    }
    int k = N / 2;
    while (k <= j) {
      j -= k;
      k /= 2;
    }
    j += k;
  }

  // Cooley-Tukey Radix-2
  for (int len = 2; len <= N; len <<= 1) {
    float angle = static_cast<float>(-2.0 * M_PI / len);
    float wlen_r = std::cos(angle);
    float wlen_i = std::sin(angle);
    int half_len = len >> 1;

    for (int i = 0; i < N; i += len) {
      float wr = 1.0f;
      float wi = 0.0f;
      for (int k = 0; k < half_len; ++k) {
        float tr = wr * real[i + k + half_len] - wi * imag[i + k + half_len];
        float ti = wr * imag[i + k + half_len] + wi * real[i + k + half_len];

        real[i + k + half_len] = real[i + k] - tr;
        imag[i + k + half_len] = imag[i + k] - ti;
        real[i + k] += tr;
        imag[i + k] += ti;

        float next_wr = wr * wlen_r - wi * wlen_i;
        float next_wi = wr * wlen_i + wi * wlen_r;
        wr = next_wr;
        wi = next_wi;
      }
    }
  }
}

std::vector<float> AudioPreprocessor::computeFbank(const float *chunk, size_t num_samples,
                                                   const float *mel_banks, const float *window) {
  constexpr size_t FRAME_LEN = 400; // 25ms @ 16kHz
  constexpr size_t HOP_LEN = 160;   // 10ms @ 16kHz
  constexpr size_t N_FFT = 512;
  constexpr size_t NUM_MEL_BINS = 80;

  if (num_samples < FRAME_LEN) {
    return {};
  }

  size_t num_frames = (num_samples - FRAME_LEN) / HOP_LEN + 1; // 998 for 160,000
  std::vector<float> fbank(num_frames * NUM_MEL_BINS, 0.0f);

  float real[N_FFT];
  float imag[N_FFT];
  float power[256];

  for (size_t f = 0; f < num_frames; ++f) {
    const float *frame_src = chunk + f * HOP_LEN;

    // 1. Scale by 32768.0 and remove DC offset
    float mean_val = 0.0f;
    for (size_t i = 0; i < FRAME_LEN; ++i) {
      mean_val += frame_src[i] * 32768.0f;
    }
    mean_val /= FRAME_LEN;

    float frame_buf[FRAME_LEN];
    for (size_t i = 0; i < FRAME_LEN; ++i) {
      frame_buf[i] = frame_src[i] * 32768.0f - mean_val;
    }

    // 2. Preemphasis: frame[0] -= 0.97 * frame[0], frame[i] -= 0.97 * frame[i-1]
    float prev = frame_buf[0];
    frame_buf[0] -= 0.97f * frame_buf[0];
    for (size_t i = 1; i < FRAME_LEN; ++i) {
      float curr = frame_buf[i];
      frame_buf[i] -= 0.97f * prev;
      prev = curr;
    }

    // 3. Windowing & zero padding to 512
    for (size_t i = 0; i < FRAME_LEN; ++i) {
      real[i] = frame_buf[i] * window[i];
      imag[i] = 0.0f;
    }
    for (size_t i = FRAME_LEN; i < N_FFT; ++i) {
      real[i] = 0.0f;
      imag[i] = 0.0f;
    }

    // 4. FFT & Power spectrum for 256 bins
    fft512(real, imag);
    for (size_t k = 0; k < 256; ++k) {
      power[k] = real[k] * real[k] + imag[k] * imag[k];
    }

    // 5. Mel filterbank multiplication & Log
    for (size_t m = 0; m < NUM_MEL_BINS; ++m) {
      const float *mel_row = mel_banks + m * 256;
      float sum = 0.0f;
      for (size_t k = 0; k < 256; ++k) {
        sum += mel_row[k] * power[k];
      }
      fbank[m * num_frames + f] = std::log(std::max(sum, 1.1920929e-7f));
    }
  }

  // 6. Mean-centering across time frames for each mel bin
  for (size_t m = 0; m < NUM_MEL_BINS; ++m) {
    float bin_mean = 0.0f;
    for (size_t f = 0; f < num_frames; ++f) {
      bin_mean += fbank[m * num_frames + f];
    }
    bin_mean /= num_frames;
    for (size_t f = 0; f < num_frames; ++f) {
      fbank[m * num_frames + f] -= bin_mean;
    }
  }

  return fbank;
}

} // namespace speaker_diarization
