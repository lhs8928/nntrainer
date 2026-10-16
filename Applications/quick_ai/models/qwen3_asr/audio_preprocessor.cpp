// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   audio_preprocessor.cpp
 * @date   14 September 2026
 * @brief  Lightweight C++ WAV decoder and log-Mel spectrogram feature extractor for Qwen3-ASR
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 */

#include "audio_preprocessor.h"
#define MINIMP3_IMPLEMENTATION
#include "minimp3_ex.h"
#include <fstream>
#include <cmath>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cstdint>

namespace quick_ai {

std::vector<float> AudioPreprocessor::loadWav(const std::string &file_path) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open WAV file: " + file_path);
  }

  // 1. Read RIFF header
  char riff[4];
  uint32_t chunk_size;
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

  // 2. Loop over sub-chunks robustly
  char sub_chunk_id[4];
  uint32_t sub_chunk_size;

  while (file.read(sub_chunk_id, 4) && file.read(reinterpret_cast<char*>(&sub_chunk_size), 4)) {
    if (std::strncmp(sub_chunk_id, "fmt ", 4) == 0) {
      uint16_t audio_format;
      file.read(reinterpret_cast<char*>(&audio_format), 2);
      file.read(reinterpret_cast<char*>(&num_channels), 2);
      file.read(reinterpret_cast<char*>(&sample_rate), 4);
      
      // skip byte_rate (4) + block_align (2)
      file.seekg(6, std::ios::cur);
      file.read(reinterpret_cast<char*>(&bits_per_sample), 2);

      if (audio_format != 1) {
        throw std::runtime_error("Unsupported WAV format: only uncompressed PCM (1) is supported.");
      }

      // Skip remaining format bytes if sub_chunk_size is larger than 16 (e.g. 18 for fmt extension)
      if (sub_chunk_size > 16) {
        file.seekg(sub_chunk_size - 16, std::ios::cur);
      }
    } else if (std::strncmp(sub_chunk_id, "data", 4) == 0) {
      data_size = sub_chunk_size;
      raw_data.resize(data_size);
      file.read(raw_data.data(), data_size);
      break; // stop reading after data chunk
    } else {
      // Unknown chunk (like LIST, JUNK, BEXT, etc.), skip its payload safely
      file.seekg(sub_chunk_size, std::ios::cur);
    }
  }

  if (raw_data.empty()) {
    throw std::runtime_error("No 'data' chunk found in WAV file: " + file_path);
  }

  unsigned int bytes_per_sample = bits_per_sample / 8;
  unsigned int num_samples = data_size / (num_channels * bytes_per_sample);

  std::vector<float> pcm_mono_16k;
  pcm_mono_16k.reserve(num_samples);

  unsigned int stride = 1;
  if (sample_rate == 48000) {
    stride = 3; // downsample 48kHz to 16kHz
  } else if (sample_rate != 16000) {
    throw std::runtime_error("Unsupported sample rate: " + std::to_string(sample_rate) + "Hz. Only 16kHz or 48kHz are supported.");
  }

  for (unsigned int i = 0; i < num_samples; i += stride) {
    float sample_avg = 0.0f;
    for (unsigned int c = 0; c < num_channels; ++c) {
      size_t byte_idx = (i * num_channels + c) * bytes_per_sample;
      if (byte_idx + bytes_per_sample > raw_data.size()) {
        break;
      }

      float sample_val = 0.0f;
      if (bits_per_sample == 16) {
        int16_t val;
        std::memcpy(&val, &raw_data[byte_idx], 2);
        sample_val = val / 32768.0f;
      } else if (bits_per_sample == 24) {
        int32_t val = 0;
        uint8_t b0 = (uint8_t)raw_data[byte_idx];
        uint8_t b1 = (uint8_t)raw_data[byte_idx + 1];
        uint8_t b2 = (uint8_t)raw_data[byte_idx + 2];
        val = b0 | (b1 << 8) | (b2 << 16);
        if (val & 0x800000) {
          val |= 0xFF000000;
        }
        sample_val = val / 8388608.0f;
      } else if (bits_per_sample == 32) {
        float val;
        std::memcpy(&val, &raw_data[byte_idx], 4);
        sample_val = val;
      }
      sample_avg += sample_val;
    }
    pcm_mono_16k.push_back(sample_avg / num_channels);
  }

  return pcm_mono_16k;
}

std::vector<float> AudioPreprocessor::loadMp3(const std::string &file_path) {
  mp3dec_t mp3d;
  mp3dec_file_info_t info;

  if (mp3dec_load(&mp3d, file_path.c_str(), &info, NULL, NULL)) {
    throw std::runtime_error("Failed to decode MP3 file: " + file_path);
  }

  if (info.samples == 0 || info.channels == 0 || info.hz == 0) {
    if (info.buffer) free(info.buffer);
    throw std::runtime_error("Invalid or empty MP3 stream in: " + file_path);
  }

  size_t total_frames = info.samples / info.channels;
  std::vector<float> pcm_mono(total_frames);

  // Convert to Mono float in [-1.0, 1.0]
  if (info.channels == 1) {
    for (size_t i = 0; i < total_frames; ++i) {
      pcm_mono[i] = static_cast<float>(info.buffer[i]) / 32768.0f;
    }
  } else if (info.channels == 2) {
    for (size_t i = 0; i < total_frames; ++i) {
      float left = static_cast<float>(info.buffer[2 * i]) / 32768.0f;
      float right = static_cast<float>(info.buffer[2 * i + 1]) / 32768.0f;
      pcm_mono[i] = 0.5f * (left + right);
    }
  } else {
    float inv_ch = 1.0f / static_cast<float>(info.channels);
    for (size_t i = 0; i < total_frames; ++i) {
      float sum = 0.0f;
      for (int c = 0; c < info.channels; ++c) {
        sum += static_cast<float>(info.buffer[i * info.channels + c]);
      }
      pcm_mono[i] = (sum * inv_ch) / 32768.0f;
    }
  }

  int orig_hz = info.hz;
  int orig_ch = info.channels;
  free(info.buffer);

  // Resample to 16,000 Hz if necessary
  if (orig_hz == 16000) {
    return pcm_mono;
  }

  double ratio = 16000.0 / static_cast<double>(orig_hz);
  size_t target_samples = static_cast<size_t>(std::round(static_cast<double>(total_frames) * ratio));
  std::vector<float> resampled(target_samples);

  double step = static_cast<double>(orig_hz) / 16000.0;
  for (size_t i = 0; i < target_samples; ++i) {
    double src_pos = static_cast<double>(i) * step;
    size_t idx0 = static_cast<size_t>(src_pos);
    size_t idx1 = std::min(idx0 + 1, total_frames - 1);
    double frac = src_pos - static_cast<double>(idx0);

    resampled[i] = static_cast<float>((1.0 - frac) * pcm_mono[idx0] + frac * pcm_mono[idx1]);
  }

  std::cout << "[AudioPreprocessor] MP3 decoded: " << file_path << " (" << orig_hz << " Hz, "
            << orig_ch << " ch) -> Resampled to 16000 Hz (" << target_samples << " samples, "
            << (static_cast<double>(target_samples) / 16000.0) << " s)" << std::endl;

  return resampled;
}

std::vector<float> AudioPreprocessor::loadAudio(const std::string &file_path) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open audio file: " + file_path);
  }
  char magic[4] = {0};
  file.read(magic, 4);
  file.close();

  if (std::strncmp(magic, "RIFF", 4) == 0) {
    return loadWav(file_path);
  } else if (std::strncmp(magic, "ID3", 3) == 0 ||
             (static_cast<uint8_t>(magic[0]) == 0xFF && (static_cast<uint8_t>(magic[1]) & 0xE0) == 0xE0)) {
    return loadMp3(file_path);
  }

  // Fallback to extension check
  std::string ext = "";
  size_t dot_pos = file_path.rfind('.');
  if (dot_pos != std::string::npos) {
    ext = file_path.substr(dot_pos);
    for (auto &c : ext) c = std::tolower(c);
  }

  if (ext == ".mp3") {
    return loadMp3(file_path);
  }
  return loadWav(file_path);
}

void AudioPreprocessor::applyHannWindow(const float *input, float *output, unsigned int N) {
  for (unsigned int i = 0; i < N; ++i) {
    // Hann window equation matching PyTorch/librosa periodic=True:
    // w[n] = 0.5 * (1 - cos(2 * pi * n / N))
    float w = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / N));
    output[i] = input[i] * w;
  }
}

float AudioPreprocessor::hzToSlaneyMel(float freq) {
  if (freq < 1000.0f) {
    return freq / (200.0f / 3.0f);
  }
  return 15.0f + std::log(freq / 1000.0f) / (std::log(6.4f) / 27.0f);
}

float AudioPreprocessor::slaneyMelToHz(float mel) {
  if (mel < 15.0f) {
    return mel * (200.0f / 3.0f);
  }
  return 1000.0f * std::exp((mel - 15.0f) * (std::log(6.4f) / 27.0f));
}

std::vector<float> AudioPreprocessor::generateMelFilterBank(unsigned int num_mel_bins,
                                                            unsigned int num_fft_bins,
                                                            float sampling_rate) {
  std::vector<float> mel_filters(num_mel_bins * num_fft_bins, 0.0f);

  float min_mel = hzToSlaneyMel(0.0f);
  float max_mel = hzToSlaneyMel(sampling_rate / 2.0f);

  std::vector<float> mel_points(num_mel_bins + 2);
  for (unsigned int i = 0; i < num_mel_bins + 2; ++i) {
    mel_points[i] = min_mel + i * (max_mel - min_mel) / (num_mel_bins + 1);
  }

  std::vector<float> hz_points(num_mel_bins + 2);
  for (unsigned int i = 0; i < num_mel_bins + 2; ++i) {
    hz_points[i] = slaneyMelToHz(mel_points[i]);
  }

  // Mapping frequencies to FFT bins
  float fft_bin_width = sampling_rate / (2.0f * (num_fft_bins - 1));

  for (unsigned int j = 0; j < num_mel_bins; ++j) {
    float f_low = hz_points[j];
    float f_center = hz_points[j + 1];
    float f_high = hz_points[j + 2];

    float norm_factor = 2.0f / (f_high - f_low);

    for (unsigned int k = 0; k < num_fft_bins; ++k) {
      float bin_freq = k * fft_bin_width;
      float weight = 0.0f;

      if (bin_freq >= f_low && bin_freq <= f_high) {
        if (bin_freq <= f_center) {
          weight = (bin_freq - f_low) / (f_center - f_low);
        } else {
          weight = (f_high - bin_freq) / (f_high - f_center);
        }
        mel_filters[j * num_fft_bins + k] = weight * norm_factor;
      }
    }
  }

  return mel_filters;
}

std::vector<float> AudioPreprocessor::computeMelSpectrogram(const std::vector<float> &waveform,
                                                            unsigned int num_mel_bins,
                                                            unsigned int n_fft,
                                                            unsigned int hop_length) {
  unsigned int num_fft_bins = 1 + n_fft / 2; // e.g. 201

  // Precompute cos/sin tables for O(N^2) unpadded 400-point DFT to ensure 100% exact numerical match
  std::vector<float> cos_table(n_fft * num_fft_bins);
  std::vector<float> sin_table(n_fft * num_fft_bins);
  for (unsigned int k = 0; k < num_fft_bins; ++k) {
    for (unsigned int n = 0; n < n_fft; ++n) {
      float angle = 2.0f * M_PI * k * n / n_fft;
      cos_table[k * n_fft + n] = std::cos(angle);
      sin_table[k * n_fft + n] = std::sin(angle);
    }
  }

  // Precompute Mel-filter bank matrix
  auto mel_filters = generateMelFilterBank(num_mel_bins, num_fft_bins, 16000.0f);

  // Padding waveform at start and end matching PyTorch STFT behavior (Reflect padding)
  // Whisper uses reflect padding of size n_fft // 2 (200) on both sides
  unsigned int pad_size = n_fft / 2;
  std::vector<float> padded_waveform(waveform.size() + 2 * pad_size);
  
  // Reflect padding left
  for (unsigned int i = 0; i < pad_size; ++i) {
    padded_waveform[pad_size - 1 - i] = waveform[i + 1];
  }
  // Copy center
  std::memcpy(&padded_waveform[pad_size], waveform.data(), waveform.size() * sizeof(float));
  // Reflect padding right
  for (unsigned int i = 0; i < pad_size; ++i) {
    padded_waveform[padded_waveform.size() - pad_size + i] = waveform[waveform.size() - 2 - i];
  }

  unsigned int num_frames = (padded_waveform.size() - n_fft) / hop_length + 1;
  unsigned int out_frames = num_frames - 1; // skip the absolute last frame to match HF log_spec[:, :-1]
  std::vector<float> mel_spectrogram(num_mel_bins * out_frames, 0.0f);

  // Compute spectrogram frame by frame
  std::vector<float> frame_windowed(n_fft);
  std::vector<float> power_spectrum(num_fft_bins);

  for (unsigned int f = 0; f < out_frames; ++f) {
    size_t start_sample = f * hop_length;
    applyHannWindow(&padded_waveform[start_sample], frame_windowed.data(), n_fft);

    // Run DFT directly using precomputed tables (highly optimized)
    for (unsigned int k = 0; k < num_fft_bins; ++k) {
      float real = 0.0f;
      float imag = 0.0f;
      size_t table_offset = k * n_fft;

      for (unsigned int n = 0; n < n_fft; ++n) {
        real += frame_windowed[n] * cos_table[table_offset + n];
        imag -= frame_windowed[n] * sin_table[table_offset + n];
      }
      power_spectrum[k] = real * real + imag * imag;
    }

    // Multiply by Mel-filter bank
    for (unsigned int m = 0; m < num_mel_bins; ++m) {
      float mel_val = 0.0f;
      size_t filter_offset = m * num_fft_bins;
      for (unsigned int k = 0; k < num_fft_bins; ++k) {
        mel_val += power_spectrum[k] * mel_filters[filter_offset + k];
      }

      // Log scaling: log10(max(mel, 1e-10))
      float log_mel = std::log10(std::max(mel_val, 1e-10f));
      
      // Store in row-major layout (out_frames x num_mel_bins)
      mel_spectrogram[f * num_mel_bins + m] = log_mel;
    }
  }

  // Final Normalization: Clamping and Scaling matching Whisper specification
  // 1. Clamp to log_spec.max() - 8.0
  float max_val = -100.0f;
  for (float val : mel_spectrogram) {
    if (val > max_val) {
      max_val = val;
    }
  }

  float clamp_threshold = max_val - 8.0f;
  for (size_t i = 0; i < mel_spectrogram.size(); ++i) {
    mel_spectrogram[i] = std::max(mel_spectrogram[i], clamp_threshold);
    // 2. Scale: (log_mel + 4.0) / 4.0
    mel_spectrogram[i] = (mel_spectrogram[i] + 4.0f) / 4.0f;
  }

  return mel_spectrogram;
}

} // namespace quick_ai
