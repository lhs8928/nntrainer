// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   main.cpp
 * @date   29 September 2026
 * @brief  CLI Application runner for Speaker Diarization in NNTrainer (Batch & Streaming)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <map>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <thread>
#include <atomic>

#if !defined(_WIN32)
#include <sys/resource.h>
#include <unistd.h>
#endif

#include "../models/weight_loader.h"
#include "../models/audio_preprocessor.h"
#include "../models/pyannet.h"
#include "../models/wespeaker_resnet34.h"
#include "../models/plda_vbx.h"
#include "../models/online_speaker_tracker.h"
#include "../models/audio_stream_buffer.h"

using namespace speaker_diarization;

// ==============================================================================
// Memory Profiler (CausalLM-compatible RSS & Peak tracking)
// ==============================================================================
class MemoryProfiler {
public:
  MemoryProfiler() = default;
  ~MemoryProfiler() = default;

  void start() {}
  void stop() {}

  static size_t readVmRssKb() {
#if defined(_WIN32)
    return 0;
#else
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
      if (line.rfind("VmRSS:", 0) == 0) {
        size_t kb = 0;
        if (sscanf(line.c_str(), "VmRSS: %zu kB", &kb) == 1) {
          return kb;
        }
      }
    }
    return 0;
#endif
  }

  static size_t readPeakVmHwmKb() {
#if defined(_WIN32)
    return 0;
#else
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
      if (line.rfind("VmHWM:", 0) == 0) {
        size_t kb = 0;
        if (sscanf(line.c_str(), "VmHWM: %zu kB", &kb) == 1) {
          return kb;
        }
      }
    }
    return 0;
#endif
  }

  static size_t readRssAnonKb() {
#if defined(_WIN32)
    return 0;
#else
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
      if (line.rfind("RssAnon:", 0) == 0) {
        size_t kb = 0;
        if (sscanf(line.c_str(), "RssAnon: %zu kB", &kb) == 1) {
          return kb;
        }
      }
    }
    return 0;
#endif
  }

  static size_t readGetrusageMaxRssKb() {
#if defined(_WIN32)
    return 0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
      return static_cast<size_t>(usage.ru_maxrss);
    }
    return 0;
#endif
  }
};

// ==============================================================================
// Debug Tensor Serialization
// ==============================================================================
struct SavedTensor {
  std::vector<size_t> shape;
  std::vector<float> data;
};

static void write_debug_binary(const std::string &filename, const std::map<std::string, SavedTensor> &tensors) {
  std::ofstream out(filename, std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "Failed to open debug binary file: " << filename << std::endl;
    return;
  }

  uint32_t num_tensors = static_cast<uint32_t>(tensors.size());
  out.write(reinterpret_cast<const char*>(&num_tensors), sizeof(uint32_t));

  for (const auto &pair : tensors) {
    const std::string &name = pair.first;
    const SavedTensor &st = pair.second;

    uint32_t name_len = static_cast<uint32_t>(name.size());
    out.write(reinterpret_cast<const char*>(&name_len), sizeof(uint32_t));
    out.write(name.data(), name_len);

    uint32_t rank = static_cast<uint32_t>(st.shape.size());
    out.write(reinterpret_cast<const char*>(&rank), sizeof(uint32_t));
    for (size_t dim : st.shape) {
      uint32_t d32 = static_cast<uint32_t>(dim);
      out.write(reinterpret_cast<const char*>(&d32), sizeof(uint32_t));
    }

    uint32_t num_elements = static_cast<uint32_t>(st.data.size());
    out.write(reinterpret_cast<const char*>(&num_elements), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(st.data.data()), num_elements * sizeof(float));
  }

  std::cout << "\n[Debug] Dumped " << num_tensors << " intermediate tensors to " << filename << std::endl;
}

static void print_tensor_row_by_row(const std::string &tag, const std::string &name, const float *data, const std::vector<size_t> &shape) {
  std::cout << tag << " " << name << " shape=[";
  for (size_t i = 0; i < shape.size(); ++i) {
    std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "");
  }
  std::cout << "]\n";

  size_t total_elements = 1;
  for (size_t d : shape) total_elements *= d;
  if (total_elements == 0) return;

  size_t width = shape.back();
  size_t height = total_elements / width;

  auto print_row = [&](size_t r) {
    const float *row = data + r * width;
    std::cout << "  row " << std::setw(3) << r << ": ";
    if (width > 8) {
      for (size_t i = 0; i < 3; ++i) {
        std::cout << std::fixed << std::setprecision(6) << row[i] << ", ";
      }
      std::cout << "... ";
      for (size_t i = width - 3; i < width; ++i) {
        std::cout << std::fixed << std::setprecision(6) << row[i] << (i + 1 < width ? ", " : "");
      }
    } else {
      for (size_t i = 0; i < width; ++i) {
        std::cout << std::fixed << std::setprecision(6) << row[i] << (i + 1 < width ? ", " : "");
      }
    }
    std::cout << "\n";
  };

  if (height <= 6) {
    for (size_t r = 0; r < height; ++r) print_row(r);
  } else {
    for (size_t r = 0; r < 3; ++r) print_row(r);
    std::cout << "    ...\n";
    for (size_t r = height - 3; r < height; ++r) print_row(r);
  }
}

// ==============================================================================
// Continuous Streaming Pipeline Execution
// ==============================================================================
static void runStreamingPipeline(
    const std::string &audio_path,
    bool read_stdin,
    double window_sec,
    double step_sec,
    double sim_speed,
    AudioPreprocessor &preprocessor,
    PyanNet &seg_model,
    WeSpeakerResNet34 &emb_model,
    PldaVBx &plda_vbx,
    const float *resample_kernel,
    const float *mel_banks,
    const float *fbank_window) {

  std::cout << "\n============================================================\n";
  std::cout << "  Starting Continuous Audio Streaming Diarization\n";
  std::cout << "  Input Source:    " << (read_stdin ? "stdin (raw PCM / pipe)" : audio_path) << "\n";
  std::cout << "  Window Duration: " << window_sec << " seconds\n";
  std::cout << "  Slide Step:      " << step_sec << " seconds\n";
  std::cout << "============================================================\n\n";

  AudioStreamBuffer stream_buffer(16000, window_sec, step_sec);
  OnlineSpeakerTracker tracker(0.65f, 0.85f);

  std::vector<float> full_audio;
  size_t file_feed_offset = 0;
  if (!read_stdin) {
    full_audio = preprocessor.loadWav(audio_path, resample_kernel);
    std::cout << "[Stream Feeder] Loaded " << full_audio.size() << " samples ("
              << std::fixed << std::setprecision(2) << (full_audio.size() / 16000.0)
              << "s) for continuous streaming simulation.\n\n";
  }

  constexpr size_t CHUNK_FEED_SAMPLES = 16000; // feed 1.0 second per tick
  std::map<std::string, float> last_confirmed_per_spk;
  size_t stream_window_idx = 0;
  std::vector<SpeakerSegment> all_confirmed_segments;

  auto t_stream_start = std::chrono::high_resolution_clock::now();
  double total_pure_model_ms = 0.0;

  bool stream_active = true;
  while (stream_active) {
    // 1. Ingest audio samples
    if (read_stdin) {
      int16_t pcm_buf[CHUNK_FEED_SAMPLES];
      std::cin.read(reinterpret_cast<char*>(pcm_buf), sizeof(pcm_buf));
      std::streamsize bytes_read = std::cin.gcount();
      if (bytes_read <= 0) {
        stream_active = false;
        break;
      }
      size_t samples_read = bytes_read / sizeof(int16_t);
      std::vector<float> float_chunk(samples_read);
      for (size_t i = 0; i < samples_read; ++i) {
        float_chunk[i] = static_cast<float>(pcm_buf[i]) / 32768.0f;
      }
      stream_buffer.pushSamples(float_chunk.data(), float_chunk.size());
    } else {
      if (file_feed_offset >= full_audio.size()) {
        stream_active = false;
        break;
      }
      size_t feed_count = std::min(CHUNK_FEED_SAMPLES, full_audio.size() - file_feed_offset);
      stream_buffer.pushSamples(full_audio.data() + file_feed_offset, feed_count);
      file_feed_offset += feed_count;

      if (sim_speed > 0.0) {
        double sleep_ms = (feed_count / 16000.0) * 1000.0 / sim_speed;
        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<long long>(sleep_ms)));
      }
    }

    // 2. Process windows as they become available
    while (stream_buffer.hasWindow()) {
      std::vector<float> window_audio;
      double window_start_time = 0.0;
      if (!stream_buffer.getWindow(window_audio, window_start_time)) break;

      auto t_win_start = std::chrono::high_resolution_clock::now();

      // Slice window into 10-second chunks with 1-second step
      auto chunks = preprocessor.sliceChunks(window_audio);
      size_t num_chunks = chunks.size();
      constexpr size_t FRAMES_PER_CHUNK = 589;
      constexpr double STEP_SEC = 1.0;
      constexpr double FRAME_STEP_SEC = 0.016875;

      size_t last_chunk_start_frame = static_cast<size_t>(std::round((num_chunks - 1) * STEP_SEC / FRAME_STEP_SEC));
      size_t total_frames = last_chunk_start_frame + FRAMES_PER_CHUNK;

      // PyanNet segmentation
      std::vector<std::vector<float>> segmentations(num_chunks);
      #pragma omp parallel for schedule(dynamic)
      for (size_t c = 0; c < num_chunks; ++c) {
        segmentations[c] = seg_model.forwardChunk(chunks[c].data(), c, nullptr);
      }

      std::vector<uint8_t> speaker_counting(total_frames, 0);
      std::vector<float> frame_sums(total_frames, 0.0f);
      std::vector<float> frame_weights(total_frames, 0.0f);

      for (size_t c = 0; c < num_chunks; ++c) {
        size_t chunk_start_frame = static_cast<size_t>(std::round(c * STEP_SEC / FRAME_STEP_SEC));
        for (size_t f = 0; f < FRAMES_PER_CHUNK; ++f) {
          size_t global_frame = chunk_start_frame + f;
          if (global_frame >= total_frames) break;
          float spk_sum = segmentations[c][f * 3 + 0]
                        + segmentations[c][f * 3 + 1]
                        + segmentations[c][f * 3 + 2];
          frame_sums[global_frame] += spk_sum;
          frame_weights[global_frame] += 1.0f;
        }
      }
      for (size_t f = 0; f < total_frames; ++f) {
        speaker_counting[f] = (frame_weights[f] > 0.0f) ? static_cast<uint8_t>(std::rint(frame_sums[f] / frame_weights[f])) : 0;
      }

      // WeSpeaker ResNet34 embeddings
      std::vector<float> all_embeddings;
      all_embeddings.reserve(num_chunks * 3 * 256);
      std::vector<std::vector<float>> chunk_embeddings(num_chunks, std::vector<float>(3 * 256));

      #pragma omp parallel for schedule(dynamic)
      for (size_t c = 0; c < num_chunks; ++c) {
        auto fbank = preprocessor.computeFbank(chunks[c].data(), chunks[c].size(), mel_banks, fbank_window);
        std::vector<float> chunk_feat(256 * 10 * 125);
        emb_model.forwardBackbone(fbank.data(), chunk_feat.data(), c, nullptr);

        for (size_t s = 0; s < 3; ++s) {
          std::vector<float> spk_mask(589);
          for (size_t f = 0; f < 589; ++f) {
            spk_mask[f] = segmentations[c][f * 3 + s];
          }
          auto emb = emb_model.forwardPool(chunk_feat.data(), spk_mask.data(), c, s, nullptr);
          std::memcpy(chunk_embeddings[c].data() + s * 256, emb.data(), 256 * sizeof(float));
        }
      }

      for (size_t c = 0; c < num_chunks; ++c) {
        all_embeddings.insert(all_embeddings.end(), chunk_embeddings[c].begin(), chunk_embeddings[c].end());
      }

      // Stream Diarization with OnlineSpeakerTracker
      auto window_segments = plda_vbx.diarizeStreaming(
          all_embeddings, segmentations, speaker_counting, total_frames,
          window_start_time, tracker);

      auto t_win_end = std::chrono::high_resolution_clock::now();
      double win_ms = std::chrono::duration<double, std::milli>(t_win_end - t_win_start).count();
      total_pure_model_ms += win_ms;

      // Commit segments up to (window_start_time + step_sec)
      double commit_horizon = window_start_time + step_sec;
      for (const auto &seg : window_segments) {
        float last_t = last_confirmed_per_spk[seg.speaker];
        if (seg.start_time >= last_t && seg.end_time <= commit_horizon + 0.1f) {
          std::cout << "[STREAM " << std::fixed << std::setprecision(1) << window_start_time << "s~"
                    << (window_start_time + window_sec) << "s | win#" << stream_window_idx << "] "
                    << seg.speaker << ": "
                    << std::fixed << std::setprecision(3) << seg.start_time << "s ~ "
                    << seg.end_time << "s (" << seg.duration << "s)\n" << std::flush;
          all_confirmed_segments.push_back(seg);
          last_confirmed_per_spk[seg.speaker] = seg.end_time;
        }
      }

      stream_window_idx++;
      stream_buffer.advanceStep();
    }
  }

  // Flush remaining speech at end of stream
  std::vector<float> remaining_audio;
  double remaining_start = 0.0;
  if (stream_buffer.flushRemaining(remaining_audio, remaining_start)) {
    auto chunks = preprocessor.sliceChunks(remaining_audio);
    size_t num_chunks = chunks.size();
    if (num_chunks > 0) {
      constexpr size_t FRAMES_PER_CHUNK = 589;
      constexpr double STEP_SEC = 1.0;
      constexpr double FRAME_STEP_SEC = 0.016875;
      size_t last_chunk_start_frame = static_cast<size_t>(std::round((num_chunks - 1) * STEP_SEC / FRAME_STEP_SEC));
      size_t total_frames = last_chunk_start_frame + FRAMES_PER_CHUNK;

      std::vector<std::vector<float>> segmentations(num_chunks);
      #pragma omp parallel for schedule(dynamic)
      for (size_t c = 0; c < num_chunks; ++c) {
        segmentations[c] = seg_model.forwardChunk(chunks[c].data(), c, nullptr);
      }

      std::vector<uint8_t> speaker_counting(total_frames, 0);
      std::vector<float> frame_sums(total_frames, 0.0f);
      std::vector<float> frame_weights(total_frames, 0.0f);
      for (size_t c = 0; c < num_chunks; ++c) {
        size_t chunk_start_frame = static_cast<size_t>(std::round(c * STEP_SEC / FRAME_STEP_SEC));
        for (size_t f = 0; f < FRAMES_PER_CHUNK; ++f) {
          size_t global_frame = chunk_start_frame + f;
          if (global_frame >= total_frames) break;
          float spk_sum = segmentations[c][f * 3 + 0] + segmentations[c][f * 3 + 1] + segmentations[c][f * 3 + 2];
          frame_sums[global_frame] += spk_sum;
          frame_weights[global_frame] += 1.0f;
        }
      }
      for (size_t f = 0; f < total_frames; ++f) {
        speaker_counting[f] = (frame_weights[f] > 0.0f) ? static_cast<uint8_t>(std::rint(frame_sums[f] / frame_weights[f])) : 0;
      }

      std::vector<float> all_embeddings;
      all_embeddings.reserve(num_chunks * 3 * 256);
      std::vector<std::vector<float>> chunk_embeddings(num_chunks, std::vector<float>(3 * 256));
      #pragma omp parallel for schedule(dynamic)
      for (size_t c = 0; c < num_chunks; ++c) {
        auto fbank = preprocessor.computeFbank(chunks[c].data(), chunks[c].size(), mel_banks, fbank_window);
        std::vector<float> chunk_feat(256 * 10 * 125);
        emb_model.forwardBackbone(fbank.data(), chunk_feat.data(), c, nullptr);
        for (size_t s = 0; s < 3; ++s) {
          std::vector<float> spk_mask(589);
          for (size_t f = 0; f < 589; ++f) spk_mask[f] = segmentations[c][f * 3 + s];
          auto emb = emb_model.forwardPool(chunk_feat.data(), spk_mask.data(), c, s, nullptr);
          std::memcpy(chunk_embeddings[c].data() + s * 256, emb.data(), 256 * sizeof(float));
        }
      }
      for (size_t c = 0; c < num_chunks; ++c) {
        all_embeddings.insert(all_embeddings.end(), chunk_embeddings[c].begin(), chunk_embeddings[c].end());
      }

      auto final_segments = plda_vbx.diarizeStreaming(all_embeddings, segmentations, speaker_counting, total_frames, remaining_start, tracker);
      for (const auto &seg : final_segments) {
        float last_t = last_confirmed_per_spk[seg.speaker];
        if (seg.start_time >= last_t) {
          std::cout << "[STREAM FINAL] " << seg.speaker << ": "
                    << std::fixed << std::setprecision(3) << seg.start_time << "s ~ "
                    << seg.end_time << "s (" << seg.duration << "s)\n" << std::flush;
          all_confirmed_segments.push_back(seg);
          last_confirmed_per_spk[seg.speaker] = seg.end_time;
        }
      }
    }
  }

  auto t_stream_end = std::chrono::high_resolution_clock::now();
  double e2e_ms = std::chrono::duration<double, std::milli>(t_stream_end - t_stream_start).count();
  double total_audio_sec = stream_buffer.totalSamplesIngested() / 16000.0;

  std::cout << "\n============================================================\n";
  std::cout << "  Continuous Streaming Diarization Summary Report\n";
  std::cout << "============================================================\n";
  std::cout << "  Total Audio Processed:   " << std::fixed << std::setprecision(2) << total_audio_sec << " seconds ("
            << (total_audio_sec / 60.0) << " minutes)\n";
  std::cout << "  Unique Speakers Tracked: " << tracker.numSpeakers() << "\n";
  for (const auto &prof : tracker.getProfiles()) {
    std::cout << "    - " << prof.name << ": first seen at " << std::fixed << std::setprecision(2)
              << prof.first_seen_sec << "s, last seen at " << prof.last_seen_sec << "s ("
              << prof.count << " chunks)\n";
  }
  std::cout << "  Finalized Segments:      " << all_confirmed_segments.size() << " turns\n";
  std::cout << "  Pure Model Time:         " << total_pure_model_ms << " ms\n";
  std::cout << "  Total Streaming E2E Time:" << e2e_ms << " ms\n";
  double rtf = (total_audio_sec > 0.0) ? ((total_pure_model_ms / 1000.0) / total_audio_sec) : 0.0;
  std::cout << "  Streaming RTF:           " << std::fixed << std::setprecision(4) << rtf
            << "x (" << (rtf > 0.0 ? (1.0 / rtf) : 0.0) << "x Real-Time)\n";
  std::cout << "============================================================\n";
}

#if defined(USE_BLAS)
extern "C" void openblas_set_num_threads(int num_threads) __attribute__((weak));
#endif

int main(int argc, char **argv) {
#if defined(USE_BLAS)
  if (openblas_set_num_threads) {
    openblas_set_num_threads(1);
  }
#endif
  setenv("OPENBLAS_NUM_THREADS", "1", 1);

  std::string audio_path = "sample.wav";
  std::string weights_path = "weights/speaker_diarization.bin";
  bool debug_mode = false;
  bool stream_mode = false;
  bool read_stdin = false;
  double window_sec = 20.0;
  double step_sec = 10.0;
  double sim_speed = 0.0;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--audio" && i + 1 < argc) {
      audio_path = argv[++i];
    } else if (arg == "--weights" && i + 1 < argc) {
      weights_path = argv[++i];
    } else if (arg == "--debug") {
      debug_mode = true;
    } else if (arg == "--stream") {
      stream_mode = true;
    } else if (arg == "--stdin") {
      read_stdin = true;
      stream_mode = true;
    } else if (arg == "--window" && i + 1 < argc) {
      window_sec = std::stod(argv[++i]);
    } else if (arg == "--step" && i + 1 < argc) {
      step_sec = std::stod(argv[++i]);
    } else if (arg == "--sim-speed" && i + 1 < argc) {
      sim_speed = std::stod(argv[++i]);
    }
  }

  std::cout << "============================================================\n";
  std::cout << "  NNTrainer Speaker Diarization System\n";
  std::cout << "  Mode:    " << (stream_mode ? "Continuous Streaming" : "Batch File") << "\n";
  std::cout << "  Audio:   " << (read_stdin ? "stdin" : audio_path) << "\n";
  std::cout << "  Weights: " << weights_path << "\n";
  std::cout << "  Debug:   " << (debug_mode ? "Enabled" : "Disabled") << "\n";
  std::cout << "============================================================\n\n";

  // Start background memory profiler
  MemoryProfiler mem_profiler;
  mem_profiler.start();

  auto t_e2e_start = std::chrono::high_resolution_clock::now();

  // 1. Load weights
  WeightLoader loader;
  if (!loader.load(weights_path)) {
    std::cerr << "Fatal: Failed to load weights from: " << weights_path << "\n";
    return 1;
  }

  // 2. Initialize Models
  AudioPreprocessor preprocessor;
  PyanNet seg_model;
  WeSpeakerResNet34 emb_model;
  PldaVBx plda_vbx;

  if (!seg_model.init(loader)) {
    std::cerr << "Fatal: Failed to initialize PyanNet segmentation model!\n";
    return 1;
  }
  if (!emb_model.init(loader)) {
    std::cerr << "Fatal: Failed to initialize WeSpeaker ResNet34 model!\n";
    return 1;
  }
  if (!plda_vbx.init(loader)) {
    std::cerr << "Fatal: Failed to initialize PLDA & VBx module!\n";
    return 1;
  }

  const float *resample_kernel = loader.getTensor("frontend.resample_kernel");
  const float *mel_banks       = loader.getTensor("frontend.fbank_mel_banks");
  const float *fbank_window    = loader.getTensor("frontend.fbank_window");

  // Route to Streaming Pipeline if requested
  if (stream_mode) {
    runStreamingPipeline(audio_path, read_stdin, window_sec, step_sec, sim_speed,
                         preprocessor, seg_model, emb_model, plda_vbx,
                         resample_kernel, mel_banks, fbank_window);
    return 0;
  }

  // ==============================================================================
  // Offline Batch Pipeline
  // ==============================================================================
  std::cout << "[Pipeline] Loading audio file: " << audio_path << " ...\n";
  auto waveform = preprocessor.loadWav(audio_path, resample_kernel);
  float audio_duration_sec = waveform.size() / 16000.0f;
  std::cout << "[Pipeline] Loaded " << waveform.size() << " samples at 16kHz ("
            << std::fixed << std::setprecision(3) << audio_duration_sec << " seconds)\n";

  if (debug_mode) {
    print_tensor_row_by_row("[NNTrainer Input]", "audio_waveform_16k", waveform.data(), {1, waveform.size()});
  }

  // 4. Slice Audio Chunks
  auto chunks = preprocessor.sliceChunks(waveform);
  std::cout << "[Pipeline] Sliced into " << chunks.size() << " chunks of 10.0s (160,000 samples)\n";

  // 5. Run Segmentation Model (PyanNet)
  std::cout << "[Pipeline] Running PyanNet segmentation on " << chunks.size() << " chunks...\n";
  std::vector<std::vector<float>> segmentations;
  std::vector<uint8_t> speaker_counting;

  std::map<std::string, SavedTensor> saved_debug_tensors;

  DebugTensorCallback debug_cb = nullptr;
  if (debug_mode) {
    saved_debug_tensors["audio_waveform_16k"] = SavedTensor{std::vector<size_t>{1, waveform.size()}, waveform};
    debug_cb = [&](const std::string &name, const float *data, const std::vector<size_t> &shape) {
      print_tensor_row_by_row("[NNTrainer Output]", name, data, shape);
      size_t count = 1;
      for (size_t d : shape) count *= d;
      saved_debug_tensors[name] = SavedTensor{shape, std::vector<float>(data, data + count)};
    };
  }

  size_t num_chunks = chunks.size();
  constexpr size_t FRAMES_PER_CHUNK = 589;
  constexpr double STEP_SEC = 1.0;
  constexpr double FRAME_STEP_SEC = 0.016875;

  size_t last_chunk_start_frame = static_cast<size_t>(std::round((num_chunks - 1) * STEP_SEC / FRAME_STEP_SEC));
  size_t total_frames = last_chunk_start_frame + FRAMES_PER_CHUNK;

  auto t_seg_start = std::chrono::high_resolution_clock::now();
  if (debug_mode) {
    seg_model.runInference(chunks, segmentations, speaker_counting, debug_cb);
  } else {
    segmentations.resize(num_chunks);
    #pragma omp parallel for schedule(dynamic)
    for (size_t c = 0; c < num_chunks; ++c) {
      segmentations[c] = seg_model.forwardChunk(chunks[c].data(), c, nullptr);
    }

    speaker_counting.assign(total_frames, 0);
    std::vector<float> frame_sums(total_frames, 0.0f);
    std::vector<float> frame_weights(total_frames, 0.0f);

    for (size_t c = 0; c < num_chunks; ++c) {
      size_t chunk_start_frame = static_cast<size_t>(std::round(c * STEP_SEC / FRAME_STEP_SEC));
      for (size_t f = 0; f < FRAMES_PER_CHUNK; ++f) {
        size_t global_frame = chunk_start_frame + f;
        if (global_frame >= total_frames) break;

        float spk_sum = segmentations[c][f * 3 + 0]
                      + segmentations[c][f * 3 + 1]
                      + segmentations[c][f * 3 + 2];
        frame_sums[global_frame] += spk_sum;
        frame_weights[global_frame] += 1.0f;
      }
    }

    for (size_t f = 0; f < total_frames; ++f) {
      if (frame_weights[f] > 0.0f) {
        speaker_counting[f] = static_cast<uint8_t>(std::rint(frame_sums[f] / frame_weights[f]));
      } else {
        speaker_counting[f] = 0;
      }
    }
  }
  auto t_seg_end = std::chrono::high_resolution_clock::now();

  // 6. Extract Speaker Embeddings for Active Speakers
  std::cout << "[Pipeline] Extracting WeSpeaker ResNet34 embeddings...\n";
  std::vector<float> all_embeddings;
  all_embeddings.reserve(num_chunks * 3 * 256);

  auto t_emb_start = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<float>> chunk_embeddings(num_chunks, std::vector<float>(3 * 256));

  if (debug_mode) {
    for (size_t c = 0; c < num_chunks; ++c) {
      auto fbank = preprocessor.computeFbank(chunks[c].data(), chunks[c].size(), mel_banks, fbank_window);
      std::vector<float> chunk_feat(256 * 10 * 125);
      emb_model.forwardBackbone(fbank.data(), chunk_feat.data(), c, debug_cb);

      for (size_t s = 0; s < 3; ++s) {
        std::vector<float> spk_mask(589);
        for (size_t f = 0; f < 589; ++f) {
          spk_mask[f] = segmentations[c][f * 3 + s];
        }
        auto emb = emb_model.forwardPool(chunk_feat.data(), spk_mask.data(), c, s, debug_cb);
        std::memcpy(chunk_embeddings[c].data() + s * 256, emb.data(), 256 * sizeof(float));
      }
    }
  } else {
    #pragma omp parallel for schedule(dynamic)
    for (size_t c = 0; c < num_chunks; ++c) {
      auto fbank = preprocessor.computeFbank(chunks[c].data(), chunks[c].size(), mel_banks, fbank_window);
      std::vector<float> chunk_feat(256 * 10 * 125);
      emb_model.forwardBackbone(fbank.data(), chunk_feat.data(), c, nullptr);

      for (size_t s = 0; s < 3; ++s) {
        std::vector<float> spk_mask(589);
        for (size_t f = 0; f < 589; ++f) {
          spk_mask[f] = segmentations[c][f * 3 + s];
        }
        auto emb = emb_model.forwardPool(chunk_feat.data(), spk_mask.data(), c, s, nullptr);
        std::memcpy(chunk_embeddings[c].data() + s * 256, emb.data(), 256 * sizeof(float));
      }
    }
  }

  for (size_t c = 0; c < num_chunks; ++c) {
    all_embeddings.insert(all_embeddings.end(), chunk_embeddings[c].begin(), chunk_embeddings[c].end());
  }
  auto t_emb_end = std::chrono::high_resolution_clock::now();

  if (debug_mode) {
    std::vector<float> flat_seg;
    for (const auto &chunk_seg : segmentations) {
      flat_seg.insert(flat_seg.end(), chunk_seg.begin(), chunk_seg.end());
    }
    saved_debug_tensors["pipeline_segmentation"] = SavedTensor{std::vector<size_t>{num_chunks, 589, 3}, flat_seg};

    std::vector<float> float_counting(speaker_counting.begin(), speaker_counting.end());
    saved_debug_tensors["pipeline_speaker_counting"] = SavedTensor{std::vector<size_t>{speaker_counting.size(), 1}, float_counting};

    saved_debug_tensors["pipeline_embeddings"] = SavedTensor{std::vector<size_t>{num_chunks, 3, 256}, all_embeddings};

    write_debug_binary("nntrainer_debug_tensors.bin", saved_debug_tensors);
  }

  // 7. PLDA & Clustering & Diarization Timeline
  std::cout << "[Pipeline] Performing PLDA transformation & timeline reconstruction...\n";
  auto t_cluster_start = std::chrono::high_resolution_clock::now();
  auto segments = plda_vbx.diarize(all_embeddings, segmentations, speaker_counting, total_frames);
  auto t_cluster_end = std::chrono::high_resolution_clock::now();

  auto t_e2e_end = std::chrono::high_resolution_clock::now();

  // Stop memory tracker
  mem_profiler.stop();

  // Print Diarization Segments
  std::cout << "\n============================================================\n";
  std::cout << "  Final Speaker Diarization Segments (" << segments.size() << " segments):\n";
  std::cout << "============================================================\n";
  for (const auto &seg : segments) {
    std::cout << "  " << seg.speaker << ": "
              << std::fixed << std::setprecision(3) << seg.start_time << "s ~ "
              << seg.end_time << "s (" << seg.duration << "s)\n";
  }
  std::cout << "============================================================\n";

  // Calculate timing metrics
  double seg_ms = std::chrono::duration<double, std::milli>(t_seg_end - t_seg_start).count();
  double emb_ms = std::chrono::duration<double, std::milli>(t_emb_end - t_emb_start).count();
  double cluster_ms = std::chrono::duration<double, std::milli>(t_cluster_end - t_cluster_start).count();
  double pure_model_ms = seg_ms + emb_ms + cluster_ms;
  double e2e_ms = std::chrono::duration<double, std::milli>(t_e2e_end - t_e2e_start).count();

  double rtf_pure = (audio_duration_sec > 0.0f) ? ((pure_model_ms / 1000.0) / audio_duration_sec) : 0.0;
  double rtf_e2e  = (audio_duration_sec > 0.0f) ? ((e2e_ms / 1000.0) / audio_duration_sec) : 0.0;

  size_t vm_rss_kb   = MemoryProfiler::readVmRssKb();
  size_t peak_hwm_kb = MemoryProfiler::readPeakVmHwmKb();
  size_t anon_rss_kb = MemoryProfiler::readRssAnonKb();

  std::cout << "\n============================================================\n";
  std::cout << "  Performance & Memory Profiling Report\n";
  std::cout << "============================================================\n";
  std::cout << "  Audio Duration           : " << std::fixed << std::setprecision(3) << audio_duration_sec << " s (" << waveform.size() << " samples)\n";
  std::cout << "  Processed Chunks         : " << num_chunks << " chunks (10.0s window / 1.0s step)\n";
  std::cout << "  ----------------------------------------------------------\n";
  std::cout << "  [Pure Model Latency]:\n";
  std::cout << "    - PyanNet Segmentation : " << std::fixed << std::setprecision(2) << seg_ms << " ms  ("
            << (seg_ms / num_chunks) << " ms/chunk)\n";
  std::cout << "    - WeSpeaker ResNet34   : " << std::fixed << std::setprecision(2) << emb_ms << " ms  ("
            << (emb_ms / num_chunks) << " ms/chunk)\n";
  std::cout << "    - PLDA & Clustering    : " << std::fixed << std::setprecision(2) << cluster_ms << " ms\n";
  std::cout << "    * Total Pure Model     : " << std::fixed << std::setprecision(2) << pure_model_ms << " ms  ("
            << "RTF: " << std::setprecision(4) << rtf_pure << "x, "
            << std::setprecision(1) << (rtf_pure > 0 ? 1.0 / rtf_pure : 0.0) << "x Real-Time)\n";
  std::cout << "  ----------------------------------------------------------\n";
  std::cout << "  [End-to-End Latency]     : " << std::fixed << std::setprecision(2) << e2e_ms << " ms  ("
            << "RTF: " << std::setprecision(4) << rtf_e2e << "x, "
            << std::setprecision(1) << (rtf_e2e > 0 ? 1.0 / rtf_e2e : 0.0) << "x Real-Time)\n";
  std::cout << "  ----------------------------------------------------------\n";
  std::cout << "  [RSS Memory Usage]:\n";
  std::cout << "    - Current VmRSS        : " << vm_rss_kb << " KB (" << std::fixed << std::setprecision(2) << (vm_rss_kb / 1024.0) << " MB)\n";
  std::cout << "    - Peak Resident (maxrss): " << peak_hwm_kb << " KB (" << std::fixed << std::setprecision(2) << (peak_hwm_kb / 1024.0) << " MB)\n";
  std::cout << "    - Private RSS          : " << anon_rss_kb << " KB (" << std::fixed << std::setprecision(2) << (anon_rss_kb / 1024.0) << " MB)\n";
  std::cout << "============================================================\n";

  return 0;
}
