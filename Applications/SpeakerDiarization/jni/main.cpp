// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   main.cpp
 * @date   29 September 2026
 * @brief  CLI Application runner for Speaker Diarization in NNTrainer with Performance & RSS Profiling
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
    return readPeakMaxRssKb();
#endif
  }

  static size_t readPeakMaxRssKb() {
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

  static size_t readPrivateRssKb() {
#if defined(_WIN32)
    return 0;
#else
    std::ifstream smaps("/proc/self/smaps_rollup");
    if (!smaps.is_open()) {
      return readVmRssKb();
    }
    std::string line;
    size_t total = 0;
    while (std::getline(smaps, line)) {
      if (line.rfind("Private_Clean:", 0) == 0 || line.rfind("Private_Dirty:", 0) == 0) {
        size_t kb = 0;
        if (sscanf(line.c_str(), "%*s %zu", &kb) == 1) {
          total += kb;
        }
      }
    }
    return total > 0 ? total : readVmRssKb();
#endif
  }
};

// ==============================================================================
// Debug Binary Writer
// ==============================================================================
struct SavedTensor {
  std::vector<size_t> shape;
  std::vector<float> data;
};

static void write_debug_binary(const std::string &filename, const std::map<std::string, SavedTensor> &tensors) {
  std::ofstream f(filename, std::ios::binary | std::ios::trunc);
  if (!f.is_open()) return;

  std::vector<uint8_t> header;
  const char magic[8] = {'N','N','T','R','D','I','A','R'};
  header.insert(header.end(), magic, magic + 8);

  uint32_t num_tensors = static_cast<uint32_t>(tensors.size());
  uint8_t *nt_ptr = reinterpret_cast<uint8_t*>(&num_tensors);
  header.insert(header.end(), nt_ptr, nt_ptr + 4);

  uint64_t current_offset = 0;
  struct Entry {
    std::string name;
    std::vector<uint32_t> dims;
    uint64_t size;
    uint64_t offset;
  };
  std::vector<Entry> entries;

  for (const auto &kv : tensors) {
    Entry e;
    e.name = kv.first;
    for (size_t d : kv.second.shape) e.dims.push_back(static_cast<uint32_t>(d));
    e.size = kv.second.data.size() * sizeof(float);
    e.offset = current_offset;
    current_offset += e.size;
    entries.push_back(e);
  }

  for (const auto &e : entries) {
    uint32_t name_len = static_cast<uint32_t>(e.name.size());
    uint8_t *nl_ptr = reinterpret_cast<uint8_t*>(&name_len);
    header.insert(header.end(), nl_ptr, nl_ptr + 4);
    header.insert(header.end(), e.name.begin(), e.name.end());

    uint32_t num_dims = static_cast<uint32_t>(e.dims.size());
    uint8_t *nd_ptr = reinterpret_cast<uint8_t*>(&num_dims);
    header.insert(header.end(), nd_ptr, nd_ptr + 4);
    for (uint32_t d : e.dims) {
      uint8_t *d_ptr = reinterpret_cast<uint8_t*>(&d);
      header.insert(header.end(), d_ptr, d_ptr + 4);
    }
    const uint8_t *sz_ptr = reinterpret_cast<const uint8_t*>(&e.size);
    header.insert(header.end(), sz_ptr, sz_ptr + 8);
    const uint8_t *off_ptr = reinterpret_cast<const uint8_t*>(&e.offset);
    header.insert(header.end(), off_ptr, off_ptr + 8);
  }

  size_t pad_len = (64 - (header.size() % 64)) % 64;
  header.insert(header.end(), pad_len, 0);

  f.write(reinterpret_cast<const char*>(header.data()), header.size());
  for (const auto &kv : tensors) {
    f.write(reinterpret_cast<const char*>(kv.second.data.data()), kv.second.data.size() * sizeof(float));
  }
  std::cout << "[Debug] Saved " << tensors.size() << " intermediate tensors to " << filename << "\n";
}

static void print_tensor_row_by_row(const std::string &title, const std::string &name,
                                    const float *data, const std::vector<size_t> &shape) {
  size_t total_elements = 1;
  for (size_t d : shape) total_elements *= d;

  if (total_elements == 0) return;

  float t_min = data[0];
  float t_max = data[0];
  double t_sum = 0.0;
  double t_sq_sum = 0.0;

  for (size_t i = 0; i < total_elements; ++i) {
    float v = data[i];
    if (v < t_min) t_min = v;
    if (v > t_max) t_max = v;
    t_sum += v;
    t_sq_sum += static_cast<double>(v) * v;
  }
  float t_mean = static_cast<float>(t_sum / total_elements);
  float t_l2 = static_cast<float>(std::sqrt(t_sq_sum));

  std::cout << "\n" << title << " Name: " << name << " | Dim: [";
  for (size_t i = 0; i < shape.size(); ++i) {
    std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "");
  }
  std::cout << "] | Min: " << std::fixed << std::setprecision(6) << t_min
            << " | Max: " << t_max << " | Mean: " << t_mean << " | L2: " << t_l2 << "\n";

  size_t width = (shape.size() >= 2) ? shape.back() : total_elements;
  size_t height = total_elements / width;

  auto print_row = [&](size_t r) {
    std::cout << "    [Row " << r << "]: ";
    const float *row = data + r * width;
    if (width > 6) {
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

int main(int argc, char **argv) {
  std::string audio_path = "/hdd/workspace/temp/qwen3-asr/nntrainer/asr_en.wav";
  std::string weights_path = "weights/speaker_diarization.bin";
  bool debug_mode = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--audio" && i + 1 < argc) {
      audio_path = argv[++i];
    } else if (arg == "--weights" && i + 1 < argc) {
      weights_path = argv[++i];
    } else if (arg == "--debug") {
      debug_mode = true;
    }
  }

  std::cout << "============================================================\n";
  std::cout << "  NNTrainer Speaker Diarization System\n";
  std::cout << "  Audio:   " << audio_path << "\n";
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

  // 3. Load Audio
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

  auto t_seg_start = std::chrono::high_resolution_clock::now();
  if (debug_mode) {
    seg_model.runInference(chunks, segmentations, speaker_counting, debug_cb);
  } else {
    segmentations.resize(num_chunks);
    #pragma omp parallel for schedule(dynamic)
    for (size_t c = 0; c < num_chunks; ++c) {
      segmentations[c] = seg_model.forwardChunk(chunks[c].data(), c, nullptr);
    }

    constexpr size_t TOTAL_FRAMES = 949;
    constexpr size_t FRAMES_PER_CHUNK = 589;
    constexpr double STEP_SEC = 1.0;
    constexpr double FRAME_STEP_SEC = 0.016875;

    speaker_counting.assign(TOTAL_FRAMES, 0);
    std::vector<float> frame_sums(TOTAL_FRAMES, 0.0f);
    std::vector<float> frame_weights(TOTAL_FRAMES, 0.0f);

    for (size_t c = 0; c < num_chunks; ++c) {
      size_t chunk_start_frame = static_cast<size_t>(std::round(c * STEP_SEC / FRAME_STEP_SEC));
      for (size_t f = 0; f < FRAMES_PER_CHUNK; ++f) {
        size_t global_frame = chunk_start_frame + f;
        if (global_frame >= TOTAL_FRAMES) break;

        float spk_sum = segmentations[c][f * 3 + 0]
                      + segmentations[c][f * 3 + 1]
                      + segmentations[c][f * 3 + 2];
        frame_sums[global_frame] += spk_sum;
        frame_weights[global_frame] += 1.0f;
      }
    }

    for (size_t f = 0; f < TOTAL_FRAMES; ++f) {
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
  auto segments = plda_vbx.diarize(all_embeddings, segmentations, speaker_counting);
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

  size_t curr_rss_kb = MemoryProfiler::readVmRssKb();
  size_t peak_max_rss_kb = MemoryProfiler::readPeakMaxRssKb();
  size_t private_rss_kb = MemoryProfiler::readPrivateRssKb();

  // Print Performance & Profiling Report
  std::cout << "\n============================================================\n";
  std::cout << "  Performance & Memory Profiling Report\n";
  std::cout << "============================================================\n";
  std::cout << "  Audio Duration           : " << std::fixed << std::setprecision(3) << audio_duration_sec << " s (" << waveform.size() << " samples)\n";
  std::cout << "  Processed Chunks         : " << num_chunks << " chunks (10.0s window / 1.0s step)\n";
  std::cout << "  ----------------------------------------------------------\n";
  std::cout << "  [Pure Model Latency]:\n";
  std::cout << "    - PyanNet Segmentation : " << std::fixed << std::setprecision(2) << seg_ms << " ms  (" << (seg_ms / num_chunks) << " ms/chunk)\n";
  std::cout << "    - WeSpeaker ResNet34   : " << std::fixed << std::setprecision(2) << emb_ms << " ms  (" << (emb_ms / num_chunks) << " ms/chunk)\n";
  std::cout << "    - PLDA & Clustering    : " << std::fixed << std::setprecision(2) << cluster_ms << " ms\n";
  std::cout << "    * Total Pure Model     : " << std::fixed << std::setprecision(2) << pure_model_ms << " ms  (RTF: " << std::setprecision(4) << rtf_pure << "x, " << std::setprecision(1) << (1.0 / rtf_pure) << "x Real-Time)\n";
  std::cout << "  ----------------------------------------------------------\n";
  std::cout << "  [End-to-End Latency]     : " << std::fixed << std::setprecision(2) << e2e_ms << " ms  (RTF: " << std::setprecision(4) << rtf_e2e << "x, " << std::setprecision(1) << (1.0 / rtf_e2e) << "x Real-Time)\n";
  std::cout << "  ----------------------------------------------------------\n";
  std::cout << "  [RSS Memory Usage]:\n";
  std::cout << "    - Current VmRSS        : " << curr_rss_kb << " KB (" << std::fixed << std::setprecision(2) << (curr_rss_kb / 1024.0) << " MB)\n";
  std::cout << "    - Peak Resident (maxrss): " << peak_max_rss_kb << " KB (" << std::fixed << std::setprecision(2) << (peak_max_rss_kb / 1024.0) << " MB)\n";
  if (private_rss_kb > 0) {
    std::cout << "    - Private RSS          : " << private_rss_kb << " KB (" << std::fixed << std::setprecision(2) << (private_rss_kb / 1024.0) << " MB)\n";
  }
  std::cout << "============================================================\n\n";

  return 0;
}
