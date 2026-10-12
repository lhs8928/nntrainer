/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	main.cpp
 * @date	23 July 2025
 * @brief	This is a main file for CausalLM application
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "json.hpp"
#include <app_context.h>
#include <factory.h>

#include "ced/ced_transformer.h"
#include "qwen3_asr_causallm.h"
#include "audio_stream.h"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <csignal>
#include <deque>
#include <mutex>
#include <thread>
#include "audio_preprocessor.h"
#if !defined(_WIN32)
#include <sys/resource.h>
#endif

#include <atomic>
#include <chrono>
#include <filesystem>
#include <thread>

using json = nlohmann::json;

std::atomic<size_t> peak_rss_kb{0};
std::atomic<bool> tracking_enabled{true};

namespace {

inline unsigned int get_feat_extract_output_lengths(unsigned int input_lengths) {
  unsigned int input_lengths_leave = input_lengths % 100;
  unsigned int tail_tokens = 0;
  if (input_lengths_leave > 0) {
    unsigned int feat_lengths = (input_lengths_leave - 1) / 2 + 1;
    tail_tokens = ((feat_lengths - 1) / 2 + 1 - 1) / 2 + 1;
  }
  return tail_tokens + (input_lengths / 100) * 13;
}

void resolveNntrConfigPath(json &nntr_cfg, const std::string &key,
                           const std::string &model_path) {
  if (!nntr_cfg.contains(key) || !nntr_cfg[key].is_string())
    return;

  std::filesystem::path path = nntr_cfg[key].get<std::string>();
  if (path.empty() || path.is_absolute())
    return;

  nntr_cfg[key] = (std::filesystem::path(model_path) / path).string();
}

} // namespace

/**
 * @brief Print the maximum resident set size for the current process.
 */
void printMemoryUsage() {
#if defined(_WIN32)
  std::cout << "Max Resident Set Size: unavailable on Windows" << std::endl;
#else
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  std::cout << "Max Resident Set Size: " << usage.ru_maxrss << " KB"
            << std::endl;
#endif
}

/**
 * @brief Read the current process resident set size on Linux.
 */
size_t read_vm_rss_kb() {
#if defined(_WIN32)
  return 0;
#else
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.find("VmRSS:") == 0) {
      size_t kb = 0;
      sscanf(line.c_str(), "VmRSS: %zu kB", &kb);
      return kb;
    }
  }
  return 0;
#endif
}

/**
 * @brief Read private resident memory from smaps_rollup on Linux.
 */
size_t read_private_rss_kb() {
#if defined(_WIN32)
  return 0;
#else
  std::ifstream smaps("/proc/self/smaps_rollup");
  std::string line;
  size_t total = 0;
  while (std::getline(smaps, line)) {
    if (line.find("Private_Clean:") == 0 || line.find("Private_Dirty:") == 0) {
      size_t kb;
      sscanf(line.c_str(), "%*s %zu", &kb);
      total += kb;
    }
  }
  return total;
#endif
}

/**
 * @brief Start a background sampler for peak private RSS.
 */
void start_peak_tracker() {
  std::thread([] {
    while (tracking_enabled.load()) {
      size_t current = read_private_rss_kb();
      size_t prev = peak_rss_kb.load();
      if (current > prev) {
        peak_rss_kb.store(current);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }).detach();
}

/**
 * @brief Stop the memory sampler and print the observed peak.
 */
void stop_and_print_peak() {
  tracking_enabled.store(false);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  std::cout << "Peak memory usage (VmRSS): " << peak_rss_kb.load() << " KB"
            << std::endl;
}

/**
 * @brief Resolve config architecture names to registered model factory names.
 */
std::string resolve_architecture(std::string model_type,
                                 const std::string &architecture) {
  std::transform(model_type.begin(), model_type.end(), model_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (model_type == "embedding") {
    if (architecture == "Qwen3ForCausalLM") {
      return "Qwen3Embedding";
    } else if (architecture == "Gemma3ForCausalLM" ||
               architecture == "Gemma3TextModel") {
      return "EmbeddingGemma";
    } else if (architecture == "Qwen2Model") {
      return "Qwen2Embedding";
    } else if (architecture == "BertForMaskedLM") {
      return "MultilingualTinyBert";
    } else if (architecture == "XLMRobertaForMaskedLM" ||
               architecture == "XLMRobertaModel") {
      return "XLMRobertaForMaskedLM";
    } else if (architecture == "deberta-v2" ||
               architecture == "DebertaV2Model" ||
               architecture == "DebertaV2ForMaskedLM") {
      return "DebertaV2";
    } else {
      throw std::invalid_argument(
        "Unsupported architecture for embedding model: " + architecture);
    }
  }

  if (architecture == "Gemma4ForConditionalGeneration") {
    return "Gemma4ForCausalLM";
  }

  return architecture;
}


namespace {

std::atomic<bool> g_stream_stop{false};
void onStreamSignal(int) { g_stream_stop.store(true); }

/** @brief mute std::cout while alive (the model prints a long report per run) */
struct CoutMute {
  std::streambuf *old;
  explicit CoutMute(bool on) : old(on ? std::cout.rdbuf(nullptr) : nullptr),
                               on_(on) {}
  ~CoutMute() {
    if (on_) {
      std::cout.rdbuf(old);
      std::cout.clear();
      std::cout.width(0);
    }
  }
  bool on_;
};

/**
 * @brief Segment-mode live loop: a capture thread reads PCM from `src` and cuts
 *        it into segments (pause-aware), the calling thread runs the resident
 *        model on every segment and prints its text.
 */
void runStreamMode(quick_ai::Qwen3ASRCausalLM &asr, quick_ai::PcmSource &src,
                   float seg_min_sec, float silence_rms,
                   const std::function<std::string(const std::string &)>
                     &make_prompt,
                   quick_ai::ContextCarry &ctx_carry, bool do_sample,
                   const std::string &head, const std::string &tail,
                   bool verbose, unsigned int max_waiting) {
  using clk = std::chrono::steady_clock;
  struct Item {
    std::vector<float> pcm;
    size_t start;
    clk::time_point ready;
  };
  const size_t max_len = static_cast<size_t>(asr.getAudioSeqLen()) * 160;
  const size_t min_len = static_cast<size_t>(seg_min_sec * 16000.0f);
  std::mutex mtx;
  std::condition_variable cv;
  std::deque<Item> q;
  bool done = false;
  size_t max_backlog = 0, n_dropped = 0;
  const auto t_begin = clk::now();

  std::thread capture([&] {
    quick_ai::Segmenter seg(min_len, max_len);
    std::vector<float> buf(1600);
    size_t n;
    auto enqueue = [&](std::vector<float> &&s, size_t st) {
      std::lock_guard<std::mutex> lk(mtx);
      while (max_waiting > 0 && q.size() >= max_waiting) {
        // too slow for real time: drop the oldest waiting segment
        const Item &old = q.front();
        // cerr: std::cout may be muted while the model is running
        std::cerr << "[Stream] WARNING: dropped segment "
                  << old.start / 16000.0 << "s-"
                  << (old.start + old.pcm.size()) / 16000.0
                  << "s (processing is slower than real time)" << std::endl;
        ++n_dropped;
        q.pop_front();
      }
      q.push_back({std::move(s), st, clk::now()});
      max_backlog = std::max(max_backlog, q.size());
      cv.notify_one();
    };
    std::vector<float> s;
    size_t st = 0;
    while (!g_stream_stop.load() &&
           (n = src.read(buf.data(), buf.size())) > 0) {
      seg.push(buf.data(), n);
      while (seg.pop(s, st))
        enqueue(std::move(s), st);
    }
    if (seg.flush(s, st) && s.size() >= 4800) // ignore < 0.3 s tails
      enqueue(std::move(s), st);
    std::lock_guard<std::mutex> lk(mtx);
    done = true;
    cv.notify_one();
  });

  std::signal(SIGINT, onStreamSignal);
  std::signal(SIGTERM, onStreamSignal);
  std::cout << "[Stream] segment length " << seg_min_sec << "-"
            << max_len / 16000.0 << " s, silence threshold " << silence_rms
            << " (RMS). Speak now; Ctrl-C to stop." << std::endl;

  double audio_s = 0.0, proc_s = 0.0, max_delay_s = 0.0;
  size_t n_seg = 0, n_skipped = 0;
  while (true) {
    Item it;
    {
      std::unique_lock<std::mutex> lk(mtx);
      cv.wait(lk, [&] { return !q.empty() || done; });
      if (q.empty())
        break;
      it = std::move(q.front());
      q.pop_front();
    }
    const double t0 = it.start / 16000.0;
    const double t1 = t0 + it.pcm.size() / 16000.0;
    audio_s += it.pcm.size() / 16000.0;
    ++n_seg;
    const float peak = quick_ai::segmentPeakRms(it.pcm);
    if (peak < silence_rms) {
      ++n_skipped;
      if (verbose)
        std::cout << "[" << std::fixed << std::setprecision(1) << t0 << "s-"
                  << t1 << "s] (silence, skipped)" << std::endl;
      continue;
    }
    it.pcm.resize(max_len, 0.0f); // pad to the fixed model length with silence
    asr.setAudioPcm(std::move(it.pcm));
    const auto p0 = clk::now();
    {
      CoutMute mute(!verbose);
      asr.run(make_prompt(ctx_carry.get()), do_sample, head, tail);
    }
    const auto p1 = clk::now();
    const double ms = std::chrono::duration<double, std::milli>(p1 - p0).count();
    const double delay_s =
      std::chrono::duration<double>(p1 - it.ready).count();
    proc_s += ms / 1000.0;
    max_delay_s = std::max(max_delay_s, delay_s);
    ctx_carry.add(asr.getLastTranscription());
    std::cout << "[" << std::fixed << std::setprecision(1) << t0 << "s-" << t1
              << "s] " << asr.getLastTranscription() << "   (proc "
              << std::setprecision(1) << ms / 1000.0 << " s, text ready "
              << delay_s << " s after the segment ended)" << std::endl;
  }
  capture.join();
  const double wall = std::chrono::duration<double>(clk::now() - t_begin).count();
  std::cout << "[Stream] done: " << std::setprecision(1) << audio_s
            << " s of audio in " << n_seg << " segment(s) (" << n_skipped
            << " silent, " << n_dropped << " dropped), processing " << proc_s << " s (RTF "
            << std::setprecision(2) << (audio_s > 0 ? proc_s / audio_s : 0.0)
            << "), max text delay " << std::setprecision(1) << max_delay_s
            << " s, max queue backlog " << max_backlog << ", wall " << wall
            << " s" << std::endl;
}

} // namespace

/**
 * @brief Entry point for loading, initializing, and running a CausalLM model.
 */
int main(int argc, char *argv[]) {
  // Parse and consume --audio option if present
  std::string audio_path = "";
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--audio") == 0 && i + 1 < argc) {
      audio_path = argv[i + 1];
      for (int j = i; j < argc - 2; ++j) {
        argv[j] = argv[j + 2];
      }
      argc -= 2;
      break;
    }
  }

  // Parse and consume --segment-sec <S>: process the --audio file in
  // consecutive S-second segments with ONE resident model (segment mode).
  float segment_sec = 0.0f;
  float segment_min_sec = 0.0f;
  float silence_rms = 0.005f;
  std::string stream_file, mic_device;
  // --language <Name>: force the output language (e.g. Korean, English) by
  // appending "language <Name><asr_text>" to the prompt, so the model does not
  // spend ~5 decode tokens (~0.5 s) per run on the language tag.
  std::string forced_language;
  // --max-backlog N: if the model falls behind real time and more than N
  // segments are waiting, drop the oldest ones (bounded latency) and say so.
  unsigned int max_backlog_opt = 2;
  // --context-chars N: give the last N characters of the previous segments'
  // text to the next segment as context (system prompt). 0 = off.
  size_t context_chars = 0;
  float stream_speed = 1.0f; // --stream-speed X: replay --stream-file X times faster (stress test)
  bool stream_mic = false, stream_stdin = false, stream_verbose = false;
  auto take_opt = [&](const char *name, bool has_value, std::string &val) {
    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], name) == 0 && (!has_value || i + 1 < argc)) {
        const int n = has_value ? 2 : 1;
        if (has_value)
          val = argv[i + 1];
        for (int j = i; j < argc - n; ++j)
          argv[j] = argv[j + n];
        argc -= n;
        return true;
      }
    }
    return false;
  };
  {
    std::string v;
    if (take_opt("--segment-sec", true, v))
      segment_sec = std::stof(v);
    if (take_opt("--segment-min-sec", true, v))
      segment_min_sec = std::stof(v);
    if (take_opt("--silence-rms", true, v))
      silence_rms = std::stof(v);
    if (take_opt("--stream-file", true, v))
      stream_file = v;
    if (take_opt("--mic-device", true, v))
      mic_device = v;
    if (take_opt("--mic", false, v))
      stream_mic = true;
    if (take_opt("--stdin-pcm", false, v))
      stream_stdin = true;
    if (take_opt("--stream-verbose", false, v))
      stream_verbose = true;
    if (take_opt("--language", true, v))
      forced_language = v;
    if (take_opt("--context-chars", true, v))
      context_chars = static_cast<size_t>(std::stoul(v));
    if (take_opt("--stream-speed", true, v))
      stream_speed = std::stof(v);
    if (take_opt("--max-backlog", true, v))
      max_backlog_opt = static_cast<unsigned int>(std::stoul(v));
  }
  const bool stream_mode = !stream_file.empty() || stream_mic || stream_stdin;
  if (stream_mode && segment_sec <= 0.0f)
    segment_sec = 10.0f;
  if (stream_mode && segment_min_sec <= 0.0f)
    segment_min_sec = 0.9f * segment_sec;

  auto start_time = std::chrono::high_resolution_clock::now();

  /** Register CED model */
  quick_ai::Factory::Instance().registerModel(
    "CedForAudioClassification",
    [](json &cfg, json &generation_cfg, json &nntr_cfg) {
      return std::make_unique<quick_ai::CedTransformer>(cfg, generation_cfg,
                                                        nntr_cfg);
    });

  quick_ai::Factory::Instance().registerModel(
    "Qwen3ASRForConditionalGeneration",
    [](json &cfg, json &generation_cfg, json &nntr_cfg) {
      return std::make_unique<quick_ai::Qwen3ASRCausalLM>(cfg, generation_cfg,
                                                          nntr_cfg);
    });

  // Validate arguments
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model_path> [input_prompt]\n"
              << "  <model_path>   : Path to model directory\n"
              << "  [input_prompt] : Optional input text (uses sample_input or "
                 "chat_input if omitted)\n";
    return EXIT_FAILURE;
  }

  const std::string model_path = argv[1];
  std::string input_text;
  std::function<std::string(const std::string &)> make_prompt;
  std::string system_head_prompt = "";
  std::string system_tail_prompt = "";

  std::cout << model_path << std::endl;

  try {
    // Load configuration files
    json cfg = quick_ai::LoadJsonFile(model_path + "/config.json");
    json text_cfg = json::object();
    if (cfg.contains("text_config") && cfg["text_config"].is_object()) {
      text_cfg = cfg["text_config"];
    } else if (cfg.contains("thinker_config") && cfg["thinker_config"].is_object() &&
               cfg["thinker_config"].contains("text_config") && cfg["thinker_config"]["text_config"].is_object()) {
      text_cfg = cfg["thinker_config"]["text_config"];
    }

    if (!text_cfg.empty()) {
      for (auto& [key, val] : text_cfg.items()) {
        if (key != "architectures" && key != "model_type" && !val.is_null()) {
          cfg[key] = val;
        }
      }
    }

    json generation_cfg = json::object();
    std::string generation_config_path = model_path + "/generation_config.json";
    if (std::filesystem::exists(generation_config_path)) {
      generation_cfg = quick_ai::LoadJsonFile(generation_config_path);
    }
    json nntr_cfg = quick_ai::LoadJsonFile(model_path + "/nntr_config.json");

    if (!audio_path.empty() || stream_mode) {
      quick_ai::AudioPreprocessor preprocessor;
      std::vector<float> pcm;
      if (!audio_path.empty())
        pcm = preprocessor.loadWav(audio_path);
      unsigned int audio_frames = pcm.size() / 160;
      constexpr unsigned int MAX_AUDIO_FRAMES = 1200 * 100; // 20 minutes (1200s * 100fps)
      if (audio_frames > MAX_AUDIO_FRAMES) {
        audio_frames = MAX_AUDIO_FRAMES;
      }
      if (segment_sec > 0.0f) {
        // segment mode: the model is built once for exactly one segment
        audio_frames = static_cast<unsigned int>(segment_sec * 100.0f + 0.5f);
      }
      nntr_cfg["audio_seq_len"] = audio_frames;
      unsigned int audio_seq_len = get_feat_extract_output_lengths(audio_frames);
      unsigned int base_seq_len = nntr_cfg.value("init_seq_len", 15);
      if (!forced_language.empty())
        base_seq_len += 8; // room for the "language X<asr_text>" prompt tokens
      if (context_chars > 0)
        base_seq_len += 2 * context_chars + 8; // room for the context tokens
      unsigned int num_to_generate = nntr_cfg.value("num_to_generate", 5);
      nntr_cfg["init_seq_len"] = base_seq_len + audio_seq_len;
      nntr_cfg["max_seq_len"] = base_seq_len + audio_seq_len + num_to_generate;
      nntr_cfg["sliding_window"] = base_seq_len + audio_seq_len + num_to_generate;
    }
    // Resolve relative paths in nntr_config.json against the model directory.
    // Iterate by value (const std::string) rather than by reference: the
    // initializer list holds const char * literals that are converted to
    // temporary std::string objects, and binding those temporaries to a
    // const reference is flagged by -Werror=range-loop-construct.
    for (const std::string key :
         {"tokenizer_file", "embedding_file_name", "subsampler_model_file_name",
          "ple_file_name", "sample_input", "yolo_ref_dir"}) {
      resolveNntrConfigPath(nntr_cfg, key, model_path);
    }

    if (nntr_cfg.contains("subsampler") && nntr_cfg["subsampler"].is_object()) {
      resolveNntrConfigPath(nntr_cfg["subsampler"], "model_file_name", model_path);
    }

    if (nntr_cfg.contains("system_prompt")) {
      system_head_prompt =
        nntr_cfg["system_prompt"]["head_prompt"].get<std::string>();
      system_tail_prompt =
        nntr_cfg["system_prompt"]["tail_prompt"].get<std::string>();
    }

    // Construct weight file path
    const std::string weight_file =
      model_path + "/" + nntr_cfg["model_file_name"].get<std::string>();

    std::cout << weight_file << std::endl;

    // Initialize and run model
    std::string architecture;
    if (cfg.contains("architectures") && cfg["architectures"].is_array() &&
        !cfg["architectures"].empty()) {
      architecture = cfg["architectures"].get<std::vector<std::string>>()[0];
    } else if (cfg.contains("architecture") &&
               cfg["architecture"].is_string()) {
      architecture = cfg["architecture"].get<std::string>();
    } else if (cfg.contains("model_type") && cfg["model_type"].is_string()) {
      architecture = cfg["model_type"].get<std::string>();
    } else {
      throw std::invalid_argument(
        "config.json must contain 'architectures', 'architecture', or "
        "'model_type'.");
    }

    if (nntr_cfg.contains("model_type")) {
      std::string model_type = nntr_cfg["model_type"].get<std::string>();
      architecture = resolve_architecture(model_type, architecture);
    }

    // Determine input text
    if (argc >= 3) {
      input_text = argv[2];
    } else {
      input_text = nntr_cfg["sample_input"].get<std::string>();
    }

    auto model = quick_ai::Factory::Instance().create(architecture, cfg,
                                                      generation_cfg, nntr_cfg);
    if (!model) {
      std::cerr << "Unknown architecture: " << architecture << std::endl;
      std::cerr << "Registered architectures:";
      quick_ai::Factory::Instance().printRegistered(std::cerr);
      std::cerr << std::endl;
      return EXIT_FAILURE;
    }
    if (architecture == "Qwen3ASRForConditionalGeneration" &&
        (!audio_path.empty() || stream_mode)) {
      auto *asr_model = dynamic_cast<quick_ai::Qwen3ASRCausalLM*>(model.get());
      if (asr_model) {
        asr_model->setAudioPath(audio_path);
        
        quick_ai::AudioPreprocessor preprocessor;
        std::vector<float> pcm;
        if (!audio_path.empty())
          pcm = preprocessor.loadWav(audio_path);
        unsigned int audio_frames = pcm.size() / 160;
        constexpr unsigned int MAX_AUDIO_FRAMES = 1200 * 100; // 20 minutes (1200s * 100fps)
        if (audio_frames > MAX_AUDIO_FRAMES) {
          audio_frames = MAX_AUDIO_FRAMES;
        }
        if (segment_sec > 0.0f) {
          audio_frames = static_cast<unsigned int>(segment_sec * 100.0f + 0.5f);
        }
        unsigned int audio_seq_len = get_feat_extract_output_lengths(audio_frames);
        
        std::string pad_tokens = "";
        for (unsigned int idx = 0; idx < audio_seq_len; ++idx) {
          pad_tokens += "<|audio_pad|>";
        }
        // the prompt of a segment: the optional context goes into the system
        // message (the official Qwen3-ASR way of giving context)
        const std::string user_text = input_text;
        make_prompt = [pad_tokens, user_text,
                       forced_language](const std::string &ctx) {
          std::string p = "<|im_start|>system\n" + ctx +
                          "<|im_end|>\n<|im_start|>user\n<|audio_start|>" +
                          pad_tokens + "<|audio_end|>" + user_text +
                          "<|im_end|>\n<|im_start|>assistant\n";
          if (!forced_language.empty())
            p += "language " + forced_language + "<asr_text>";
          return p;
        };
        input_text = make_prompt("");
        
        system_head_prompt.clear();
        system_tail_prompt.clear();
      }
    }

    model->initialize();
    model->load_weight(weight_file);
    model->repack_weight();

    bool do_sample = generation_cfg.value("do_sample", false);

#ifdef PROFILE
    start_peak_tracker();
#endif
    auto *asr_seg_model =
      dynamic_cast<quick_ai::Qwen3ASRCausalLM *>(model.get());
    if (stream_mode && asr_seg_model != nullptr) {
      std::unique_ptr<quick_ai::PcmSource> source;
      if (!stream_file.empty()) {
        quick_ai::AudioPreprocessor sp;
        source = std::make_unique<quick_ai::WavRealtimeSource>(
          sp.loadWav(stream_file), stream_speed);
      } else if (stream_mic) {
        const std::string dev = mic_device.empty() ? "default" : mic_device;
        source = std::make_unique<quick_ai::PipeSource>(
          "arecord -q -D " + dev + " -f S16_LE -r 16000 -c 1 -t raw");
      } else {
        source = std::make_unique<quick_ai::PipeSource>("");
      }
      quick_ai::ContextCarry ctx_carry(context_chars);
      runStreamMode(*asr_seg_model, *source, segment_min_sec, silence_rms,
                    make_prompt, ctx_carry, do_sample, system_head_prompt,
                    system_tail_prompt, stream_verbose, max_backlog_opt);
    } else if (segment_sec > 0.0f && asr_seg_model != nullptr &&
               !audio_path.empty()) {
      // ---- segment mode: one resident model, many segments ----------------
      quick_ai::AudioPreprocessor seg_pre;
      std::vector<float> all = seg_pre.loadWav(audio_path);
      const size_t seg_samples =
        static_cast<size_t>(asr_seg_model->getAudioSeqLen()) * 160;
      const size_t num_segs = (all.size() + seg_samples - 1) / seg_samples;
      std::cout << "[Segment] " << all.size() / 16000.0 << " s of audio, "
                << num_segs << " segment(s) of " << seg_samples / 16000.0
                << " s" << std::endl;
      double total_proc_ms = 0.0;
      std::string full_text;
      quick_ai::ContextCarry ctx_carry(context_chars);
      for (size_t k = 0; k < num_segs; ++k) {
        const size_t b = k * seg_samples;
        const size_t e = std::min(all.size(), b + seg_samples);
        std::vector<float> seg(all.begin() + b, all.begin() + e);
        seg.resize(seg_samples, 0.0f); // pad the last segment with silence
        asr_seg_model->setAudioPcm(std::move(seg));
        auto t0 = std::chrono::high_resolution_clock::now();
        model->run(make_prompt(ctx_carry.get()), do_sample, system_head_prompt,
                   system_tail_prompt);
        double ms = std::chrono::duration<double, std::milli>(
                      std::chrono::high_resolution_clock::now() - t0)
                      .count();
        total_proc_ms += ms;
        const std::string &txt = asr_seg_model->getLastTranscription();
        ctx_carry.add(txt);
        std::cout << "[SEG " << k << "] " << b / 16000.0 << "s-" << e / 16000.0
                  << "s processed in " << ms << " ms: " << txt << std::endl;
        if (!full_text.empty() && !txt.empty())
          full_text += " ";
        full_text += txt;
      }
      std::cout << "[Segment] total processing " << total_proc_ms
                << " ms for " << all.size() / 16000.0 << " s of audio (RTF "
                << total_proc_ms / 1000.0 / (all.size() / 16000.0) << ")\n"
                << "[Segment] FULL TEXT: " << full_text << std::endl;
    } else {
#if defined(_WIN32)
    model->run(input_text.c_str(), do_sample, system_head_prompt.c_str(),
               system_tail_prompt.c_str());
#else
    model->run(input_text, do_sample, system_head_prompt, system_tail_prompt);
#endif
    }
#ifdef PROFILE
    stop_and_print_peak();
#endif
    auto finish_time = std::chrono::high_resolution_clock::now();
    auto e2e_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      finish_time - start_time);
    std::cout << "[e2e time]: " << e2e_duration.count() << " ms \n";
    printMemoryUsage();

  } catch (const std::exception &e) {
    std::cerr << "\n[!] FATAL ERROR: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
