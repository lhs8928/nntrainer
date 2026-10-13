// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   main.cpp
 * @date   12 October 2026
 * @brief  Speech recognition + speaker diarization in one NNTrainer application
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * One audio stream feeds two lanes in the same process:
 *
 *   audio source -> fan-out -+-> ASR lane (segmenter + resident Qwen3-ASR) -+
 *                            |                                             +->
 * merger -> lines
 *                            +-> SD lane  (streaming diarizer)  ----------+
 *
 * The lanes run on their own threads and can be given their own CPUs and thread
 * counts, so the two models do not fight over the cores.
 */

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#if defined(__linux__)
#include <sys/resource.h>
#endif

#include "asr_lane.h"
#include "audio_fanout.h"
#include "cpu_affinity.h"
#include "events.h"
#include "merger.h"
#include "sd_lane.h"

namespace {

void onSignal(int) { discretion::AudioFanout::requestStop(); }

void usage(const char *prog) {
  std::fprintf(
    stderr,
    "Usage: %s <asr_model_dir> --sd-weights <speaker_diarization.bin>\n"
    "          (--stream-file <wav> [--stream-speed X] | --mic [--mic-device "
    "D] | "
    "--stdin-pcm)\n"
    "\n"
    "ASR:   --segment-sec S (8)  --segment-min-sec S  --silence-rms R (0.005)\n"
    "       --language NAME (English, \"\" = automatic)  --max-backlog N (2)\n"
    "SD:    --window S (20)  --step S (10)\n"
    "Cores: --asr-cpus LIST  --sd-cpus LIST (e.g. 0,1 / 2-3)\n"
    "       --asr-threads N  --sd-threads N\n"
    "       --no-word-timestamps (label whole segments instead of words)\n"
    "Other: --seconds N (stop after N s of audio)  --verbose\n",
    prog);
}

struct Args {
  std::string model_dir;
  discretion::AsrOptions asr;
  discretion::SdOptions sd;
  discretion::AudioFanout::Options audio;
  bool verbose = false;
};

/** @brief parse the command line; false (after printing why) on error */
bool parseArgs(int argc, char **argv, Args &a) {
  if (argc < 2 || std::string(argv[1]).rfind("--", 0) == 0) {
    usage(argv[0]);
    return false;
  }
  a.model_dir = argv[1];
  a.asr.model_dir = a.model_dir;
  try {
    for (int i = 2; i < argc; ++i) {
      const std::string k = argv[i];
      auto next = [&]() -> std::string {
        if (i + 1 >= argc)
          throw std::invalid_argument("missing value for " + k);
        return argv[++i];
      };
      if (k == "--sd-weights")
        a.sd.weights = next();
      else if (k == "--stream-file")
        a.audio.file = next();
      else if (k == "--stream-speed")
        a.audio.speed = std::stod(next());
      else if (k == "--mic")
        a.audio.mic = true;
      else if (k == "--mic-device")
        a.audio.mic_device = next();
      else if (k == "--stdin-pcm")
        a.audio.stdin_pcm = true;
      else if (k == "--seconds")
        a.audio.max_seconds = std::stod(next());
      else if (k == "--segment-sec")
        a.asr.segment_sec = std::stof(next());
      else if (k == "--segment-min-sec")
        a.asr.segment_min_sec = std::stof(next());
      else if (k == "--silence-rms")
        a.asr.silence_rms = std::stof(next());
      else if (k == "--language")
        a.asr.language = next();
      else if (k == "--max-backlog")
        a.asr.max_backlog = static_cast<unsigned>(std::stoul(next()));
      else if (k == "--window")
        a.sd.window_sec = std::stod(next());
      else if (k == "--step")
        a.sd.step_sec = std::stod(next());
      else if (k == "--asr-cpus")
        a.asr.cpus = discretion::parseCpuList(next());
      else if (k == "--sd-cpus")
        a.sd.cpus = discretion::parseCpuList(next());
      else if (k == "--asr-threads")
        a.asr.threads = std::stoi(next());
      else if (k == "--sd-threads")
        a.sd.threads = std::stoi(next());
      else if (k == "--no-word-timestamps")
        a.asr.word_timestamps = false;
      else if (k == "--verbose")
        a.verbose = true;
      else
        throw std::invalid_argument("unknown option " + k);
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    usage(argv[0]);
    return false;
  }
  if (a.sd.weights.empty()) {
    std::fprintf(stderr, "error: --sd-weights is required\n");
    return false;
  }
  if (a.sd.step_sec <= 0.0 || a.sd.window_sec < a.sd.step_sec) {
    std::fprintf(stderr, "error: need 0 < --step <= --window\n");
    return false;
  }
  if (a.asr.segment_sec <= 0.0f) {
    std::fprintf(stderr, "error: --segment-sec must be positive\n");
    return false;
  }
  return true;
}

std::string cpuText(const std::vector<int> &cpus) {
  if (cpus.empty())
    return "any";
  std::string s;
  for (size_t i = 0; i < cpus.size(); ++i)
    s += (i ? "," : "") + std::to_string(cpus[i]);
  return s;
}

} // namespace

int main(int argc, char **argv) {
  // One OpenBLAS thread: both lanes manage their own parallelism.
  setenv("OPENBLAS_NUM_THREADS", "1", 0);

  Args args;
  if (!parseArgs(argc, argv, args))
    return EXIT_FAILURE;

  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);

  std::printf(
    "==============================================================\n"
    "  NNTrainer ASR + Speaker Diarization (discretion)\n"
    "  ASR model : %s  (segments up to %.1f s, language '%s')\n"
    "  SD weights: %s  (window %.0f s, step %.0f s)\n"
    "  cores     : ASR %s (threads %d)  |  SD %s (threads %d)\n"
    "==============================================================\n",
    args.model_dir.c_str(), args.asr.segment_sec, args.asr.language.c_str(),
    args.sd.weights.c_str(), args.sd.window_sec, args.sd.step_sec,
    cpuText(args.asr.cpus).c_str(), args.asr.threads,
    cpuText(args.sd.cpus).c_str(), args.sd.threads);
  std::fflush(stdout);

  // Load both models first: stream time zero is the start of the audio feed.
  const auto t_load = discretion::Clock::now();
  discretion::SdLane sd;
  if (!sd.init(args.sd)) {
    std::fprintf(stderr, "error: cannot initialize the diarization models\n");
    return EXIT_FAILURE;
  }
  discretion::AsrLane asr;
  std::string error;
  if (!asr.init(args.asr, error)) {
    std::fprintf(stderr, "error: cannot initialize the ASR model: %s\n",
                 error.c_str());
    return EXIT_FAILURE;
  }
  discretion::AudioFanout fanout;
  if (!fanout.open(args.audio, error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    return EXIT_FAILURE;
  }
  std::printf(
    "models loaded in %.1f s. Streaming ...  (Ctrl-C to stop)\n\n",
    std::chrono::duration<double>(discretion::Clock::now() - t_load).count());
  std::fflush(stdout);

  discretion::StreamClock clock;
  discretion::BlockQueue asr_in, sd_in;
  discretion::Merger merger(clock, stdout, args.verbose);

  asr.start(asr_in, clock, merger);
  sd.start(sd_in, clock, merger);
  fanout.start(clock, {&asr_in, &sd_in});

  fanout.join();
  const double audio_end = clock.now();
  asr.join();
  sd.join();
  const double done = clock.now();

  const auto &as = asr.stats();
  const auto &ss = sd.stats();
  const auto ms = merger.summary();
  const double audio = fanout.audioSeconds();

  double cpu = 0.0;
  long peak_kb = 0;
#if defined(__linux__)
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) == 0) {
    cpu = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec / 1e6 + ru.ru_stime.tv_sec +
          ru.ru_stime.tv_usec / 1e6;
    peak_kb = ru.ru_maxrss;
  }
#endif

  std::printf(
    "\n==============================================================\n"
    "  Summary\n"
    "==============================================================\n"
    "  audio            : %.1f s, fed in %.1f s; everything finished "
    "%.1f s after the audio ended\n"
    "  ASR              : %zu transcribed, %zu silent, %zu dropped | "
    "model time %.1f s (RTF %.2f) | max text delay %.1f s | max backlog %zu\n"
    "  SD               : %zu turns, %zu windows, %zu speakers | "
    "model time %.1f s (RTF %.2f)\n"
    "  merged lines     : %zu | lag after segment end: mean %.1f s, "
    "max %.1f s (ASR alone: mean %.1f s)\n"
    "  process          : CPU time %.1f core-seconds (%.2f cores on "
    "average over %.1f s) | peak RSS %.0f MB\n"
    "==============================================================\n",
    audio, audio_end, done - audio_end, as.segments, as.silent, as.dropped,
    as.proc_sec, audio > 0 ? as.proc_sec / audio : 0.0, as.max_delay_sec,
    as.max_backlog, ss.turns, ss.windows, ss.speakers, ss.model_ms / 1000.0,
    audio > 0 ? ss.model_ms / 1000.0 / audio : 0.0, ms.lines, ms.mean_lag,
    ms.max_lag, ms.mean_asr_lag, cpu, done > 0 ? cpu / done : 0.0, done,
    peak_kb / 1024.0);
  return EXIT_SUCCESS;
}
