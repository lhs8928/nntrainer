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
#include <ctime>
#include <iostream>
#include <memory>
#include <string>

#if defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

#include "asr_lane.h"
#include "audio_fanout.h"
#include "batch_pipeline.h"
#include "cpu_affinity.h"
#include "events.h"
#include "merger.h"
#include "sd_lane.h"
#include "speech_ops.h"

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
    "       --vad (cut segments at pauses too; min length 1.5 s)\n"
    "       --language NAME (English, \"\" = automatic)  --max-backlog N (2)\n"
    "SD:    --window S (20)  --step S (20)\n"
    "       --no-sd (ASR only: no speaker labels, no SD weights needed)\n"
    "Cores: --asr-cpus LIST  --sd-cpus LIST (e.g. 0,1 / 2-3)\n"
    "       --asr-threads N  --sd-threads N\n"
    "       --no-word-timestamps (label whole segments instead of words)\n"
    "Serial: default. ASR and SD take turns (each call uses all its threads);\n"
    "        --parallel runs both lanes at once on their own cores instead\n"
    "Other: --seconds N (stop after N s of audio)  --verbose\n"
    "Output: --jsonl FILE (also write each merged line as one JSON object)\n"
    "Batch: --batch transcribes a whole file the WhisperX way: PyanNet voice\n"
    "        activity -> chunks of up to --chunk-sec S (30) -> one transcript\n"
    "        per chunk -> sentences -> speaker of the longest overlap with a\n"
    "        whole-file diarization. Needs --sd-weights even with --no-sd\n"
    "        (voice activity). --vad-onset X (0.5) --vad-offset X (0.363)\n"
    "        --vad-step S (1, PyanNet hop when not diarizing)\n"
    "        --pause-sec S (0: off; a pause this long always ends a chunk)\n"
    "        --line-speakers (split lines at speaker changes, then cluster "
    "one\n"
    "        embedding per line; --line-threshold X (0.5) cosine to merge)\n"
    "Speech: --serve --speech --sd-weights W [--asr-sec 8,12.5] answers\n"
    "        ASR|EMBED|SEGMENT<TAB>PCM16 file<TAB>JSON file per line: text "
    "and\n"
    "        word times, a speaker embedding, PyanNet frame classes per 10 s\n"
    "        window. One ASR graph per length; longer audio is cut to fit.\n"
    "Serve: --serve keeps the ASR model loaded and reads requests from stdin,\n"
    "        one per line: WAV<TAB>JSONL<TAB>0|1 (1 = with speaker labels).\n"
    "        Replies on stdout: READY, then OK or ERR <reason> per request.\n"
    "        The language is fixed at start-up; the banner goes to stderr.\n",
    prog);
}

struct Args {
  std::string model_dir;
  discretion::AsrOptions asr;
  discretion::SdOptions sd;
  discretion::AudioFanout::Options audio;
  bool verbose = false;
  bool no_sd = false;     // --no-sd: skip the diarization lane entirely
  bool serve = false;     // --serve: resident model, requests on stdin
  bool serial = true;     // one model call at a time (--parallel to turn off)
  std::string jsonl_path; // --jsonl: JSON Lines copy of the merged lines
  bool batch = false;     // --batch: WhisperX-style whole-file pipeline
  bool speech = false;    // --serve --speech: ASR / EMBED / SEGMENT requests
  std::vector<float> asr_secs; // --asr-sec: model graph lengths (speech)
  discretion::BatchOptions batch_opt;
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
      else if (k == "--vad")
        a.asr.vad = true;
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
      else if (k == "--no-sd")
        a.no_sd = true;
      else if (k == "--serve")
        a.serve = true;
      else if (k == "--verbose")
        a.verbose = true;
      else if (k == "--jsonl")
        a.jsonl_path = next();
      else if (k == "--serial")
        a.serial = true;
      else if (k == "--parallel")
        a.serial = false;
      else if (k == "--speech")
        a.speech = true;
      else if (k == "--asr-sec") {
        std::string list = next();
        for (size_t p = 0; p < list.size();) {
          const size_t q = list.find(',', p);
          a.asr_secs.push_back(std::stof(list.substr(p, q - p)));
          p = q == std::string::npos ? list.size() : q + 1;
        }
      } else if (k == "--batch")
        a.batch = true;
      else if (k == "--chunk-sec")
        a.batch_opt.chunk_sec = std::stod(next());
      else if (k == "--vad-onset")
        a.batch_opt.onset = std::stof(next());
      else if (k == "--vad-offset")
        a.batch_opt.offset = std::stof(next());
      else if (k == "--vad-step")
        a.batch_opt.vad_step_sec = std::stod(next());
      else if (k == "--pause-sec")
        a.batch_opt.pause_sec = std::stod(next());
      else if (k == "--line-speakers")
        a.batch_opt.line_speakers = true;
      else if (k == "--line-threshold")
        a.batch_opt.line_threshold = std::stof(next());
      else
        throw std::invalid_argument("unknown option " + k);
    }
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    usage(argv[0]);
    return false;
  }
  if (!a.no_sd && a.sd.weights.empty()) {
    std::fprintf(stderr, "error: --sd-weights is required (or use --no-sd)\n");
    return false;
  }
  if (a.batch) {
    if (a.sd.weights.empty()) {
      std::fprintf(stderr, "error: --batch needs --sd-weights (its voice "
                           "activity runs on PyanNet)\n");
      return false;
    }
    if (a.batch_opt.chunk_sec <= 0.0 || a.batch_opt.vad_step_sec <= 0.0 ||
        a.batch_opt.offset > a.batch_opt.onset) {
      std::fprintf(stderr, "error: need --chunk-sec > 0, --vad-step > 0 and "
                           "--vad-offset <= --vad-onset\n");
      return false;
    }
    if (!a.serve && a.audio.file.empty()) {
      std::fprintf(stderr, "error: --batch needs --stream-file (or --serve)\n");
      return false;
    }
    // The model is built for one chunk: the longest one.
    a.asr.segment_sec = static_cast<float>(a.batch_opt.chunk_sec);
    a.batch_opt.verbose = a.verbose;
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

/**
 * @brief one --serve request: transcribe one WAV with the resident ASR model,
 *        writing the merged lines to a JSONL file. Lanes, merger and (when
 *        diarizing) the diarizer are made per request, so no state carries
 * over.
 */
bool serveRequest(const Args &args,
                  const std::shared_ptr<discretion::AsrEngine> &engine,
                  const std::string &wav, const std::string &jsonl_path,
                  bool diarize, std::string &error) {
  discretion::AudioFanout::Options audio = args.audio;
  audio.file = wav;
  discretion::AudioFanout fanout;
  if (!fanout.open(audio, error))
    return false;
  std::FILE *jsonl = std::fopen(jsonl_path.c_str(), "w");
  if (!jsonl) {
    error = "cannot open " + jsonl_path;
    return false;
  }

  discretion::StreamClock clock;
  discretion::BlockQueue asr_in, sd_in;
  discretion::Merger merger(clock, stderr, args.verbose);
  char session[32];
  const std::time_t now = std::time(nullptr);
  std::strftime(session, sizeof(session), "%Y-%m-%dT%H:%M:%S",
                std::localtime(&now));
  merger.setJsonl(jsonl, session);

  discretion::AsrLane asr;
  asr.adopt(engine, args.asr);
  std::unique_ptr<discretion::SdLane> sd;
  if (diarize) {
    sd = std::make_unique<discretion::SdLane>();
    if (!sd->init(args.sd)) {
      error = "cannot initialize the diarization models";
      std::fclose(jsonl);
      return false;
    }
  } else {
    // Nothing to wait for: see the one-shot path in main().
    merger.onSdFinished(0.0);
  }

  asr.start(asr_in, clock, merger);
  if (sd) {
    sd->start(sd_in, clock, merger);
    fanout.start(clock, {&asr_in, &sd_in});
  } else {
    fanout.start(clock, {&asr_in});
  }
  fanout.join();
  asr.join();
  if (sd)
    sd->join();
  std::fclose(jsonl);
  return true;
}

/**
 * @brief one --serve --batch request: the WhisperX-style pipeline over one
 *        file, with the models that stay loaded
 */
bool serveBatchRequest(discretion::BatchPipeline &batch, const std::string &wav,
                       const std::string &jsonl_path, bool diarize,
                       std::string &error) {
  std::vector<float> pcm;
  if (!discretion::loadAudioFile(wav, pcm, error))
    return false;
  std::FILE *jsonl = std::fopen(jsonl_path.c_str(), "w");
  if (!jsonl) {
    error = "cannot open " + jsonl_path;
    return false;
  }
  char session[32];
  const std::time_t now = std::time(nullptr);
  std::strftime(session, sizeof(session), "%Y-%m-%dT%H:%M:%S",
                std::localtime(&now));
  discretion::BatchStats st;
  const auto lines = batch.run(pcm, diarize, &st);
  discretion::writeBatchLines(lines, stderr, jsonl, session);
  std::fclose(jsonl);
  std::fprintf(
    stderr,
    "batch: %.1f s audio, %zu regions, %zu chunks (%zu split again), "
    "%zu lines | VAD %.1f s, SD %.1f s, ASR %.1f s\n",
    st.audio_sec, st.regions, st.chunks, st.retries, lines.size(), st.vad_sec,
    st.sd_sec, st.asr_sec);
  return true;
}

/**
 * @brief --serve --speech: the speech models of another program's pipeline;
 *        requests OP<TAB>PCM16<TAB>JSON on stdin, READY / OK / ERR on stdout
 */
int speechServeMain(Args &args) {
  const int proto_fd = dup(STDOUT_FILENO);
  dup2(STDERR_FILENO, STDOUT_FILENO);
  std::FILE *proto = fdopen(proto_fd, "w");
  if (!proto) {
    std::fprintf(stderr, "error: cannot open the reply channel\n");
    return EXIT_FAILURE;
  }
  discretion::SpeechOptions opt;
  opt.asr = args.asr;
  opt.asr_secs = args.asr_secs;
  opt.sd_weights = args.sd.weights;
  discretion::SpeechOps ops;
  std::string error;
  if (!ops.init(opt, error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    return EXIT_FAILURE;
  }
  std::fprintf(proto, "READY %s\n", ops.describe().c_str());
  std::fflush(proto);
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty())
      continue;
    const size_t t1 = line.find('\t');
    const size_t t2 = t1 == std::string::npos ? t1 : line.find('\t', t1 + 1);
    std::string reply = "OK";
    if (t2 == std::string::npos) {
      reply = "ERR bad request (expected OP<TAB>PCM16<TAB>JSON)";
    } else if (!ops.handle(line.substr(0, t1), line.substr(t1 + 1, t2 - t1 - 1),
                           line.substr(t2 + 1), error)) {
      reply = "ERR " + error;
      for (char &c : reply)
        if (c == '\n' || c == '\r')
          c = ' ';
    }
    std::fprintf(proto, "%s\n", reply.c_str());
    std::fflush(proto);
  }
  std::fclose(proto);
  return EXIT_SUCCESS;
}

/** @brief --serve: load the ASR model once, then answer requests from stdin */
int serveMain(Args &args) {
  // The model prints its log to stdout. Point stdout at stderr so the log stays
  // out of the protocol, and keep the protocol on a copy of the original
  // stdout.
  const int proto_fd = dup(STDOUT_FILENO);
  dup2(STDERR_FILENO, STDOUT_FILENO);
  std::FILE *proto = fdopen(proto_fd, "w");
  if (!proto) {
    std::fprintf(stderr, "error: cannot open the reply channel\n");
    return EXIT_FAILURE;
  }

  // Same gate as the one-shot path: one model call at a time.
  std::mutex gate;
  if (args.serial) {
    args.asr.gate = &gate;
    args.sd.gate = &gate;
  }
  discretion::AsrLane boot;
  std::string error;
  if (!boot.init(args.asr, error)) {
    std::fprintf(stderr, "error: cannot initialize the ASR model: %s\n",
                 error.c_str());
    return EXIT_FAILURE;
  }
  const auto engine = boot.engine();
  std::unique_ptr<discretion::BatchPipeline> batch;
  if (args.batch) {
    batch = discretion::BatchPipeline::create(engine, args.sd.weights,
                                              args.batch_opt, error);
    if (!batch) {
      std::fprintf(stderr, "error: %s\n", error.c_str());
      return EXIT_FAILURE;
    }
  }
  std::fprintf(stderr, "serving with %s (language '%s'%s)\n",
               args.model_dir.c_str(), args.asr.language.c_str(),
               args.no_sd ? ", no speaker labels" : "");
  std::fprintf(proto, "READY\n");
  std::fflush(proto);

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty())
      continue;
    // request: WAV<TAB>JSONL<TAB>0|1
    const size_t t1 = line.find('\t');
    const size_t t2 = t1 == std::string::npos ? t1 : line.find('\t', t1 + 1);
    std::string reply;
    if (t2 == std::string::npos) {
      reply = "ERR bad request (expected WAV<TAB>JSONL<TAB>0|1)";
    } else {
      const std::string wav = line.substr(0, t1);
      const std::string jsonl = line.substr(t1 + 1, t2 - t1 - 1);
      const bool diarize = line.substr(t2 + 1) == "1";
      std::string err;
      if (diarize && args.no_sd)
        err = "speaker labels requested but the server runs with --no-sd";
      else if (batch ? !serveBatchRequest(*batch, wav, jsonl, diarize, err)
                     : !serveRequest(args, engine, wav, jsonl, diarize, err))
        err = err.empty() ? "failed" : err;
      reply = err.empty() ? "OK" : "ERR " + err;
      for (char &c : reply)
        if (c == '\n' || c == '\r')
          c = ' ';
    }
    std::fprintf(proto, "%s\n", reply.c_str());
    std::fflush(proto);
  }
  std::fclose(proto);
  return EXIT_SUCCESS;
}

/** @brief --batch without --serve: the WhisperX-style pipeline over one file */
int batchMain(Args &args) {
  std::printf(
    "==============================================================\n"
    "  NNTrainer ASR + Speaker Diarization (discretion, batch)\n"
    "  ASR model : %s  (chunks up to %.1f s, language '%s')\n"
    "  VAD       : onset %.3f, offset %.3f%s\n"
    "==============================================================\n",
    args.model_dir.c_str(), args.batch_opt.chunk_sec, args.asr.language.c_str(),
    args.batch_opt.onset, args.batch_opt.offset,
    args.no_sd ? " | no speaker labels (--no-sd)"
               : " | whole-file diarization");
  std::fflush(stdout);

  std::FILE *jsonl = nullptr;
  if (!args.jsonl_path.empty()) {
    jsonl = std::fopen(args.jsonl_path.c_str(), "w");
    if (!jsonl) {
      std::fprintf(stderr, "error: cannot open %s\n", args.jsonl_path.c_str());
      return EXIT_FAILURE;
    }
  }
  const auto t_load = discretion::Clock::now();
  discretion::AsrLane boot;
  std::string error;
  if (!boot.init(args.asr, error)) {
    std::fprintf(stderr, "error: cannot initialize the ASR model: %s\n",
                 error.c_str());
    return EXIT_FAILURE;
  }
  auto batch = discretion::BatchPipeline::create(boot.engine(), args.sd.weights,
                                                 args.batch_opt, error);
  if (!batch) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    return EXIT_FAILURE;
  }
  std::vector<float> pcm;
  if (!discretion::loadAudioFile(args.audio.file, pcm, error)) {
    std::fprintf(stderr, "error: %s\n", error.c_str());
    return EXIT_FAILURE;
  }
  std::printf(
    "models loaded in %.1f s.\n\n",
    std::chrono::duration<double>(discretion::Clock::now() - t_load).count());
  std::fflush(stdout);

  char session[32];
  const std::time_t now = std::time(nullptr);
  std::strftime(session, sizeof(session), "%Y-%m-%dT%H:%M:%S",
                std::localtime(&now));
  discretion::BatchStats st;
  const auto t_run = discretion::Clock::now();
  const auto lines = batch->run(pcm, !args.no_sd, &st);
  const double took =
    std::chrono::duration<double>(discretion::Clock::now() - t_run).count();
  discretion::writeBatchLines(lines, stdout, jsonl, session);
  if (jsonl)
    std::fclose(jsonl);

  std::printf(
    "\n==============================================================\n"
    "  Summary\n"
    "==============================================================\n"
    "  audio            : %.1f s | %zu speech regions -> %zu chunks (%zu split "
    "again) -> %zu lines\n"
    "  time             : VAD %.1f s, SD %.1f s (%zu speakers), ASR %.1f s | "
    "total %.1f s (RTF %.2f)\n"
    "==============================================================\n",
    st.audio_sec, st.regions, st.chunks, st.retries, lines.size(), st.vad_sec,
    st.sd_sec, st.speakers, st.asr_sec, took,
    st.audio_sec > 0 ? took / st.audio_sec : 0.0);
  return EXIT_SUCCESS;
}

} // namespace

#if defined(USE_BLAS)
extern "C" void openblas_set_num_threads(int num_threads) __attribute__((weak));
#endif

int main(int argc, char **argv) {
  // One OpenBLAS thread: both lanes manage their own parallelism. The library
  // is already loaded, so the environment variable alone is not enough.
#if defined(USE_BLAS)
  if (openblas_set_num_threads) {
    openblas_set_num_threads(1);
  }
#endif
  setenv("OPENBLAS_NUM_THREADS", "1", 0);

  Args args;
  if (!parseArgs(argc, argv, args))
    return EXIT_FAILURE;

  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);

  if (args.serve && args.speech)
    return speechServeMain(args);
  if (args.serve)
    return serveMain(args);
  if (args.batch)
    return batchMain(args);

  std::printf(
    "==============================================================\n"
    "  NNTrainer ASR + Speaker Diarization (discretion)\n"
    "  ASR model : %s  (segments up to %.1f s%s, language '%s')\n"
    "  SD weights: %s  (window %.0f s, step %.0f s)\n"
    "  cores     : ASR %s (threads %d)  |  SD %s (threads %d)\n"
    "==============================================================\n",
    args.model_dir.c_str(), args.asr.segment_sec,
    args.asr.vad ? ", VAD: cut at pauses" : "", args.asr.language.c_str(),
    args.no_sd ? "off (--no-sd)" : args.sd.weights.c_str(), args.sd.window_sec,
    args.sd.step_sec, cpuText(args.asr.cpus).c_str(), args.asr.threads,
    args.no_sd ? "off" : cpuText(args.sd.cpus).c_str(), args.sd.threads);
  std::fflush(stdout);

  // Serial mode: the same gate is held by both lanes around each model call.
  std::mutex gate;
  if (args.serial) {
    args.asr.gate = &gate;
    args.sd.gate = &gate;
  }

  // Opened before the models load, so a bad --jsonl path fails at once.
  std::FILE *jsonl = nullptr;
  if (!args.jsonl_path.empty()) {
    jsonl = std::fopen(args.jsonl_path.c_str(), "w");
    if (!jsonl) {
      std::fprintf(stderr, "error: cannot open %s\n", args.jsonl_path.c_str());
      return EXIT_FAILURE;
    }
  }

  // Load both models first: stream time zero is the start of the audio feed.
  const auto t_load = discretion::Clock::now();
  discretion::SdLane sd;
  if (!args.no_sd && !sd.init(args.sd)) {
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
  if (jsonl) {
    char session[32];
    const std::time_t now = std::time(nullptr);
    std::strftime(session, sizeof(session), "%Y-%m-%dT%H:%M:%S",
                  std::localtime(&now));
    merger.setJsonl(jsonl, session);
  }

  asr.start(asr_in, clock, merger);
  if (args.no_sd) {
    // Nothing to wait for: the merger otherwise holds text until the diarizer
    // has confirmed it. Lines then carry the label "(no speaker)".
    merger.onSdFinished(0.0);
    fanout.start(clock, {&asr_in});
  } else {
    sd.start(sd_in, clock, merger);
    fanout.start(clock, {&asr_in, &sd_in});
  }

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
  if (jsonl)
    std::fclose(jsonl);
  return EXIT_SUCCESS;
}
