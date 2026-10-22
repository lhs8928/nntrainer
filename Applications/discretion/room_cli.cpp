// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   room_cli.cpp
 * @date   22 October 2026
 * @brief  nntr_room: the room recognizer on a file or stdin, for tests
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Drives libnntr_discretion_jni.so through its C interface only, as the app
 * does, and writes every event as one JSON line on stdout:
 *
 *   nntr_room --config '{"model_dir":..,"sd_weights":..,"silero":..}'
 *             [--enroll NAME=voice.wav ...] [--realtime] [--seconds S] INPUT
 *
 * INPUT is a 16 kHz mono PCM16 WAV file, '-' for raw PCM16 on stdin (a live
 * stream relayed from another machine), or 'mic' for the phone's microphone
 * (AAudio, voice recognition input). --realtime feeds a file at its own pace,
 * as a microphone would. Listening ends at the end of the input, after
 * --seconds, or on SIGINT / SIGTERM / SIGHUP; what was heard is still
 * recognized and written before the program exits.
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#ifdef __ANDROID__
#include <aaudio/AAudio.h>
#endif

#include "room_api.h"

namespace {

std::atomic<bool> stop_listening{false};

std::atomic<int> stop_signal{0};

void onSignal(int sig) {
  stop_signal = sig;
  stop_listening = true;
}

#ifdef __ANDROID__
/** @brief the phone's microphone, 16 kHz mono PCM16, read in blocks */
class Mic {
public:
  ~Mic() {
    if (stream_) {
      AAudioStream_requestStop(stream_);
      AAudioStream_close(stream_);
    }
  }
  bool open(std::string &error) {
    AAudioStreamBuilder *b = nullptr;
    if (AAudio_createStreamBuilder(&b) != AAUDIO_OK) {
      error = "AAudio unavailable";
      return false;
    }
    AAudioStreamBuilder_setDirection(b, AAUDIO_DIRECTION_INPUT);
    AAudioStreamBuilder_setSampleRate(b, 16000);
    AAudioStreamBuilder_setChannelCount(b, 1);
    AAudioStreamBuilder_setFormat(b, AAUDIO_FORMAT_PCM_I16);
    AAudioStreamBuilder_setSharingMode(b, AAUDIO_SHARING_MODE_SHARED);
    AAudioStreamBuilder_setInputPreset(b,
                                       AAUDIO_INPUT_PRESET_VOICE_RECOGNITION);
    const aaudio_result_t r = AAudioStreamBuilder_openStream(b, &stream_);
    AAudioStreamBuilder_delete(b);
    if (r != AAUDIO_OK) {
      error = std::string("cannot open the microphone: ") +
              AAudio_convertResultToText(r);
      stream_ = nullptr;
      return false;
    }
    if (AAudioStream_getSampleRate(stream_) != 16000 ||
        AAudioStream_getChannelCount(stream_) != 1 ||
        AAudioStream_getFormat(stream_) != AAUDIO_FORMAT_PCM_I16) {
      error = "the microphone gives " +
              std::to_string(AAudioStream_getSampleRate(stream_)) + " Hz, " +
              std::to_string(AAudioStream_getChannelCount(stream_)) +
              " channels, not 16 kHz mono PCM16";
      return false;
    }
    if (AAudioStream_requestStart(stream_) != AAUDIO_OK) {
      error = "cannot start the microphone";
      return false;
    }
    return true;
  }
  /** @brief n samples (waits for them); -1 on error */
  long read(int16_t *x, size_t n) {
    return AAudioStream_read(stream_, x, static_cast<int32_t>(n), 1000000000LL);
  }

private:
  AAudioStream *stream_ = nullptr;
};
#endif

bool readWav(const std::string &path, std::vector<int16_t> &pcm,
             std::string &error) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    error = "cannot open " + path;
    return false;
  }
  std::vector<char> raw((std::istreambuf_iterator<char>(f)), {});
  auto u16 = [&](size_t i) {
    return static_cast<unsigned>(static_cast<unsigned char>(raw[i]) |
                                 static_cast<unsigned char>(raw[i + 1]) << 8);
  };
  auto u32 = [&](size_t i) { return u16(i) | u16(i + 2) << 16; };
  if (raw.size() < 12 || std::memcmp(raw.data(), "RIFF", 4) ||
      std::memcmp(raw.data() + 8, "WAVE", 4)) {
    error = path + " is not a WAV file";
    return false;
  }
  bool fmt_ok = false;
  for (size_t i = 12; i + 8 <= raw.size();) {
    const size_t size = u32(i + 4);
    if (!std::memcmp(raw.data() + i, "fmt ", 4) && i + 24 <= raw.size()) {
      fmt_ok = u16(i + 8) == 1 && u16(i + 10) == 1 && u32(i + 12) == 16000 &&
               u16(i + 22) == 16;
    } else if (!std::memcmp(raw.data() + i, "data", 4)) {
      if (!fmt_ok) {
        error = path + ": 16 kHz mono PCM16 only (ffmpeg -ar 16000 -ac 1)";
        return false;
      }
      const size_t n = std::min(size, raw.size() - i - 8) / 2;
      pcm.resize(n);
      std::memcpy(pcm.data(), raw.data() + i + 8, n * 2);
      return true;
    }
    i += 8 + size + (size & 1);
  }
  error = path + ": no audio data";
  return false;
}

std::string text(nd_room *room, int (*get)(nd_room *, char *, int)) {
  std::vector<char> buf(1 << 16);
  while (true) {
    const int n = get(room, buf.data(), static_cast<int>(buf.size()));
    if (n < 0)
      return "";
    if (n < static_cast<int>(buf.size()))
      return std::string(buf.data(), n);
    buf.resize(n + 1);
  }
}

/** @brief one JSON line per event of the array nd_poll gave */
std::FILE *events = stdout;

void print(const std::string &array) {
  // the events are JSON objects: split at the top level
  int depth = 0;
  bool in_str = false, esc = false;
  std::string cur;
  for (size_t i = 1; i + 1 < array.size(); ++i) {
    const char c = array[i];
    if (in_str) {
      cur += c;
      if (esc)
        esc = false;
      else if (c == '\\')
        esc = true;
      else if (c == '"')
        in_str = false;
      continue;
    }
    if (c == '"')
      in_str = true;
    if (c == '{')
      ++depth;
    if (c == '}')
      --depth;
    if (c == ',' && depth == 0) {
      std::fprintf(events, "%s\n", cur.c_str());
      cur.clear();
      continue;
    }
    cur += c;
  }
  if (!cur.empty())
    std::fprintf(events, "%s\n", cur.c_str());
  std::fflush(events);
}

} // namespace

int main(int argc, char **argv) {
  std::string config, input;
  std::vector<std::pair<std::string, std::string>> enroll;
  bool realtime = false;
  double seconds = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--config" && i + 1 < argc)
      config = argv[++i];
    else if (a == "--enroll" && i + 1 < argc) {
      const std::string v = argv[++i];
      const size_t eq = v.find('=');
      if (eq == std::string::npos) {
        std::fprintf(stderr, "--enroll NAME=voice.wav\n");
        return 2;
      }
      enroll.emplace_back(v.substr(0, eq), v.substr(eq + 1));
    } else if (a == "--realtime")
      realtime = true;
    else if (a == "--seconds" && i + 1 < argc)
      seconds = std::stod(argv[++i]);
    else
      input = a;
  }
  if (config.empty() || input.empty()) {
    std::fprintf(stderr, "usage: nntr_room --config JSON [--enroll NAME=WAV] "
                         "[--realtime] [--seconds S] INPUT.wav|-|mic\n");
    return 2;
  }
  // the models log to stdout: the events keep stdout to themselves
  std::fflush(stdout);
  events = fdopen(dup(STDOUT_FILENO), "w");
  dup2(STDERR_FILENO, STDOUT_FILENO);
  if (!events) {
    std::fprintf(stderr, "error: cannot open the event channel\n");
    return 1;
  }
  char err[512] = {0};
  const auto t0 = std::chrono::steady_clock::now();
  nd_room *room = nd_create(config.c_str(), err, sizeof(err));
  if (!room) {
    std::fprintf(stderr, "error: %s\n", err);
    return 1;
  }
  std::fprintf(
    stderr, "loaded in %.1f s: %s\n",
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count(),
    text(room, nd_status).c_str());
  for (const auto &[name, path] : enroll) {
    std::vector<int16_t> pcm;
    std::string error;
    if (!readWav(path, pcm, error) ||
        nd_enroll(room, name.c_str(), pcm.data(), static_cast<int>(pcm.size()),
                  err, sizeof(err)) != 0) {
      std::fprintf(stderr, "enroll %s: %s\n", name.c_str(),
                   error.empty() ? err : error.c_str());
      nd_destroy(room);
      return 1;
    }
  }

  std::signal(SIGINT, onSignal);
  std::signal(SIGTERM, onSignal);
  std::signal(SIGHUP, onSignal);
  const size_t CHUNK = 1600; // 100 ms
  const auto start = std::chrono::steady_clock::now();
  auto over = [&] {
    return stop_listening ||
           (seconds > 0 && std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - start)
                               .count() >= seconds);
  };
  size_t fed = 0;
  auto give = [&](const int16_t *x, size_t n) {
    while (nd_feed(room, x, static_cast<int>(n), -1) == 1) {
      print(text(room, nd_poll));
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    fed += n;
    print(text(room, nd_poll));
  };
  if (input == "mic") {
#ifdef __ANDROID__
    Mic mic;
    std::string error;
    if (!mic.open(error)) {
      std::fprintf(stderr, "error: %s\n", error.c_str());
      nd_destroy(room);
      return 1;
    }
    std::fprintf(stderr, "listening (stop with Ctrl+C or SIGINT)\n");
    std::vector<int16_t> buf(CHUNK);
    double sum = 0;
    size_t count = 0;
    while (!over()) {
      const long got = mic.read(buf.data(), CHUNK);
      if (got < 0) {
        std::fprintf(stderr, "error: the microphone stopped (%ld)\n", got);
        break;
      }
      for (long i = 0; i < got; ++i)
        sum += static_cast<double>(buf[i]) * buf[i];
      count += got;
      if (count >= 5 * 16000) { // the level every 5 s, to see it hears
        std::fprintf(stderr, "mic level %.4f\n",
                     std::sqrt(sum / count) / 32768);
        sum = 0;
        count = 0;
      }
      if (got)
        give(buf.data(), got);
    }
    if (stop_signal)
      std::fprintf(stderr, "stopped by signal %d\n", stop_signal.load());
#else
    std::fprintf(stderr, "error: 'mic' is for Android only\n");
    nd_destroy(room);
    return 1;
#endif
  } else if (input == "-") {
    std::vector<int16_t> buf(CHUNK);
    while (!over()) {
      const size_t got = std::fread(buf.data(), 2, CHUNK, stdin);
      if (got)
        give(buf.data(), got);
      if (got < CHUNK)
        break;
    }
  } else {
    std::vector<int16_t> pcm;
    std::string error;
    if (!readWav(input, pcm, error)) {
      std::fprintf(stderr, "error: %s\n", error.c_str());
      nd_destroy(room);
      return 1;
    }
    for (size_t i = 0; i < pcm.size() && !over(); i += CHUNK) {
      if (realtime)
        std::this_thread::sleep_until(
          start + std::chrono::microseconds(i * 1000000 / 16000));
      give(pcm.data() + i, std::min(CHUNK, pcm.size() - i));
    }
  }
  nd_flush(room);
  print(text(room, nd_poll));
  const double wall =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
  std::fprintf(stderr, "audio %.1f s, wall %.1f s (RTF %.2f)\nstatus: %s\n",
               fed / 16000.0, wall, wall / std::max(1e-9, fed / 16000.0),
               text(room, nd_status).c_str());
  nd_destroy(room);
  return 0;
}
