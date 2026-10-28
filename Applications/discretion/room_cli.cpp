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
 *   nntr_room --config '{"model_dir":..,"segmentation_weights":..,
 *                        "speaker_weights":..,"silero":..}'
 *             [--enroll NAME=voice.wav ...] [--enroll-vector NAME=v.json ...]
 *             [--pin LINE=NAME ...] [--voices NAME,NAME] [--commands]
 *             [--pcm-out PATH] [--realtime] [--seconds S] INPUT
 *
 * INPUT is a 16 kHz mono PCM16 WAV file, '-' for raw PCM16 on stdin (a live
 * stream relayed from another machine), or 'mic' for the phone's microphone
 * (AAudio, voice recognition input). --realtime feeds a file at its own pace,
 * as a microphone would. Listening ends at the end of the input, after
 * --seconds, or on SIGINT / SIGTERM / SIGHUP; what was heard is still
 * recognized and written before the program exits. --pin corrects a line as a
 * person would, as soon as it is written (NAME '-' = nobody), to see what the
 * diarizer makes of the lines after it. --enroll-vector enrolls a voice the
 * server keeps (a JSON array of the voice model's numbers: 192 ERes2NetV2,
 * 256 WeSpeaker; or an object with "vector");
 * --voices writes, at the end, {"type":"voice","person","vector"} for each
 * name (the voice to remember, nothing when there is none). --pcm-out also
 * writes the audio it recognizes, as raw PCM16 in the order it is fed, to PATH
 * (a file or a FIFO another process reads): the app's job of sending the same
 * audio to the server while the phone recognizes it (ALICE's ASR 비교 tab).
 * Sample k of that output is time k / 16 ms of the line events.
 *
 * --commands reads corrections from stdin while it listens (not with '-'),
 * one JSON object per line, as an app gets them from the server:
 *   {"op":"pin","line":"audio-20","person":"홍길동"|null}
 *   {"op":"map_speaker","label":"화자 2","person":"홍길동"}
 *   {"op":"enroll","person":"홍길동","vector":[192 or 256 numbers]}
 *   {"op":"remember","person":"홍길동","on":true}   its voice at the end
 *   {"op":"record","person":"홍길동","seconds":6}  the next seconds are theirs
 * and answers each with {"type":"ack","op":..,"rc":..} (rc as the C function;
 * "k" is copied when given).
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#ifdef __ANDROID__
#include <aaudio/AAudio.h>
#endif

#include "json.hpp"
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

std::string text(nntr_room *room, int (*get)(nntr_room *, char *, int)) {
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

/** @brief one JSON line per event of the array nntr_room_poll gave */
std::FILE *events = stdout;
/** @brief --pin: line id -> person ("-" = nobody), applied once written */
std::map<std::string, std::string> pins;
nntr_room *pinned_room = nullptr;
/** @brief events come from the main thread and the command reader */
std::mutex events_m;
/** @brief people whose voice is written at the end (--voices, remember) */
std::set<std::string> remembered;
std::mutex remembered_m;

void writeEvent(const nlohmann::json &e) {
  std::lock_guard<std::mutex> lk(events_m);
  std::fprintf(events, "%s\n", e.dump().c_str());
  std::fflush(events);
}

/** @brief --commands: corrections from stdin, one JSON object per line */
void readCommands(nntr_room *room) {
  using nlohmann::json;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.find_first_not_of(" \t\r") == std::string::npos)
      continue;
    json ack = {{"type", "ack"}, {"rc", -1}};
    try {
      const json c = json::parse(line);
      const std::string op = c.value("op", "");
      ack["op"] = op;
      if (c.contains("k"))
        ack["k"] = c["k"];
      const bool someone = c.contains("person") && c["person"].is_string();
      const std::string person = someone ? c["person"].get<std::string>() : "";
      if (op == "pin") {
        const std::string id = c.value("line", "");
        ack["line"] = id;
        ack["rc"] =
          nntr_room_pin(room, id.c_str(), someone ? person.c_str() : nullptr);
      } else if (op == "map_speaker" && someone) {
        const std::string label = c.value("label", "");
        ack["label"] = label;
        ack["rc"] = nntr_room_map_speaker(room, label.c_str(), person.c_str());
      } else if (op == "enroll" && someone) {
        const auto v = c.at("vector").get<std::vector<float>>();
        char err[512] = {0};
        const int rc =
          nntr_room_enroll_vector(room, person.c_str(), v.data(),
                                  static_cast<int>(v.size()), err, sizeof(err));
        ack["rc"] = rc;
        if (rc != 0)
          ack["error"] = err;
      } else if (op == "record" && someone) {
        ack["rc"] = nntr_room_begin_enrollment(room, person.c_str(),
                                               c.value("seconds", 6.0));
      } else if (op == "remember" && someone) {
        std::lock_guard<std::mutex> lk(remembered_m);
        if (c.value("on", true))
          remembered.insert(person);
        else
          remembered.erase(person);
        ack["rc"] = 0;
      }
      if (someone)
        ack["person"] = person;
    } catch (const std::exception &e) {
      ack["error"] = e.what();
    }
    writeEvent(ack);
  }
}

void write(const std::string &event) {
  {
    std::lock_guard<std::mutex> lk(events_m);
    std::fprintf(events, "%s\n", event.c_str());
  }
  for (auto it = pins.begin(); it != pins.end(); ++it)
    if (event.find("\"type\":\"line\"") != std::string::npos &&
        event.find("\"id\":\"" + it->first + "\"") != std::string::npos) {
      const int rc =
        nntr_room_pin(pinned_room, it->first.c_str(),
                      it->second == "-" ? nullptr : it->second.c_str());
      std::fprintf(stderr, "pin %s -> %s: %d\n", it->first.c_str(),
                   it->second.c_str(), rc);
      pins.erase(it);
      break;
    }
}

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
      write(cur);
      cur.clear();
      continue;
    }
    cur += c;
  }
  if (!cur.empty())
    write(cur);
  std::fflush(events);
}

/** @brief the numbers of the first JSON array in a file */
bool readVector(const std::string &path, std::vector<float> &v) {
  std::ifstream in(path);
  const std::string s((std::istreambuf_iterator<char>(in)),
                      std::istreambuf_iterator<char>());
  const size_t a = s.find('['), b = s.find(']', a);
  if (a == std::string::npos || b == std::string::npos)
    return false;
  const char *p = s.c_str() + a + 1, *end = s.c_str() + b;
  while (p < end) {
    char *q = nullptr;
    const float x = std::strtof(p, &q);
    if (q == p) {
      ++p;
      continue;
    }
    v.push_back(x);
    p = q;
  }
  return !v.empty();
}

} // namespace

int main(int argc, char **argv) {
  std::string config, input;
  std::vector<std::pair<std::string, std::string>> enroll, vectors;
  std::vector<std::string> voices;
  bool commands = false;
  bool realtime = false;
  std::string pcm_out;
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
    } else if (a == "--enroll-vector" && i + 1 < argc) {
      const std::string v = argv[++i];
      const size_t eq = v.find('=');
      if (eq == std::string::npos) {
        std::fprintf(stderr, "--enroll-vector NAME=vector.json\n");
        return 2;
      }
      vectors.emplace_back(v.substr(0, eq), v.substr(eq + 1));
    } else if (a == "--voices" && i + 1 < argc) {
      std::string v = argv[++i];
      for (size_t at = 0; at <= v.size();) {
        const size_t c = std::min(v.find(',', at), v.size());
        if (c > at)
          voices.push_back(v.substr(at, c - at));
        at = c + 1;
      }
    } else if (a == "--commands") {
      commands = true;
    } else if (a == "--pcm-out" && i + 1 < argc) {
      pcm_out = argv[++i];
    } else if (a == "--pin" && i + 1 < argc) {
      const std::string v = argv[++i];
      const size_t eq = v.find('=');
      if (eq == std::string::npos) {
        std::fprintf(stderr, "--pin LINE=NAME\n");
        return 2;
      }
      pins[v.substr(0, eq)] = v.substr(eq + 1);
    } else if (a == "--realtime")
      realtime = true;
    else if (a == "--seconds" && i + 1 < argc)
      seconds = std::stod(argv[++i]);
    else
      input = a;
  }
  if (config.empty() || input.empty()) {
    std::fprintf(stderr,
                 "usage: nntr_room --config JSON [--enroll NAME=WAV] "
                 "[--enroll-vector NAME=JSON] [--pin LINE=NAME] [--voices A,B] "
                 "[--commands] [--pcm-out PATH] [--realtime] [--seconds S] "
                 "INPUT.wav|-|mic\n");
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
  nntr_room *room = nntr_room_create(config.c_str(), err, sizeof(err));
  if (!room) {
    std::fprintf(stderr, "error: %s\n", err);
    return 1;
  }
  pinned_room = room;
  std::fprintf(
    stderr, "loaded in %.1f s: %s\n",
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count(),
    text(room, nntr_room_status).c_str());
  for (const auto &[name, path] : enroll) {
    std::vector<int16_t> pcm;
    std::string error;
    if (!readWav(path, pcm, error) ||
        nntr_room_enroll(room, name.c_str(), pcm.data(),
                         static_cast<int>(pcm.size()), err, sizeof(err)) != 0) {
      std::fprintf(stderr, "enroll %s: %s\n", name.c_str(),
                   error.empty() ? err : error.c_str());
      nntr_room_destroy(room);
      return 1;
    }
  }
  for (const auto &[name, path] : vectors) {
    std::vector<float> v;
    if (!readVector(path, v) ||
        nntr_room_enroll_vector(room, name.c_str(), v.data(),
                                static_cast<int>(v.size()), err,
                                sizeof(err)) != 0) {
      std::fprintf(stderr, "enroll-vector %s: %s\n", name.c_str(),
                   v.empty() ? "no numbers in the file" : err);
      nntr_room_destroy(room);
      return 1;
    }
  }
  if (commands && input != "-")
    std::thread(readCommands, room).detach();
  else if (commands)
    std::fprintf(stderr, "--commands: stdin carries the audio, ignored\n");

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
  // the audio as it is fed (a FIFO waits here until its reader opens it); a
  // reader that goes away ends the copy, never the recognition
  FILE *copy = nullptr;
  if (!pcm_out.empty()) {
    std::signal(SIGPIPE, SIG_IGN);
    copy = std::fopen(pcm_out.c_str(), "wb");
    if (!copy) {
      std::fprintf(stderr, "error: cannot write %s\n", pcm_out.c_str());
      nntr_room_destroy(room);
      return 1;
    }
  }
  size_t fed = 0;
  auto give = [&](const int16_t *x, size_t n) {
    if (copy && (std::fwrite(x, 2, n, copy) != n || std::fflush(copy) != 0)) {
      std::fprintf(stderr, "pcm-out: the reader went away, copy stopped\n");
      std::fclose(copy);
      copy = nullptr;
    }
    while (nntr_room_feed(room, x, static_cast<int>(n), -1) == 1) {
      print(text(room, nntr_room_poll));
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    fed += n;
    print(text(room, nntr_room_poll));
  };
  if (input == "mic") {
#ifdef __ANDROID__
    Mic mic;
    std::string error;
    if (!mic.open(error)) {
      std::fprintf(stderr, "error: %s\n", error.c_str());
      nntr_room_destroy(room);
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
    nntr_room_destroy(room);
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
      nntr_room_destroy(room);
      return 1;
    }
    for (size_t i = 0; i < pcm.size() && !over(); i += CHUNK) {
      if (realtime)
        std::this_thread::sleep_until(
          start + std::chrono::microseconds(i * 1000000 / 16000));
      give(pcm.data() + i, std::min(CHUNK, pcm.size() - i));
    }
  }
  if (copy)
    std::fclose(copy);
  nntr_room_flush(room);
  print(text(room, nntr_room_poll));
  {
    std::lock_guard<std::mutex> lk(remembered_m);
    remembered.insert(voices.begin(), voices.end());
  }
  std::set<std::string> people;
  {
    std::lock_guard<std::mutex> lk(remembered_m);
    people = remembered;
  }
  for (const auto &who : people) {
    std::vector<float> v(256);
    const int n = nntr_room_voice_of(room, who.c_str(), v.data(),
                                     static_cast<int>(v.size()));
    if (n <= 0 || n > static_cast<int>(v.size()))
      continue;
    v.resize(n);
    writeEvent({{"type", "voice"}, {"person", who}, {"vector", v}});
  }
  const double wall =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
  std::fprintf(stderr, "audio %.1f s, wall %.1f s (RTF %.2f)\nstatus: %s\n",
               fed / 16000.0, wall, wall / std::max(1e-9, fed / 16000.0),
               text(room, nntr_room_status).c_str());
  nntr_room_destroy(room);
  return 0;
}
