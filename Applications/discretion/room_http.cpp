// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   room_http.cpp
 * @date   2 November 2026
 * @brief  nntr_room_http: the room recognizer for a web page on the same phone
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The facilitator opens ALICE's host page in the phone's browser; the page
 * gives the microphone audio it captures to this process on the loopback and
 * takes back the lines, so the phone recognizes the meeting without an app
 * (docs/browser-phone-asr.ko.md). Drives libnntr_discretion_jni.so through
 * room_api.h only, as nntr_room does:
 *
 *   nntr_room_http --config JSON [--port 8031] [--origin URL ...]
 *
 *   GET  /health   {"status":"loading"|"ready"|"error","session":..}
 *   POST /session  {"meeting":KEY,"labels_in_use":[..]} (both optional): a
 *                  recognizer for this meeting; the same KEY again keeps the
 *                  one it has (a page reloaded mid-meeting). Answers
 *                  {"session":KEY,"resumed":bool,"clock_ms":end of the audio
 *                  fed so far}
 *   POST /feed     raw PCM16 16 kHz mono; X-At-Ms: where it starts on the
 *                  recognizer's clock (ms). {"rc":0} taken, 1 busy (give it
 *                  again), -1 error
 *   GET  /poll     {"events":[..]} since the last poll (room_api.h)
 *   GET  /status   nntr_room_status
 *   POST /flush    recognize everything fed (waits)
 *
 * It listens on 127.0.0.1 only. With --origin, only pages of those origins
 * may call it (a request without an Origin, e.g. curl on the phone, may too);
 * without it, any page the browser lets through. Chrome asks the person once
 * per site before a page may reach the loopback.
 */

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <netinet/in.h>
#include <set>
#include <string>
#include <sys/socket.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "json.hpp"
#include "room_api.h"

namespace {

using json = nlohmann::json;
constexpr size_t MAX_HEADER_BYTES = 16 * 1024;
constexpr size_t MAX_FEED_BYTES = 10 * 16000 * 2; // ten seconds at once

std::string base_config;       // --config
std::set<std::string> origins; // --origin; empty: any

std::mutex g_room; // the recognizer and the session fields below
nntr_room *room = nullptr;
std::string session;         // the meeting key of the recognizer
bool fed = false;            // the recognizer has been given audio
double clock_ms = 0;         // the end of the audio fed so far
std::atomic<int> state{0};   // 0 loading, 1 ready, 2 error
std::string load_error;      // when state is 2 (set before it)
std::atomic<long> fed_ms{0}; // audio taken, for the log
std::atomic<long> busy_feeds{0};

std::string lower(std::string s) {
  for (char &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

std::string header(const std::string &head, const std::string &name) {
  size_t pos = head.find("\r\n");
  while (pos != std::string::npos && pos + 2 < head.size()) {
    const size_t end = head.find("\r\n", pos + 2);
    const std::string line = head.substr(pos + 2, end - pos - 2);
    const size_t colon = line.find(':');
    if (colon != std::string::npos && lower(line.substr(0, colon)) == name) {
      size_t v = colon + 1;
      while (v < line.size() && line[v] == ' ')
        ++v;
      return line.substr(v);
    }
    pos = end;
  }
  return "";
}

bool sendAll(int fd, const std::string &data) {
  size_t sent = 0;
  while (sent < data.size()) {
    const ssize_t n =
      send(fd, data.data() + sent, data.size() - sent, MSG_NOSIGNAL);
    if (n <= 0)
      return false;
    sent += static_cast<size_t>(n);
  }
  return true;
}

/** @brief a reply the calling page may read (CORS for its origin) */
void reply(int fd, int status, const std::string &body,
           const std::string &origin) {
  std::string head = "HTTP/1.1 " + std::to_string(status) +
                     (status == 200   ? " OK"
                      : status == 204 ? " No Content"
                      : status == 403 ? " Forbidden"
                      : status == 404 ? " Not Found"
                      : status == 503 ? " Service Unavailable"
                                      : " Bad Request") +
                     "\r\nContent-Type: application/json; charset=utf-8\r\n";
  if (!origin.empty())
    head += "Access-Control-Allow-Origin: " + origin +
            "\r\nVary: Origin\r\nAccess-Control-Allow-Methods: GET, POST, "
            "OPTIONS\r\nAccess-Control-Allow-Headers: Content-Type, "
            "X-At-Ms\r\nAccess-Control-Allow-Private-Network: "
            "true\r\nAccess-Control-Max-Age: 600\r\n";
  head += "Content-Length: " + std::to_string(body.size()) +
          "\r\nConnection: close\r\n\r\n";
  sendAll(fd, head + body);
}

std::string message(const std::string &text) {
  return json{{"message", text}}.dump();
}

/** @brief the whole text of a room_api call that fills a buffer */
template <typename F> std::string text(F call) {
  std::vector<char> buf(1 << 16);
  while (true) {
    const int n = call(buf.data(), static_cast<int>(buf.size()));
    if (n < 0)
      return "";
    if (n < static_cast<int>(buf.size()))
      return std::string(buf.data(), static_cast<size_t>(n));
    buf.resize(static_cast<size_t>(n) + 1);
  }
}

/** @brief a new recognizer for `key` (g_room held); "" or the error */
std::string open(const std::string &key, const json &labels) {
  json config = json::parse(base_config);
  if (labels.is_array() && !labels.empty())
    config["labels_in_use"] = labels;
  if (room) {
    nntr_room_destroy(room);
    room = nullptr;
  }
  char err[512] = {0};
  const auto t0 = std::chrono::steady_clock::now();
  room = nntr_room_create(config.dump().c_str(), err, sizeof(err));
  if (!room)
    return err[0] ? err : "cannot load the models";
  session = key;
  fed = false;
  clock_ms = 0;
  std::fprintf(
    stderr, "session %s: loaded in %.1f s\n", key.c_str(),
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
      .count());
  return "";
}

void handle(int fd) {
  timeval tv{30, 0};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  std::string data;
  char buf[65536];
  size_t head_end;
  while ((head_end = data.find("\r\n\r\n")) == std::string::npos) {
    if (data.size() > MAX_HEADER_BYTES)
      return reply(fd, 400, message("header too large"), "");
    const ssize_t n = recv(fd, buf, sizeof(buf), 0);
    if (n <= 0)
      return;
    data.append(buf, static_cast<size_t>(n));
  }
  const std::string head = data.substr(0, head_end);
  std::string body = data.substr(head_end + 4);
  const size_t sp1 = head.find(' '), sp2 = head.find(' ', sp1 + 1);
  if (sp1 == std::string::npos || sp2 == std::string::npos)
    return reply(fd, 400, message("bad request line"), "");
  const std::string method = head.substr(0, sp1);
  std::string path = head.substr(sp1 + 1, sp2 - sp1 - 1);
  path = path.substr(0, path.find('?'));

  const std::string origin = header(head, "origin");
  if (!origin.empty() && !origins.empty() && !origins.count(origin)) {
    std::fprintf(stderr, "refused %s %s from %s\n", method.c_str(),
                 path.c_str(), origin.c_str());
    return reply(fd, 403, message("origin not allowed"), "");
  }
  const std::string allow = origin.empty() ? "" : origin;
  if (method == "OPTIONS")
    return reply(fd, 204, "", allow);

  const std::string length_text = header(head, "content-length");
  const size_t length =
    length_text.empty() ? 0 : std::strtoul(length_text.c_str(), nullptr, 10);
  if (length > MAX_FEED_BYTES)
    return reply(fd, 400, message("body too large"), allow);
  while (body.size() < length) {
    const ssize_t n = recv(fd, buf, sizeof(buf), 0);
    if (n <= 0)
      return;
    body.append(buf, static_cast<size_t>(n));
  }
  body.resize(length);

  if (method == "GET" && path == "/health") {
    const int s = state.load();
    json out = {{"status", s == 0 ? "loading" : s == 1 ? "ready" : "error"}};
    if (s == 2)
      out["error"] = load_error;
    std::unique_lock<std::mutex> lk(g_room, std::try_to_lock);
    if (lk.owns_lock()) { // busy (loading, flushing): only the status
      out["session"] = session.empty() ? json(nullptr) : json(session);
      out["clock_ms"] = clock_ms;
    }
    return reply(fd, 200, out.dump(), allow);
  }
  if (state.load() != 1)
    return reply(fd, 503,
                 message(state.load() == 0 ? "loading the models"
                                           : "models failed: " + load_error),
                 allow);
  {
    std::lock_guard<std::mutex> lk(g_room);
    if (!room && path != "/session")
      return reply(fd, 503, message("no recognizer: POST /session"), allow);
  }

  if (method == "POST" && path == "/session") {
    json req = json::object();
    if (!body.empty()) {
      req = json::parse(body, nullptr, false);
      if (!req.is_object())
        return reply(fd, 400, message("JSON object expected"), allow);
    }
    const std::string key =
      req.contains("meeting") && req["meeting"].is_string()
        ? req["meeting"].get<std::string>()
        : "";
    const json labels = req.value("labels_in_use", json::array());
    std::lock_guard<std::mutex> lk(g_room);
    const bool resumed = room && fed && !key.empty() && key == session;
    if (room && !fed && labels.empty())
      session = key; // a recognizer that has heard nothing yet serves as new
    else if (!resumed) {
      if (fed) // what the last meeting still had in the queue is dropped
        std::fprintf(stderr, "session %s ends\n", session.c_str());
      const std::string error = open(key, labels);
      if (!error.empty())
        return reply(fd, 400, message(error), allow);
    }
    return reply(
      fd, 200,
      json{{"session", session}, {"resumed", resumed}, {"clock_ms", clock_ms}}
        .dump(),
      allow);
  }
  if (method == "POST" && path == "/feed") {
    if (length == 0 || length % 2)
      return reply(fd, 400, message("body must be PCM16"), allow);
    const std::string at_text = header(head, "x-at-ms");
    const double at =
      at_text.empty() ? -1 : std::strtod(at_text.c_str(), nullptr);
    const int n = static_cast<int>(length / 2);
    std::lock_guard<std::mutex> lk(g_room);
    const int rc = nntr_room_feed(
      room, reinterpret_cast<const int16_t *>(body.data()), n, at);
    if (rc == 0) {
      fed = true;
      const double start = at >= 0 ? at : clock_ms;
      clock_ms = std::max(clock_ms, start + n / 16.0);
      fed_ms += n / 16;
    } else if (rc == 1)
      ++busy_feeds;
    return reply(fd, 200, json{{"rc", rc}}.dump(), allow);
  }
  if (method == "GET" && path == "/poll") {
    std::lock_guard<std::mutex> lk(g_room);
    const std::string events =
      text([](char *out, int cap) { return nntr_room_poll(room, out, cap); });
    return reply(
      fd, 200, "{\"events\":" + (events.empty() ? "[]" : events) + "}", allow);
  }
  if (method == "GET" && path == "/status") {
    std::lock_guard<std::mutex> lk(g_room);
    const std::string s =
      text([](char *out, int cap) { return nntr_room_status(room, out, cap); });
    return reply(fd, 200, s.empty() ? "{}" : s, allow);
  }
  if (method == "POST" && path == "/flush") {
    std::lock_guard<std::mutex> lk(g_room);
    const int rc = nntr_room_flush(room);
    return reply(fd, 200, json{{"rc", rc}}.dump(), allow);
  }
  reply(fd, 404, message("not found"), allow);
}

} // namespace

int main(int argc, char **argv) {
  int port = 8031;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--config" && i + 1 < argc)
      base_config = argv[++i];
    else if (a == "--port" && i + 1 < argc)
      port = std::atoi(argv[++i]);
    else if (a == "--origin" && i + 1 < argc)
      origins.insert(argv[++i]);
    else {
      std::fprintf(stderr, "usage: nntr_room_http --config JSON [--port 8031] "
                           "[--origin URL ...]\n");
      return 2;
    }
  }
  if (base_config.empty() || !json::accept(base_config)) {
    std::fprintf(stderr, "--config needs the recognizer's JSON settings\n");
    return 2;
  }
  // the models log to stdout; this process has nothing else to say there
  dup2(STDERR_FILENO, STDOUT_FILENO);

  const int server = socket(AF_INET, SOCK_STREAM, 0);
  const int on = 1;
  setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port));
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (server < 0 ||
      bind(server, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 ||
      listen(server, 16) != 0) {
    std::perror("bind");
    return 1;
  }
  std::fprintf(stderr, "nntr_room_http on 127.0.0.1:%d, origins: %s\n", port,
               origins.empty() ? "any"
                               : std::to_string(origins.size()).c_str());

  // the models load while the page can already ask how far they are
  std::thread([] {
    std::lock_guard<std::mutex> lk(g_room);
    load_error = open("", json::array());
    state = load_error.empty() ? 1 : 2;
    if (!load_error.empty())
      std::fprintf(stderr, "error: %s\n", load_error.c_str());
  }).detach();
  std::thread([] { // one line a minute while audio comes
    long last = 0;
    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(60));
      if (fed_ms.load() != last)
        std::fprintf(stderr, "fed %.1f s in all, busy %ld\n",
                     fed_ms.load() / 1000.0, busy_feeds.load());
      last = fed_ms.load();
    }
  })
    .detach();

  while (true) {
    const int fd = accept(server, nullptr, nullptr);
    if (fd < 0)
      continue;
    std::thread([fd] {
      handle(fd);
      close(fd);
    }).detach();
  }
}
