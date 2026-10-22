// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   speech_http.cpp
 * @date   22 October 2026
 * @brief  HTTP front of the speech models, inside the executable
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "speech_http.h"

#include <arpa/inet.h>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>

#include "json.hpp"
#include "speech_ops.h"

namespace discretion {

namespace {

using json = nlohmann::json;
constexpr size_t MAX_HEADER_BYTES = 16 * 1024;
constexpr size_t MAX_BODY_BYTES = 120 * 16000 * 2; // two minutes of audio

std::mutex g_model;            // the models answer one request at a time
std::atomic<int> g_waiting{0}; // requests waiting for or using the models

const char *reason(int status) {
  switch (status) {
  case 100:
    return "Continue";
  case 200:
    return "OK";
  case 400:
    return "Bad Request";
  case 401:
    return "Unauthorized";
  case 404:
    return "Not Found";
  case 413:
    return "Payload Too Large";
  case 422:
    return "Unprocessable Entity";
  default:
    return "Internal Server Error";
  }
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

void reply(int fd, int status, const std::string &body) {
  char head[256];
  std::snprintf(head, sizeof(head),
                "HTTP/1.1 %d %s\r\nContent-Type: application/json; "
                "charset=utf-8\r\nContent-Length: %zu\r\nConnection: "
                "close\r\n\r\n",
                status, reason(status), body.size());
  sendAll(fd, head + body);
}

void fail(int fd, int status, const std::string &detail) {
  reply(fd, status, json{{"detail", detail}}.dump());
}

std::string lower(std::string s) {
  for (char &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

/** @brief the value of header `name` (lower case) in the header block */
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

bool sameToken(const std::string &a, const std::string &b) {
  if (a.size() != b.size())
    return false;
  unsigned char diff = 0;
  for (size_t i = 0; i < a.size(); ++i)
    diff |= static_cast<unsigned char>(a[i] ^ b[i]);
  return diff == 0;
}

void handle(int fd, SpeechOps &ops, const HttpOptions &opt,
            const std::string &peer) {
  timeval tv{30, 0}; // a client that stops sending does not hold a thread
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  std::string data;
  char buf[65536];
  size_t head_end;
  while ((head_end = data.find("\r\n\r\n")) == std::string::npos) {
    if (data.size() > MAX_HEADER_BYTES)
      return fail(fd, 400, "header too large");
    const ssize_t n = recv(fd, buf, sizeof(buf), 0);
    if (n <= 0)
      return;
    data.append(buf, static_cast<size_t>(n));
  }
  const std::string head = data.substr(0, head_end);
  std::string body = data.substr(head_end + 4);
  const size_t sp1 = head.find(' '), sp2 = head.find(' ', sp1 + 1);
  if (sp1 == std::string::npos || sp2 == std::string::npos)
    return fail(fd, 400, "bad request line");
  const std::string method = head.substr(0, sp1);
  const std::string path = head.substr(sp1 + 1, sp2 - sp1 - 1);

  if (!opt.token.empty() &&
      !sameToken(header(head, "authorization"), "Bearer " + opt.token))
    return fail(fd, 401, "token required");

  if (method == "GET" && path == "/health") {
    const std::string info = ops.describe(); // a JSON object
    return reply(
      fd, 200,
      "{\"status\":\"ready\",\"queue\":" + std::to_string(g_waiting.load()) +
        (info.size() > 2 ? "," + info.substr(1) : "}"));
  }
  std::string op;
  if (path == "/asr")
    op = "ASR";
  else if (path == "/embed")
    op = "EMBED";
  else if (path == "/segment")
    op = "SEGMENT";
  if (method != "POST" || op.empty())
    return fail(fd, 404, "not found");

  const std::string length_text = header(head, "content-length");
  const size_t length =
    length_text.empty() ? 0 : std::strtoul(length_text.c_str(), nullptr, 10);
  if (length > MAX_BODY_BYTES)
    return fail(fd, 413, "body must be 16 kHz mono PCM16, up to two minutes");
  if (length == 0 || length % 2)
    return fail(fd, 400, "body must be 16 kHz mono PCM16, up to two minutes");
  if (lower(header(head, "expect")) == "100-continue" && body.empty())
    sendAll(fd, "HTTP/1.1 100 Continue\r\n\r\n");
  while (body.size() < length) {
    const ssize_t n = recv(fd, buf, sizeof(buf), 0);
    if (n <= 0)
      return;
    body.append(buf, static_cast<size_t>(n));
  }
  body.resize(length);

  const auto t0 = std::chrono::steady_clock::now();
  std::string answer, error;
  bool ok;
  ++g_waiting;
  {
    std::lock_guard<std::mutex> lk(g_model);
    ok = ops.run(op, body, answer, error);
  }
  --g_waiting;
  const double ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - t0)
                      .count();
  std::fprintf(stderr, "%s %s %zu bytes %.0f ms%s\n", peer.c_str(), op.c_str(),
               length, ms, ok ? "" : (" ERR " + error).c_str());
  if (!ok)
    return fail(fd, 422, error);
  reply(fd, 200, answer);
}

} // namespace

int serveHttp(SpeechOps &ops, const HttpOptions &opt) {
  const int server = socket(AF_INET, SOCK_STREAM, 0);
  if (server < 0) {
    std::perror("socket");
    return EXIT_FAILURE;
  }
  const int on = 1;
  setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(opt.port));
  if (inet_pton(AF_INET, opt.host.c_str(), &addr.sin_addr) != 1) {
    std::fprintf(stderr, "error: bad --http-host %s\n", opt.host.c_str());
    return EXIT_FAILURE;
  }
  if (bind(server, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 ||
      listen(server, 16) != 0) {
    std::perror("bind");
    close(server);
    return EXIT_FAILURE;
  }
  std::fprintf(stderr, "nntr speech server on %s:%d %s\n", opt.host.c_str(),
               opt.port, ops.describe().c_str());
  while (true) {
    sockaddr_in peer{};
    socklen_t len = sizeof(peer);
    const int fd = accept(server, reinterpret_cast<sockaddr *>(&peer), &len);
    if (fd < 0)
      continue;
    char ip[INET_ADDRSTRLEN] = "?";
    inet_ntop(AF_INET, &peer.sin_addr, ip, sizeof(ip));
    std::thread([fd, &ops, &opt, who = std::string(ip)] {
      handle(fd, ops, opt, who);
      close(fd);
    }).detach();
  }
}

} // namespace discretion
