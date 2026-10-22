// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   room_api.cpp
 * @date   22 October 2026
 * @brief  C interface of the room recognizer (libnntr_discretion_jni.so)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "room_api.h"

#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>

#include "json.hpp"
#include "room_recognizer.h"
#include "speech_ops.h"

namespace discretion {

namespace {

using json = nlohmann::json;
constexpr size_t QUEUE_SAMPLES = 60 * 16000; // ALICE's QUEUE_SECONDS

/** @brief SpeechOps as the pipeline's three models */
class OpsModels : public SpeechModels {
public:
  explicit OpsModels(SpeechOps &ops) : ops_(ops) {}
  Recognized recognize(const std::vector<float> &pcm) override {
    const AsrResult r = ops_.recognize(pcm);
    Recognized out;
    out.text = r.text;
    out.ms = r.ms;
    for (const auto &w : r.words)
      out.words.push_back({w.start_ms, w.end_ms, w.text});
    return out;
  }
  std::vector<float> embed(const std::vector<float> &pcm) override {
    if (pcm.size() < 8000) // NntrSpeaker: under 0.5 s is no voice
      throw std::invalid_argument("화자 구분을 위한 음성이 너무 짧습니다.");
    return ops_.embedding(pcm);
  }
  std::vector<std::vector<uint8_t>>
  segment(const std::vector<float> &pcm) override {
    return ops_.segmentation(pcm);
  }

private:
  SpeechOps &ops_;
};

} // namespace

static int copyOut(const std::string &s, char *out, int cap) {
  const int n = static_cast<int>(s.size());
  if (out && cap > n) {
    std::memcpy(out, s.data(), s.size());
    out[n] = '\0';
  }
  return n;
}

static void setError(char *err, int cap, const std::string &msg) {
  if (err && cap > 0) {
    const size_t n = std::min<size_t>(cap - 1, msg.size());
    std::memcpy(err, msg.data(), n);
    err[n] = '\0';
  }
}

Room::~Room() {
  {
    std::lock_guard<std::mutex> lk(m_);
    stop_ = true;
  }
  cv_.notify_all();
  if (worker_.joinable())
    worker_.join();
}

bool Room::init(const std::string &config, std::string &error) {
  json c;
  try {
    c = json::parse(config.empty() ? "{}" : config);
  } catch (const std::exception &e) {
    error = std::string("config is not JSON: ") + e.what();
    return false;
  }
  SpeechOptions opt;
  opt.asr.model_dir = c.value("model_dir", "");
  opt.sd_weights = c.value("sd_weights", "");
  opt.asr.language = c.value("language", "Korean");
  opt.asr.threads = c.value("threads", 0);
  // ALICE's segments run up to 12 s (more with the bridge): a 12.5 s graph
  // hears them whole, an 8 s one decodes the short ones faster
  opt.asr_secs = c.contains("asr_secs")
                   ? c["asr_secs"].get<std::vector<float>>()
                   : std::vector<float>{8.0f, 12.5f};
  if (opt.asr.model_dir.empty() || opt.sd_weights.empty()) {
    error = "model_dir and sd_weights are needed";
    return false;
  }
  if (!ops_.init(opt, error))
    return false;
  models_ = std::make_unique<OpsModels>(ops_);
  DiarizerOptions d;
  d.same = d.threshold = c.value("speaker_threshold", .6);
  registry_ = std::make_unique<RoomDiarizer>(d);
  std::unique_ptr<VoiceGate> gate;
  if (c.value("vad", "silero") == "silero")
    gate = makeVoiceGate(c.value("silero", ""), c.value("silero_threshold", .3),
                         gate_error_);
  else
    gate = std::make_unique<EnergyGate>();
  gate_ = gate->name();
  pipeline_ = std::make_unique<RoomPipeline>(
    *models_, *registry_, std::move(gate),
    [this](const RoomLine &l) {
      json e;
      e["type"] = "line";
      e["id"] = l.id;
      e["speaker"] = l.speaker.empty() ? json(nullptr) : json(l.speaker);
      e["start_ms"] = l.start_ms;
      e["end_ms"] = l.end_ms;
      e["text"] = l.text;
      e["doubt"] = l.doubt;
      e["info"] = json::parse(l.info);
      push(e);
    },
    [this](const std::map<std::string, std::string> &moved) {
      push({{"type", "moved"}, {"moved", moved}});
    });
  worker_ = std::thread([this] { run(); });
  return true;
}

void Room::push(const json &event) {
  std::lock_guard<std::mutex> lk(out_m_);
  events_.push_back(event.dump());
}

int Room::feed(const int16_t *pcm, size_t n, double at_ms) {
  std::lock_guard<std::mutex> lk(m_);
  if (queued_ + n > QUEUE_SAMPLES)
    return 1;
  Item item;
  item.pcm.assign(pcm, pcm + n);
  item.at_ms = at_ms;
  queued_ += n;
  queue_.push_back(std::move(item));
  cv_.notify_all();
  return 0;
}

void Room::call(std::function<void()> task) {
  std::promise<void> done;
  auto f = done.get_future();
  {
    std::lock_guard<std::mutex> lk(m_);
    Item item;
    item.task = [&] {
      try {
        task();
        done.set_value();
      } catch (...) {
        done.set_exception(std::current_exception());
      }
    };
    queue_.push_back(std::move(item));
  }
  cv_.notify_all();
  f.get();
}

void Room::enroll(const std::string &person, const int16_t *pcm, size_t n) {
  // ALICE's AudioRuntime.enroll_pcm
  if (n < 3 * 16000 || n > 15 * 16000)
    throw std::invalid_argument("등록 음성은 3~15초여야 합니다.");
  long long s = 0;
  for (size_t i = 0; i < n; ++i)
    s += static_cast<long long>(pcm[i]) * pcm[i];
  if (std::sqrt(static_cast<double>(s) / n) / 32768 < .008)
    throw std::invalid_argument("목소리를 충분히 크게 녹음하세요.");
  std::vector<float> x(n);
  for (size_t i = 0; i < n; ++i)
    x[i] = pcm[i] / 32768.0f;
  call([&] { registry_->enroll(person, models_->embed(x)); });
}

void Room::mapSpeaker(const std::string &label, const std::string &person) {
  registry_->mapSpeaker(label, person);
}

void Room::flush() {
  call([this] { pipeline_->flush(); });
}

std::string Room::poll() {
  std::lock_guard<std::mutex> lk(out_m_);
  if (staged_.empty()) {
    std::string s = "[";
    for (size_t i = 0; i < events_.size(); ++i)
      s += (i ? "," : "") + events_[i];
    s += "]";
    events_.clear();
    staged_ = s;
  }
  return staged_;
}

void Room::consumed() {
  std::lock_guard<std::mutex> lk(out_m_);
  staged_.clear();
}

std::string Room::status() {
  json j;
  {
    std::lock_guard<std::mutex> lk(m_);
    j["queued_seconds"] = std::round(queued_ / 1600.0) / 10.0;
  }
  j["gate"] = gate_;
  if (!gate_error_.empty())
    j["gate_error"] = gate_error_;
  j["segments"] = pipeline_->segments();
  j["looped_lines"] = pipeline_->looped();
  j["unknown_speakers"] = registry_->unknown();
  j["models"] = json::parse(ops_.describe());
  return j.dump();
}

void Room::run() {
  while (true) {
    Item item;
    {
      std::unique_lock<std::mutex> lk(m_);
      cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
      if (stop_)
        return; // closed: what still waits is dropped
      item = std::move(queue_.front());
      queue_.pop_front();
    }
    if (item.task) {
      item.task();
      continue;
    }
    try {
      pipeline_->feed(item.pcm.data(), item.pcm.size(), item.at_ms);
    } catch (const std::exception &e) {
      push({{"type", "error"},
            {"message", std::string("음성 처리 실패: ") + e.what()}});
    }
    std::lock_guard<std::mutex> lk(m_);
    queued_ -= item.pcm.size();
  }
}

} // namespace discretion

using discretion::copyOut;
using discretion::Room;
using discretion::setError;

struct nd_room {
  Room room;
};

extern "C" {

nd_room *nd_create(const char *config_json, char *err, int err_cap) {
  auto r = std::make_unique<nd_room>();
  std::string error;
  try {
    if (!r->room.init(config_json ? config_json : "", error)) {
      setError(err, err_cap, error);
      return nullptr;
    }
  } catch (const std::exception &e) {
    setError(err, err_cap, e.what());
    return nullptr;
  }
  return r.release();
}

int nd_feed(nd_room *room, const int16_t *pcm, int n, double at_ms) {
  if (!room || !pcm || n < 0)
    return -1;
  return room->room.feed(pcm, static_cast<size_t>(n), at_ms);
}

int nd_enroll(nd_room *room, const char *person, const int16_t *pcm, int n,
              char *err, int err_cap) {
  if (!room || !person || !pcm || n <= 0) {
    setError(err, err_cap, "bad arguments");
    return -1;
  }
  try {
    room->room.enroll(person, pcm, static_cast<size_t>(n));
    return 0;
  } catch (const std::exception &e) {
    setError(err, err_cap, e.what());
    return -1;
  }
}

int nd_map_speaker(nd_room *room, const char *label, const char *person) {
  if (!room || !label || !person)
    return -1;
  try {
    room->room.mapSpeaker(label, person);
    return 0;
  } catch (const std::exception &) {
    return -1;
  }
}

int nd_poll(nd_room *room, char *out, int cap) {
  if (!room)
    return -1;
  const std::string s = room->room.poll();
  const int n = copyOut(s, out, cap);
  if (out && cap > n)
    room->room.consumed();
  return n;
}

int nd_status(nd_room *room, char *out, int cap) {
  if (!room)
    return -1;
  return copyOut(room->room.status(), out, cap);
}

int nd_flush(nd_room *room) {
  if (!room)
    return -1;
  try {
    room->room.flush();
    return 0;
  } catch (const std::exception &) {
    return -1;
  }
}

void nd_destroy(nd_room *room) { delete room; }

} // extern "C"
