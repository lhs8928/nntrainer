// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   room_recognizer.h
 * @date   22 October 2026
 * @brief  The room recognizer behind room_api.h and the JNI wrapper
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The models, the whole-meeting clustering and the pipeline, with one worker
 * thread that runs everything that touches the models in order: fed audio,
 * enrollment and the final flush (ALICE's AudioRuntime worker).
 */

#ifndef __DISCRETION_ROOM_RECOGNIZER_H__
#define __DISCRETION_ROOM_RECOGNIZER_H__

#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "json.hpp"
#include "room_diarizer.h"
#include "room_pipeline.h"
#include "speech_ops.h"

namespace discretion {

class Room {
public:
  Room() = default;
  ~Room();
  Room(const Room &) = delete;
  Room &operator=(const Room &) = delete;

  /** @brief see room_api.h for the config keys */
  bool init(const std::string &config_json, std::string &error);
  /** @brief 0 taken, 1 busy */
  int feed(const int16_t *pcm, size_t n, double at_ms);
  /** @brief throws with the reason */
  void enroll(const std::string &person, const int16_t *pcm, size_t n);
  void mapSpeaker(const std::string &label, const std::string &person);
  /** @brief waits until everything fed is recognized */
  void flush();
  /** @brief the pending events as a JSON array, kept until consumed() */
  std::string poll();
  void consumed();
  std::string status();

private:
  struct Item {
    std::vector<int16_t> pcm;
    double at_ms = -1;
    std::function<void()> task; // set: run this instead of feeding
  };
  void run();
  void call(std::function<void()> task);
  void push(const nlohmann::json &event);

  SpeechOps ops_;
  std::unique_ptr<SpeechModels> models_;
  std::unique_ptr<RoomDiarizer> registry_;
  std::unique_ptr<RoomPipeline> pipeline_;
  std::string gate_, gate_error_;

  std::mutex m_;
  std::condition_variable cv_;
  std::deque<Item> queue_;
  size_t queued_ = 0;
  bool stop_ = false;
  std::thread worker_;

  std::mutex out_m_;
  std::vector<std::string> events_;
  std::string staged_;
};

} // namespace discretion

#endif // __DISCRETION_ROOM_RECOGNIZER_H__
