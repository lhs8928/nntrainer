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
#include <map>
#include <memory>
#include <mutex>
#include <optional>
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
  /**
   * @brief the next seconds of the room are this person's voice (ALICE's
   *        begin_enrollment): not transcribed, an "enrolled" event when done;
   *        false while another person's enrollment records
   */
  bool beginEnrollment(const std::string &person, double seconds);
  /** @brief a voice the server keeps (of the voice model); throws why not */
  void enrollVector(const std::string &person,
                    const std::vector<float> &vector);
  /** @brief the person's voice over this meeting (ALICE's voice_of) */
  std::optional<std::vector<float>> voiceOf(const std::string &person);
  /**
   * @brief a line a person corrected: every diarizer turn it holds stays
   *        theirs (nullopt: nobody); false when the line is unknown
   */
  bool pin(const std::string &line, const std::optional<std::string> &person);
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
  /** @brief enroll_pcm's checks and the embedding; throws why not */
  std::vector<float> voiceOfRecording(const int16_t *pcm, size_t n);
  void finishEnrollment();

  SpeechOps ops_;
  std::unique_ptr<SpeechModels> models_;
  std::unique_ptr<RoomDiarizer> registry_;
  std::unique_ptr<RoomPipeline> pipeline_;
  std::string gate_, gate_error_;

  std::mutex parts_m_;
  std::map<std::string, std::vector<std::string>> parts_; // line -> turns

  std::mutex m_;
  std::condition_variable cv_;
  std::deque<Item> queue_;
  size_t queued_ = 0;
  bool stop_ = false;
  struct Enrolling {
    std::string person;
    size_t samples = 0;
    std::vector<int16_t> pcm;
  };
  std::optional<Enrolling> enrolling_; // guarded by m_
  std::thread worker_;

  std::mutex out_m_;
  std::vector<std::string> events_;
  std::string staged_;
};

} // namespace discretion

#endif // __DISCRETION_ROOM_RECOGNIZER_H__
