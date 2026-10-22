// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   voice_gate.h
 * @date   22 October 2026
 * @brief  Speech or not for each 20 ms frame (ALICE's voice gates)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Ports of ambient/audio.py EnergyGate and SileroGate (ambient-agent-alice),
 * with the same thresholds, so a phone cuts segments where the server would.
 */

#ifndef __DISCRETION_VOICE_GATE_H__
#define __DISCRETION_VOICE_GATE_H__

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace discretion {

/** @brief one 20 ms frame (320 samples at 16 kHz) at a time */
class VoiceGate {
public:
  virtual ~VoiceGate() = default;
  /** @param rms the frame's RMS (0..1), @param frame its samples */
  virtual bool voiced(double rms, const int16_t *frame, size_t n) = 0;
  virtual const char *name() const = 0;
};

/**
 * @brief loudness above the room's noise floor: the floor follows the quietest
 *        frames at once and rises slowly under steady noise; a frame is speech
 *        when it is `above` times the floor and above `minimum`
 */
class EnergyGate : public VoiceGate {
public:
  explicit EnergyGate(double minimum = .008, double above = 2.0,
                      double rise = .005) :
    minimum_(minimum), above_(above), rise_(rise) {}
  bool voiced(double rms, const int16_t *, size_t) override;
  const char *name() const override { return "energy"; }

private:
  double minimum_, above_, rise_, floor_ = 0.0;
};

/**
 * @brief Silero VAD v6 (the ONNX model ALICE takes from faster-whisper) run on
 *        the stream with its state carried over: speech from `threshold`,
 *        held down to `threshold - 0.15`
 */
class SileroGate : public VoiceGate {
public:
  SileroGate(const std::string &model_path, double threshold);
  ~SileroGate() override;
  bool voiced(double rms, const int16_t *frame, size_t n) override;
  const char *name() const override { return "silero"; }
  /** @brief probability of the last 32 ms block */
  float last() const { return last_; }

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  double threshold_;
  bool on_ = false;
  float last_ = 0.0f;
};

/**
 * @brief Silero when the model loads (built with ONNX Runtime), else the
 *        energy gate, as ambient.audio.vad_gate falls back
 * @param error why Silero was not used (empty when it is)
 */
std::unique_ptr<VoiceGate> makeVoiceGate(const std::string &silero_model,
                                         double threshold, std::string &error);

} // namespace discretion

#endif // __DISCRETION_VOICE_GATE_H__
