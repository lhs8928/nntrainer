// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   voice_gate.cpp
 * @date   22 October 2026
 * @brief  Speech or not for each 20 ms frame (ALICE's voice gates)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "voice_gate.h"

#include "../SpeakerDiarization/models/silero_vad.h"

#include <algorithm>
#include <array>
#include <stdexcept>

#ifdef ENABLE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace discretion {

bool EnergyGate::voiced(double rms, const int16_t *, size_t) {
  const bool speech = rms >= std::max(minimum_, floor_ * above_);
  floor_ += (rms - floor_) * (rms < floor_ ? .5 : rise_);
  return speech;
}

/** @brief one 512-sample block in, its speech probability out */
struct SileroGate::Impl {
  static constexpr size_t BLOCK = speaker_diarization::SileroVad::BLOCK;
  std::vector<float> pending;
  virtual ~Impl() = default;
  virtual float run(const float *block) = 0;
};

namespace {

/** @brief the weights from export_silero_vad.py, run by this repository */
struct NativeSilero : SileroGate::Impl {
  speaker_diarization::SileroVad vad;
  explicit NativeSilero(const std::string &path) {
    std::string error;
    if (!vad.load(path, &error))
      throw std::runtime_error(error);
  }
  float run(const float *block) override { return vad.run(block); }
};

#ifdef ENABLE_ONNXRUNTIME

/** @brief the ONNX file, run by ONNX Runtime */
struct OnnxSilero : SileroGate::Impl {
  static constexpr size_t CONTEXT = 64, STATE = 128;
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "silero"};
  Ort::Session session{nullptr};
  std::array<float, STATE> h{}, c{};
  std::array<float, CONTEXT> context{};

  explicit OnnxSilero(const std::string &path) {
    Ort::SessionOptions o;
    o.SetIntraOpNumThreads(1);
    o.SetInterOpNumThreads(1);
    session = Ort::Session(env, path.c_str(), o);
  }

  float run(const float *block) override {
    std::array<float, CONTEXT + BLOCK> input;
    std::copy(context.begin(), context.end(), input.begin());
    std::copy(block, block + BLOCK, input.begin() + CONTEXT);
    const auto mem =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const int64_t in_shape[2] = {1, CONTEXT + BLOCK};
    const int64_t st_shape[3] = {1, 1, STATE};
    Ort::Value inputs[3] = {
      Ort::Value::CreateTensor<float>(mem, input.data(), input.size(), in_shape,
                                      2),
      Ort::Value::CreateTensor<float>(mem, h.data(), h.size(), st_shape, 3),
      Ort::Value::CreateTensor<float>(mem, c.data(), c.size(), st_shape, 3)};
    const char *in_names[3] = {"input", "h", "c"};
    const char *out_names[3] = {"speech_probs", "hn", "cn"};
    auto out =
      session.Run(Ort::RunOptions{nullptr}, in_names, inputs, 3, out_names, 3);
    const float p = out[0].GetTensorData<float>()[0];
    std::copy_n(out[1].GetTensorData<float>(), STATE, h.begin());
    std::copy_n(out[2].GetTensorData<float>(), STATE, c.begin());
    std::copy(block + BLOCK - CONTEXT, block + BLOCK, context.begin());
    return p;
  }
};

#endif

bool onnx_file(const std::string &path) {
  return path.size() >= 5 && path.compare(path.size() - 5, 5, ".onnx") == 0;
}

std::unique_ptr<SileroGate::Impl> open_silero(const std::string &path) {
  if (!onnx_file(path))
    return std::make_unique<NativeSilero>(path);
#ifdef ENABLE_ONNXRUNTIME
  return std::make_unique<OnnxSilero>(path);
#else
  throw std::runtime_error("built without ONNX Runtime: give the .bin made by "
                           "export_silero_vad.py instead of the .onnx");
#endif
}

} // namespace

SileroGate::SileroGate(const std::string &model_path, double threshold) :
  impl_(open_silero(model_path)), threshold_(threshold) {}

SileroGate::~SileroGate() = default;

bool SileroGate::voiced(double, const int16_t *frame, size_t n) {
  for (size_t i = 0; i < n; ++i)
    impl_->pending.push_back(frame[i] / 32768.0f);
  size_t used = 0;
  while (impl_->pending.size() - used >= Impl::BLOCK) {
    last_ = impl_->run(impl_->pending.data() + used);
    used += Impl::BLOCK;
    on_ = last_ >= threshold_ || (on_ && last_ >= threshold_ - .15);
  }
  impl_->pending.erase(impl_->pending.begin(), impl_->pending.begin() + used);
  return on_;
}

std::unique_ptr<VoiceGate> makeVoiceGate(const std::string &silero_model,
                                         double threshold, std::string &error) {
  error.clear();
  if (!silero_model.empty()) {
    try {
      return std::make_unique<SileroGate>(silero_model, threshold);
    } catch (const std::exception &e) {
      error = e.what();
    }
  } else {
    error = "no Silero model given";
  }
  return std::make_unique<EnergyGate>();
}

} // namespace discretion
