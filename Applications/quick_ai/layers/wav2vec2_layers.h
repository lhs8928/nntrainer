// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   wav2vec2_layers.h
 * @date   04 November 2026
 * @brief  wav2vec 2.0 feature encoder and positional convolution layers
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * wav2vec2_feature_encoder
 *   input : (1, 1, 1, max_samples) 16 kHz waveform
 *   output: (1, 1, max_frames, conv_dim[last]) frames x channels
 *   The stack of 1-D convolutions of HuggingFace's Wav2Vec2FeatureEncoder:
 *   every layer conv -> norm -> GELU, the norm being a LayerNorm over the
 *   channels of every frame ("layer", XLS-R) or, for the first layer only,
 *   a GroupNorm with one group per channel over the frames ("group", base
 *   models). The real length of the waveform is set at run time with the
 *   property num_samples; only its frames are computed (an incremental step
 *   [0, frames) as the rest of the graph). FP32 only.
 *   weights: conv<i>:weight (out, in, 1, k), conv<i>:bias, norm<i>:gamma/beta
 *   (stored as "<layer>:conv<i>_weight" etc.)
 *
 * wav2vec2_pos_conv
 *   input/output: (1, 1, frames, channels)
 *   GELU(grouped conv1d(kernel, padding kernel / 2, groups)) along the
 *   frames, the last output frame dropped for an even kernel (SamePad): the
 *   positional embedding of Wav2Vec2PositionalConvEmbedding, with the weight
 *   norm folded into the weight by the converter. Frames past the step are
 *   zero, as the padding of the reference. FP32 only.
 */

#ifndef __WAV2VEC2_LAYERS_H__
#define __WAV2VEC2_LAYERS_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <string>
#include <vector>

#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace quick_ai {

/**
 * @class Wav2Vec2FeatureEncoderLayer
 * @brief waveform -> frames of conv features
 */
WIN_EXPORT class Wav2Vec2FeatureEncoderLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT Wav2Vec2FeatureEncoderLayer() = default;
  WIN_EXPORT ~Wav2Vec2FeatureEncoderLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override {
  }
  WIN_EXPORT bool supportBackwarding() const override { return false; }
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}
  WIN_EXPORT const std::string getType() const override {
    return Wav2Vec2FeatureEncoderLayer::type;
  }

  /**
   * @brief properties: conv_dim, conv_kernel, conv_stride ("512,512,...",
   *        one per layer), conv_bias (true), norm ("layer" | "group"),
   *        epsilon (1e-5), num_samples (the waveform's length, at run time)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /** @brief frames of a waveform of n samples */
  WIN_EXPORT static size_t framesOf(size_t n, const std::vector<int> &kernel,
                                    const std::vector<int> &stride);

  inline static const std::string type = "wav2vec2_feature_encoder";

private:
  std::vector<int> dim_, kernel_, stride_;
  bool bias_ = true;
  std::string norm_ = "layer";
  float eps_ = 1e-5f;
  size_t num_samples_ = 0;
  std::vector<unsigned int> w_idx_, b_idx_, g_idx_, beta_idx_;
};

/**
 * @class Wav2Vec2PosConvLayer
 * @brief GELU(grouped conv over the frames), wav2vec 2.0 positions
 */
WIN_EXPORT class Wav2Vec2PosConvLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT Wav2Vec2PosConvLayer() = default;
  WIN_EXPORT ~Wav2Vec2PosConvLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override {
  }
  WIN_EXPORT bool supportBackwarding() const override { return false; }
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}
  WIN_EXPORT const std::string getType() const override {
    return Wav2Vec2PosConvLayer::type;
  }

  /** @brief properties: kernel_size (128), groups (16) */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "wav2vec2_pos_conv";

private:
  unsigned int kernel_ = 128;
  unsigned int groups_ = 16;
  unsigned int w_idx_ = 0, b_idx_ = 0;
};

} // namespace quick_ai

#endif // __cplusplus
#endif // __WAV2VEC2_LAYERS_H__
