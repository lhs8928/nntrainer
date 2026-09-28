// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   multimodal_scatter.h
 * @date   14 September 2026
 * @brief  Multimodal scatter/fusion layer for Qwen3-ASR
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 */

#ifndef __MULTIMODAL_SCATTER_LAYER_H__
#define __MULTIMODAL_SCATTER_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace quick_ai {

/**
 * @brief Multimodal scatter layer to fuse text embeddings and audio features.
 *        Inputs:
 *        - input0: Text embeddings (shape: [batch, 1, seq_len, hidden_size])
 *        - input1: Audio features (shape: [batch, 1, audio_seq_len, hidden_size])
 *        - input2: Input token IDs (shape: [batch, 1, 1, seq_len])
 *        Outputs:
 *        - output0: Fused embeddings (shape: [batch, 1, seq_len, hidden_size])
 */
WIN_EXPORT class MultimodalScatterLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT MultimodalScatterLayer();
  WIN_EXPORT ~MultimodalScatterLayer() {}

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  WIN_EXPORT bool supportBackwarding() const override { return false; }

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}

  WIN_EXPORT const std::string getType() const override {
    return MultimodalScatterLayer::type;
  }

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "multimodal_scatter";

private:
  unsigned int audio_token_id;
};

} // namespace quick_ai

#endif // __cplusplus
#endif // __MULTIMODAL_SCATTER_LAYER_H__
