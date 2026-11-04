// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   position_embedding.h
 * @date   04 November 2026
 * @brief  Adds a table of position embeddings by the rows' absolute position
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * out[t] = in[t] + table[from + t]: rows of an incremental step get the
 * embedding of their position in the sequence, not of their row in the step.
 * The table (max_position x width, FP32) is a weight named "weight", so it is
 * loaded like any other weight (Whisper's learned decoder positions and the
 * encoder's fixed sinusoids).
 */

#ifndef __POSITION_EMBEDDING_LAYER_H__
#define __POSITION_EMBEDDING_LAYER_H__
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
 * @class PositionEmbeddingLayer
 * @brief in + table[position]
 */
WIN_EXPORT class PositionEmbeddingLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT PositionEmbeddingLayer() = default;
  WIN_EXPORT ~PositionEmbeddingLayer() = default;

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
    return PositionEmbeddingLayer::type;
  }

  /** @brief property: max_position (rows of the table) */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "position_embedding";

private:
  unsigned int max_position = 0;
  unsigned int weight_idx = 0;
};

} // namespace quick_ai

#endif // __cplusplus
#endif // __POSITION_EMBEDDING_LAYER_H__
