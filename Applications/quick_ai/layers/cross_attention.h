// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   cross_attention.h
 * @date   04 November 2026
 * @brief  Encoder-decoder (cross) attention over a fixed key/value memory
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The encoder output, already projected to every decoder layer's keys and
 * values, is the "memory". It is computed once per utterance by the encoder
 * model, whose last layer (cross_memory_writer) stores it under a name, and
 * then only read by the decoder's cross_attention layers at every step. The
 * two models are separate graphs, so the memory lives in a registry here
 * rather than in graph inputs.
 *
 * cross_memory_writer
 *   inputs: key_0, value_0, key_1, value_1, ... (B = 1, 1, S, width)
 *   output: (1, 1, 1, 1), only so that the graph keeps the layer
 *   property memory_prefix: the memory of input pair i is "<prefix>_l<i>"
 *
 * cross_attention
 *   input 0: query (B, 1, T, num_heads * head_dim)
 *   output : (B, 1, T, num_heads * head_dim)
 *   properties: num_heads, memory_id, layer_index (key of the capture),
 *               capture_heads ("none" = none, "3,7" = record these heads)
 *
 * Every query row attends to all S memory rows (no mask). For word
 * timestamps the layer can record, for chosen heads, the scaled attention
 * logits (q.k / sqrt(head_dim)) of every row it computes, keyed by the row's
 * absolute position (crossAttentionCapture()).
 */

#ifndef __CROSS_ATTENTION_LAYER_H__
#define __CROSS_ATTENTION_LAYER_H__
#ifdef __cplusplus

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace quick_ai {

/**
 * @brief one layer's memory: rows x width keys and values (FP16 when the
 *        build has FP16, else FP32)
 */
struct CrossMemory {
#ifdef ENABLE_FP16
  using value_type = _FP16;
#else
  using value_type = float;
#endif
  std::vector<value_type> key;
  std::vector<value_type> value;
  size_t rows = 0;
  size_t width = 0;
};

/** @brief the memory stored under `id` (created empty when missing) */
WIN_EXPORT CrossMemory &crossMemory(const std::string &id);

/** @brief free every memory whose id starts with `prefix` */
WIN_EXPORT void crossMemoryRelease(const std::string &prefix);

/**
 * @brief captured logits: (layer index, head) -> position -> S logits
 */
using CrossAttentionRows =
  std::map<std::pair<int, int>, std::map<unsigned int, std::vector<float>>>;

/** @brief drop everything captured so far */
WIN_EXPORT void crossAttentionCaptureClear();

/** @brief the rows captured since the last crossAttentionCaptureClear() */
WIN_EXPORT const CrossAttentionRows &crossAttentionCapture();

/**
 * @class CrossMemoryWriterLayer
 * @brief stores (key, value) input pairs as named memories
 */
WIN_EXPORT class CrossMemoryWriterLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT CrossMemoryWriterLayer() = default;
  WIN_EXPORT ~CrossMemoryWriterLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override {
    forwarding(context, training);
  }

  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override {
  }

  WIN_EXPORT bool supportBackwarding() const override { return false; }

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}

  WIN_EXPORT const std::string getType() const override {
    return CrossMemoryWriterLayer::type;
  }

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "cross_memory_writer";

private:
  std::string memory_prefix;
};

/**
 * @class CrossAttentionLayer
 * @brief softmax(q k^T / sqrt(d)) v per head over a named memory
 */
WIN_EXPORT class CrossAttentionLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT CrossAttentionLayer() = default;
  WIN_EXPORT ~CrossAttentionLayer() = default;

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
    return CrossAttentionLayer::type;
  }

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "cross_attention";

private:
  unsigned int num_heads = 1;
  int layer_index = 0;
  std::string memory_id;
  std::vector<int> capture_heads;
};

} // namespace quick_ai

#endif // __cplusplus
#endif // __CROSS_ATTENTION_LAYER_H__
