// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nhwc_to_nchw.h
 * @date   02 October 2026
 * @brief  Format conversion layer from NHWC to NCHW for Qwen3-ASR Subsampler
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 */

#ifndef __NHWC_TO_NCHW_LAYER_H__
#define __NHWC_TO_NCHW_LAYER_H__
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
 * @brief NHWCToNCHWLayer converts physical NHWC layout [H, W, C] to NCHW layout [C, H, W]
 */
WIN_EXPORT class NHWCToNCHWLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT NHWCToNCHWLayer();
  WIN_EXPORT ~NHWCToNCHWLayer() = default;

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override {
    forwarding(context, training);
  }

  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  WIN_EXPORT bool supportBackwarding() const override { return false; }

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}

  WIN_EXPORT const std::string getType() const override {
    return NHWCToNCHWLayer::type;
  }

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "nhwc_to_nchw";
};

} // namespace quick_ai

#endif // __cplusplus
#endif // __NHWC_TO_NCHW_LAYER_H__
