// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   custom_multiply.h
 * @date   02 April 2026
 * @brief  Custom multiply layer for CausalLM
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __CUSTOM_MULTIPLY_LAYER_H__
#define __CUSTOM_MULTIPLY_LAYER_H__
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

namespace causallm {

/**
 * @brief Custom elementwise multiply layer.
 *
 * @note Supports the same broadcasting rule as nntrainer's multiply layer.
 *       incremental_forwarding() is overridden to avoid the generic binary-op
 *       incremental slicing assumption on height axis.
 *       Supports in-place execution when inplace=true property is set.
 */
WIN_EXPORT class CustomMultiplyLayer final : public nntrainer::Layer {
public:
  WIN_EXPORT CustomMultiplyLayer();
  WIN_EXPORT ~CustomMultiplyLayer() {}

  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  WIN_EXPORT bool supportBackwarding() const override {
    return support_backwarding;
  }

  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}

  WIN_EXPORT const std::string getType() const override {
    return CustomMultiplyLayer::type;
  }

  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /**
   * @brief Get the inplace direction for the tensor operation layer
   * @return InPlaceDirection
   */
  nntrainer::InPlaceDirection getInPlaceDirection() override {
    if (!supportInPlace())
      return nntrainer::InPlaceDirection::NONE;
    if (std::get<nntrainer::props::InPlaceDirectionProp>(multiply_props)
          .empty() ||
        (std::get<nntrainer::props::InPlaceDirectionProp>(multiply_props)
           .get() == "left")) {
      return nntrainer::InPlaceDirection::LEFT;
    } else {
      return nntrainer::InPlaceDirection::RIGHT;
    }
  }

  /**
   * @brief Initialize the in-place settings of the layer
   * @return InPlaceType
   */
  nntrainer::InPlaceType initializeInPlace() final {
    if (std::get<nntrainer::props::InPlaceProp>(multiply_props).empty() ||
        !std::get<nntrainer::props::InPlaceProp>(multiply_props).get()) {
      is_inplace = false;
      support_backwarding = true;
    } else {
      is_inplace = true;
      support_backwarding = false;
    }

    if (!supportInPlace())
      return nntrainer::InPlaceType::NONE;
    else
      return nntrainer::InPlaceType::NON_RESTRICTING;
  }

  inline static const std::string type = "custom_multiply";

protected:
  std::tuple<nntrainer::props::Print, nntrainer::props::InPlaceProp,
             nntrainer::props::InPlaceDirectionProp>
    multiply_props;
  bool support_backwarding; /**< support backwarding */

private:
  static constexpr size_t OUT_IDX = 0;
  static constexpr size_t INPUT_IDX_0 = 0;
  static constexpr size_t INPUT_IDX_1 = 1;
};

} // namespace causallm

#endif /* __cplusplus */
#endif /* __CUSTOM_MULTIPLY_LAYER_H__ */
