// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   batchwise_dot_product_layer.h
 * @date   29 November 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Batchwise dot product Layer Class for Neural Network
 *
 */

#ifndef __BATCHWISE_DOT_PRODUCT_LAYER_H__
#define __BATCHWISE_DOT_PRODUCT_LAYER_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   Batchwise Dot product Layer
 * @brief   Batchwise Dot product Layer
 */
class BatchwiseDotproductLayer : public virtual Layer {
public:
  /**
   * @brief     Constructor of Batchwise Dot product Layer
   */
  BatchwiseDotproductLayer();

  /**
   * @brief     Destructor of Batchwise Dot product Layer
   */
  ~BatchwiseDotproductLayer();

  /**
   *  @brief  Move constructor of BatchwiseDotproductLayer.
   *  @param[in] BatchwiseDotproductLayer &&
   */
  BatchwiseDotproductLayer(BatchwiseDotproductLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs BatchwiseDotproductLayer to be moved.
   */
  BatchwiseDotproductLayer &operator=(BatchwiseDotproductLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return BatchwiseDotproductLayer::type;
  };

  inline static const std::string type = "batchwise_dot_product";

protected:
  /**
   * @brief     Finalize the batchwise dot product layer with the given context
   * @param[in] context InitLayerContext
   *
   * @note This function provides the basic finalize details which can be shared
   * with derived classes as well
   */
  void finalizeCommon(InitLayerContext &context);

  std::tuple<props::ScaledDotProduct, props::TransposeQuery,
             props::TransposeKey>
    batchwise_dot_product_props;

private:
  ActiFunc sm; /** softmax activation operation */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __BATCHWISE_DOT_PRODUCT_LAYER_H__ */
