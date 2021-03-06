// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	tflite_layer.h
 * @date	3 November 2020
 * @brief	This is class to encapsulate tflite as a layer of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __TENSORFLOW_LITE_H__
#define __TENSORFLOW_LITE_H__
#ifdef __cplusplus

#include <layer_internal.h>
#include <tensor.h>

#include <tensorflow/contrib/lite/interpreter.h>
#include <tensorflow/contrib/lite/kernels/register.h>
#include <tensorflow/contrib/lite/model.h>

namespace nntrainer {

/**
 * @class   TfLiteLayer
 * @brief   Tensorflow Lite layer
 */
class TfLiteLayer : public Layer {
public:
  /**
   * @brief     Constructor of NNStreamer Layer
   */
  TfLiteLayer(std::string model = "") :
    Layer(LayerType::LAYER_BACKBONE_TFLITE),
    modelfile(model),
    interpreter(nullptr),
    model(nullptr) {
    trainable = false;
  }

  /**
   * @brief     Destructor of NNStreamer Layer
   */
  ~TfLiteLayer() = default;

  /**
   * @copydoc Layer::forwarding(sharedConstTensors in)
   */
  sharedConstTensors forwarding(sharedConstTensors in);

  /**
   * @copydoc Layer::backwarding(sharedConstTensors in, int iteration)
   */
  sharedConstTensors backwarding(sharedConstTensors in, int iteration);

  /**
   * @copydoc Layer::copy(std::shared_ptr<layer> l)
   */
  void copy(std::shared_ptr<Layer> l);

  /**
   * @copydoc Layer::initialize()
   */
  int initialize();

  /**
   * @copydoc Layer::setTrainable(bool train)
   */
  void setTrainable(bool train);

  /**
   * @brief     get the base name for the layer
   * @retval    base name of the layer
   */
  std::string getBaseName() { return "BackboneTFLite"; };

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type, const std::string &value = "");

private:
  std::string modelfile;
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::unique_ptr<tflite::FlatBufferModel> model;

  void setDimensions(const std::vector<int> &tensor_idx_list,
                     std::vector<TensorDim> &dim, bool is_output);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TENSORFLOW_LITE_H__ */
