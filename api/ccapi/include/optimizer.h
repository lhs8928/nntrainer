// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	optimizer.h
 * @date	14 October 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is optimizers interface for c++ API
 *
 * @note This is experimental API and not stable.
 */

#ifndef __ML_TRAIN_OPTIMIZER_H__
#define __ML_TRAIN_OPTIMIZER_H__

#if __cplusplus >= MIN_CPP_VERSION

#include <string>
#include <vector>

#include <nntrainer-api-common.h>

namespace ml {
namespace train {

/**
 * @brief     Enumeration of optimizer type
 */
enum class OptimizerType {
  ADAM = ML_TRAIN_OPTIMIZER_TYPE_ADAM,      /** adam */
  SGD = ML_TRAIN_OPTIMIZER_TYPE_SGD,        /** sgd */
  UNKNOWN = ML_TRAIN_OPTIMIZER_TYPE_UNKNOWN /** unknown */
};

/**
 * @class   Optimizer Base class for optimizers
 * @brief   Base class for all optimizers
 */
class Optimizer {
public:
  /**
   * @brief     Destructor of Optimizer Class
   */
  virtual ~Optimizer() = default;

  /**
   * @brief     get Optimizer Type
   * @retval    Optimizer type
   */
  virtual OptimizerType getType() = 0;

  /**
   * @brief     get Learning Rate
   * @retval    Learning rate
   */
  virtual float getLearningRate() = 0;

  /**
   * @brief     get Decay Rate for learning rate decay
   * @retval    decay rate
   */
  virtual float getDecayRate() = 0;

  /**
   * @brief     get Decay Steps for learning rate decay
   * @retval    decay steps
   */
  virtual float getDecaySteps() = 0;

  /**
   * @brief     set Optimizer Parameters
   * @param[in] values Optimizer Parameter list
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setProperty(std::vector<std::string> values) = 0;

  /**
   * @brief     Property Enumeration
   * learning_rate : float ,
   * decay_rate : float,
   * decay_steps : float,
   * beta1 : float,
   * beta2 : float,
   * epsilon : float,
   */
  enum class PropertyType {
    learning_rate = 0,
    decay_rate = 1,
    decay_steps = 2,
    beta1 = 3,
    beta2 = 4,
    epsilon = 5,
    continue_train = 6,
    unknown = 7,
  };

  /**
   * @brief setProperty by PropertyType
   * @note By passing empty string, this can validate if @a type is valid
   * @param[in] type property type to be passed
   * @param[in] value value to be passed, if empty string is passed, do nothing
   * but throws error when @a type is invalid
   * @exception exception::not_supported     when property type is not valid for
   * the particular layer
   * @exception std::invalid_argument invalid argument
   */
  virtual void setProperty(const PropertyType type,
                           const std::string &value = "") = 0;

  /**
   * @brief     validate the optimizer
   */
  virtual void checkValidation() = 0;
};

/**
 * @brief Factory creator with constructor for optimizer
 */
std::unique_ptr<Optimizer>
createOptimizer(OptimizerType type,
                const std::vector<std::string> &properties = {});

} // namespace train
} // namespace ml

#else
#error "CPP versions c++14 or over are only supported"
#endif // __cpluscplus
#endif // __ML_TRAIN_OPTIMIZER_H__
