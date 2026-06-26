// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   causal_lm.h
 * @date   22 July 2025
 * @see    https://github.com/nntrainer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  CausalLM Factory to support registration and creation of various
 * CausalLM models
 */

#ifndef __CAUSALLM_FACTORY_H__
#define __CAUSALLM_FACTORY_H__

#include <ostream>
#include <transformer.h>
#include <unordered_map>

namespace causallm {

/**
 * @brief Factory class
 */
class Factory {
public:
  using Creator =
    std::function<std::shared_ptr<Transformer>(json &, json &, json &)>;

  static Factory &Instance() {
    static Factory factory;
    return factory;
  }

  void registerModel(const std::string &key, Creator creator) {
    creators[key] = creator;
  }

  std::shared_ptr<Transformer>
  create(const std::string &key, json &cfg, json &generation_cfg,
         json &nntr_cfg) {
    auto creator_it = creators.find(key);
    if (creator_it != creators.end())
      return (creator_it->second)(cfg, generation_cfg, nntr_cfg);
    return nullptr;
  }

  void printRegistered(std::ostream &os) const {
    for (const auto &pair : creators) {
      os << "\n\t" << pair.first;
    }
  }

private:
  std::unordered_map<std::string, Creator> creators;
};

} // namespace causallm

#endif
