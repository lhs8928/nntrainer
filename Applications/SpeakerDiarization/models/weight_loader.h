// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   weight_loader.h
 * @date   29 September 2026
 * @brief  Binary weight loader for Speaker Diarization in NNTrainer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __WEIGHT_LOADER_H__
#define __WEIGHT_LOADER_H__

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>

namespace speaker_diarization {

struct TensorMeta {
  std::string name;
  std::vector<uint32_t> dims;
  uint64_t byte_size;
  uint64_t offset;
};

class WeightLoader {
public:
  WeightLoader() = default;
  ~WeightLoader();

  /**
   * @brief Load weights from unified NNTrainer binary file.
   * @param bin_path Path to speaker_diarization.bin
   * @return true if successful, false otherwise.
   */
  bool load(const std::string &bin_path);

  /**
   * @brief Get pointer to float tensor data.
   * @param name Name of tensor.
   * @param dims Optional output vector to receive tensor dimensions.
   * @return Pointer to float data, or nullptr if not found.
   */
  const float *getTensor(const std::string &name, std::vector<uint32_t> *dims = nullptr) const;

  /**
   * @brief Check if a tensor exists.
   */
  bool hasTensor(const std::string &name) const;

  /**
   * @brief Get all tensor names.
   */
  std::vector<std::string> getTensorNames() const;

private:
  int fd_ = -1;
  void *mmap_ptr_ = nullptr;
  size_t file_size_ = 0;
  const uint8_t *data_base_ = nullptr;
  std::unordered_map<std::string, TensorMeta> tensors_;
  std::vector<uint8_t> buffer_; // Fallback buffer if mmap fails
};

} // namespace speaker_diarization

#endif // __WEIGHT_LOADER_H__
