// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   weight_loader.cpp
 * @date   29 September 2026
 * @brief  Binary weight loader implementation for Speaker Diarization
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "weight_loader.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace speaker_diarization {

WeightLoader::~WeightLoader() {
  if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED) {
    munmap(mmap_ptr_, file_size_);
    mmap_ptr_ = nullptr;
  }
  if (fd_ != -1) {
    close(fd_);
    fd_ = -1;
  }
}

bool WeightLoader::load(const std::string &bin_path) {
  tensors_.clear();

  fd_ = open(bin_path.c_str(), O_RDONLY);
  if (fd_ == -1) {
    std::cerr << "[WeightLoader] Failed to open: " << bin_path << std::endl;
    return false;
  }

  struct stat sb;
  if (fstat(fd_, &sb) == -1) {
    std::cerr << "[WeightLoader] Failed to fstat: " << bin_path << std::endl;
    close(fd_);
    fd_ = -1;
    return false;
  }
  file_size_ = sb.st_size;

  mmap_ptr_ = mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
  const uint8_t *raw_ptr = nullptr;
  if (mmap_ptr_ == MAP_FAILED) {
    // Fallback to std::ifstream read
    mmap_ptr_ = nullptr;
    std::ifstream f(bin_path, std::ios::binary);
    if (!f.is_open()) return false;
    buffer_.resize(file_size_);
    f.read(reinterpret_cast<char*>(buffer_.data()), file_size_);
    raw_ptr = buffer_.data();
  } else {
    raw_ptr = static_cast<const uint8_t*>(mmap_ptr_);
  }

  // Parse header
  if (file_size_ < 12) {
    std::cerr << "[WeightLoader] File too small: " << file_size_ << std::endl;
    return false;
  }

  if (std::memcmp(raw_ptr, "NNTRDIAR", 8) != 0) {
    std::cerr << "[WeightLoader] Invalid magic header!" << std::endl;
    return false;
  }

  uint32_t num_tensors = 0;
  std::memcpy(&num_tensors, raw_ptr + 8, 4);

  size_t curr_pos = 12;
  for (uint32_t i = 0; i < num_tensors; ++i) {
    if (curr_pos + 4 > file_size_) return false;
    uint32_t name_len = 0;
    std::memcpy(&name_len, raw_ptr + curr_pos, 4);
    curr_pos += 4;

    if (curr_pos + name_len > file_size_) return false;
    std::string name(reinterpret_cast<const char*>(raw_ptr + curr_pos), name_len);
    curr_pos += name_len;

    if (curr_pos + 4 > file_size_) return false;
    uint32_t num_dims = 0;
    std::memcpy(&num_dims, raw_ptr + curr_pos, 4);
    curr_pos += 4;

    std::vector<uint32_t> dims(num_dims);
    for (uint32_t d = 0; d < num_dims; ++d) {
      if (curr_pos + 4 > file_size_) return false;
      std::memcpy(&dims[d], raw_ptr + curr_pos, 4);
      curr_pos += 4;
    }

    if (curr_pos + 16 > file_size_) return false;
    uint64_t byte_size = 0;
    uint64_t offset = 0;
    std::memcpy(&byte_size, raw_ptr + curr_pos, 8);
    curr_pos += 8;
    std::memcpy(&offset, raw_ptr + curr_pos, 8);
    curr_pos += 8;

    tensors_[name] = TensorMeta{name, dims, byte_size, offset};
  }

  // 64-byte alignment for data section
  size_t pad_len = (64 - (curr_pos % 64)) % 64;
  curr_pos += pad_len;
  data_base_ = raw_ptr + curr_pos;

  std::cout << "[WeightLoader] Successfully loaded " << tensors_.size()
            << " tensors from " << bin_path << " (data offset: " << curr_pos << ")" << std::endl;
  return true;
}

const float *WeightLoader::getTensor(const std::string &name, std::vector<uint32_t> *dims) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    std::cerr << "[WeightLoader] Tensor not found: " << name << std::endl;
    return nullptr;
  }

  if (dims) {
    *dims = it->second.dims;
  }

  return reinterpret_cast<const float*>(data_base_ + it->second.offset);
}

bool WeightLoader::hasTensor(const std::string &name) const {
  return tensors_.find(name) != tensors_.end();
}

std::vector<std::string> WeightLoader::getTensorNames() const {
  std::vector<std::string> names;
  names.reserve(tensors_.size());
  for (const auto &kv : tensors_) {
    names.push_back(kv.first);
  }
  return names;
}

} // namespace speaker_diarization
