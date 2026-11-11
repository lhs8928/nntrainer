// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   cross_attention.cpp
 * @date   04 November 2026
 * @brief  Encoder-decoder (cross) attention over a fixed key/value memory
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>

#include <cross_attention.h>
#include <nntrainer_error.h>
#include <thread_manager.h>

namespace quick_ai {

namespace {

using DataType = ml::train::TensorDim::DataType;
using MemT = CrossMemory::value_type;

std::mutex g_memory_mutex;
std::map<std::string, CrossMemory> g_memory;
CrossAttentionRows g_capture;

/** @brief n values of a tensor from element `offset` on, as float */
void readFloats(const nntrainer::Tensor &t, size_t offset, size_t n,
                float *dst) {
  if (t.getDataType() == DataType::FP32) {
    std::memcpy(dst, t.getData<float>() + offset, n * sizeof(float));
    return;
  }
#ifdef ENABLE_FP16
  if (t.getDataType() == DataType::FP16) {
    const _FP16 *src = t.getData<_FP16>() + offset;
    for (size_t i = 0; i < n; ++i)
      dst[i] = static_cast<float>(src[i]);
    return;
  }
#endif
  throw std::invalid_argument("cross_attention: unsupported data type");
}

/** @brief write n floats into a tensor from element `offset` on */
void writeFloats(nntrainer::Tensor &t, size_t offset, size_t n,
                 const float *src) {
  if (t.getDataType() == DataType::FP32) {
    std::memcpy(t.getData<float>() + offset, src, n * sizeof(float));
    return;
  }
#ifdef ENABLE_FP16
  if (t.getDataType() == DataType::FP16) {
    _FP16 *dst = t.getData<_FP16>() + offset;
    for (size_t i = 0; i < n; ++i)
      dst[i] = static_cast<_FP16>(src[i]);
    return;
  }
#endif
  throw std::invalid_argument("cross_attention: unsupported output type");
}

/** @brief copy a tensor's first `n` values into memory storage */
void storeTensor(const nntrainer::Tensor &t, size_t n, std::vector<MemT> &dst) {
  dst.resize(n);
  if (t.getDataType() == DataType::FP32) {
    const float *src = t.getData<float>();
    for (size_t i = 0; i < n; ++i)
      dst[i] = static_cast<MemT>(src[i]);
    return;
  }
#ifdef ENABLE_FP16
  if (t.getDataType() == DataType::FP16) {
    const _FP16 *src = t.getData<_FP16>();
    for (size_t i = 0; i < n; ++i)
      dst[i] = static_cast<MemT>(src[i]);
    return;
  }
#endif
  throw std::invalid_argument("cross_memory_writer: unsupported data type");
}

} // namespace

CrossMemory &crossMemory(const std::string &id) {
  std::lock_guard<std::mutex> lock(g_memory_mutex);
  return g_memory[id];
}

void crossMemoryRelease(const std::string &prefix) {
  std::lock_guard<std::mutex> lock(g_memory_mutex);
  for (auto it = g_memory.begin(); it != g_memory.end();) {
    if (it->first.rfind(prefix, 0) == 0)
      it = g_memory.erase(it);
    else
      ++it;
  }
}

void crossAttentionCaptureClear() { g_capture.clear(); }

const CrossAttentionRows &crossAttentionCapture() { return g_capture; }

/* ----------------------------- memory writer ----------------------------- */

void CrossMemoryWriterLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() == 0 || context.getNumInputs() % 2 != 0,
                std::invalid_argument)
    << "cross_memory_writer takes (key, value) pairs";
  NNTR_THROW_IF(memory_prefix.empty(), std::invalid_argument)
    << "cross_memory_writer needs memory_prefix";
  const auto &dims = context.getInputDimensions();
  for (size_t i = 0; i < dims.size(); ++i)
    NNTR_THROW_IF(dims[i] != dims[0], std::invalid_argument)
      << "cross_memory_writer: every key and value must have the same shape";

  nntrainer::TensorDim out_dim(
    1, 1, 1, 1,
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     context.getActivationDataType()));
  context.setOutputDimensions({out_dim});
}

void CrossMemoryWriterLayer::forwarding(nntrainer::RunLayerContext &context,
                                        bool training) {
  incremental_forwarding(context, 0, 0, training);
}

void CrossMemoryWriterLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  const unsigned int pairs = context.getNumInputs() / 2;
  for (unsigned int i = 0; i < pairs; ++i) {
    const nntrainer::Tensor &k = context.getInput(2 * i);
    const nntrainer::Tensor &v = context.getInput(2 * i + 1);
    // batch 1: the encoder runs one utterance at a time; a step's rows start
    // at row 0
    const size_t height = k.getDim().height();
    const size_t rows =
      to > from ? std::min<size_t>(to - from, height) : height;
    const size_t width = k.getDim().width();
    CrossMemory &mem = crossMemory(memory_prefix + "_l" + std::to_string(i));
    storeTensor(k, rows * width, mem.key);
    storeTensor(v, rows * width, mem.value);
    mem.rows = rows;
    mem.width = width;
  }
  context.getOutput(0).setZero();
}

void CrossMemoryWriterLayer::setProperty(
  const std::vector<std::string> &values) {
  for (const auto &property : values) {
    std::string key, value;
    if (nntrainer::getKeyValue(property, key, value) != 0)
      throw std::invalid_argument("cross_memory_writer: bad property " +
                                  property);
    if (key == "memory_prefix")
      memory_prefix = value;
  }
}

/* ---------------------------- cross attention ---------------------------- */

void CrossAttentionLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "cross_attention takes the query only; the memory is named by "
       "memory_id";
  NNTR_THROW_IF(memory_id.empty(), std::invalid_argument)
    << "cross_attention needs memory_id";

  const auto &q_dim = context.getInputDimensions()[0];
  NNTR_THROW_IF(num_heads == 0 || q_dim.width() % num_heads != 0,
                std::invalid_argument)
    << "cross_attention: width " << q_dim.width()
    << " is not divisible by num_heads " << num_heads;

  nntrainer::TensorDim out_dim = q_dim;
  out_dim.setTensorType({context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions({out_dim});
}

void CrossAttentionLayer::forwarding(nntrainer::RunLayerContext &context,
                                     bool training) {
  const unsigned int rows = context.getInput(0).getDim().height();
  incremental_forwarding(context, 0, rows, training);
}

void CrossAttentionLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor &query = context.getInput(0);
  nntrainer::Tensor &output = context.getOutput(0);
  const CrossMemory &mem = crossMemory(memory_id);

  const auto &q_dim = query.getDim();
  const size_t width = q_dim.width();
  NNTR_THROW_IF(mem.rows == 0 || mem.width != width, std::runtime_error)
    << "cross_attention: memory " << memory_id << " is not set";

  const size_t head_dim = width / num_heads;
  const size_t mem_rows = mem.rows;
  // A decode step's input holds only its own rows, from row 0 on.
  const size_t steps = std::min<size_t>(to - from, q_dim.height());
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  std::vector<float> q(steps * width), out(steps * width);
  std::vector<float> scores(static_cast<size_t>(num_heads) * steps * mem_rows);

  for (unsigned int b = 0; b < q_dim.batch(); ++b) {
    readFloats(query, b * q_dim.getFeatureLen(), steps * width, q.data());

    auto &tm = nntrainer::ThreadManager::Global();
    tm.parallel_for(0, num_heads, [&](size_t h) {
      const size_t off = h * head_dim;
      for (size_t t = 0; t < steps; ++t) {
        const float *qh = q.data() + t * width + off;
        float *s = scores.data() + (h * steps + t) * mem_rows;
        for (size_t j = 0; j < mem_rows; ++j) {
          const MemT *k = mem.key.data() + j * width + off;
          float acc = 0.0f;
          for (size_t d = 0; d < head_dim; ++d)
            acc += qh[d] * static_cast<float>(k[d]);
          s[j] = acc * scale;
        }
      }
    });

    if (b == 0) {
      for (int h : capture_heads) {
        if (h < 0 || static_cast<unsigned int>(h) >= num_heads)
          continue;
        auto &by_pos = g_capture[{layer_index, h}];
        for (size_t t = 0; t < steps; ++t) {
          const float *s = scores.data() + (h * steps + t) * mem_rows;
          by_pos[from + static_cast<unsigned int>(t)].assign(s, s + mem_rows);
        }
      }
    }

    tm.parallel_for(0, num_heads, [&](size_t h) {
      const size_t off = h * head_dim;
      for (size_t t = 0; t < steps; ++t) {
        float *s = scores.data() + (h * steps + t) * mem_rows;
        const float mx = *std::max_element(s, s + mem_rows);
        float sum = 0.0f;
        for (size_t j = 0; j < mem_rows; ++j) {
          s[j] = std::exp(s[j] - mx);
          sum += s[j];
        }
        const float inv = 1.0f / sum;
        float *o = out.data() + t * width + off;
        std::fill(o, o + head_dim, 0.0f);
        for (size_t j = 0; j < mem_rows; ++j) {
          const MemT *v = mem.value.data() + j * width + off;
          const float w = s[j] * inv;
          for (size_t d = 0; d < head_dim; ++d)
            o[d] += w * static_cast<float>(v[d]);
        }
      }
    });

    writeFloats(output, b * output.getDim().getFeatureLen(), steps * width,
                out.data());
  }
}

void CrossAttentionLayer::setProperty(const std::vector<std::string> &values) {
  for (const auto &property : values) {
    std::string key, value;
    if (nntrainer::getKeyValue(property, key, value) != 0)
      throw std::invalid_argument("cross_attention: bad property " + property);
    if (key == "num_heads") {
      num_heads = static_cast<unsigned int>(std::stoul(value));
    } else if (key == "layer_index") {
      layer_index = std::stoi(value);
    } else if (key == "memory_id") {
      memory_id = value;
    } else if (key == "capture_heads") {
      capture_heads.clear();
      std::stringstream ss(value);
      std::string item;
      while (std::getline(ss, item, ',')) {
        if (!item.empty() && item != "none")
          capture_heads.push_back(std::stoi(item));
      }
    }
  }
}

} // namespace quick_ai
