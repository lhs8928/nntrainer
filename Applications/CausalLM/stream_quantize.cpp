// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   stream_quantize.cpp
 * @date   11 November 2026
 * @brief  nntr_stream_quantize: write an NNTrainer weight file from a stream
 *         of FP32 tensors, quantizing the marked ones to Q4_0 on the fly
 * @see    https://github.com/nntrainer/nntrainer
 * @author Claude <noreply@anthropic.com>
 * @bug    No known bugs except for NYI items
 *
 * nntr_quantize builds the model and loads the whole FP32 weight file before
 * it re-saves it, which does not fit when the FP32 model is larger than the
 * memory (a 30B MoE is ~130 GB). This tool lets a converter script stream the
 * weights one tensor at a time instead, in file order, and writes each record
 * exactly as Layer::save() does, so the result is byte-identical to the
 * nntr_quantize output.
 *
 * Usage:  <producer> | nntr_stream_quantize <output.bin> [--isa
 * X86|ARM|DEFAULT]
 *
 * Input stream (little endian), repeated, then the 4-byte tag "NTQE":
 *   char[4]  "NTQR"
 *   uint32   kind      0 = copy as FP32, 1 = quantize to Q4_0
 *   uint32   rows      K (input features of a [K, N] fully_connected weight)
 *   uint32   cols      N
 *   float32  data[K * N]  row major, the NNTrainer [in, out] layout
 */

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cpu_backend.h>
#include <tensor.h>
#include <tensor_dim.h>

namespace {

using ml::train::TensorDim;

bool readExact(std::istream &in, void *dst, size_t n) {
  in.read(static_cast<char *>(dst), static_cast<std::streamsize>(n));
  return static_cast<size_t>(in.gcount()) == n;
}

ml::train::ISA parseIsa(const std::string &s) {
  if (s == "X86" || s == "x86")
    return ml::train::ISA::X86;
  if (s == "ARM" || s == "arm")
    return ml::train::ISA::ARM;
  if (s == "DEFAULT" || s == "default")
    return ml::train::ISA::DEFAULT;
  throw std::invalid_argument("unknown --isa " + s);
}

} // namespace

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <output.bin> [--isa X86|ARM|DEFAULT] < stream\n";
    return 1;
  }
  try {
    const std::string out_path = argv[1];
    ml::train::ISA isa = ml::train::ISA::DEFAULT;
    for (int i = 2; i < argc; ++i) {
      if (std::string(argv[i]) == "--isa" && i + 1 < argc)
        isa = parseIsa(argv[++i]);
      else
        throw std::invalid_argument(std::string("unknown argument ") + argv[i]);
    }

    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out.is_open())
      throw std::runtime_error("cannot open " + out_path);
    std::cin.sync_with_stdio(false);

    size_t records = 0, quantized = 0;
    uint64_t in_bytes = 0;
    while (true) {
      char tag[4];
      if (!readExact(std::cin, tag, 4))
        throw std::runtime_error("stream ended without the NTQE tag");
      if (std::memcmp(tag, "NTQE", 4) == 0)
        break;
      if (std::memcmp(tag, "NTQR", 4) != 0)
        throw std::runtime_error("bad record tag in the stream");
      uint32_t hdr[3];
      if (!readExact(std::cin, hdr, sizeof(hdr)))
        throw std::runtime_error("truncated record header");
      const uint32_t kind = hdr[0], K = hdr[1], N = hdr[2];

      nntrainer::Tensor weight(
        1, 1, K, N,
        TensorDim::TensorType(TensorDim::Format::NCHW,
                              TensorDim::DataType::FP32));
      const size_t bytes = static_cast<size_t>(K) * N * sizeof(float);
      if (!readExact(std::cin, weight.getData<float>(), bytes))
        throw std::runtime_error("truncated record data");
      in_bytes += bytes;

      // Same policy and calls as Layer::save() for TensorDim::DataType::Q4_0
      if (kind == 0 || K == 1) {
        weight.save(out);
      } else if (kind == 1) {
        if (N % 32 != 0 || K % 32 != 0)
          throw std::invalid_argument(
            "Q4_0 needs both dimensions divisible by 32, got " +
            std::to_string(K) + "x" + std::to_string(N));
        nntrainer::Tensor weight_t = weight.transpose("0:2:1");
        nntrainer::Tensor quant_weight(
          1, 1, K, N,
          TensorDim::TensorType(TensorDim::Format::NCHW,
                                TensorDim::DataType::Q4_0));
        std::vector<char> tmp(quant_weight.size());
        nntrainer::quantize_q4_0(weight_t.getData<float>(), tmp.data(), N, K,
                                 nullptr);
        nntrainer::repack_q4_0(quant_weight.getData<uint8_t>(), tmp.data(),
                               quant_weight.size(), N, K, isa);
        quant_weight.save(out);
        ++quantized;
      } else {
        throw std::runtime_error("unknown record kind " + std::to_string(kind));
      }
      ++records;
      if (!out)
        throw std::runtime_error("write failed (disk full?)");
    }
    out.close();
    std::cerr << "[nntr_stream_quantize] " << records << " tensors ("
              << quantized << " Q4_0), " << in_bytes / (1 << 20)
              << " MiB in -> " << out_path << "\n";
  } catch (const std::exception &e) {
    std::cerr << "[nntr_stream_quantize] error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
