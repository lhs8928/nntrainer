// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   huggingface_tokenizer_stub.cpp
 * @date   25 August 2026
 * @brief  Tokenizer factories for builds without the Rust tokenizer library
 * @see    https://github.com/nntrainer/nntrainer
 * @author Contributors
 * @bug    No known bugs
 *
 * The HuggingFace tokenizer is a prebuilt Rust static library and
 * Applications/quick_ai/lib only carries an x86_64 one, so a cross build for
 * armv7l cannot link it. Models that set "skip_tokenizer" never construct a
 * tokenizer -- an audio classifier such as CED takes a waveform, not text --
 * so those targets can drop it entirely. Every factory here throws, which
 * turns "the tokenizer was silently missing" into a message that names the
 * build option to flip.
 */
#include <stdexcept>
#include <string>

#include <tokenizers_cpp.h>

namespace tokenizers {

namespace {
[[noreturn]] void unavailable(const char *what) {
  throw std::runtime_error(
    std::string("Tokenizer::") + what +
    " is unavailable: this build was configured with "
    "-Dquick_ai-tokenizer=false. Rebuild with the tokenizer enabled to run a "
    "model that needs one.");
}
} // namespace

std::unique_ptr<Tokenizer> Tokenizer::FromBlobJSON(const std::string &) {
  unavailable("FromBlobJSON");
}

std::unique_ptr<Tokenizer> Tokenizer::FromBlobByteLevelBPE(const std::string &,
                                                           const std::string &,
                                                           const std::string &) {
  unavailable("FromBlobByteLevelBPE");
}

std::unique_ptr<Tokenizer>
Tokenizer::FromBlobSentencePiece(const std::string &) {
  unavailable("FromBlobSentencePiece");
}

std::unique_ptr<Tokenizer> Tokenizer::FromBlobRWKVWorld(const std::string &) {
  unavailable("FromBlobRWKVWorld");
}

} // namespace tokenizers
