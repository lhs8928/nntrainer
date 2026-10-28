// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   eres2netv2.h
 * @date   28 October 2026
 * @brief  ERes2NetV2 speaker embedding in NNTrainer (float32)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The 3D-Speaker model iic/speech_eres2netv2_sv_zh-cn_16k-common as the
 * sherpa-onnx release runs it (the speaker model of the ALICE server), with
 * that release's front end, so the same samples give the same 192 numbers:
 * Kaldi fbank of samples in [-1, 1] (80 mel bins, 25 ms / 10 ms frames, povey
 * window, frames not snipped at the edges), each bin's mean over the frames
 * removed. The whole span is used as it is, however long.
 *
 * Network: conv 3x3 (64) -> four stages of Res2Net bottlenecks of scale 2
 * (3, 4, 6, 3 blocks; widths 26, 52, 104, 208; the second half joins the first
 * by addition in stages 1-2 and by attentional feature fusion in stages 3-4)
 * -> stage 3 downsampled and fused with stage 4 -> mean and standard deviation
 * over time -> linear 20480 -> 192. Batch norms are folded into the
 * convolutions, as in the ONNX release.
 */

#ifndef __ERES2NETV2_H__
#define __ERES2NETV2_H__

#include <cstddef>
#include <string>
#include <vector>

#include "weight_loader.h"

namespace speaker_diarization {

class ERes2NetV2 {
public:
  static constexpr size_t DIM = 192;
  static constexpr size_t MEL = 80;

  /**
   * @brief bind the weights (export_eres2netv2.py names them by the module
   *        path of the 3D-Speaker model, e.g. "layer3.0.conv1.weight")
   * @return false with the first missing tensor in `missing`
   */
  bool init(const WeightLoader &loader, std::string *missing = nullptr);

  /** @brief the embedding of 16 kHz samples in [-1, 1]; empty when too short */
  std::vector<float> embed(const float *samples, size_t n) const;

  /**
   * @brief sherpa-onnx's front end: [frames x 80] log mel energies with each
   *        bin's mean over the frames removed
   */
  static std::vector<float> fbank(const float *samples, size_t n,
                                  size_t &frames);

  /** @brief the network on [frames x 80] features */
  std::vector<float> forward(const float *feats, size_t frames) const;

private:
  struct Conv {
    const float *w = nullptr;
    const float *b = nullptr; // may be null
    size_t out = 0, in = 0, k = 1, stride = 1;
  };
  struct Fuse {
    Conv att0, att3;
  };
  struct Block {
    Conv conv1, convs0, convs1, conv3, shortcut; // shortcut.w null: identity
    Fuse fuse;
    bool fused = false;
  };

  bool bind(const WeightLoader &loader, const std::string &name, size_t stride,
            Conv &c, bool required, std::string *missing) const;

  Conv conv1_, layer3_ds_;
  std::vector<Block> blocks_; // layer1.0 .. layer4.2 in order
  size_t layer3_last_ = 0;    // index of layer3's last block
  Fuse fuse34_;
  const float *seg_w_ = nullptr, *seg_b_ = nullptr;
};

} // namespace speaker_diarization

#endif // __ERES2NETV2_H__
