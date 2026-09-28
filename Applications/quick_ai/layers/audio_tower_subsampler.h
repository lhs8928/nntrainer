// SPDX-License-Identifier: Apache-2.0
/**
 * @file   audio_tower_subsampler.h
 * @brief  Qwen3-ASR Audio Tower Chunked Subsampling Layer (PyTorch 100-frame chunking)
 */

#ifndef __AUDIO_TOWER_SUBSAMPLER_H__
#define __AUDIO_TOWER_SUBSAMPLER_H__

#include <layer_impl.h>
#include <tensor.h>
#include <vector>

namespace quick_ai {

/**
 * @class   AudioTowerSubsamplerLayer
 * @brief   Custom layer executing chunked 100-frame audio convolutions (Conv1 -> GELU -> Conv2 -> GELU -> Conv3 -> GELU -> ConvOut -> PosEmbed)
 *          producing exact downsampled tokens aligned with PyTorch for arbitrary audio lengths up to 20 minutes.
 */
class AudioTowerSubsamplerLayer final : public nntrainer::LayerImpl {
public:
  AudioTowerSubsamplerLayer();
  ~AudioTowerSubsamplerLayer() override = default;

  void finalize(nntrainer::InitLayerContext &context) override;
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;
  void calcDerivative(nntrainer::RunLayerContext &context) override {}
  bool supportBackwarding() const override { return false; }
  void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "audio_tower_subsampler";
  const std::string getType() const override { return type; }

private:
  unsigned int downsampled_len = 0;

  enum WeightIdx {
    CONV1_W = 0,
    CONV1_B,
    CONV2_W,
    CONV2_B,
    CONV3_W,
    CONV3_B,
    CONV_OUT_W,
    POS_EMBED_W,
    NUM_WEIGHTS
  };
  std::array<unsigned int, NUM_WEIGHTS> wt_idx;

  // Cached FP32 weights for conv_out and pos_embed to enable fast SGEMM
  std::vector<float> cached_conv_out_fp32;
  std::vector<float> cached_pos_embed_fp32;
  bool weights_cached = false;
};

} // namespace quick_ai

#endif // __AUDIO_TOWER_SUBSAMPLER_H__
