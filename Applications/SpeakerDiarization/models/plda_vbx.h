// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   plda_vbx.h
 * @date   29 September 2026
 * @brief  PLDA transformation, VBx clustering, and timeline reconstruction in NNTrainer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __PLDA_VBX_H__
#define __PLDA_VBX_H__

#include <vector>
#include <string>
#include "weight_loader.h"

namespace speaker_diarization {

struct SpeakerSegment {
  std::string speaker;
  float start_time;
  float end_time;
  float duration;
};

class PldaVBx {
public:
  PldaVBx() = default;
  ~PldaVBx() = default;

  /**
   * @brief Initialize PLDA and transformation weights.
   */
  bool init(const WeightLoader &loader);

  /**
   * @brief Transform 256-dim embedding vector into 128-dim PLDA latent space.
   * @param in_embedding 256-dim embedding.
   * @param out_plda_fea Output 128-dim PLDA feature.
   */
  void transformPLDA(const float *in_embedding, float *out_plda_fea);

  /**
   * @brief Cluster speaker embeddings across chunks and construct continuous timeline segments.
   * @param embeddings Flat array of embeddings [num_chunks * num_local_speakers, 256].
   * @param segmentations Segmentation probabilities [num_chunks, 589, 3].
   * @param speaker_counting Active speaker count per frame [949].
   * @return List of diarization speaker segments.
   */
  std::vector<SpeakerSegment> diarize(
    const std::vector<float> &embeddings,
    const std::vector<std::vector<float>> &segmentations,
    const std::vector<uint8_t> &speaker_counting);

private:
  // xvec_transform.npz
  const float *mean1_ = nullptr; // [256]
  const float *lda_   = nullptr; // [256, 128]
  const float *mean2_ = nullptr; // [128]

  // plda.npz
  const float *mu_    = nullptr; // [128]
  const float *tr_    = nullptr; // [128, 128]
  const float *psi_   = nullptr; // [128]
};

} // namespace speaker_diarization

#endif // __PLDA_VBX_H__
