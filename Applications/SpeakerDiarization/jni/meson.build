# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file   meson.build
# @author Hyeonseok Lee <hs89.lee@samsung.com>
# @date   29 September 2026
# @brief  Meson build file for Speaker Diarization in NNTrainer
##

speaker_diarization_sources = [
  'main.cpp',
  '../models/weight_loader.cpp',
  '../models/audio_preprocessor.cpp',
  '../models/pyannet.cpp',
  '../models/wespeaker_resnet34.cpp',
  '../models/plda_vbx.cpp',
]

openmp_dep = dependency('openmp', required: false)

speaker_diarization_dependencies = [
  app_utils_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

if openmp_dep.found()
  speaker_diarization_dependencies += openmp_dep
endif

e = executable('nntrainer_speaker_diarization',
  speaker_diarization_sources,
  include_directories: [include_directories('.'), include_directories('../models')],
  dependencies: speaker_diarization_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
