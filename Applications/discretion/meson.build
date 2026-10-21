# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file   meson.build
# @author Hyeonseok Lee <hs89.lee@samsung.com>
# @date   12 October 2026
# @brief  Meson build file for the combined ASR + speaker diarization app
##

# The app pins threads to CPUs (sched_setaffinity) and replays/records audio
# through POSIX pipes, so it is only built for plain Linux targets. Android,
# Tizen and Windows builds are left untouched.
if host_machine.system() == 'linux' and get_option('platform') == 'none' and get_option('quick_ai-tokenizer')

  # The speaker diarization models are compiled straight into this executable
  # (by path), so the diarization application itself keeps its own build file.
  sd_models = '../SpeakerDiarization/models'
  discretion_sources = [
    'main.cpp',
    'asr_engine.cpp',
    'asr_lane.cpp',
    'sd_lane.cpp',
    'audio_fanout.cpp',
    'merger.cpp',
    'batch_pipeline.cpp',
    sd_models / 'weight_loader.cpp',
    sd_models / 'audio_preprocessor.cpp',
    sd_models / 'pyannet.cpp',
    sd_models / 'wespeaker_resnet34.cpp',
    sd_models / 'plda_vbx.cpp',
    sd_models / 'online_speaker_tracker.cpp',
    sd_models / 'audio_stream_buffer.cpp',
    sd_models / 'streaming_diarizer.cpp',
    sd_models / 'batch_diarizer.cpp',
  ]

  discretion_dependencies = [
    app_utils_dep,
    nntrainer_dep,
    nntrainer_ccapi_dep,
    quick_ai_layer_dependencies,
    quick_ai_dep,
    dependency('threads'),
  ]

  discretion_openmp_dep = dependency('openmp', required: false)
  if discretion_openmp_dep.found()
    discretion_dependencies += discretion_openmp_dep
  endif

  # No SD model directory in the include path on purpose: the diarization and
  # quick_ai models both have an audio_preprocessor.h with the same include
  # guard. Each side's headers are only reached through the including file's own
  # directory, and no source file includes both.
  e_discretion = executable('nntr_discretion',
    discretion_sources,
    include_directories: [include_directories('.')] + quick_ai_inc + quick_ai_external_incs,
    dependencies: discretion_dependencies,
    install: get_option('install-app'),
    install_dir: application_install_dir,
    # libquick_ai.so is installed next to the executable
    install_rpath: application_install_dir
  )
endif
