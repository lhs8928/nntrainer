# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file   Android.mk
# @author Hyeonseok Lee <hs89.lee@samsung.com>
# @date   22 October 2026
# @brief  Android NDK build of nntr_discretion (ASR + speaker diarization)
#
# Needs the Android builds of nntrainer (builddir/android_build_result) and of
# quick_ai (Applications/quick_ai/jni/libs, Applications/quick_ai/lib):
# run Applications/quick_ai/build_android.sh first.
##

LOCAL_PATH := $(call my-dir)

ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../..
endif

QUICK_AI := $(NNTRAINER_ROOT)/Applications/quick_ai

# The installed nntrainer headers, the nntrainer sources the diarization models
# reach into (CPU backend), and the quick_ai headers. The diarization model
# directory is left out on purpose: it and quick_ai both have an
# audio_preprocessor.h with the same include guard, and each side's headers are
# reached through the including file's own directory (as in meson.build).
DISCRETION_INCLUDES := \
    $(NNTRAINER_ROOT)/builddir/android_build_result/include/nntrainer \
    $(NNTRAINER_ROOT)/nntrainer \
    $(NNTRAINER_ROOT)/nntrainer/tensor \
    $(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend \
    $(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/cblas_interface \
    $(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/fallback \
    $(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/arm \
    $(NNTRAINER_ROOT)/nntrainer/utils \
    $(NNTRAINER_ROOT)/api \
    $(NNTRAINER_ROOT)/api/ccapi/include \
    $(NNTRAINER_ROOT)/ml_api_common/include \
    $(LOCAL_PATH)/.. \
    $(QUICK_AI) \
    $(QUICK_AI)/layers \
    $(QUICK_AI)/models \
    $(QUICK_AI)/models/qwen3_asr \
    $(QUICK_AI)/third_party/minja/include \
    $(QUICK_AI)/third_party

include $(CLEAR_VARS)
LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/builddir/android_build_result/lib/$(TARGET_ARCH_ABI)/libnntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/builddir/android_build_result/lib/$(TARGET_ARCH_ABI)/libccapi-nntrainer.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := quick_ai_core
LOCAL_SRC_FILES := $(QUICK_AI)/jni/libs/$(TARGET_ARCH_ABI)/libquick_ai_core.so
include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := tokenizers_c
LOCAL_SRC_FILES := $(QUICK_AI)/lib/libtokenizers_android_c.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_ARM_NEON := true
LOCAL_ARM_MODE := arm
LOCAL_MODULE := nntr_discretion
LOCAL_CFLAGS += -O3 -pthread -fexceptions -fopenmp -DENABLE_FP16=1 -DUSE__FP16=1 \
    -D__ARM_NEON__=1 -DUSE_NEON=1 -march=armv8.2-a+fp16+dotprod+i8mm \
    -mtune=cortex-a76 -Wno-nan-infinity-disabled -Wno-deprecated-literal-operator
LOCAL_CXXFLAGS += -std=c++20 -frtti
LOCAL_LDFLAGS += -fexceptions -fopenmp -static-openmp
LOCAL_LDLIBS := -llog -landroid

SD_MODELS := ../../SpeakerDiarization/models
LOCAL_SRC_FILES := \
    ../main.cpp \
    ../asr_engine.cpp \
    ../asr_lane.cpp \
    ../sd_lane.cpp \
    ../audio_fanout.cpp \
    ../merger.cpp \
    ../batch_pipeline.cpp \
    ../speech_ops.cpp \
    $(SD_MODELS)/weight_loader.cpp \
    $(SD_MODELS)/audio_preprocessor.cpp \
    $(SD_MODELS)/pyannet.cpp \
    $(SD_MODELS)/wespeaker_resnet34.cpp \
    $(SD_MODELS)/plda_vbx.cpp \
    $(SD_MODELS)/online_speaker_tracker.cpp \
    $(SD_MODELS)/audio_stream_buffer.cpp \
    $(SD_MODELS)/streaming_diarizer.cpp \
    $(SD_MODELS)/batch_diarizer.cpp

LOCAL_SHARED_LIBRARIES := quick_ai_core nntrainer ccapi-nntrainer
LOCAL_STATIC_LIBRARIES := tokenizers_c
LOCAL_C_INCLUDES += $(DISCRETION_INCLUDES)
include $(BUILD_EXECUTABLE)
