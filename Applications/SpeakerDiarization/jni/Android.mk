# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# @file   Android.mk
# @author Hyeonseok Lee <hs89.lee@samsung.com>
# @date   29 September 2026
# @brief  Android NDK build makefile for Speaker Diarization in NNTrainer
##

LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

ifndef ANDROID_NDK
$(error ANDROID_NDK is not defined!)
endif

ifndef NNTRAINER_ROOT
NNTRAINER_ROOT := $(LOCAL_PATH)/../../..
endif

ML_API_COMMON_INCLUDES := $(NNTRAINER_ROOT)/ml_api_common/include
NNTRAINER_INCLUDES := $(NNTRAINER_ROOT)/nntrainer \
        $(NNTRAINER_ROOT)/nntrainer/dataset \
        $(NNTRAINER_ROOT)/nntrainer/models \
        $(NNTRAINER_ROOT)/nntrainer/layers \
        $(NNTRAINER_ROOT)/nntrainer/compiler \
        $(NNTRAINER_ROOT)/nntrainer/graph \
        $(NNTRAINER_ROOT)/nntrainer/optimizers \
        $(NNTRAINER_ROOT)/nntrainer/tensor \
        $(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend \
        $(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/cblas_interface \
        $(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/fallback \
        $(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/arm \
        $(NNTRAINER_ROOT)/nntrainer/utils \
        $(NNTRAINER_ROOT)/api \
        $(NNTRAINER_ROOT)/api/ccapi/include \
        $(LOCAL_PATH)/../models \
        $(ML_API_COMMON_INCLUDES)

include $(CLEAR_VARS)

LOCAL_MODULE := nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/libs/$(TARGET_ARCH_ABI)/libnntrainer.so

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE := ccapi-nntrainer
LOCAL_SRC_FILES := $(NNTRAINER_ROOT)/libs/$(TARGET_ARCH_ABI)/libccapi-nntrainer.so

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_CFLAGS += -std=c++17 -O3 -pthread -fexceptions
LOCAL_CXXFLAGS += -std=c++17 -frtti -fexceptions
LOCAL_LDFLAGS += -fexceptions
LOCAL_MODULE := nntrainer_speaker_diarization
LOCAL_LDLIBS := -llog -landroid

LOCAL_SRC_FILES := main.cpp \
        ../models/weight_loader.cpp \
        ../models/audio_preprocessor.cpp \
        ../models/pyannet.cpp \
        ../models/wespeaker_resnet34.cpp \
        ../models/plda_vbx.cpp

LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer
LOCAL_C_INCLUDES += $(NNTRAINER_INCLUDES)

include $(BUILD_EXECUTABLE)
