// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   room_jni.cpp
 * @date   22 October 2026
 * @brief  JNI methods of ai.nntrainer.discretion.Recognizer
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The Java class holds the native handle (nntr_room *) as a long; errors are
 * thrown as IllegalStateException (create, enroll) or returned as codes.
 */

#include <jni.h>

#include <string>
#include <vector>

#include "room_api.h"

namespace {

std::string str(JNIEnv *env, jstring s) {
  if (!s)
    return "";
  const char *c = env->GetStringUTFChars(s, nullptr);
  std::string out(c ? c : "");
  if (c)
    env->ReleaseStringUTFChars(s, c);
  return out;
}

void raise(JNIEnv *env, const std::string &msg) {
  jclass c = env->FindClass("java/lang/IllegalStateException");
  if (c)
    env->ThrowNew(c, msg.c_str());
}

nntr_room *room(jlong h) { return reinterpret_cast<nntr_room *>(h); }

/** @brief a JSON text from nntr_room_poll / nntr_room_status, however long */
jstring text(JNIEnv *env, nntr_room *r, int (*get)(nntr_room *, char *, int)) {
  std::vector<char> buf(4096);
  while (true) {
    const int n = get(r, buf.data(), static_cast<int>(buf.size()));
    if (n < 0)
      return nullptr;
    if (n < static_cast<int>(buf.size()))
      return env->NewStringUTF(buf.data());
    buf.resize(n + 1);
  }
}

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_ai_nntrainer_discretion_Recognizer_nativeCreate(
  JNIEnv *env, jclass, jstring config) {
  char err[512] = {0};
  nntr_room *r = nntr_room_create(str(env, config).c_str(), err, sizeof(err));
  if (!r)
    raise(env, err);
  return reinterpret_cast<jlong>(r);
}

JNIEXPORT jint JNICALL Java_ai_nntrainer_discretion_Recognizer_nativeFeed(
  JNIEnv *env, jclass, jlong h, jshortArray pcm, jint n, jdouble at_ms) {
  const jsize len = env->GetArrayLength(pcm);
  if (n < 0 || n > len)
    return -1;
  jshort *x = env->GetShortArrayElements(pcm, nullptr);
  const int rc =
    nntr_room_feed(room(h), reinterpret_cast<const int16_t *>(x), n, at_ms);
  env->ReleaseShortArrayElements(pcm, x, JNI_ABORT);
  return rc;
}

JNIEXPORT void JNICALL Java_ai_nntrainer_discretion_Recognizer_nativeEnroll(
  JNIEnv *env, jclass, jlong h, jstring person, jshortArray pcm) {
  const jsize len = env->GetArrayLength(pcm);
  jshort *x = env->GetShortArrayElements(pcm, nullptr);
  char err[512] = {0};
  const int rc = nntr_room_enroll(room(h), str(env, person).c_str(),
                                  reinterpret_cast<const int16_t *>(x), len,
                                  err, sizeof(err));
  env->ReleaseShortArrayElements(pcm, x, JNI_ABORT);
  if (rc != 0)
    raise(env, err);
}

JNIEXPORT jint JNICALL Java_ai_nntrainer_discretion_Recognizer_nativeMapSpeaker(
  JNIEnv *env, jclass, jlong h, jstring label, jstring person) {
  return nntr_room_map_speaker(room(h), str(env, label).c_str(),
                               str(env, person).c_str());
}

JNIEXPORT jstring JNICALL Java_ai_nntrainer_discretion_Recognizer_nativePoll(
  JNIEnv *env, jclass, jlong h) {
  return text(env, room(h), nntr_room_poll);
}

JNIEXPORT jstring JNICALL Java_ai_nntrainer_discretion_Recognizer_nativeStatus(
  JNIEnv *env, jclass, jlong h) {
  return text(env, room(h), nntr_room_status);
}

JNIEXPORT jint JNICALL
Java_ai_nntrainer_discretion_Recognizer_nativeFlush(JNIEnv *, jclass, jlong h) {
  return nntr_room_flush(room(h));
}

JNIEXPORT void JNICALL Java_ai_nntrainer_discretion_Recognizer_nativeDestroy(
  JNIEnv *, jclass, jlong h) {
  nntr_room_destroy(room(h));
}

} // extern "C"
