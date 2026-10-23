// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   Recognizer.java
 * @date   22 October 2026
 * @brief  The phone as the meeting room's microphone: lines with speakers
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Usage (the app captures 16 kHz mono PCM16 with AudioRecord):
 *
 *   Recognizer r = new Recognizer(configJson);   // loads the models (seconds)
 *   r.enroll("김철수", threeSentences);          // optional, 3-15 s each
 *   // audio callback: r.feed(buffer, n, -1)  (1 = busy: keep it, retry)
 *   // a timer: String events = r.poll();     JSON array, see room_api.h
 *   r.flush(); r.poll(); r.close();              // the meeting ends
 *
 * Each "line" event becomes one numbered line for ALICE's
 * POST /api/device/lines: {n, speaker, start_ms, end_ms, text, doubt}.
 */

package ai.nntrainer.discretion;

public final class Recognizer implements AutoCloseable {
  static { System.loadLibrary("nntr_discretion_jni"); }

  private long handle;

  /** @param configJson model_dir, sd_weights, silero, ... (room_api.h) */
  public Recognizer(String configJson) {
    handle = nativeCreate(configJson);
  }

  /**
   * @param atMs where the chunk starts on the meeting clock, -1: right after
   *             the previous one
   * @return 0 taken, 1 busy (give the same chunk again later), -1 error
   */
  public int feed(short[] pcm, int n, double atMs) {
    return nativeFeed(check(), pcm, n, atMs);
  }

  /** @brief a participant's voice; throws IllegalStateException with why */
  public void enroll(String person, short[] pcm) {
    nativeEnroll(check(), person, pcm);
  }

  /**
   * @brief a voice the server keeps (256 numbers, WeSpeaker ResNet34), kept
   *        in memory only; throws IllegalStateException with why
   */
  public void enrollVector(String person, float[] vector) {
    nativeEnrollVector(check(), person, vector);
  }

  /**
   * @brief the person's voice over this meeting, to be remembered (call
   *        after flush); null when there is none
   */
  public float[] voiceOf(String person) {
    return nativeVoiceOf(check(), person);
  }

  /** @brief "이건 나예요": a provisional label (화자 N) is this person */
  public boolean mapSpeaker(String label, String person) {
    return nativeMapSpeaker(check(), label, person) == 0;
  }

  /**
   * @brief a line a person corrected stays theirs (person null: nobody)
   * @return false when the recognizer does not know the line
   */
  public boolean pin(String lineId, String person) {
    return nativePin(check(), lineId, person) == 0;
  }

  /** @brief events since the last poll, a JSON array */
  public String poll() { return nativePoll(check()); }

  public String status() { return nativeStatus(check()); }

  /** @brief listening ends: blocks until everything fed is recognized */
  public void flush() { nativeFlush(check()); }

  @Override
  public synchronized void close() {
    if (handle != 0) {
      nativeDestroy(handle);
      handle = 0;
    }
  }

  private long check() {
    if (handle == 0)
      throw new IllegalStateException("Recognizer is closed");
    return handle;
  }

  private static native long nativeCreate(String config);
  private static native int nativeFeed(long h, short[] pcm, int n, double atMs);
  private static native void nativeEnroll(long h, String person, short[] pcm);
  private static native void nativeEnrollVector(long h, String person,
                                                float[] vector);
  private static native float[] nativeVoiceOf(long h, String person);
  private static native int nativeMapSpeaker(long h, String label,
                                             String person);
  private static native int nativePin(long h, String lineId, String person);
  private static native String nativePoll(long h);
  private static native String nativeStatus(long h);
  private static native int nativeFlush(long h);
  private static native void nativeDestroy(long h);
}
