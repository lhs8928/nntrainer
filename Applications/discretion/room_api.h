// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   room_api.h
 * @date   22 October 2026
 * @brief  C interface of the room recognizer (libnntr_discretion_jni.so)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * One phone is the meeting room's microphone: the app gives it 16 kHz mono
 * PCM16 as it is captured and polls for finished lines, each with its
 * speaker (the whole meeting clustered, ALICE's rules), times on the meeting
 * clock and why its speaker may be wrong. Recognition runs on a worker
 * thread of its own, so nntr_room_feed never blocks the audio callback.
 *
 * config_json (all optional but the two model paths):
 *   model_dir          Qwen3-ASR model directory
 *   sd_weights         speaker_diarization.bin (PyanNet + WeSpeaker)
 *   speaker_model      "wespeaker" (default) or "eres2netv2": the voice model
 *                      (ERes2NetV2 is the ALICE server's: the same samples
 *                      give the server's 192 numbers)
 *   speaker_weights    speaker_eres2netv2.bin, for "eres2netv2"
 *   language           "Korean" (default), "" = the model decides
 *   asr_secs           model graph lengths (default [8, 12.5])
 *   threads            worker threads of the models (0 = default)
 *   vad                "silero" (default) or "energy"
 *   silero             silero_vad_v6.onnx path (energy gate when missing)
 *   silero_threshold   0.3
 *   speaker_threshold  ALICE's value for the voice model: 0.6 (WeSpeaker),
 *                      0.66 (ERes2NetV2)
 *   labels_in_use      labels an earlier run of this meeting showed, e.g.
 *                      ["화자 1","화자 2"]: a new voice never gets them
 *                      (the app restarted mid-meeting)
 *
 * nntr_room_poll gives a JSON array of events, oldest first:
 *   {"type":"line","id":"audio-3","speaker":"화자 1"|null,"start_ms":..,
 *    "end_ms":..,"text":..,"doubt":[..],"info":{..}}
 *   {"type":"moved","moved":{"audio-1":"화자 2"}}   earlier lines relabelled
 *     (only by nntr_room_map_speaker / nntr_room_pin: a label once shown
 *     stays; an id may be a part inside a joined line)
 *   {"type":"enrolled","person":..,"ok":true|false,"error":..}
 *     an enrollment from the room (nntr_room_begin_enrollment) is done, or
 *     listening ended first (nntr_room_flush: ok false)
 *   {"type":"error","message":..}
 */

#ifndef __DISCRETION_ROOM_API_H__
#define __DISCRETION_ROOM_API_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nntr_room nntr_room;

/** @brief load the models; NULL on error (the reason in err) */
nntr_room *nntr_room_create(const char *config_json, char *err, int err_cap);

/**
 * @brief audio as it is captured
 * @param at_ms where the chunk starts on the meeting clock (ms since the
 *        meeting began); < 0: right after the previous chunk
 * @return 0 taken, 1 busy (over 60 s waits for recognition: keep the chunk
 *         and give it again), -1 error
 */
int nntr_room_feed(nntr_room *room, const int16_t *pcm, int n, double at_ms);

/**
 * @brief a person's voice from 3-15 s of their speech (waits for the
 *        worker); 0 or -1 with the reason in err
 */
int nntr_room_enroll(nntr_room *room, const char *person, const int16_t *pcm,
                     int n, char *err, int err_cap);

/**
 * @brief the next seconds (3-15, ALICE uses 6) of the room are this person's
 *        voice: not transcribed (ask the others to be quiet), an "enrolled"
 *        event tells how it went
 * @return 0 started, 1 another person's enrollment is recording, -1 error
 */
int nntr_room_begin_enrollment(nntr_room *room, const char *person,
                               double seconds);

/**
 * @brief a voice the server keeps (a profile or remembered voice): n numbers
 *        of the voice model (256 WeSpeaker, 192 ERes2NetV2), kept in memory
 *        only; 0 or -1 with the
 *        reason in err
 */
int nntr_room_enroll_vector(nntr_room *room, const char *person,
                            const float *vector, int n, char *err, int err_cap);

/**
 * @brief the person's voice over this meeting, to be remembered: the
 *        duration-weighted mean of their lines (5 s or more), else their
 *        enrolled voice (call after nntr_room_flush)
 * @return its length (copied when cap is enough), 0 when there is none, -1
 */
int nntr_room_voice_of(nntr_room *room, const char *person, float *out,
                       int cap);

/** @brief "이건 나예요": a provisional label is this person; 0 or -1 */
int nntr_room_map_speaker(nntr_room *room, const char *label,
                          const char *person);

/**
 * @brief a line a person corrected ("이 말은 ○○", "내 말 아님"): it stays
 *        theirs when the diarizer clusters again, and so do the parts a
 *        joined line holds
 * @param line_id the line's "id" from nntr_room_poll
 * @param person the person, NULL = nobody (the line leaves clustering)
 * @return 0 applied, 1 unknown line (ignored), -1 error
 */
int nntr_room_pin(nntr_room *room, const char *line_id, const char *person);

/**
 * @brief the events since the last poll as a JSON array
 * @return the length of the array; when it is cap or more nothing is copied
 *         or consumed (call again with a larger buffer)
 */
int nntr_room_poll(nntr_room *room, char *out, int cap);

/** @brief queue, speakers and gate as JSON; length as nntr_room_poll */
int nntr_room_status(nntr_room *room, char *out, int cap);

/** @brief listening ends: wait until everything fed is recognized */
int nntr_room_flush(nntr_room *room);

void nntr_room_destroy(nntr_room *room);

#ifdef __cplusplus
}
#endif

#endif // __DISCRETION_ROOM_API_H__
