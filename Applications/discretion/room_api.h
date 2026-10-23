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
 *   language           "Korean" (default), "" = the model decides
 *   asr_secs           model graph lengths (default [8, 12.5])
 *   threads            worker threads of the models (0 = default)
 *   vad                "silero" (default) or "energy"
 *   silero             silero_vad_v6.onnx path (energy gate when missing)
 *   silero_threshold   0.3
 *   speaker_threshold  0.6 (WeSpeaker ResNet34, ALICE's value for it)
 *
 * nntr_room_poll gives a JSON array of events, oldest first:
 *   {"type":"line","id":"audio-3","speaker":"화자 1"|null,"start_ms":..,
 *    "end_ms":..,"text":..,"doubt":[..],"info":{..}}
 *   {"type":"moved","moved":{"audio-1":"화자 2"}}   earlier lines relabelled
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

/** @brief "이건 나예요": a provisional label is this person; 0 or -1 */
int nntr_room_map_speaker(nntr_room *room, const char *label,
                          const char *person);

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
