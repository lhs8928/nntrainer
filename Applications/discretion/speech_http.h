// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   speech_http.h
 * @date   22 October 2026
 * @brief  HTTP front of the speech models, inside the executable
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The same contract as speech_server.py, for devices without Python (an
 * Android phone). Bodies are 16 kHz mono PCM16; answers are JSON:
 * - GET /health: {"status": "ready", "queue": n, ...}
 * - POST /asr: {"text", "words": [[start_ms, end_ms, word], ...], ...}
 * - POST /embed: {"vector": [...], "ms"}
 * - POST /segment: {"frame_ms", "windows": [...], "ms"}
 *
 * Every connection gets its own thread; the models answer one request at a
 * time, so /health answers while a segment is being recognized.
 */

#ifndef __DISCRETION_SPEECH_HTTP_H__
#define __DISCRETION_SPEECH_HTTP_H__

#include <string>

namespace discretion {

class SpeechOps;

struct HttpOptions {
  std::string host = "127.0.0.1"; // 0.0.0.0 to answer other machines
  int port = 8030;
  std::string token; // non-empty: requests need Authorization: Bearer TOKEN
};

/**
 * @brief listen and answer until the process is stopped
 * @return EXIT_FAILURE when the address cannot be bound
 */
int serveHttp(SpeechOps &ops, const HttpOptions &opt);

} // namespace discretion

#endif // __DISCRETION_SPEECH_HTTP_H__
