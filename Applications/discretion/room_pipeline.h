// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   room_pipeline.h
 * @date   22 October 2026
 * @brief  Meeting lines from one room microphone (ALICE's AudioPipeline)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * A port of ambient/audio.py AudioPipeline (ambient-agent-alice, the 'pause'
 * endpointing with the diarize registry) and the helpers it uses:
 * - 20 ms frames on a voice gate: a segment ends after an 0.8 s pause, starts
 *   0.3 s before the onset (or where the last one ended, within 8 s), ends at
 *   a dip near its 12 s cap or at the quietest moment of its last 2 s;
 * - in a segment, the segmentation model's speaker changes (Segmenter.detail)
 *   kept only where both sides last a second and sound different; the text
 *   is divided by word times (align.split_by_time) or by speaking time
 *   (share); lines nobody could say in their time are dropped (plausible);
 * - each line's voice goes to the whole-meeting clustering (RoomDiarizer) and
 *   carries why its label may be wrong (speaker_doubt).
 * ALICE's own-voice gating, barge-in cuts and recognizer context are left out:
 * the phone does not speak, and Qwen3-ASR takes no context.
 */

#ifndef __DISCRETION_ROOM_PIPELINE_H__
#define __DISCRETION_ROOM_PIPELINE_H__

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "room_diarizer.h"
#include "voice_gate.h"

namespace discretion {

/** @brief the three speech models the pipeline calls (16 kHz floats) */
class SpeechModels {
public:
  struct Word {
    long start_ms, end_ms;
    std::string text;
  };
  struct Recognized {
    std::string text;
    std::vector<Word> words;
    double ms = 0;
  };
  virtual ~SpeechModels() = default;
  virtual Recognized recognize(const std::vector<float> &pcm) = 0;
  virtual std::vector<float> embed(const std::vector<float> &pcm) = 0;
  /** @brief powerset class per frame (10000/589 ms), per 10 s window */
  virtual std::vector<std::vector<uint8_t>>
  segment(const std::vector<float> &pcm) = 0;
};

/** @brief one emitted line and how it was recognized and labelled */
struct RoomLine {
  std::string id;      // audio-N
  std::string speaker; // "" = no speaker
  long start_ms = 0, end_ms = 0;
  std::string text;
  std::vector<std::string> doubt;
  std::string info; // JSON object (ALICE's line_info)
};

/** @brief a local speaker turn inside one segment (Segmenter.detail) */
struct LocalTurn {
  long start_ms, end_ms;
  std::string who;
  long overlap_ms = 0, folded_ms = 0;
};

/** @brief Segmenter.detail on the powerset classes of each 10 s window */
std::vector<LocalTurn>
detailTurns(const std::vector<std::vector<uint8_t>> &windows, size_t samples,
            long min_ms = 500);
/** @brief audio.share: one text divided by the parts' speaking time */
std::vector<std::string> shareText(const std::string &text,
                                   const std::vector<long> &durations);
/** @brief align.split_by_time; nullopt when the timing cannot be trusted */
std::optional<std::vector<std::string>>
splitByTime(const std::string &text, const std::vector<SpeechModels::Word> &w,
            const std::vector<std::pair<long, long>> &turns);
/** @brief audio.plausible: at most 25 characters a second, plus 10 */
bool plausible(const std::string &text, long ms);
/** @brief audio.normalize_text: lower-case letters and digits only */
std::u32string normalizeText(const std::string &text);

class RoomPipeline {
public:
  using Emit = std::function<void(const RoomLine &)>;
  using Moved = std::function<void(const std::map<std::string, std::string> &)>;

  RoomPipeline(SpeechModels &models, RoomDiarizer &registry,
               std::unique_ptr<VoiceGate> gate, Emit emit, Moved moved = {});

  /**
   * @brief PCM16 samples in order; recognition runs inside (the caller keeps
   *        it off the audio thread)
   * @param at_ms where the chunk starts on the meeting clock, < 0: right after
   *        the last one (see align)
   */
  void feed(const int16_t *pcm, size_t n, double at_ms = -1,
            bool exact = false);
  /** @brief listening ends: what is still open is recognized and emitted */
  void flush();
  /** @brief audio missing for longer than GAP_MS is meeting time */
  void align(double at_ms, bool exact = false);

  long cursorMs() const { return static_cast<long>(cursor_ * 1000 / RATE); }
  size_t segments() const { return segments_; }
  size_t looped() const { return looped_; }

  static constexpr long RATE = 16000;

private:
  using Frame = std::vector<int16_t>;
  struct Segment {
    std::vector<int16_t> data;
    long long start, end, lead;
    size_t group;
  };

  void step(const Frame &frame);
  size_t quietCut(const std::vector<Frame> &frames) const;
  void cap();
  void quietWindow(bool final = false);
  void finish();
  std::optional<Segment> take();
  void complete(const Segment &seg);
  std::vector<LocalTurn> confirmed(std::vector<LocalTurn> local,
                                   const std::vector<float> &speech);

  SpeechModels &models_;
  RoomDiarizer &registry_;
  std::unique_ptr<VoiceGate> vad_;
  EnergyGate floor_;
  Emit emit_;
  Moved moved_;

  size_t silence_frames_, max_frames_, soft_frames_, preroll_, bridge_;
  long long cursor_ = 0, start_ = 0, last_voice_ = 0, onset_ = 0;
  std::vector<int16_t> pending_;
  std::vector<Frame> active_, before_;
  size_t silent_ = 0, counter_ = 0, segments_ = 0, looped_ = 0;
  bool closed_ = false, heard_sound_ = false;
  std::optional<long long> first_sound_;
};

} // namespace discretion

#endif // __DISCRETION_ROOM_PIPELINE_H__
