// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   room_diarizer.h
 * @date   22 October 2026
 * @brief  Speaker separation over the whole meeting (ALICE's Diarizer)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * A port of ambient/diarize.py Diarizer (ambient-agent-alice, non-legacy
 * mode) with its constants: every turn keeps its voice embedding and the whole
 * meeting is clustered again after each segment (average-linkage
 * agglomerative clustering on cosine similarity, cut at `same`), then
 * - a cluster with less than 4 s of speech joins the nearest voice unless it
 *   is unlike every voice (FAR); the least distinct pair merges while a pair is
 *   not clearly distinct (DISTINCT) or alike as whole voices (ONE_VOICE);
 * - a label once shown is never taken back; once two voices have enough
 *   speech, the lines clearly theirs are kept and kept voices never merge;
 * - turns one segment gave to different local speakers never share a
 *   cluster; enrolled and pinned voices name their clusters, others get the
 *   smallest free '화자 N'.
 */

#ifndef __DISCRETION_ROOM_DIARIZER_H__
#define __DISCRETION_ROOM_DIARIZER_H__

#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace discretion {

/** @brief ALICE's constants (ambient/diarize.py), tuned for ERes2NetV2 */
struct DiarizerOptions {
  double same = .66;      // cosine at which two clusters are one voice
  double threshold = .66; // per-person match (enrolled voices)
  double margin = .08;    // over the next person
  double min_cluster_ms = 4000, far = .4, distinct = .12, one_voice = .75;
  double keep_score = .6, keep_margin = .1;
  int max_voices = 12;
  // labels an earlier run of this meeting showed: a new voice never gets one
  // (the app restarted, the server still names lines by them)
  std::set<std::string> reserved;
};

/** @brief why a line has its label (ALICE's line info) */
struct TurnInfo {
  std::optional<double> speaker_score, speaker_margin;
  std::string basis; // pinned, claimed, enrolled, cluster, short
  std::optional<int> cluster_size;
  std::string absorbed; // short, not_distinct, cap ("" when not absorbed)
  int voices = 0;
};

class RoomDiarizer {
public:
  explicit RoomDiarizer(const DiarizerOptions &opt = {});

  struct NewTurn {
    std::string id;
    std::vector<float> vector; // empty: no embedding of its own
    double ms = 0;
    std::string group; // the segment it came from
    std::string local; // the local speaker the segmentation gave it
  };
  struct Result {
    std::map<std::string, std::string> labels; // "" = no speaker
    std::map<std::string, std::string> moved;  // earlier lines relabelled
    std::map<std::string, TurnInfo> info;
  };

  /** @brief the turns of one segment */
  Result add(const std::vector<NewTurn> &turns);
  /** @brief a registered person's voice (also a pinned example turn) */
  void enroll(const std::string &person, const std::vector<float> &vector);
  /** @brief "이건 나예요" on a provisional label */
  void mapSpeaker(const std::string &label, const std::string &person);
  /** @brief a corrected line: its person, or nobody (leaves clustering) */
  void pin(const std::string &turn, const std::optional<std::string> &person);
  /**
   * @brief the person's voice over this meeting (the duration-weighted mean
   *        of their lines), to be remembered; their enrolled voice when the
   *        meeting heard less than least_ms of them; nullopt when neither
   */
  std::optional<std::vector<float>> voiceOf(const std::string &person,
                                            double least_ms = 5000) const;
  /** @brief provisional labels and how many lines each has */
  std::map<std::string, int> unknown() const;
  double same() const { return opt_.same; }

private:
  enum class Pin { None, Person, Nobody };
  struct Turn {
    std::vector<double> v; // unit vector, empty when none
    double ms = 0;
    std::string group, local;
    Pin pin = Pin::None;
    std::string person; // when pin == Person
    std::string label;  // "" = none yet
    bool example = false;
    std::string kept;
  };
  using Cluster = std::vector<std::string>;

  static std::vector<double> unit(const std::vector<double> &v);
  void recluster();
  void identify();
  std::vector<Cluster> cluster(const std::vector<std::string> &ids);
  std::vector<Cluster> cut(const std::vector<std::string> &ids);
  std::vector<Cluster> split(const std::vector<Cluster> &clusters);
  std::vector<Cluster> settle(std::vector<Cluster> clusters);
  void name(const std::vector<Cluster> &clusters,
            std::map<size_t, std::string> &names,
            std::map<size_t, std::string> &source);
  void keep(const std::vector<Cluster> &clusters,
            std::map<size_t, std::string> &names);
  std::string key(const std::string &id) const;
  double weight(const std::string &id) const;
  bool pinned(const Turn &t) const { return t.pin == Pin::Person; }

  DiarizerOptions opt_;
  mutable std::recursive_mutex m_;
  std::map<std::string, Turn> turns_;
  std::vector<std::string> order_;
  std::map<std::string, std::vector<double>> profiles_;
  std::vector<std::string> profile_order_;
  std::map<std::string, std::string> aliases_;
  std::map<std::string, TurnInfo> info_;
  std::map<std::string, std::string> absorbed_;
};

} // namespace discretion

#endif // __DISCRETION_ROOM_DIARIZER_H__
