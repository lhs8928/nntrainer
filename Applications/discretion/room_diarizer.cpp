// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   room_diarizer.cpp
 * @date   22 October 2026
 * @brief  Speaker separation over the whole meeting (ALICE's Diarizer)
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Function by function as ambient/diarize.py; comments name the Python method.
 */

#include "room_diarizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>

namespace discretion {

namespace {

using Vec = std::vector<double>;

double dot(const Vec &a, const Vec &b) {
  double s = 0.0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i)
    s += a[i] * b[i];
  return s;
}

double norm(const Vec &a) { return std::sqrt(dot(a, a)); }

void addScaled(Vec &acc, const Vec &v, double w) {
  if (acc.empty())
    acc.assign(v.size(), 0.0);
  for (size_t i = 0; i < v.size(); ++i)
    acc[i] += v[i] * w;
}

Vec normalized(Vec v) {
  const double n = std::max(norm(v), 1e-9);
  for (double &x : v)
    x /= n;
  return v;
}

double round3(double x) { return std::round(x * 1000.0) / 1000.0; }

bool provisional(const std::string &label) {
  return label.rfind("화자 ", 0) == 0;
}

} // namespace

RoomDiarizer::RoomDiarizer(const DiarizerOptions &opt) : opt_(opt) {}

Vec RoomDiarizer::unit(const Vec &v) {
  const double n = norm(v);
  if (v.empty() || !std::isfinite(n) || n < 1e-8)
    throw std::invalid_argument("유효한 목소리 특징이 없습니다.");
  Vec out(v);
  for (double &x : out)
    x /= n;
  return out;
}

double RoomDiarizer::weight(const std::string &id) const {
  const Turn &t = turns_.at(id);
  return t.example ? 3000.0 : std::max(1000.0, t.ms);
}

// _key: the person who pinned it, else the kept voice (an identified kept
// provisional voice is that person)
std::string RoomDiarizer::key(const std::string &id) const {
  const Turn &t = turns_.at(id);
  if (pinned(t))
    return t.person;
  if (!t.kept.empty()) {
    auto a = aliases_.find(t.kept);
    return a != aliases_.end() ? a->second : t.kept;
  }
  return "";
}

void RoomDiarizer::enroll(const std::string &person,
                          const std::vector<float> &vector) {
  std::lock_guard<std::recursive_mutex> lk(m_);
  const Vec v = unit(Vec(vector.begin(), vector.end()));
  if (!profiles_.count(person))
    profile_order_.push_back(person);
  profiles_[person] = v;
  const std::string id = "enroll:" + person;
  Turn t;
  t.v = v;
  t.group = id;
  t.local = "0";
  t.pin = Pin::Person;
  t.person = person;
  t.label = person;
  t.example = true;
  turns_[id] = t;
  if (std::find(order_.begin(), order_.end(), id) == order_.end())
    order_.push_back(id);
}

void RoomDiarizer::mapSpeaker(const std::string &label,
                              const std::string &person) {
  std::lock_guard<std::recursive_mutex> lk(m_);
  std::vector<std::string> members;
  for (const auto &id : order_)
    if (turns_[id].label == label && !turns_[id].example)
      members.push_back(id);
  if (members.empty())
    throw std::invalid_argument("알 수 없는 화자 라벨입니다.");
  Vec sum;
  size_t n = 0;
  for (const auto &id : members)
    if (!turns_[id].v.empty()) {
      addScaled(sum, turns_[id].v, 1.0);
      ++n;
    }
  Vec centroid;
  if (n)
    centroid = unit(sum);
  for (const auto &id : members) {
    Turn &t = turns_[id];
    if (centroid.empty() || t.v.empty() || dot(t.v, centroid) >= opt_.same) {
      t.pin = Pin::Person;
      t.person = person;
    }
  }
  aliases_[label] = person;
  if (!profiles_.count(person) && !centroid.empty()) {
    profiles_[person] = centroid;
    profile_order_.push_back(person);
  }
}

void RoomDiarizer::pin(const std::string &turn,
                       const std::optional<std::string> &person) {
  std::lock_guard<std::recursive_mutex> lk(m_);
  auto it = turns_.find(turn);
  if (it == turns_.end())
    return;
  if (person) {
    it->second.pin = Pin::Person;
    it->second.person = *person;
  } else {
    it->second.pin = Pin::Nobody;
  }
}

std::map<std::string, int> RoomDiarizer::unknown() const {
  std::lock_guard<std::recursive_mutex> lk(m_);
  std::map<std::string, int> out;
  for (const auto &id : order_) {
    const Turn &t = turns_.at(id);
    if (provisional(t.label) && !t.example)
      ++out[t.label];
  }
  return out;
}

RoomDiarizer::Result RoomDiarizer::add(const std::vector<NewTurn> &turns) {
  std::lock_guard<std::recursive_mutex> lk(m_);
  for (const auto &n : turns) {
    Turn t;
    if (!n.vector.empty())
      t.v = unit(Vec(n.vector.begin(), n.vector.end()));
    t.ms = n.ms;
    t.group = n.group;
    t.local = n.local;
    turns_[n.id] = t;
    order_.push_back(n.id);
  }
  std::map<std::string, std::string> before;
  for (const auto &id : order_)
    before[id] = turns_[id].label;
  recluster();
  Result r;
  for (const auto &id : order_) {
    const Turn &t = turns_[id];
    if (!before[id].empty() && before[id] != t.label && !t.example)
      r.moved[id] = t.label;
  }
  for (const auto &n : turns) {
    r.labels[n.id] = turns_[n.id].label;
    auto it = info_.find(n.id);
    if (it != info_.end())
      r.info[n.id] = it->second;
  }
  return r;
}

// _recluster
void RoomDiarizer::recluster() {
  identify();
  std::vector<std::string> ids;
  for (const auto &id : order_)
    if (!turns_[id].v.empty() && turns_[id].pin != Pin::Nobody)
      ids.push_back(id);
  auto clusters = split(cluster(ids));
  std::map<size_t, std::string> names, source;
  name(clusters, names, source);
  if (!clusters.empty()) {
    // each turn's similarity to its own cluster's centroid and to the closest
    // other one (unweighted sums, as the Python)
    std::vector<Vec> cents;
    for (const auto &c : clusters) {
      Vec s;
      for (const auto &id : c)
        addScaled(s, turns_[id].v, 1.0);
      cents.push_back(normalized(s));
    }
    for (size_t k = 0; k < clusters.size(); ++k) {
      for (const auto &id : clusters[k]) {
        Turn &t = turns_[id];
        const double own = dot(t.v, cents[k]);
        std::optional<double> other;
        for (size_t j = 0; j < clusters.size(); ++j)
          if (j != k)
            other = std::max(other.value_or(-1e9), dot(t.v, cents[j]));
        // a label once shown stays
        if (pinned(t))
          t.label = t.person;
        else if (t.label.empty())
          t.label = names[k];
        TurnInfo info;
        info.speaker_score = round3(own);
        if (other)
          info.speaker_margin = round3(own - *other);
        info.basis =
          pinned(t) ? "pinned" : (source.count(k) ? source[k] : "cluster");
        info.cluster_size = static_cast<int>(clusters[k].size());
        auto a = absorbed_.find(id);
        info.absorbed = a != absorbed_.end() ? a->second : "";
        info.voices = static_cast<int>(clusters.size());
        info_[id] = info;
      }
    }
  }
  // turns without an embedding: their segment-mate with the same local
  // speaker, else the previous labelled turn
  for (size_t k = 0; k < order_.size(); ++k) {
    const std::string &id = order_[k];
    Turn &t = turns_[id];
    if (!t.v.empty() && t.pin != Pin::Nobody)
      continue;
    if (t.pin == Pin::Nobody) {
      t.label.clear();
      continue;
    }
    if (pinned(t)) {
      t.label = t.person;
      continue;
    }
    if (!t.label.empty())
      continue; // shown already: it stays
    std::string mate;
    for (const auto &j : order_) {
      const Turn &u = turns_[j];
      if (j != id && u.group == t.group && u.local == t.local && !u.v.empty()) {
        mate = u.label;
        break;
      }
    }
    if (mate.empty()) {
      for (size_t j = k; j-- > 0;) {
        if (!turns_[order_[j]].label.empty()) {
          mate = turns_[order_[j]].label;
          break;
        }
      }
    }
    t.label = mate;
    TurnInfo info;
    info.basis = "short";
    info.voices = static_cast<int>(clusters.size());
    info_[id] = info;
  }
  keep(clusters, names);
}

// _keep: once two voices have enough speech, each keeps the lines clearly its
// own for the rest of the meeting
void RoomDiarizer::keep(const std::vector<Cluster> &clusters,
                        std::map<size_t, std::string> &names) {
  auto voice = [&](const Cluster &c) {
    double ms = 0.0;
    bool pin = false;
    for (const auto &id : c) {
      const Turn &t = turns_[id];
      if (!t.example)
        ms += t.ms;
      pin = pin || pinned(t);
    }
    return ms >= opt_.min_cluster_ms || pin;
  };
  size_t voices = 0;
  for (const auto &c : clusters)
    voices += voice(c);
  if (voices < 2)
    return;
  std::vector<Vec> sums, units;
  for (const auto &c : clusters) {
    Vec s;
    for (const auto &id : c)
      addScaled(s, turns_[id].v, weight(id));
    sums.push_back(s);
    units.push_back(normalized(s));
  }
  for (size_t k = 0; k < clusters.size(); ++k) {
    const Cluster &c = clusters[k];
    if (c.size() < 2 || !voice(c))
      continue;
    for (const auto &id : c) {
      Turn &t = turns_[id];
      if (!t.kept.empty() || pinned(t) || t.example || absorbed_.count(id) ||
          t.label != names[k])
        continue;
      Vec rest = sums[k];
      addScaled(rest, t.v, -weight(id));
      const double own = dot(t.v, rest) / std::max(norm(rest), 1e-9);
      double other = -1.0;
      for (size_t j = 0; j < clusters.size(); ++j)
        if (j != k)
          other = std::max(other, dot(t.v, units[j]));
      if (own >= opt_.keep_score && own - other >= opt_.keep_margin)
        t.kept = names[k];
    }
  }
}

// _identify: a kept provisional voice that matches an enrolled person is them
void RoomDiarizer::identify() {
  if (profiles_.empty())
    return;
  std::vector<std::string> labels;
  std::map<std::string, Vec> sums;
  for (const auto &id : order_) {
    const Turn &t = turns_[id];
    if (!t.kept.empty() && !t.v.empty()) {
      if (!sums.count(t.kept))
        labels.push_back(t.kept);
      addScaled(sums[t.kept], t.v, std::max(1000.0, t.ms));
    }
  }
  std::set<std::string> own;
  for (const auto &l : labels) {
    auto a = aliases_.find(l);
    own.insert(a != aliases_.end() ? a->second : l);
  }
  for (const auto &label : labels) {
    if (!provisional(label) || aliases_.count(label))
      continue;
    const Vec c = unit(sums[label]);
    std::vector<std::pair<double, std::string>> scores;
    for (const auto &p : profile_order_)
      scores.emplace_back(dot(c, profiles_[p]), p);
    std::sort(scores.rbegin(), scores.rend());
    if (scores[0].first >= opt_.threshold &&
        (scores.size() == 1 ||
         scores[0].first - scores[1].first >= opt_.margin) &&
        !own.count(scores[0].second)) {
      aliases_[label] = scores[0].second;
      own.insert(scores[0].second);
    }
  }
}

// _cluster
std::vector<RoomDiarizer::Cluster>
RoomDiarizer::cluster(const std::vector<std::string> &ids) {
  absorbed_.clear();
  return settle(split(cut(ids)));
}

// _cut: average-linkage clustering cut at `same`; the turns one segment gave
// to one local speaker enter as one unit, and two units of one segment with
// different local speakers never stay in one cluster
std::vector<RoomDiarizer::Cluster>
RoomDiarizer::cut(const std::vector<std::string> &ids) {
  if (ids.empty())
    return {};
  std::vector<std::pair<std::string, std::string>> keys;
  std::map<std::pair<std::string, std::string>, std::vector<std::string>> units;
  for (const auto &id : ids) {
    const auto k = std::make_pair(turns_[id].group, turns_[id].local);
    if (!units.count(k))
      keys.push_back(k);
    units[k].push_back(id);
  }
  const size_t n = keys.size();
  std::vector<Vec> U(n);
  for (size_t i = 0; i < n; ++i) {
    Vec s;
    for (const auto &id : units[keys[i]])
      addScaled(s, turns_[id].v, 1.0);
    U[i] = normalized(s);
  }
  // average linkage: merge the closest pair while its distance <= 1 - same
  std::vector<int> assign(n);
  std::iota(assign.begin(), assign.end(), 0);
  if (n > 1) {
    std::vector<std::vector<double>> D(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i)
      for (size_t j = i + 1; j < n; ++j)
        D[i][j] = D[j][i] = std::clamp(1.0 - dot(U[i], U[j]), 0.0, 2.0);
    std::vector<int> size(n, 1);
    std::vector<bool> alive(n, true);
    const double t = 1.0 - opt_.same;
    while (true) {
      double best = std::numeric_limits<double>::infinity();
      size_t bi = 0, bj = 0;
      for (size_t i = 0; i < n; ++i) {
        if (!alive[i])
          continue;
        for (size_t j = i + 1; j < n; ++j)
          if (alive[j] && D[i][j] < best) {
            best = D[i][j];
            bi = i;
            bj = j;
          }
      }
      if (!std::isfinite(best) || best > t)
        break;
      for (size_t k = 0; k < n; ++k) {
        if (!alive[k] || k == bi || k == bj)
          continue;
        D[bi][k] = D[k][bi] =
          (D[bi][k] * size[bi] + D[bj][k] * size[bj]) / (size[bi] + size[bj]);
      }
      size[bi] += size[bj];
      alive[bj] = false;
      for (auto &a : assign)
        if (a == static_cast<int>(bj))
          a = static_cast<int>(bi);
    }
  }
  // clusters in the order of their first unit
  std::vector<std::pair<int, std::vector<size_t>>> clusters;
  for (size_t u = 0; u < n; ++u) {
    auto it = std::find_if(clusters.begin(), clusters.end(),
                           [&](const auto &c) { return c.first == assign[u]; });
    if (it == clusters.end())
      clusters.emplace_back(assign[u], std::vector<size_t>(1, u));
    else
      it->second.push_back(u);
  }
  int next_key = 0;
  for (const auto &c : clusters)
    next_key = std::max(next_key, c.first + 1);
  auto wt = [&](size_t u) { return units[keys[u]].size(); };
  const size_t count = clusters.size();
  for (size_t ci = 0; ci < count; ++ci) {
    std::vector<std::pair<std::string, std::vector<size_t>>> by_group;
    for (size_t u : clusters[ci].second) {
      auto it = std::find_if(by_group.begin(), by_group.end(),
                             [&](auto &g) { return g.first == keys[u].first; });
      if (it == by_group.end())
        by_group.push_back({keys[u].first, {u}});
      else
        it->second.push_back(u);
    }
    for (const auto &[group, ns] : by_group) {
      if (ns.size() < 2)
        continue;
      size_t keepu = ns[0];
      for (size_t u : ns)
        if (wt(u) > wt(keepu))
          keepu = u;
      for (size_t u : ns) {
        if (u == keepu)
          continue;
        auto &mine = clusters[ci].second;
        mine.erase(std::find(mine.begin(), mine.end(), u));
        double best = -std::numeric_limits<double>::infinity();
        int best_key = -1;
        for (const auto &o : clusters) {
          if (o.first == clusters[ci].first || o.second.empty())
            continue;
          bool same_group = false;
          for (size_t m : o.second)
            same_group = same_group || keys[m].first == group;
          if (same_group)
            continue;
          double s = 0.0;
          for (size_t m : o.second)
            s += dot(U[m], U[u]);
          s /= o.second.size();
          if (s > best || (s == best && o.first > best_key)) {
            best = s;
            best_key = o.first;
          }
        }
        if (best_key >= 0 && best >= opt_.same) {
          for (auto &o : clusters)
            if (o.first == best_key)
              o.second.push_back(u);
        } else {
          clusters.emplace_back(next_key++, std::vector<size_t>(1, u));
        }
      }
    }
  }
  std::vector<Cluster> out;
  for (const auto &c : clusters) {
    if (c.second.empty())
      continue;
    Cluster ids_of;
    for (size_t u : c.second)
      for (const auto &id : units[keys[u]])
        ids_of.push_back(id);
    out.push_back(ids_of);
  }
  return out;
}

// _split: a cluster holding turns that must stay with two voices is divided
std::vector<RoomDiarizer::Cluster>
RoomDiarizer::split(const std::vector<Cluster> &clusters) {
  std::vector<Cluster> out;
  for (const auto &c : clusters) {
    std::set<std::string> ks;
    for (const auto &id : c)
      if (!key(id).empty())
        ks.insert(key(id));
    if (ks.size() <= 1) {
      out.push_back(c);
      continue;
    }
    std::map<std::string, Vec> cents;
    for (const auto &p : ks) {
      Vec s;
      size_t n = 0;
      for (const auto &id : c)
        if (key(id) == p) {
          addScaled(s, turns_[id].v, 1.0);
          ++n;
        }
      for (double &x : s)
        x /= n;
      cents[p] = unit(s);
    }
    std::map<std::string, Cluster> parts;
    for (const auto &id : c) {
      std::string p = key(id);
      if (p.empty()) {
        double best = -1e9;
        for (const auto &q : ks)
          if (dot(turns_[id].v, cents[q]) > best) {
            best = dot(turns_[id].v, cents[q]);
            p = q;
          }
      }
      parts[p].push_back(id);
    }
    for (const auto &[p, part] : parts)
      if (!part.empty())
        out.push_back(part);
  }
  return out;
}

// _settle: the prior that a meeting has few speakers
std::vector<RoomDiarizer::Cluster>
RoomDiarizer::settle(std::vector<Cluster> clusters) {
  if (clusters.empty())
    return clusters;
  auto total = [&](const Cluster &c) {
    double s = 0.0;
    for (const auto &id : c)
      s += weight(id);
    return s;
  };
  auto centroid = [&](const Cluster &c) {
    Vec s;
    for (const auto &id : c)
      addScaled(s, turns_[id].v, weight(id));
    return normalized(s);
  };
  auto margin = [&](const Cluster &a, const Cluster &b) {
    const Vec cb = centroid(b);
    if (a.size() < 2)
      return opt_.far + opt_.distinct - dot(turns_[a[0]].v, cb);
    Vec sa;
    for (const auto &id : a)
      addScaled(sa, turns_[id].v, weight(id));
    double m = 0.0;
    for (const auto &id : a) {
      Vec rest = sa;
      addScaled(rest, turns_[id].v, -weight(id));
      rest = normalized(rest);
      m += weight(id) * (dot(turns_[id].v, rest) - dot(turns_[id].v, cb));
    }
    return m / total(a);
  };
  auto apart = [&](const Cluster &a, const Cluster &b) {
    return dot(centroid(a), centroid(b)) >= opt_.one_voice
             ? -1.0
             : std::min(margin(a, b), margin(b, a));
  };
  auto joinable = [&](const Cluster &a, const Cluster &b) {
    std::set<std::string> ka, kb;
    for (const auto &id : a)
      if (!key(id).empty())
        ka.insert(key(id));
    for (const auto &id : b)
      if (!key(id).empty())
        kb.insert(key(id));
    return !(!ka.empty() && !kb.empty() && ka != kb);
  };
  const size_t cap = std::max<size_t>(opt_.max_voices, profiles_.size());
  for (int round = 0; round < 12; ++round) {
    bool changed = false;
    // too little speech to be a voice: the nearest voice takes it, unless
    // unlike all
    std::vector<size_t> order(clusters.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) {
      return total(clusters[x]) < total(clusters[y]);
    });
    // indices into `work` keep the Python's object identity: a cluster that
    // took another keeps growing and is visited with its new total
    std::vector<Cluster> work(clusters);
    std::vector<bool> gone(work.size(), false);
    for (const size_t ci : order) {
      size_t live = 0;
      for (bool g : gone)
        live += !g;
      if (live < 2 || total(work[ci]) >= opt_.min_cluster_ms)
        break;
      const Vec cc = centroid(work[ci]);
      double best = -1e9;
      int near = -1;
      for (size_t o = 0; o < work.size(); ++o) {
        if (o == ci || gone[o] || !joinable(work[ci], work[o]))
          continue;
        const double s = dot(cc, centroid(work[o]));
        if (s > best) {
          best = s;
          near = static_cast<int>(o);
        }
      }
      if (near < 0 || best < opt_.far)
        continue;
      for (const auto &id : work[ci])
        absorbed_.emplace(id, "short");
      work[near].insert(work[near].end(), work[ci].begin(), work[ci].end());
      gone[ci] = true;
      changed = true;
    }
    clusters.clear();
    for (size_t i = 0; i < work.size(); ++i)
      if (!gone[i])
        clusters.push_back(work[i]);
    // not clearly two voices, or more voices than people: the least distinct
    // pair merges
    while (clusters.size() > 1) {
      double best = std::numeric_limits<double>::infinity();
      size_t bx = 0, by = 0;
      bool any = false;
      for (size_t x = 0; x < clusters.size(); ++x)
        for (size_t y = x + 1; y < clusters.size(); ++y) {
          if (!joinable(clusters[x], clusters[y]))
            continue;
          const double s = apart(clusters[x], clusters[y]);
          if (!any || s < best) {
            best = s;
            bx = x;
            by = y;
            any = true;
          }
        }
      if (!any)
        break;
      if (best >= opt_.distinct && clusters.size() <= cap)
        break;
      const std::string why = best == -1.0           ? "one_voice"
                              : best < opt_.distinct ? "not_distinct"
                                                     : "cap";
      if (why != "one_voice") {
        const Cluster &smaller = total(clusters[by]) < total(clusters[bx])
                                   ? clusters[by]
                                   : clusters[bx];
        for (const auto &id : smaller)
          absorbed_.emplace(id, why);
      }
      clusters[bx].insert(clusters[bx].end(), clusters[by].begin(),
                          clusters[by].end());
      clusters.erase(clusters.begin() + by);
      changed = true;
    }
    if (!changed)
      break;
  }
  return clusters;
}

// _name: pins, kept voices and enrolled voices name clusters; provisional
// labels stay with the cluster holding most of their earlier turns; the rest
// get the smallest number no line shows
void RoomDiarizer::name(const std::vector<Cluster> &clusters,
                        std::map<size_t, std::string> &names,
                        std::map<size_t, std::string> &source) {
  for (size_t k = 0; k < clusters.size(); ++k) {
    const Cluster &c = clusters[k];
    std::set<std::string> pins;
    for (const auto &id : c)
      if (pinned(turns_[id]))
        pins.insert(turns_[id].person);
    if (!pins.empty()) {
      const std::string nm = *pins.begin();
      names[k] = nm;
      bool claimed = false;
      for (const auto &id : c)
        claimed = claimed || (pinned(turns_[id]) && turns_[id].person == nm &&
                              !turns_[id].example);
      source[k] = claimed ? "claimed" : "enrolled";
      continue;
    }
    std::set<std::string> kept;
    for (const auto &id : c)
      if (!key(id).empty())
        kept.insert(key(id));
    if (!kept.empty()) {
      names[k] = *kept.begin();
      if (profiles_.count(names[k]))
        source[k] = "enrolled";
      continue;
    }
    Vec s;
    for (const auto &id : c)
      addScaled(s, turns_[id].v, 1.0);
    for (double &x : s)
      x /= c.size();
    const Vec cent = unit(s);
    std::vector<std::pair<double, std::string>> scores;
    for (const auto &p : profile_order_)
      scores.emplace_back(dot(cent, profiles_[p]), p);
    std::sort(scores.rbegin(), scores.rend());
    if (!scores.empty() && scores[0].first >= opt_.threshold &&
        (scores.size() == 1 ||
         scores[0].first - scores[1].first >= opt_.margin)) {
      names[k] = scores[0].second;
      source[k] = "enrolled";
    }
  }
  struct Vote {
    int count;
    size_t k;
    std::string old;
  };
  std::vector<Vote> votes;
  for (size_t k = 0; k < clusters.size(); ++k) {
    if (names.count(k))
      continue;
    std::set<std::string> olds;
    for (const auto &id : clusters[k])
      if (provisional(turns_[id].label))
        olds.insert(turns_[id].label);
    for (const auto &old : olds) {
      int count = 0;
      for (const auto &id : clusters[k])
        count += turns_[id].label == old;
      votes.push_back({count, k, old});
    }
  }
  std::sort(votes.begin(), votes.end(), [](const Vote &a, const Vote &b) {
    if (a.count != b.count)
      return a.count > b.count;
    if (a.k != b.k)
      return a.k > b.k;
    return a.old > b.old;
  });
  std::set<std::string> taken;
  for (const auto &[k, nm] : names)
    taken.insert(nm);
  for (const auto &v : votes) {
    if (names.count(v.k) || taken.count(v.old))
      continue;
    names[v.k] = v.old;
    taken.insert(v.old);
  }
  for (size_t k = 0; k < clusters.size(); ++k) {
    if (names.count(k))
      continue;
    std::set<std::string> used;
    for (const auto &[kk, nm] : names)
      used.insert(nm);
    for (const auto &[label, person] : aliases_)
      used.insert(label);
    for (const auto &[id, t] : turns_)
      if (!t.label.empty())
        used.insert(t.label);
    int n = 1;
    while (used.count("화자 " + std::to_string(n)))
      ++n;
    names[k] = "화자 " + std::to_string(n);
  }
}

} // namespace discretion
