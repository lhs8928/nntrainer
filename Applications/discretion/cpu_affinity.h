// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   cpu_affinity.h
 * @date   12 October 2026
 * @brief  Pin the calling thread (and the threads it creates) to some CPUs
 * @see    https://github.com/nntrainer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * A thread that is created by a pinned thread inherits its CPU set. The ASR
 * thread pool (nntrainer ThreadManager) and the diarization OpenMP team are
 * both created lazily by the first thread that uses them, so each lane pins its
 * own thread before it touches them and the lanes end up on separate cores.
 * Linux only; on other systems the calls do nothing.
 */

#ifndef __DISCRETION_CPU_AFFINITY_H__
#define __DISCRETION_CPU_AFFINITY_H__

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

namespace discretion {

/**
 * @brief Parse a CPU list such as "0,1", "2-3" or "0-1,3" into CPU numbers.
 * @return empty vector for an empty or invalid list
 */
inline std::vector<int> parseCpuList(const std::string &spec) {
  std::vector<int> cpus;
  std::stringstream ss(spec);
  std::string part;
  while (std::getline(ss, part, ',')) {
    if (part.empty())
      continue;
    const auto dash = part.find('-');
    try {
      if (dash == std::string::npos) {
        cpus.push_back(std::stoi(part));
      } else {
        const int a = std::stoi(part.substr(0, dash));
        const int b = std::stoi(part.substr(dash + 1));
        if (b < a)
          return {};
        for (int c = a; c <= b; ++c)
          cpus.push_back(c);
      }
    } catch (const std::exception &) {
      return {};
    }
  }
  return cpus;
}

/**
 * @brief Restrict the calling thread to `cpus` for the lifetime of the object
 *        and restore the previous CPU set afterwards. Empty list = no change.
 */
class ScopedAffinity {
public:
  explicit ScopedAffinity(const std::vector<int> &cpus) {
#if defined(__linux__)
    if (cpus.empty())
      return;
    if (sched_getaffinity(0, sizeof(old_), &old_) != 0)
      return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus)
      if (c >= 0 && c < CPU_SETSIZE)
        CPU_SET(c, &set);
    active_ = (sched_setaffinity(0, sizeof(set), &set) == 0);
#else
    (void)cpus;
#endif
  }

  ~ScopedAffinity() {
#if defined(__linux__)
    if (active_)
      sched_setaffinity(0, sizeof(old_), &old_);
#endif
  }

  ScopedAffinity(const ScopedAffinity &) = delete;
  ScopedAffinity &operator=(const ScopedAffinity &) = delete;

  /** @brief true if the CPU set was really changed */
  bool active() const { return active_; }

private:
#if defined(__linux__)
  cpu_set_t old_;
#endif
  bool active_ = false;
};

} // namespace discretion

#endif // __DISCRETION_CPU_AFFINITY_H__
