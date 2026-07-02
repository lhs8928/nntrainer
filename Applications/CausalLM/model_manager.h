// SPDX-License-Identifier: Apache-2.0
/**
 * @file   model_manager.h
 * @brief  Model lifecycle + execution orchestration for the CausalLM app.
 *         A ModelService owns one model family (a weight-owning base + a
 *         request-driven worker pool that creates a fresh shared instance per
 *         request and discards it). The ModelManager owns the sharing policy,
 *         a process-global creation mutex, and routes interactive commands.
 */
#ifndef __CAUSALLM_MODEL_MANAGER_H__
#define __CAUSALLM_MODEL_MANAGER_H__

#include <string>

// Process-global string representing the prompt currently displayed on screen.
// Used to clear and restore the user's input line when async background logs are printed.
extern std::string g_current_prompt;

#include <atomic>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "model_runtime.h"

/** Summary of one serving session. */
struct ServingStats {
  int pool_size = 0;
  int served = 0;
  int max_concurrency = 0;
  double avg_ms = 0.0;
  double p95_ms = 0.0;
  size_t peak_rss_kb = 0;
};

/**
 * @brief One model family: a weight-owning base (does not serve) plus a worker
 *        pool. Each request creates a fresh instance (sharing the base's
 *        weights when enabled), runs once, and discards it — so no generation
 *        state is reused across requests.
 */
class ModelService {
public:
  ModelService(std::string model_path, int pool_size, bool share); // parse config/arch
  ~ModelService();

  void loadBase(std::mutex &global_create_mtx);   // create base (loads weights)
  void start(std::mutex &global_create_mtx);
  void submit(const std::string &prompt);          // enqueue a request
  ServingStats stop();                             // stop workers, return stats

  const std::string &path() const { return model_path_; }
  const std::string &arch() const { return architecture_; }
  int poolSize() const { return pool_size_; }
  bool share() const { return share_; }
  bool isStarted() const { return !workers_.empty(); }
  int getServedCount() const { return served_.load(); }

private:
  void workerLoop(int worker_id, bool share);

  Config config_;
  std::string architecture_;
  std::string model_path_;
  std::shared_ptr<causallm::Transformer> base_; // ref / weight owner

  std::queue<std::string> queue_;
  std::mutex qm_;
  std::condition_variable cv_;
  bool stop_ = false;
  std::vector<std::thread> workers_;
  std::mutex io_mtx_;                    // serialize console output
  std::mutex *create_mtx_ = nullptr;     // process-global creation lock (borrowed)

  int pool_size_ = 0;
  bool share_ = true;
  std::atomic<int> active_{0};
  std::atomic<int> max_active_{0};
  std::atomic<int> served_{0};
  std::vector<double> latencies_ms_;
  std::mutex stats_m_;
};

/**
 * @brief Owns multiple ModelServices, the single sharing policy, and a
 *        process-global creation mutex. Routes interactive commands.
 */
class ModelManager {
public:
  explicit ModelManager(bool sharing = true) : sharing_(sharing) {}

  int addService(const std::string &model_path, int pool_size = 2); // create service (+base if sharing)
  void serve(int id);              // interactive serving loop
  void runSingle(int id, const std::string &prompt); // synchronous single-shot prompt run
  void list();
  void remove(int id);
  bool has(int id) const;

private:
  std::map<int, std::unique_ptr<ModelService>> services_;
  int next_id_ = 0;
  bool sharing_ = true;
  std::mutex global_create_mtx_; // serialize all model creation process-wide
};

#endif // __CAUSALLM_MODEL_MANAGER_H__
