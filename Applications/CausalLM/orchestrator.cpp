// SPDX-License-Identifier: Apache-2.0
/**
 * @file   orchestrator.cpp
 * @brief  Implementation of ModelService and Orchestrator (see orchestrator.h).
 */
#include "orchestrator.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>

std::string g_current_prompt = "";

/* ============================ ModelService ============================ */

ModelService::ModelService(std::string model_path, int pool_size, bool share)
  : model_path_(std::move(model_path)), pool_size_(pool_size), share_(share) {
  config_ = getConfigs(model_path_);
  architecture_ = getArchitecture(config_);
}

ModelService::~ModelService() {
  // Safety net: if a serving session was started but never stopped, tear it
  // down so std::thread destructors do not terminate on joinable threads.
  if (!workers_.empty())
    stop();
}

void ModelService::loadBase(std::mutex &global_create_mtx) {
  std::lock_guard<std::mutex> lk(global_create_mtx);
  std::stringstream null_stream;
  auto old_cout = std::cout.rdbuf(null_stream.rdbuf());
  auto old_cerr = std::cerr.rdbuf(null_stream.rdbuf());

  try {
    base_ = buildInstance(config_, architecture_, model_path_, nullptr);
  } catch (...) {
    std::cout.rdbuf(old_cout);
    std::cerr.rdbuf(old_cerr);
    throw;
  }
  std::cout.rdbuf(old_cout);
  std::cerr.rdbuf(old_cerr);

  if (!base_)
    throw std::runtime_error("Failed to create base model for " + model_path_);
}

void ModelService::start(std::mutex &global_create_mtx) {
  create_mtx_ = &global_create_mtx;
  stop_ = false;
  active_.store(0);
  max_active_.store(0);
  served_.store(0);
  {
    std::lock_guard<std::mutex> lk(stats_m_);
    latencies_ms_.clear();
  }

  // Peak-memory sampler for the whole serving session.
  peak_rss_kb.store(0);
  tracking_enabled.store(true);
  start_peak_tracker();

  workers_.clear();
  for (int wi = 0; wi < pool_size_; ++wi)
    workers_.emplace_back([this, wi]() { workerLoop(wi, share_); });
}

void ModelService::submit(const std::string &prompt) {
  {
    std::lock_guard<std::mutex> lk(qm_);
    queue_.push(prompt);
  }
  cv_.notify_one();
}

void ModelService::workerLoop(int worker_id, bool share) {
  while (true) {
    std::string prompt;
    {
      std::unique_lock<std::mutex> lk(qm_);
      cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
      if (queue_.empty() && stop_)
        return;
      prompt = std::move(queue_.front());
      queue_.pop();
    }

    int cur = ++active_;
    int pm = max_active_.load();
    while (cur > pm && !max_active_.compare_exchange_weak(pm, cur)) {
    }

    size_t pending_on_start = 0;
    {
      std::lock_guard<std::mutex> lk(qm_);
      pending_on_start = queue_.size();
    }

    {
      std::lock_guard<std::mutex> lk(io_mtx_);
      std::cout << "\r\033[2K" << std::flush;
      std::cout << "[worker " << worker_id << "] processing (active=" << cur
                << ", pending=" << pending_on_start << ")" << std::endl;
      if (!g_current_prompt.empty()) {
        std::cout << g_current_prompt << std::flush;
      }
    }

    // (a) Create a fresh instance (serialized). When sharing, reference the
    //     base's weights; otherwise build a standalone instance.
    std::shared_ptr<causallm::Transformer> inst;
    {
      std::lock_guard<std::mutex> lk(*create_mtx_);
      std::stringstream null_stream;
      auto old_cout = std::cout.rdbuf(null_stream.rdbuf());
      auto old_cerr = std::cerr.rdbuf(null_stream.rdbuf());

      try {
        inst = buildInstance(config_, architecture_, model_path_,
                             share ? base_ : nullptr);
      } catch (...) {
        std::cout.rdbuf(old_cout);
        std::cerr.rdbuf(old_cerr);
        throw;
      }
      std::cout.rdbuf(old_cout);
      std::cerr.rdbuf(old_cerr);
    }
    if (!inst) {
      --active_;
      std::lock_guard<std::mutex> lk(io_mtx_);
      std::cout << "\r\033[2K" << std::flush;
      std::cerr << "[worker " << worker_id << "] instance creation failed"
                << std::endl;
      if (!g_current_prompt.empty()) {
        std::cout << g_current_prompt << std::flush;
      }
      continue;
    }

    // (b) Run once.
    auto t0 = std::chrono::high_resolution_clock::now();
    std::string out =
      runModel(config_, architecture_, model_path_, inst, prompt);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    // (c) Discard: frees this instance's activation/KV memory; the base's
    //     shared weight pool is only referenced, never owned, so it survives.
    inst.reset();

    --active_;
    ++served_;
    {
      std::lock_guard<std::mutex> lk(stats_m_);
      latencies_ms_.push_back(ms);
    }

    size_t pending = 0;
    {
      std::lock_guard<std::mutex> lk(qm_);
      pending = queue_.size();
    }

    {
      std::lock_guard<std::mutex> lk(io_mtx_);
      std::cout << "\r\033[2K" << std::flush;
      std::cout << "\n========================================\n"
                << "[worker " << worker_id << " Input]:  " << prompt << "\n"
                << "----------------------------------------\n"
                << "[worker " << worker_id << " Output]:\n"
                << out << "\n"
                << "----------------------------------------\n"
                << "[worker " << worker_id << "] done in " << ms << " ms"
                << " (active: " << active_.load() << ", pending: " << pending << ")\n"
                << "========================================\n"
                << std::endl;
      if (!g_current_prompt.empty()) {
        std::cout << g_current_prompt << std::flush;
      }
    }
  }
}

ServingStats ModelService::stop() {
  {
    std::lock_guard<std::mutex> lk(qm_);
    stop_ = true;
  }
  cv_.notify_all();
  for (auto &t : workers_)
    if (t.joinable())
      t.join();
  workers_.clear();

  tracking_enabled.store(false);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));

  ServingStats s;
  s.pool_size = pool_size_;
  s.served = served_.load();
  s.max_concurrency = max_active_.load();
  s.peak_rss_kb = peak_rss_kb.load();
  {
    std::lock_guard<std::mutex> lk(stats_m_);
    double sum = 0.0;
    for (double v : latencies_ms_)
      sum += v;
    s.avg_ms = latencies_ms_.empty() ? 0.0 : sum / latencies_ms_.size();
    if (!latencies_ms_.empty()) {
      std::sort(latencies_ms_.begin(), latencies_ms_.end());
      size_t idx = static_cast<size_t>(latencies_ms_.size() * 0.95);
      s.p95_ms = latencies_ms_[std::min(idx, latencies_ms_.size() - 1)];
    }
  }
  return s;
}

/* ============================ Orchestrator ============================ */

static std::string extractModelName(const std::string &path) {
  std::string norm = path;
  while (!norm.empty() && (norm.back() == '/' || norm.back() == '\\')) {
    norm.pop_back();
  }
  size_t last_slash = norm.find_last_of("/\\");
  if (last_slash != std::string::npos) {
    return norm.substr(last_slash + 1);
  }
  return norm;
}

int Orchestrator::addService(const std::string &model_path, int pool_size) {
  try {
    std::string model_name = extractModelName(model_path);

    // Check if a service for this model path is already registered
    for (const auto &[id, existing_svc] : services_) {
      if (extractModelName(existing_svc->path()) == model_name) {
        std::cout << "[INFO] Service for " << model_path
                  << " already exists with ID: " << id << std::endl;
        return id;
      }
    }

    auto svc = std::make_unique<ModelService>(model_path, pool_size, sharing_);
    if (sharing_)
      svc->loadBase(global_create_mtx_);
    int id = next_id_++;
    services_[id] = std::move(svc);
    std::cout << "[SUCCESS] Service created with ID: " << id
              << " (pool_size=" << pool_size << ", sharing=" << (sharing_ ? "ON" : "OFF") << ")" << std::endl;
    return id;
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Failed to create service: " << e.what() << std::endl;
    return -1;
  }
}

void Orchestrator::serve(int id) {
  auto it = services_.find(id);
  if (it == services_.end()) {
    std::cerr << "[ERROR] Service ID " << id << " not found" << std::endl;
    return;
  }
  ModelService &svc = *it->second;

  std::cout << "[Serving] id=" << id << " pool_size=" << svc.poolSize()
            << " sharing=" << (svc.share() ? "ON" : "OFF")
            << " — type a prompt + Enter, 'quit' to stop." << std::endl;

  svc.start(global_create_mtx_);

  g_current_prompt = "Prompt > ";
  std::cout << g_current_prompt << std::flush;

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == "quit" || line == "exit")
      break;
    if (line.empty()) {
      std::cout << g_current_prompt << std::flush;
      continue;
    }
    g_current_prompt = "";
    svc.submit(line);
    g_current_prompt = "Prompt > ";
    std::cout << g_current_prompt << std::flush;
  }
  g_current_prompt = "";

  ServingStats s = svc.stop();
  std::cout << "\n====== Serving Summary ======\n"
            << "pool size        : " << s.pool_size << "\n"
            << "sharing          : " << (svc.share() ? "ON" : "OFF") << "\n"
            << "requests served  : " << s.served << "\n"
            << "max concurrency  : " << s.max_concurrency << "\n"
            << "avg latency (ms) : " << s.avg_ms << "\n"
            << "p95 latency (ms) : " << s.p95_ms << "\n"
            << "peak private RSS : " << s.peak_rss_kb << " KB" << std::endl;
  printMemoryUsage();
  std::cout << "=============================\n" << std::endl;
}

void Orchestrator::list() {
  if (services_.empty()) {
    std::cout << "No services" << std::endl;
    return;
  }
  std::cout << "\n=== Services ===" << std::endl;
  for (const auto &[id, svc] : services_)
    std::cout << "ID: " << id << " | Path: " << svc->path()
              << " | Architecture: " << svc->arch()
              << " | Pool Size: " << svc->poolSize()
              << " | Sharing: " << (svc->share() ? "ON" : "OFF") << std::endl;
  std::cout << std::endl;
}

void Orchestrator::remove(int id) {
  if (services_.erase(id) > 0)
    std::cout << "[SUCCESS] Service " << id << " removed" << std::endl;
  else
    std::cerr << "[ERROR] Service ID " << id << " not found" << std::endl;
}

bool Orchestrator::has(int id) const {
  return services_.find(id) != services_.end();
}
