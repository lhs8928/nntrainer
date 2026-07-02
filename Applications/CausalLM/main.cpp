/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the_project_root.
 *
 *
 * @file	main.cpp
 * @date	23 July 2025
 * @brief	This is a main file for CausalLM application
 * @see		https://github.com/nnstreamer/
 * @author	Eunju Yang <ej.yang@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <iostream>
#include <string>
#include <sstream>
#include <chrono>
#include <cstdlib>
#include <filesystem>

#include "model_manager.h"
#include "model_runtime.h"

std::string g_models_dir = "";
std::vector<std::string> g_discovered_models;

void displayMenu() {
  std::cout << "\n====== CausalLM Model Manager ======" << std::endl;
  std::cout << "1                       - List registered models (in folder)" << std::endl;
  std::cout << "2 <model_idx_or_name>   - Create service (from folder)" << std::endl;
  std::cout << "3                       - List active services" << std::endl;
  std::cout << "4 <id> [prompt]         - Serve service (interactive or single prompt)" << std::endl;
  std::cout << "5 <id>                  - Remove service" << std::endl;
  std::cout << "6                       - Exit" << std::endl;
  std::cout << "====================================\n" << std::endl;
}

void interactiveShell(ModelManager &orch, const std::string &init_path = "", int init_pool = 2) {
  if (!init_path.empty()) {
    int id = orch.addService(init_path, init_pool);
    if (id >= 0) {
      orch.serve(id);
    }
    return;
  }

  int last_served_id = -1;
  bool first_run = true;

  while (true) {
    if (first_run) {
      displayMenu();
      first_run = false;
    }
    g_current_prompt = "Enter choice and option: ";
    std::cout << g_current_prompt << std::flush;
    std::string line;
    std::getline(std::cin, line);
    g_current_prompt = "";
    if (line.empty())
      continue;

    // Trim leading/trailing spaces
    auto trim = [](std::string &s) {
      s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
      }));
      s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
      }).base(), s.end());
    };
    trim(line);

    if (line.empty())
      continue;

    // Help menu request check
    if (line == "help" || line == "h" || line == "menu") {
      displayMenu();
      continue;
    }

    std::istringstream iss(line);
    int choice;
    iss >> choice;

    switch (choice) {
      case 1: {
        if (g_models_dir.empty()) {
          std::cout << "No models parent directory configured on startup." << std::endl;
        } else {
          std::cout << "\n=== Registered Models in " << g_models_dir << " ===" << std::endl;
          try {
            std::vector<std::filesystem::path> subdirs;
            for (const auto &entry : std::filesystem::directory_iterator(g_models_dir)) {
              if (entry.is_directory()) {
                subdirs.push_back(entry.path());
              }
            }
            std::sort(subdirs.begin(), subdirs.end());
            
            g_discovered_models.clear();
            int count = 0;
            for (const auto &subdir : subdirs) {
              if (std::filesystem::exists(subdir / "config.json")) {
                std::cout << "  [" << count << "] " << subdir.filename().string() << std::endl;
                g_discovered_models.push_back(subdir.string());
                count++;
              }
            }
            if (count == 0) {
              std::cout << "  (No valid model subdirectories containing config.json found)" << std::endl;
            }
          } catch (const std::exception &e) {
            std::cerr << "[ERROR] Failed to scan models directory: " << e.what() << std::endl;
          }
        }
        break;
      }
      case 2: {
        std::string input_val;
        if (iss >> input_val) {
          int pool = 2;
          int pool_val;
          if (iss >> pool_val) {
            pool = pool_val;
          }

          std::string full_path = "";
          // Check if input_val is purely numeric
          bool is_numeric = true;
          for (char c : input_val) {
            if (c < '0' || c > '9') {
              is_numeric = false;
              break;
            }
          }

          if (is_numeric) {
            int idx = std::stoi(input_val);
            if (idx >= 0 && idx < static_cast<int>(g_discovered_models.size())) {
              full_path = g_discovered_models[idx];
            } else {
              std::cerr << "[ERROR] Invalid model index: " << idx << ". Use Option 1 to see valid indices." << std::endl;
              break;
            }
          } else {
            full_path = input_val;
            if (!g_models_dir.empty() && input_val[0] != '/' && input_val[0] != '\\') {
              full_path = g_models_dir + "/" + input_val;
            }
          }

          orch.addService(full_path, pool);
        } else {
          std::cerr << "Usage: 2 <model_index_or_name> [pool_size]" << std::endl;
        }
        break;
      }
      case 3:
        orch.list();
        break;
      case 4: {
        int id;
        if (iss >> id) {
          std::string prompt;
          std::getline(iss, prompt);

          // Trim leading/trailing spaces
          auto trim_inner = [](std::string &s) {
            s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
              return !std::isspace(ch);
            }));
            s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
              return !std::isspace(ch);
            }).base(), s.end());
          };
          trim_inner(prompt);

          last_served_id = id; // Update the last served model ID

          if (prompt.empty()) {
            orch.serve(id);
          } else {
            orch.runSingle(id, prompt);
          }
        } else {
          std::cerr << "Usage: 4 <id> [prompt_text]" << std::endl;
        }
        break;
      }
      case 5: {
        int id;
        if (iss >> id)
          orch.remove(id);
        else
          std::cerr << "Usage: 5 <id>" << std::endl;
        break;
      }
      case 6:
        std::cout << "Exiting..." << std::endl;
        return;
      default:
        std::cerr << "Invalid choice. Please enter 1, 2, 3, 4, 5, or 6."
                  << std::endl;
    }
  }
}

/**
 * @brief Entry point: register models, then either run the interactive
 *        model manager shell or serve a model given on the command line.
 */
int main(int argc, char *argv[]) {
  auto start_time = std::chrono::high_resolution_clock::now();

  registerCausalModels();

  // Sharing policy (orchestration-level). Override at runtime with
  // CAUSALLM_SHARING=0 to disable (non-shared baseline) without rebuilding.
  bool sharing = true;
  if (const char *e = std::getenv("CAUSALLM_SHARING"))
    sharing = !(std::string(e) == "0" || std::string(e) == "off");

  std::string init_path = (argc >= 2) ? argv[1] : "";
  int init_pool = (argc >= 3) ? std::atoi(argv[2]) : 2;

  if (!init_path.empty()) {
    try {
      if (std::filesystem::is_directory(init_path)) {
        if (std::filesystem::exists(std::filesystem::path(init_path) / "config.json")) {
          // Single model directory: do not set parent directory
        } else {
          // Parent directory containing multiple models
          g_models_dir = init_path;
          std::cout << "[INFO] Models parent directory configured: " << g_models_dir << std::endl;
          init_path = ""; // Clear so it doesn't load parent itself as a service

          // Scan parent directory alphabetically and populate g_discovered_models on boot
          std::vector<std::filesystem::path> subdirs;
          for (const auto &entry : std::filesystem::directory_iterator(g_models_dir)) {
            if (entry.is_directory()) {
              subdirs.push_back(entry.path());
            }
          }
          std::sort(subdirs.begin(), subdirs.end());
          g_discovered_models.clear();
          for (const auto &subdir : subdirs) {
            if (std::filesystem::exists(subdir / "config.json")) {
              g_discovered_models.push_back(subdir.string());
            }
          }
        }
      }
    } catch (...) {
      // Keep as-is on any filesystem errors
    }
  }

  ModelManager m_mgr(sharing);

  interactiveShell(m_mgr, init_path, init_pool);

  auto finish_time = std::chrono::high_resolution_clock::now();
  auto e2e = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_time - start_time);
  std::cout << "[e2e time]: " << e2e.count() << " ms \n";
  printMemoryUsage();
  return EXIT_SUCCESS;
}
