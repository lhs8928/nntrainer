// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   qwen3_asr_causallm.cpp
 * @brief  Qwen3 ASR multimodal conditional generation model implementation.
 * @date   14 September 2026
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 */

#include <llm_util.hpp>
#include <model.h>
#include <layer_node.h>
#include <qwen3_asr_causallm.h>

#include <app_context.h>
#include <engine.h>
#include <multimodal_scatter.h>
#include <reshaped_rms_norm.h>
#include <tie_word_embedding.h>
#include <swiglu.h>
#include <audio_preprocessor.h>
#include <api/streamer.h>
#include <performance_metrics.h>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include <fstream>

namespace quick_ai {

static unsigned int get_feat_extract_output_lengths(unsigned int input_lengths) {
  unsigned int input_lengths_leave = input_lengths % 100;
  unsigned int tail_tokens = 0;
  if (input_lengths_leave > 0) {
    unsigned int feat_lengths = (input_lengths_leave - 1) / 2 + 1;
    tail_tokens = ((feat_lengths - 1) / 2 + 1 - 1) / 2 + 1;
  }
  return tail_tokens + (input_lengths / 100) * 13;
}

std::pair<Tensor, Tensor> Qwen3ASRTransformer::constructModel() {
  // input0: text token IDs [batch, 1, 1, seq_len] always in FP32
  Tensor input0 = Tensor(nntrainer::TensorDim(1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN), nntrainer::TensorDim::Format::NCHW, nntrainer::TensorDim::DataType::FP32), "input0");

  unsigned int downsampled_len = get_feat_extract_output_lengths(audio_seq_len);

  // input1: downsampled audio embeddings [batch, 1, downsampled_len, 1024] in FP16 (produced by Qwen3ASRSubsampler sub-model!)
  Tensor input1 = Tensor(nntrainer::TensorDim(1, 1, downsampled_len, 1024, nntrainer::TensorDim::Format::NCHW, nntrainer::TensorDim::DataType::FP16), "input1");

  // text embedding
  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

  LayerHandle embedding(createLayer(
    embedding_type,
    buildEmbeddingLayerProperties("embedding0", NUM_VOCAB, DIM, EMBEDDING_DTYPE,
                                  EMBEDDING_SCALE, EMBEDDING_FILE_NAME)));
  Tensor text_embed = embedding(input0);

  // audio encoder (24 Transformer Encoder blocks + Proj)
  Tensor audio_embed = createAudioEncoder(input1);

  // multimodal scatter (fusion)
  LayerHandle scatter(createLayer(
    "multimodal_scatter",
    {withKey("name", "fusion_scatter"),
     withKey("vocab_size", std::to_string(NUM_VOCAB)),
     withKey("audio_pad_id", "151676")}));
  Tensor h = scatter({text_embed, audio_embed, input0});

  // text decoder blocks
  for (unsigned int i = 0; i < NUM_LAYERS; ++i) {
    h = createTransformerDecoderBlock(i, h);
  }

  // final rms_norm
  LayerHandle out_norm(
    createLayer("rms_norm", {withKey("name", "output_norm"),
                             withKey("epsilon", std::to_string(NORM_EPS)),
                             withKey("packed", "false")}));
  h = out_norm(h);

  return {input0, h};
}

Tensor Qwen3ASRTransformer::createAudioEncoder(Tensor audio_input) {
  unsigned int downsampled_len = audio_input.shape().height();
  Tensor h = audio_input;

  // 8. 24x Transformer Encoder Blocks
  for (int i = 0; i < 24; ++i) {
    h = createAudioAttentionBlock(i, h, downsampled_len);
  }

  // 9. ln_post: LayerNorm
  LayerHandle ln_post(createLayer(
    "layer_normalization",
    {withKey("name", "audio_tower_ln_post"),
     withKey("axis", "3"),
     withKey("epsilon", "1e-5"),
     withKey("weight_dtype", "FP16")}));
  h = ln_post(h);

  // 10. Projections
  LayerHandle proj1(createLayer(
    "fully_connected",
    {withKey("name", "audio_tower_proj1"),
     withKey("unit", "1024"),
     withKey("disable_bias", "false"),
     withKey("weight_dtype", "FP16")}));
  h = proj1(h);

  LayerHandle proj_gelu(createLayer("activation", {withKey("name", "audio_tower_proj_gelu"), withKey("activation", "gelu")}));
  h = proj_gelu(h);

  LayerHandle proj2(createLayer(
    "fully_connected",
    {withKey("name", "audio_tower_proj2"),
     withKey("unit", std::to_string(DIM)),
     withKey("disable_bias", "false"),
     withKey("weight_dtype", "FP16")}));
  h = proj2(h);

  return h;
}

Tensor Qwen3ASRTransformer::createAudioAttentionBlock(const int layer_id, Tensor input, unsigned int max_timestep) {
  const std::string prefix = "audio_tower_layer" + std::to_string(layer_id) + "_";

  LayerHandle norm(createLayer(
    "layer_normalization",
    {withKey("name", prefix + "attention_norm"),
     withKey("axis", "3"),
     withKey("epsilon", "1e-5"),
     withKey("weight_dtype", "FP16")}));
  Tensor normed = norm(input);

  LayerHandle q_proj(createLayer(
    "fully_connected",
    {withKey("name", prefix + "qkv_q"),
     withKey("unit", "1024"),
     withKey("disable_bias", "false"),
     withKey("weight_dtype", "FP16")}));
  LayerHandle k_proj(createLayer(
    "fully_connected",
    {withKey("name", prefix + "qkv_k"),
     withKey("unit", "1024"),
     withKey("disable_bias", "false"),
     withKey("weight_dtype", "FP16")}));
  LayerHandle v_proj(createLayer(
    "fully_connected",
    {withKey("name", prefix + "qkv_v"),
     withKey("unit", "1024"),
     withKey("disable_bias", "false"),
     withKey("weight_dtype", "FP16")}));

  Tensor query = q_proj(normed);
  Tensor key = k_proj(normed);
  Tensor value = v_proj(normed);

  LayerHandle q_cast_fp32(createLayer("cast", {withKey("name", prefix + "q_cast_fp32"), withKey("tensor_dtype", "FP32")}));
  LayerHandle k_cast_fp32(createLayer("cast", {withKey("name", prefix + "k_cast_fp32"), withKey("tensor_dtype", "FP32")}));
  LayerHandle v_cast_fp32(createLayer("cast", {withKey("name", prefix + "v_cast_fp32"), withKey("tensor_dtype", "FP32")}));
  query = q_cast_fp32(query);
  key = k_cast_fp32(key);
  value = v_cast_fp32(value);

  LayerHandle attention(createLayer(
    "mha_core",
    {withKey("name", prefix + "attention"),
     withKey("num_heads", "16"),
     withKey("num_heads_kv", "16"),
     withKey("max_timestep", std::to_string(max_timestep)),
     withKey("use_rope", "false"),
     withKey("is_causal", "false")}));
  Tensor context = attention({query, key, value});

  LayerHandle context_cast_fp16(createLayer(
    "cast",
    {withKey("name", prefix + "context_cast_fp16"),
     withKey("tensor_dtype", "FP16")}));
  context = context_cast_fp16(context);

  LayerHandle out_proj(createLayer(
    "fully_connected",
    {withKey("name", prefix + "attention_out"),
     withKey("unit", "1024"),
     withKey("disable_bias", "false"),
     withKey("weight_dtype", "FP16")}));
  Tensor att_out = out_proj(context);

  LayerHandle attention_res(createLayer("addition", {withKey("name", prefix + "attention_residual")}));
  Tensor residual = attention_res({input, att_out});

  LayerHandle ffn_norm(createLayer(
    "layer_normalization",
    {withKey("name", prefix + "ffn_norm"),
     withKey("axis", "3"),
     withKey("epsilon", "1e-5"),
     withKey("weight_dtype", "FP16")}));
  Tensor ffn_normed = ffn_norm(residual);

  LayerHandle ffn_up(createLayer(
    "fully_connected",
    {withKey("name", prefix + "ffn_up"),
     withKey("unit", "4096"),
     withKey("disable_bias", "false"),
     withKey("weight_dtype", "FP16")}));
  Tensor h = ffn_up(ffn_normed);

  LayerHandle ffn_gelu_cast_fp32(createLayer(
    "cast",
    {withKey("name", prefix + "ffn_gelu_cast_fp32"),
     withKey("tensor_dtype", "FP32")}));
  h = ffn_gelu_cast_fp32(h);

  LayerHandle ffn_gelu(createLayer("activation", {withKey("name", prefix + "ffn_gelu"), withKey("activation", "gelu")}));
  h = ffn_gelu(h);

  LayerHandle ffn_gelu_cast_fp16(createLayer(
    "cast",
    {withKey("name", prefix + "ffn_gelu_cast_fp16"),
     withKey("tensor_dtype", "FP16")}));
  h = ffn_gelu_cast_fp16(h);

  LayerHandle ffn_down(createLayer(
    "fully_connected",
    {withKey("name", prefix + "ffn_down"),
     withKey("unit", "1024"),
     withKey("disable_bias", "false"),
     withKey("weight_dtype", "FP16")}));
  Tensor mlp_out = ffn_down(h);

  LayerHandle ffn_res(createLayer("addition", {withKey("name", prefix + "ffn_residual")}));
  return ffn_res({residual, mlp_out});
}

std::pair<Tensor, Tensor> Qwen3ASRCausalLM::constructModel() {
  auto [x, h] = Qwen3ASRTransformer::constructModel();

  const std::string lmhead_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head";

  std::vector<std::string> lmhead_prop = {
    withKey("name", "output_of_causallm"),
    withKey("unit", NUM_VOCAB),
    withKey("disable_bias", "true"),
    withKey("weight_dtype", LMHEAD_DTYPE),
  };

  if (TIE_WORD_EMBEDDINGS)
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));

  LayerHandle lmhead(createLayer(lmhead_type, lmhead_prop));
  Tensor y = lmhead(h);

  return {x, y};
}

void Qwen3ASRCausalLM::initialize() {
  std::cout << "[Qwen3-ASR] Initializing Qwen3ASRSubsampler sub-model..." << std::endl;
  subsampler.initialize(MODEL_TENSOR_TYPE);

  std::cout << "[Qwen3-ASR] Initializing Main CausalLM model..." << std::endl;
  Transformer::initialize();
}

void Qwen3ASRCausalLM::load_weight(const std::string &path) {
  std::cout << "[Qwen3-ASR] Loading weights for Subsampler sub-model from: " << path << std::endl;
  subsampler.load_weight(path);

  std::cout << "[Qwen3-ASR] Loading weights for Main CausalLM model from: " << path << std::endl;
  Transformer::load_weight(path);
}

void Qwen3ASRCausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();

  static std::once_flag registered_flag;
  std::call_once(registered_flag, []() {
    const auto &ct_engine = nntrainer::Engine::Global();
    const auto app_context = static_cast<nntrainer::AppContext *>(
      ct_engine.getRegisteredContext("cpu"));

    try {
      app_context->registerFactory(
        nntrainer::createLayer<quick_ai::MultimodalScatterLayer>);
      app_context->registerFactory(
        nntrainer::createLayer<quick_ai::ReshapedRMSNormLayer>);
      app_context->registerFactory(
        nntrainer::createLayer<quick_ai::TieWordEmbedding>);
      app_context->registerFactory(
        nntrainer::createLayer<quick_ai::SwiGLULayer>);
    } catch (const std::invalid_argument &e) {
      std::cerr << "failed to register factory, reason: " << e.what()
                << std::endl;
    }
  });
}

void Qwen3ASRCausalLM::run(const WSTR prompt, bool do_sample,
                           const WSTR system_prompt, const WSTR tail_prompt,
                           bool log_output) {
  std::cout << "[Dtype Size Debug] sizeof(_FP16): " << sizeof(_FP16) 
            << ", sizeof(_Float16): " << sizeof(_Float16) 
            << ", sizeof(float): " << sizeof(float) << std::endl;

  if (!is_initialized) {
    throw std::runtime_error("Qwen3ASRCausalLM model is not initialized.");
  }

  // 1. Check/load audio file and extract Mel features matching compiled audio_seq_len
  std::vector<float> mel_features_transposed(static_cast<size_t>(audio_seq_len) * 128, 0.0f);
  bool loaded_pytorch_mel = false;

  std::string pytorch_mel_path = "/data/local/tmp/nntrainer/causallm/pytorch_mel.bin";
  if (std::filesystem::exists(pytorch_mel_path)) {
    std::cout << "[Qwen3-ASR] [Bypass Preprocessor] Loading official PyTorch Mel features from: " << pytorch_mel_path << std::endl;
    std::ifstream mel_file(pytorch_mel_path, std::ios::in | std::ios::binary);
    if (mel_file.is_open()) {
      mel_file.read(reinterpret_cast<char*>(mel_features_transposed.data()), mel_features_transposed.size() * sizeof(float));
      mel_file.close();
      loaded_pytorch_mel = true;
    }
  }

  if (!loaded_pytorch_mel) {
    std::vector<float> mel_features;
    quick_ai::AudioPreprocessor preprocessor;
    if (!audio_path.empty()) {
      std::cout << "[Qwen3-ASR] Loading and preprocessing audio WAV: " << audio_path << std::endl;
      std::vector<float> pcm = preprocessor.loadWav(audio_path);
      
      std::cout << "[Step 1 Test] NNTrainer loadWav total samples: " << pcm.size() << std::endl;
      std::cout << "First 20 samples: " << std::endl;
      for (size_t i = 0; i < std::min<size_t>(pcm.size(), 20); ++i) {
        std::cout << pcm[i] << ", ";
      }
      std::cout << std::endl;
      std::cout << "Last 20 samples: " << std::endl;
      size_t start = pcm.size() > 20 ? pcm.size() - 20 : 0;
      for (size_t i = start; i < pcm.size(); ++i) {
        std::cout << pcm[i] << ", ";
      }
      std::cout << std::endl;

      mel_features = preprocessor.computeMelSpectrogram(pcm);
      std::cout << "[Qwen3-ASR] Raw extracted Mel features: " << mel_features.size() / 128 << " frames" << std::endl;
    } else {
      // If no audio is loaded, search for asr_en_short.wav in current dir or fallback
      std::string fallback = "asr_en_short.wav";
      if (std::filesystem::exists(fallback)) {
        audio_path = fallback;
        std::vector<float> pcm = preprocessor.loadWav(audio_path);
        mel_features = preprocessor.computeMelSpectrogram(pcm);
      } else {
        std::cerr << "[Qwen3-ASR] Error: No audio path provided and fallback not found!" << std::endl;
        return;
      }
    }

    size_t required_elements = static_cast<size_t>(audio_seq_len) * 128;
    if (mel_features.size() < required_elements) {
      mel_features.resize(required_elements, 0.0f);
    } else if (mel_features.size() > required_elements) {
      mel_features.resize(required_elements);
    }
    std::cout << "[Qwen3-ASR] Normalized Mel features: " << audio_seq_len << " frames (padded/truncated)" << std::endl;

    // Transpose Mel features to match PyTorch's [128, audio_seq_len] shape layout
    for (unsigned int t = 0; t < audio_seq_len; ++t) {
      for (unsigned int f = 0; f < 128; ++f) {
        mel_features_transposed[f * audio_seq_len + t] = mel_features[t * 128 + f];
      }
    }
  }

  std::cout << "[Step 4 Test] NNTrainer mel_features_transposed shape: [128, " << audio_seq_len << "]" << std::endl;
  for (unsigned int r : {0, 1, 2, 127}) {
    std::cout << "  [Row " << r << "]: ";
    for (unsigned int k = 0; k < std::min<unsigned int>(audio_seq_len, 5); ++k) {
      std::cout << mel_features_transposed[r * audio_seq_len + k] << ", ";
    }
    std::cout << std::endl;
  }

  // Run Subsampler sub-model chunk-by-chunk to produce fused_audio_embeds [1, 1, downsampled_len, 1024]
  unsigned int downsampled_len = get_feat_extract_output_lengths(audio_seq_len);
  unsigned int num_chunks = (audio_seq_len + 99) / 100;
  if (num_chunks == 0) num_chunks = 1;

  std::vector<_Float16> fused_audio_embeds(static_cast<size_t>(downsampled_len) * 1024, static_cast<_Float16>(0.0f));
  nntrainer::Tensor chunk_tensor(nntrainer::TensorDim(1, 1, 128, 100, nntrainer::TensorDim::Format::NCHW, nntrainer::TensorDim::DataType::FP32));
  float *chunk_ptr = chunk_tensor.getData<float>();

  size_t out_token_idx = 0;
  for (unsigned int c = 0; c < num_chunks; ++c) {
    unsigned int start_frame = c * 100;
    unsigned int chunk_frames = std::min<unsigned int>(100, audio_seq_len - start_frame);

    // Extract chunk [128, 100] (0-padded on right)
    std::fill_n(chunk_ptr, 128 * 100, 0.0f);
    for (unsigned int m = 0; m < 128; ++m) {
      for (unsigned int f = 0; f < chunk_frames; ++f) {
        chunk_ptr[m * 100 + f] = mel_features_transposed[m * audio_seq_len + start_frame + f];
      }
    }

    // Forward through Subsampler sub-model (runs in FP32)
    auto chunk_out = subsampler.inference(chunk_ptr);
    const float *cout_ptr = chunk_out[0];

    // Copy valid tokens into fused_audio_embeds (converting float -> _Float16)
    unsigned int valid_tokens = (c + 1 < num_chunks) ? 13 : (downsampled_len - out_token_idx);
    for (unsigned int t = 0; t < valid_tokens; ++t) {
      _Float16 *dst = fused_audio_embeds.data() + out_token_idx * 1024;
      const float *src = cout_ptr + t * 1024;
      for (int k = 0; k < 1024; ++k) {
        dst[k] = static_cast<_Float16>(src[k]);
      }
      out_token_idx++;
    }
  }

  std::cout << "[Qwen3-ASR] Subsampler sub-model completed: " << num_chunks
            << " chunks -> " << out_token_idx << " audio tokens [1, 1, "
            << downsampled_len << ", 1024]" << std::endl;

  void *audio_input_ptr = fused_audio_embeds.data();

  // 2. Tokenize prompt
  std::string prompt_ = system_prompt + prompt + tail_prompt;
  auto _input = tokenizer->Encode(prompt_);

  std::cout << "[Qwen3-ASR Debug] Encoded Prompt Token IDs (" << _input.size() << " tokens): ";
  for (size_t i = 0; i < _input.size(); ++i) {
    std::cout << _input[i] << " ";
  }
  std::cout << std::endl;

  std::vector<int64_t> init_input;
  unsigned int _len = _input.size();
  unsigned int num_allow_str = MAX_SEQ_LEN - NUM_TO_GENERATE;
  unsigned int text_len = _len;

  if (_len > num_allow_str) {
    text_len = num_allow_str;
    std::cerr << "[Qwen3-ASR] WARNING: prompt (" << _len
              << " tokens) exceeds max prefill (" << num_allow_str << ")" << std::endl;
  }

  for (unsigned int i = 0; i < text_len; ++i) {
    init_input.push_back(_input[i]);
  }

  unsigned int init_len = init_input.size();
  float *input_sample = (float *)malloc(sizeof(float) * BATCH_SIZE * MAX_SEQ_LEN);
  std::vector<bool> eos_list(BATCH_SIZE, false);

  // Zero-initialize ids_history to clean up garbage values from malloc
  std::fill_n(ids_history, BATCH_SIZE * MAX_SEQ_LEN, 0);

  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    for (unsigned int i = 0; i < init_len; ++i) {
      input_sample[b * MAX_SEQ_LEN + i] = static_cast<float>(init_input[i]);
      ids_history[b * MAX_SEQ_LEN + i] = init_input[i];
    }
  }

  // 3. Allocate and bind KV Cache FIRST!
  allocateAndBindKVCache();
  setKVCachePosition(0);

  // 4. Build inference inputs (input0, input1, and KV caches)
  std::vector<std::pair<std::string, float *>> cache_inputs;
  cache_inputs.reserve(static_cast<size_t>(NUM_LAYERS) * 2);
  for (int i = 0; i < NUM_LAYERS; ++i) {
    cache_inputs.emplace_back(
      "cache_k_l" + std::to_string(i),
      reinterpret_cast<float *>(kv_cache.getKeyCache(i).getData()));
    cache_inputs.emplace_back(
      "cache_v_l" + std::to_string(i),
      reinterpret_cast<float *>(kv_cache.getValueCache(i).getData()));
  }
  std::sort(cache_inputs.begin(), cache_inputs.end(),
            [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

  std::vector<float *> inference_inputs;
  inference_inputs.reserve(2 + cache_inputs.size());
  inference_inputs.push_back(input_sample);       // Input 0: text token IDs
  inference_inputs.push_back(reinterpret_cast<float *>(audio_input_ptr)); // Input 1: audio Mel-spectrogram
  for (const auto &cache_input : cache_inputs) {
    inference_inputs.push_back(cache_input.second);
  }

  // 5. Run prefill step
  std::cout << "[Qwen3-ASR] Running prefill inference..." << std::endl;
  std::vector<float *> label;
  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, inference_inputs, label, init_len, 0, init_len, false);

  float *logits = output[0];

  std::cout << "[Logits Debug] Token 11528 ('language') Logit: " << logits[11528] << std::endl;
  std::cout << "[Logits Debug] Token 17408 (' ruling') Logit: " << logits[17408] << std::endl;

  std::vector<std::pair<float, int>> indexed_logits;
  for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
    indexed_logits.push_back({logits[i], i});
  }
  std::sort(indexed_logits.begin(), indexed_logits.end(), [](const auto& a, const auto& b) {
    return a.first > b.first;
  });
  std::cout << "=== C++ Prefill Top-10 Logits ===" << std::endl;
  for (int i = 0; i < 10; ++i) {
    std::cout << "Rank " << i + 1 << ": Token ID " << indexed_logits[i].second 
              << ", Logit " << indexed_logits[i].first << std::endl;
  }

  // 6. Autoregressive Generation Loop
  unsigned int generation_cnt = 0;
  auto start_gen = std::chrono::high_resolution_clock::now();

  for (unsigned int step = init_len; step < MAX_SEQ_LEN; ++step) {
    // Print top-10 logits for each step to verify directly against PyTorch
    std::vector<std::pair<float, int>> step_logits;
    step_logits.reserve(NUM_VOCAB);
    for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
      step_logits.push_back({logits[i], i});
    }
    std::sort(step_logits.begin(), step_logits.end(), [](const auto &a, const auto &b) {
      return a.first > b.first;
    });
    std::cout << "\n=== Step " << step << " Top-10 Logits ===" << std::endl;
    for (int i = 0; i < 10; ++i) {
      std::cout << "Rank " << i + 1 << ": Token ID " << step_logits[i].second
                << ", Logit " << step_logits[i].first << " ('"
                << tokenizer->Decode({static_cast<int>(step_logits[i].second)}) << "')" << std::endl;
    }

    unsigned int next_token = applyTKP(logits, NUM_VOCAB, TEMPERATURE, TOP_K, TOP_P, rng);

    // clean up prefill/last step outputs
    for (auto out : output) {
      delete[] out;
    }

    if (next_token == 151645 || next_token == 151643) { // IM_END or BOS
      std::cout << "[Qwen3-ASR] Generation completed (EOS detected)." << std::endl;
      break;
    }

    // append to history and output
    for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
      ids_history[b * MAX_SEQ_LEN + step] = next_token;
      input_sample[b * MAX_SEQ_LEN + step] = static_cast<float>(next_token);
      output_list[b] += tokenizer->Decode({static_cast<int>(next_token)});
    }

    if (log_output) {
      std::cout << "\n[Autoregress step " << step << "] next_token: " << next_token 
                << " ('" << tokenizer->Decode({static_cast<int>(next_token)}) << "')" << std::endl;
    }

    generation_cnt++;

    // Prepare next step inputs
    setKVCachePosition(step);
    inference_inputs[0] = input_sample + step;
    output = model->incremental_inference(
      BATCH_SIZE, inference_inputs, label, 1, step, step + 1, false);
    
    logits = output[0];
  }

  auto end_gen = std::chrono::high_resolution_clock::now();
  auto gen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_gen - start_gen).count();
  std::cout << "\n[Qwen3-ASR] Generated " << generation_cnt << " tokens in " << gen_ms << " ms ("
            << (generation_cnt * 1000.0 / gen_ms) << " TPS)" << std::endl;

  std::ofstream out_f("history.txt");
  for (unsigned int step = 0; step < MAX_SEQ_LEN; ++step) {
    if (ids_history[step] != 0) {
      out_f << ids_history[step] << " ";
    }
  }
  out_f << std::endl;
  out_f.close();
  std::cout << "[Qwen3-ASR] Saved generated token IDs to history.txt" << std::endl;

  free(input_sample);
}

} // namespace quick_ai
