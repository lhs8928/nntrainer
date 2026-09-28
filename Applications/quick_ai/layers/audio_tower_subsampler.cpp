// SPDX-License-Identifier: Apache-2.0
/**
 * @file   audio_tower_subsampler.cpp
 * @brief  Qwen3-ASR Audio Tower Chunked Subsampling Layer implementation
 */

#include "audio_tower_subsampler.h"
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <algorithm>

enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

extern "C" {
void cblas_sgemm(enum CBLAS_ORDER Order, enum CBLAS_TRANSPOSE TransA,
                 enum CBLAS_TRANSPOSE TransB, int M, int N, int K,
                 float alpha, const float *A, int lda,
                 const float *B, int ldb, float beta,
                 float *C, int ldc);
}

namespace quick_ai {

namespace {

inline float gelu(float x) {
  return 0.5f * x * (1.0f + std::erf(x * 0.7071067811865475f));
}

inline unsigned int get_feat_extract_output_lengths(unsigned int input_lengths) {
  unsigned int input_lengths_leave = input_lengths % 100;
  unsigned int tail_tokens = 0;
  if (input_lengths_leave > 0) {
    unsigned int feat_lengths = (input_lengths_leave - 1) / 2 + 1;
    tail_tokens = ((feat_lengths - 1) / 2 + 1 - 1) / 2 + 1;
  }
  return tail_tokens + (input_lengths / 100) * 13;
}

void conv2d_im2col_sgemm(
    const float *input, int in_c, int in_h, int in_w,
    const float *weight, const float *bias, int out_c, int k_h, int k_w,
    int stride, int pad,
    float *output, int out_h, int out_w,
    float *col_buffer) {
  int k_size = in_c * k_h * k_w;
  int out_pixels = out_h * out_w;

  for (int c = 0; c < in_c; ++c) {
    for (int kh = 0; kh < k_h; ++kh) {
      for (int kw = 0; kw < k_w; ++kw) {
        int col_row = (c * k_h + kh) * k_w + kw;
        for (int oh = 0; oh < out_h; ++oh) {
          int ih = oh * stride - pad + kh;
          for (int ow = 0; ow < out_w; ++ow) {
            int iw = ow * stride - pad + kw;
            int col_idx = col_row * out_pixels + (oh * out_w + ow);
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
              col_buffer[col_idx] = input[(c * in_h + ih) * in_w + iw];
            } else {
              col_buffer[col_idx] = 0.0f;
            }
          }
        }
      }
    }
  }

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              out_c, out_pixels, k_size,
              1.0f, weight, k_size,
              col_buffer, out_pixels,
              0.0f, output, out_pixels);

  for (int oc = 0; oc < out_c; ++oc) {
    float b = bias[oc];
    float *row = output + oc * out_pixels;
    for (int p = 0; p < out_pixels; ++p) {
      row[p] = gelu(row[p] + b);
    }
  }
}

} // namespace

AudioTowerSubsamplerLayer::AudioTowerSubsamplerLayer() {
  wt_idx.fill(std::numeric_limits<unsigned int>::max());
}

void AudioTowerSubsamplerLayer::setProperty(const std::vector<std::string> &values) {
  for (const auto &val : values) {
    auto pos = val.find("=");
    if (pos == std::string::npos) continue;
    std::string key = val.substr(0, pos);
    std::string v = val.substr(pos + 1);
    if (key == "downsampled_len") {
      downsampled_len = std::stoul(v);
    }
  }
}

void AudioTowerSubsamplerLayer::finalize(nntrainer::InitLayerContext &context) {
  const auto &input_dims = context.getInputDimensions();
  NNTR_THROW_IF(input_dims.empty(), std::invalid_argument)
    << "AudioTowerSubsamplerLayer requires 1 input tensor";

  unsigned int audio_frames = input_dims[0].width();
  if (downsampled_len == 0) {
    downsampled_len = get_feat_extract_output_lengths(audio_frames);
  }

  // Output: [1, 1, downsampled_len, 1024] in FP16
  nntrainer::TensorDim out_dim(1, 1, downsampled_len, 1024,
                              nntrainer::TensorDim::Format::NCHW,
                              nntrainer::TensorDim::DataType::FP16);
  context.setOutputDimensions({out_dim});

  // Request weights
  // conv1: weight [480, 1, 3, 3] FP32, bias [1, 1, 1, 480] FP32
  wt_idx[CONV1_W] = context.requestWeight(
    nntrainer::TensorDim(480, 1, 3, 3, nntrainer::TensorDim::Format::NCHW, nntrainer::TensorDim::DataType::FP32),
    nntrainer::props::InitializerInfo::Enum::NONE, nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "conv1_w", false);
  wt_idx[CONV1_B] = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1, 480, nntrainer::TensorDim::Format::NCHW, nntrainer::TensorDim::DataType::FP32),
    nntrainer::props::InitializerInfo::Enum::NONE, nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "conv1_b", false);

  // conv2: weight [480, 480, 3, 3] FP32, bias [1, 1, 1, 480] FP32
  wt_idx[CONV2_W] = context.requestWeight(
    nntrainer::TensorDim(480, 480, 3, 3, nntrainer::TensorDim::Format::NCHW, nntrainer::TensorDim::DataType::FP32),
    nntrainer::props::InitializerInfo::Enum::NONE, nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "conv2_w", false);
  wt_idx[CONV2_B] = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1, 480, nntrainer::TensorDim::Format::NCHW, nntrainer::TensorDim::DataType::FP32),
    nntrainer::props::InitializerInfo::Enum::NONE, nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "conv2_b", false);

  // conv3: weight [480, 480, 3, 3] FP32, bias [1, 1, 1, 480] FP32
  wt_idx[CONV3_W] = context.requestWeight(
    nntrainer::TensorDim(480, 480, 3, 3, nntrainer::TensorDim::Format::NCHW, nntrainer::TensorDim::DataType::FP32),
    nntrainer::props::InitializerInfo::Enum::NONE, nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "conv3_w", false);
  wt_idx[CONV3_B] = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1, 480, nntrainer::TensorDim::Format::NCHW, nntrainer::TensorDim::DataType::FP32),
    nntrainer::props::InitializerInfo::Enum::NONE, nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "conv3_b", false);

  // conv_out: weight [1, 1, 1024, 7680] FP16
  wt_idx[CONV_OUT_W] = context.requestWeight(
    nntrainer::TensorDim(1, 1, 1024, 7680, nntrainer::TensorDim::Format::NCHW, nntrainer::TensorDim::DataType::FP16),
    nntrainer::props::InitializerInfo::Enum::NONE, nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "conv_out_w", false);

  // pos_embed: weights [1, 1, 13, 1024] FP16
  wt_idx[POS_EMBED_W] = context.requestWeight(
    nntrainer::TensorDim(1, 1, 13, 1024, nntrainer::TensorDim::Format::NCHW, nntrainer::TensorDim::DataType::FP16),
    nntrainer::props::InitializerInfo::Enum::NONE, nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "pos_embed_w", false);
}

void AudioTowerSubsamplerLayer::forwarding(nntrainer::RunLayerContext &context, bool training) {
  nntrainer::Tensor &in = context.getInput(0);
  nntrainer::Tensor &out = context.getOutput(0);

  const float *w1 = context.getWeight(wt_idx[CONV1_W]).getData<float>();
  const float *b1 = context.getWeight(wt_idx[CONV1_B]).getData<float>();
  const float *w2 = context.getWeight(wt_idx[CONV2_W]).getData<float>();
  const float *b2 = context.getWeight(wt_idx[CONV2_B]).getData<float>();
  const float *w3 = context.getWeight(wt_idx[CONV3_W]).getData<float>();
  const float *b3 = context.getWeight(wt_idx[CONV3_B]).getData<float>();

  nntrainer::Tensor &t_cout = context.getWeight(wt_idx[CONV_OUT_W]);
  nntrainer::Tensor &t_pos = context.getWeight(wt_idx[POS_EMBED_W]);

  // Cache FP32 copies of conv_out and pos_embed on first forwarding
  if (!weights_cached) {
    cached_conv_out_fp32.resize(1024 * 7680);
    cached_pos_embed_fp32.resize(13 * 1024);

    if (t_cout.getDataType() == nntrainer::TensorDim::DataType::FP16) {
      const _Float16 *p = t_cout.getData<_Float16>();
      for (size_t i = 0; i < 1024 * 7680; ++i) cached_conv_out_fp32[i] = static_cast<float>(p[i]);
    } else {
      const float *p = t_cout.getData<float>();
      std::memcpy(cached_conv_out_fp32.data(), p, 1024 * 7680 * sizeof(float));
    }

    if (t_pos.getDataType() == nntrainer::TensorDim::DataType::FP16) {
      const _Float16 *p = t_pos.getData<_Float16>();
      for (size_t i = 0; i < 13 * 1024; ++i) cached_pos_embed_fp32[i] = static_cast<float>(p[i]);
    } else {
      const float *p = t_pos.getData<float>();
      std::memcpy(cached_pos_embed_fp32.data(), p, 13 * 1024 * sizeof(float));
    }
    weights_cached = true;
  }

  const float *w_cout = cached_conv_out_fp32.data();
  const float *pos_embed = cached_pos_embed_fp32.data();

  unsigned int audio_seq_len = in.width();
  unsigned int num_chunks = (audio_seq_len + 99) / 100;
  if (num_chunks == 0) num_chunks = 1;

  const float *feat = in.getData<float>();
  _Float16 *out_ptr = out.getData<_Float16>();

  // Thread-local scratch buffers to avoid heap allocations during inference
  static thread_local std::vector<float> col1(1 * 3 * 3 * 64 * 50);
  static thread_local std::vector<float> out1(480 * 64 * 50);
  static thread_local std::vector<float> col2(480 * 3 * 3 * 32 * 25);
  static thread_local std::vector<float> out2(480 * 32 * 25);
  static thread_local std::vector<float> col3(480 * 3 * 3 * 16 * 13);
  static thread_local std::vector<float> out3(480 * 16 * 13);
  static thread_local std::vector<float> cout_in(13 * 7680);
  static thread_local std::vector<float> cout_out(13 * 1024);
  static thread_local std::vector<float> chunk(128 * 100, 0.0f);

  size_t out_token_idx = 0;
  for (unsigned int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    unsigned int start_frame = chunk_idx * 100;
    unsigned int chunk_frames = std::min<unsigned int>(100, audio_seq_len - start_frame);

    // Extract chunk [128, 100] (0-padded on right)
    std::fill(chunk.begin(), chunk.end(), 0.0f);
    for (int mel = 0; mel < 128; ++mel) {
      for (unsigned int f = 0; f < chunk_frames; ++f) {
        chunk[mel * 100 + f] = feat[mel * audio_seq_len + start_frame + f];
      }
    }

    // Conv1: [1, 128, 100] -> [480, 64, 50]
    conv2d_im2col_sgemm(chunk.data(), 1, 128, 100,
                        w1, b1, 480, 3, 3, 2, 1,
                        out1.data(), 64, 50, col1.data());

    // Conv2: [480, 64, 50] -> [480, 32, 25]
    conv2d_im2col_sgemm(out1.data(), 480, 64, 50,
                        w2, b2, 480, 3, 3, 2, 1,
                        out2.data(), 32, 25, col2.data());

    // Conv3: [480, 32, 25] -> [480, 16, 13]
    conv2d_im2col_sgemm(out2.data(), 480, 32, 25,
                        w3, b3, 480, 3, 3, 2, 1,
                        out3.data(), 16, 13, col3.data());

    // Permute: out3 [480, 16, 13] -> cout_in [13, 480, 16] = [13, 7680]
    for (int c = 0; c < 480; ++c) {
      for (int h = 0; h < 16; ++h) {
        for (int t = 0; t < 13; ++t) {
          cout_in[t * 7680 + c * 16 + h] = out3[(c * 16 + h) * 13 + t];
        }
      }
    }

    // Conv_out: cout_out [13, 1024] = cout_in [13, 7680] * w_cout^T [7680, 1024]
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                13, 1024, 7680,
                1.0f, cout_in.data(), 7680,
                w_cout, 7680,
                0.0f, cout_out.data(), 1024);

    // Add positional embedding pos_embed [13, 1024]
    for (int t = 0; t < 13; ++t) {
      for (int d = 0; d < 1024; ++d) {
        cout_out[t * 1024 + d] += pos_embed[t * 1024 + d];
      }
    }

    // Log intermediate layers for chunk 0 to match PyTorch layer-by-layer debug format
    if (chunk_idx == 0) {
      auto print_compact = [](const std::string &name, const std::string &type,
                              const std::string &io_label, int idx,
                              unsigned int batch, unsigned int channel,
                              unsigned int height, unsigned int width,
                              const float *data) {
        std::cout << "\n[Layer Activity] Name: " << name << " Type: " << type
                  << " | from: 0 | to: " << height << std::endl;
        std::cout << "  -> " << io_label << " " << idx << " | Dim: ("
                  << batch << "," << channel << "," << height << "," << width << ")" << std::endl;

        auto print_r = [&](size_t r) {
          const float *row = data + r * width;
          std::cout << "    [Row " << r << "]: ";
          if (width <= 6) {
            for (size_t k = 0; k < width; ++k) std::cout << row[k] << ", ";
          } else {
            std::cout << row[0] << ", " << row[1] << ", " << row[2] << " ... "
                      << row[width - 3] << ", " << row[width - 2] << ", " << row[width - 1];
          }
          std::cout << std::endl;
        };

        if (height <= 6) {
          for (size_t r = 0; r < height; ++r) print_r(r);
        } else {
          for (size_t r = 0; r < 3; ++r) print_r(r);
          std::cout << "    ..." << std::endl;
          for (size_t r = height - 3; r < height; ++r) print_r(r);
        }
      };

      print_compact("audio_tower_conv1", "conv2d", "Input", 0, num_chunks, 1, 128, 100, chunk.data());
      print_compact("audio_tower_conv1", "conv2d", "Output", 0, num_chunks, 480, 64, 50, out1.data());

      print_compact("audio_tower_conv2", "conv2d", "Input", 0, num_chunks, 480, 64, 50, out1.data());
      print_compact("audio_tower_conv2", "conv2d", "Output", 0, num_chunks, 480, 32, 25, out2.data());

      print_compact("audio_tower_conv3", "conv2d", "Input", 0, num_chunks, 480, 32, 25, out2.data());
      print_compact("audio_tower_conv3", "conv2d", "Output", 0, num_chunks, 480, 16, 13, out3.data());

      print_compact("audio_tower_conv_out", "fully_connected", "Input", 0, num_chunks, 1, 13, 7680, cout_in.data());
      print_compact("audio_tower_conv_out", "fully_connected", "Output", 0, num_chunks, 1, 13, 1024, cout_out.data());

      print_compact("audio_tower_pos_add", "addition", "Output", 0, num_chunks, 1, 13, 1024, cout_out.data());
    }

    // Valid tokens: 13 for full chunks, remainder for tail chunk
    unsigned int valid_tokens = (chunk_idx + 1 < num_chunks) ? 13 : (downsampled_len - out_token_idx);
    for (unsigned int t = 0; t < valid_tokens; ++t) {
      _Float16 *dst = out_ptr + out_token_idx * 1024;
      const float *src = cout_out.data() + t * 1024;
      for (int d = 0; d < 1024; ++d) {
        dst[d] = static_cast<_Float16>(src[d]);
      }
      out_token_idx++;
    }
  }

  std::cout << "[AudioTowerSubsampler] Processed " << num_chunks << " chunks, output tokens: "
            << out_token_idx << " / " << downsampled_len << std::endl;
}

} // namespace quick_ai
