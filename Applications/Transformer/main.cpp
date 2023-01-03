#include <app_context.h>
#include <fstream>
#include <model.h>
#include <tensor.h>

#include <iostream>

#define pp(x) std::cerr << #x << "\n" << x << "\n";

const unsigned int batch_size = 128;
const unsigned int num_encoder_layer = 6;
const unsigned int num_decoder_layer = 6;
const unsigned int num_heads = 8;
const unsigned int encoder_timestep = 150;
const unsigned int decoder_timestep = 120;
const unsigned int model_dim = 512;
const unsigned int fc_unit = 2048;

bool swap = false;
bool optimize = true;
bool optimize_attention = true;
bool training = true;

std::shared_ptr<ml::train::Model> genModel() {
  std::shared_ptr<ml::train::Model> model;
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model->setProperty({"batch_size=" + std::to_string(batch_size),
                      swap ? "memory_swap=true" : "memory_swap=false"});

  std::shared_ptr<ml::train::Layer> decoder_input = ml::train::layer::Input(
    {"name=decoder_input",
     "input_shape=1:1:" + std::to_string(decoder_timestep) + ":" +
       std::to_string(model_dim)});
  model->addLayer(decoder_input);

  for (unsigned int i = 0; i < num_decoder_layer; ++i) {
    std::shared_ptr<ml::train::Layer> multiout1 = ml::train::layer::MultiOut(
      {"name=decoder_layer" + std::to_string(i) + "/multi_out1"});
    model->addLayer(multiout1);

    if (optimize) {
      std::string concat_input = "";

      for (unsigned int j = 0; j < num_heads; ++j) {
        std::shared_ptr<ml::train::Layer> masked_multi_head_attention_v_fc =
          ml::train::layer::FullyConnected(
            {"name=decoder_layer" + std::to_string(i) +
               "/masked_multi_head_attention/v_fc" +
               std::to_string(num_heads - 1 - j),
             "input_layers=decoder_layer" + std::to_string(i) + "/multi_out1(" +
               std::to_string(2 * num_heads + j) + ")",
             "unit=" + std::to_string(model_dim / num_heads)});
        model->addLayer(masked_multi_head_attention_v_fc);
      }

      for (unsigned int j = 0; j < num_heads; ++j) {
        std::shared_ptr<ml::train::Layer> masked_multi_head_attention_k_fc =
          ml::train::layer::FullyConnected(
            {"name=decoder_layer" + std::to_string(i) +
               "/masked_multi_head_attention/k_fc" +
               std::to_string(num_heads - 1 - j),
             "input_layers=decoder_layer" + std::to_string(i) + "/multi_out1(" +
               std::to_string(num_heads + j) + ")",
             "unit=" + std::to_string(model_dim / num_heads)});
        model->addLayer(masked_multi_head_attention_k_fc);
      }

      for (unsigned int j = 0; j < num_heads; ++j) {
        std::shared_ptr<ml::train::Layer> masked_multi_head_attention_q_fc =
          ml::train::layer::FullyConnected(
            {"name=decoder_layer" + std::to_string(i) +
               "/masked_multi_head_attention/q_fc" +
               std::to_string(num_heads - 1 - j),
             "input_layers=decoder_layer" + std::to_string(i) + "/multi_out1(" +
               std::to_string(j) + ")",
             "unit=" + std::to_string(model_dim / num_heads)});
        model->addLayer(masked_multi_head_attention_q_fc);
      }

      for (unsigned int j = 0; j < num_heads; ++j) {
        if (optimize_attention) {
          std::shared_ptr<ml::train::Layer> masked_multi_head_attention_bwdp1 =
            ml::train::layer::BatchwiseDotproduct(
              {"name=decoder_layer" + std::to_string(i) +
                 "/masked_multi_head_attention/bwdp1" +
                 std::to_string(num_heads - 1 - j),
               "input_layers=decoder_layer" + std::to_string(i) +
                 "/masked_multi_head_attention/q_fc" +
                 std::to_string(num_heads - 1 - j) + ",decoder_layer" +
                 std::to_string(i) + "/masked_multi_head_attention/k_fc" +
                 std::to_string(num_heads - 1 - j),
               "transpose_key=true", "scaled_dot_product=true",
               "activation=softmax"});
          model->addLayer(masked_multi_head_attention_bwdp1);

          std::shared_ptr<ml::train::Layer> masked_multi_head_attention_bwdp2 =
            ml::train::layer::BatchwiseDotproduct(
              {"name=decoder_layer" + std::to_string(i) +
                 "/masked_multi_head_attention/bwdp2" +
                 std::to_string(num_heads - 1 - j),
               "input_layers=decoder_layer" + std::to_string(i) +
                 "/masked_multi_head_attention/bwdp1" +
                 std::to_string(num_heads - 1 - j) + ",decoder_layer" +
                 std::to_string(i) + "/masked_multi_head_attention/v_fc" +
                 std::to_string(num_heads - 1 - j)});
          model->addLayer(masked_multi_head_attention_bwdp2);

          std::shared_ptr<ml::train::Layer>
            masked_multi_head_attention_attention = ml::train::layer::Identity(
              {"name=decoder_layer" + std::to_string(i) +
                 "/masked_multi_head_attention/attention" +
                 std::to_string(num_heads - 1 - j),
               "input_layers=decoder_layer" + std::to_string(i) +
                 "/masked_multi_head_attention/bwdp2" +
                 std::to_string(num_heads - 1 - j)});
          model->addLayer(masked_multi_head_attention_attention);
        } else {
          std::shared_ptr<ml::train::Layer>
            masked_multi_head_attention_attention = ml::train::layer::Attention(
              {"name=decoder_layer" + std::to_string(i) +
                 "/masked_multi_head_attention/attention" +
                 std::to_string(num_heads - 1 - j),
               "input_layers=decoder_layer" + std::to_string(i) +
                 "/masked_multi_head_attention/q_fc" +
                 std::to_string(num_heads - 1 - j) + ",decoder_layer" +
                 std::to_string(i) + "/masked_multi_head_attention/v_fc" +
                 std::to_string(num_heads - 1 - j) + ",decoder_layer" +
                 std::to_string(i) + "/masked_multi_head_attention/k_fc" +
                 std::to_string(num_heads - 1 - j),
               "scaled_dot_product=true"});
          model->addLayer(masked_multi_head_attention_attention);
        }

        concat_input += "decoder_layer" + std::to_string(i) +
                        "/masked_multi_head_attention/attention" +
                        std::to_string(j);
        if (j != num_heads - 1) {
          concat_input += ",";
        }
      }

      std::shared_ptr<ml::train::Layer> masked_multi_head_attention_concat =
        ml::train::layer::Concat({"name=decoder_layer" + std::to_string(i) +
                                    "/masked_multi_head_attention/concat",
                                  "input_layers=" + concat_input, "axis=3"});
      model->addLayer(masked_multi_head_attention_concat);

      std::shared_ptr<ml::train::Layer> masked_multi_head_attention_fc =
        ml::train::layer::FullyConnected(
          {"name=decoder_layer" + std::to_string(i) +
             "/masked_multi_head_attention/fc",
           "input_layers=decoder_layer" + std::to_string(i) +
             "/masked_multi_head_attention/concat",
           "unit=" + std::to_string(model_dim)});
      model->addLayer(masked_multi_head_attention_fc);

      std::shared_ptr<ml::train::Layer> masked_multi_head_attention =
        ml::train::layer::Identity({"name=decoder_layer" + std::to_string(i) +
                                      "/masked_multi_head_attention",
                                    "input_layers=decoder_layer" +
                                      std::to_string(i) +
                                      "/masked_multi_head_attention/fc"});
      model->addLayer(masked_multi_head_attention);
    } else {
      std::shared_ptr<ml::train::Layer> masked_multi_head_attention =
        ml::train::layer::MultiHeadAttention(
          {"name=decoder_layer" + std::to_string(i) +
             "/masked_multi_head_attention",
           "input_layers=decoder_layer" + std::to_string(i) +
             "/multi_out1(0), decoder_layer" + std::to_string(i) +
             "/multi_out1(1), decoder_layer" + std::to_string(i) +
             "/multi_out1(2)",
           "num_heads=" + std::to_string(num_heads)});
      model->addLayer(masked_multi_head_attention);
    }

    std::shared_ptr<ml::train::Layer> add1 = ml::train::layer::Addition(
      {"name=decoder_layer" + std::to_string(i) + "/add1",
       "input_layers=decoder_layer" + std::to_string(i) + "/multi_out1(" +
         std::to_string(3 * num_heads) + "), decoder_layer" +
         std::to_string(i) + "/masked_multi_head_attention"});
    model->addLayer(add1);

    std::shared_ptr<ml::train::Layer> ln1 =
      ml::train::layer::LayerNormalization(
        {"name=decoder_layer" + std::to_string(i) + "/ln1", "axis=3",
         "epsilon=1e-5"});
    model->addLayer(ln1);

    std::shared_ptr<ml::train::Layer> multiout2 = ml::train::layer::MultiOut(
      {"name=decoder_layer" + std::to_string(i) + "/multi_out2"});
    model->addLayer(multiout2);

    if (optimize) {
      std::string concat_input = "";

      for (unsigned int j = 0; j < num_heads; ++j) {
        std::shared_ptr<ml::train::Layer> multi_head_attention_v_fc =
          ml::train::layer::FullyConnected(
            {"name=decoder_layer" + std::to_string(num_decoder_layer - 1 - i) +
               "/multi_head_attention/v_fc" + std::to_string(num_heads - 1 - j),
             "input_layers=encoder_output(" +
               std::to_string(2 * (num_decoder_layer - i) * num_heads - 1 - j) +
               ")",
             "unit=" + std::to_string(model_dim / num_heads)});
        model->addLayer(multi_head_attention_v_fc);
      }

      for (unsigned int j = 0; j < num_heads; ++j) {
        std::shared_ptr<ml::train::Layer> multi_head_attention_k_fc =
          ml::train::layer::FullyConnected(
            {"name=decoder_layer" + std::to_string(num_decoder_layer - 1 - i) +
               "/multi_head_attention/k_fc" + std::to_string(num_heads - 1 - j),
             "input_layers=encoder_output(" +
               std::to_string(2 * (num_decoder_layer - i) * num_heads - 1 -
                              num_heads - j) +
               ")",
             "unit=" + std::to_string(model_dim / num_heads)});
        model->addLayer(multi_head_attention_k_fc);
      }

      for (unsigned int j = 0; j < num_heads; ++j) {
        std::shared_ptr<ml::train::Layer> multi_head_attention_q_fc =
          ml::train::layer::FullyConnected(
            {"name=decoder_layer" + std::to_string(num_decoder_layer - 1 - i) +
               "/multi_head_attention/q_fc" + std::to_string(num_heads - 1 - j),
             "input_layers=decoder_layer" +
               std::to_string(num_decoder_layer - 1 - i) + "/multi_out2(" +
               std::to_string(j) + ")",
             "unit=" + std::to_string(model_dim / num_heads)});
        model->addLayer(multi_head_attention_q_fc);
      }

      for (unsigned int j = 0; j < num_heads; ++j) {
        if (optimize_attention) {
          std::shared_ptr<ml::train::Layer> multi_head_attention_bwdp1 =
            ml::train::layer::BatchwiseDotproduct(
              {"name=decoder_layer" +
                 std::to_string(num_decoder_layer - 1 - i) +
                 "/multi_head_attention/bwdp1" +
                 std::to_string(num_heads - 1 - j),
               "input_layers=decoder_layer" +
                 std::to_string(num_decoder_layer - 1 - i) +
                 "/multi_head_attention/q_fc" +
                 std::to_string(num_heads - 1 - j) + ",decoder_layer" +
                 std::to_string(num_decoder_layer - 1 - i) +
                 "/multi_head_attention/k_fc" +
                 std::to_string(num_heads - 1 - j),
               "transpose_key=true", "scaled_dot_product=true",
               "activation=softmax"});
          model->addLayer(multi_head_attention_bwdp1);

          std::shared_ptr<ml::train::Layer> multi_head_attention_bwdp2 =
            ml::train::layer::BatchwiseDotproduct(
              {"name=decoder_layer" +
                 std::to_string(num_decoder_layer - 1 - i) +
                 "/multi_head_attention/bwdp2" +
                 std::to_string(num_heads - 1 - j),
               "input_layers=decoder_layer" +
                 std::to_string(num_decoder_layer - 1 - i) +
                 "/multi_head_attention/bwdp1" +
                 std::to_string(num_heads - 1 - j) + ",decoder_layer" +
                 std::to_string(num_decoder_layer - 1 - i) +
                 "/multi_head_attention/v_fc" +
                 std::to_string(num_heads - 1 - j)});
          model->addLayer(multi_head_attention_bwdp2);

          std::shared_ptr<ml::train::Layer> multi_head_attention_attention =
            ml::train::layer::Identity(
              {"name=decoder_layer" +
                 std::to_string(num_decoder_layer - 1 - i) +
                 "/multi_head_attention/attention" +
                 std::to_string(num_heads - 1 - j),
               "input_layers=decoder_layer" +
                 std::to_string(num_decoder_layer - 1 - i) +
                 "/multi_head_attention/bwdp2" +
                 std::to_string(num_heads - 1 - j)});
          model->addLayer(multi_head_attention_attention);
        } else {
          std::shared_ptr<ml::train::Layer> multi_head_attention_attention =
            ml::train::layer::Attention(
              {"name=decoder_layer" +
                 std::to_string(num_decoder_layer - 1 - i) +
                 "/multi_head_attention/attention" +
                 std::to_string(num_heads - 1 - j),
               "input_layers=decoder_layer" +
                 std::to_string(num_decoder_layer - 1 - i) +
                 "/multi_head_attention/q_fc" +
                 std::to_string(num_heads - 1 - j) + ",decoder_layer" +
                 std::to_string(num_decoder_layer - 1 - i) +
                 "/multi_head_attention/v_fc" +
                 std::to_string(num_heads - 1 - j) + ",decoder_layer" +
                 std::to_string(num_decoder_layer - 1 - i) +
                 "/multi_head_attention/k_fc" +
                 std::to_string(num_heads - 1 - j),
               "scaled_dot_product=true"});
          model->addLayer(multi_head_attention_attention);
        }

        concat_input += "decoder_layer" +
                        std::to_string(num_decoder_layer - 1 - i) +
                        "/multi_head_attention/attention" + std::to_string(j);
        if (j != num_heads - 1) {
          concat_input += ",";
        }
      }

      std::shared_ptr<ml::train::Layer> multi_head_attention_concat =
        ml::train::layer::Concat({"name=decoder_layer" +
                                    std::to_string(num_decoder_layer - 1 - i) +
                                    "/multi_head_attention/concat",
                                  "input_layers=" + concat_input, "axis=3"});
      model->addLayer(multi_head_attention_concat);

      std::shared_ptr<ml::train::Layer> multi_head_attention_fc =
        ml::train::layer::FullyConnected(
          {"name=decoder_layer" + std::to_string(num_decoder_layer - 1 - i) +
             "/multi_head_attention/fc",
           "input_layers=decoder_layer" +
             std::to_string(num_decoder_layer - 1 - i) +
             "/multi_head_attention/concat",
           "unit=" + std::to_string(model_dim)});
      model->addLayer(multi_head_attention_fc);

      std::shared_ptr<ml::train::Layer> multi_head_attention =
        ml::train::layer::Identity(
          {"name=decoder_layer" + std::to_string(num_decoder_layer - 1 - i) +
             "/multi_head_attention",
           "input_layers=decoder_layer" +
             std::to_string(num_decoder_layer - 1 - i) +
             "/multi_head_attention/fc"});
      model->addLayer(multi_head_attention);
    } else {
      std::shared_ptr<ml::train::Layer> multi_head_attention =
        ml::train::layer::MultiHeadAttention(
          {"name=decoder_layer" + std::to_string(i) + "/multi_head_attention",
           "input_layers=decoder_layer" + std::to_string(i) +
             "/multi_out2(0), encoder_output(0), encoder_output(1)",
           "num_heads=" + std::to_string(num_heads)});
      model->addLayer(multi_head_attention);
    }

    std::shared_ptr<ml::train::Layer> add2 = ml::train::layer::Addition(
      {"name=decoder_layer" + std::to_string(i) + "/add2",
       "input_layers=decoder_layer" + std::to_string(i) + "/multi_out2(" +
         std::to_string(num_heads) + "), decoder_layer" + std::to_string(i) +
         "/multi_head_attention"});
    model->addLayer(add2);

    std::shared_ptr<ml::train::Layer> ln2 =
      ml::train::layer::LayerNormalization(
        {"name=decoder_layer" + std::to_string(i) + "/ln2", "axis=3",
         "epsilon=1e-5"});
    model->addLayer(ln2);

    std::shared_ptr<ml::train::Layer> multiout3 = ml::train::layer::MultiOut(
      {"name=decoder_layer" + std::to_string(i) + "/multi_out3"});
    model->addLayer(multiout3);

    std::shared_ptr<ml::train::Layer> fc1 = ml::train::layer::FullyConnected(
      {"name=decoder_layer" + std::to_string(i) + "/fc1",
       "input_layers=decoder_layer" + std::to_string(i) + "/multi_out3(0)",
       "unit=" + std::to_string(fc_unit), "activation=relu"});
    model->addLayer(fc1);

    std::shared_ptr<ml::train::Layer> fc2 = ml::train::layer::FullyConnected(
      {"name=decoder_layer" + std::to_string(i) + "/fc2",
       "unit=" + std::to_string(model_dim)});
    model->addLayer(fc2);

    std::shared_ptr<ml::train::Layer> add3 = ml::train::layer::Addition(
      {"name=decoder_layer" + std::to_string(i) + "/add3",
       "input_layers=decoder_layer" + std::to_string(i) +
         "/multi_out3(1), decoder_layer" + std::to_string(i) + "/fc2"});
    model->addLayer(add3);

    std::shared_ptr<ml::train::Layer> ln3 =
      ml::train::layer::LayerNormalization(
        {"name=decoder_layer" + std::to_string(i) + "/ln3", "axis=3",
         "epsilon=1e-5"});
    model->addLayer(ln3);
  }

  std::shared_ptr<ml::train::Layer> decoder_layer_normalization =
    ml::train::layer::LayerNormalization(
      {"name=decoder_layer_normalization", "axis=3", "epsilon=1e-5"});
  model->addLayer(decoder_layer_normalization);

  std::shared_ptr<ml::train::Layer> loss = ml::train::loss::MSE({"name=loss"});
  model->addLayer(loss);

  // encoder

  std::shared_ptr<ml::train::Layer> encoder_input = ml::train::layer::Input(
    {"name=encoder_input", "input_shape=1:" + std::to_string(encoder_timestep) +
                             ":" + std::to_string(model_dim)});
  model->addLayer(encoder_input);

  for (unsigned int i = 0; i < num_encoder_layer; ++i) {
    std::shared_ptr<ml::train::Layer> multi_out1 = ml::train::layer::MultiOut(
      {"name=encoder_layer" + std::to_string(i) + "/multi_out1"});
    model->addLayer(multi_out1);

    if (optimize) {

      std::string concat_input = "";

      for (unsigned int j = 0; j < num_heads; ++j) {
        std::shared_ptr<ml::train::Layer> multi_head_attention_v_fc =
          ml::train::layer::FullyConnected(
            {"name=encoder_layer" + std::to_string(i) +
               "/multi_head_attention/v_fc" + std::to_string(num_heads - 1 - j),
             "input_layers=encoder_layer" + std::to_string(i) + "/multi_out1(" +
               std::to_string(2 * num_heads + j) + ")",
             "unit=" + std::to_string(model_dim / num_heads)});
        model->addLayer(multi_head_attention_v_fc);
      }

      for (unsigned int j = 0; j < num_heads; ++j) {
        std::shared_ptr<ml::train::Layer> multi_head_attention_k_fc =
          ml::train::layer::FullyConnected(
            {"name=encoder_layer" + std::to_string(i) +
               "/multi_head_attention/k_fc" + std::to_string(num_heads - 1 - j),
             "input_layers=encoder_layer" + std::to_string(i) + "/multi_out1(" +
               std::to_string(num_heads + j) + ")",
             "unit=" + std::to_string(model_dim / num_heads)});
        model->addLayer(multi_head_attention_k_fc);
      }

      for (unsigned int j = 0; j < num_heads; ++j) {
        std::shared_ptr<ml::train::Layer> multi_head_attention_q_fc =
          ml::train::layer::FullyConnected(
            {"name=encoder_layer" + std::to_string(i) +
               "/multi_head_attention/q_fc" + std::to_string(num_heads - 1 - j),
             "input_layers=encoder_layer" + std::to_string(i) + "/multi_out1(" +
               std::to_string(j) + ")",
             "unit=" + std::to_string(model_dim / num_heads)});
        model->addLayer(multi_head_attention_q_fc);
      }

      for (unsigned int j = 0; j < num_heads; ++j) {
        if (optimize_attention) {
          std::shared_ptr<ml::train::Layer> multi_head_attention_bwdp1 =
            ml::train::layer::BatchwiseDotproduct(
              {"name=encoder_layer" + std::to_string(i) +
                 "/multi_head_attention/bwdp1" +
                 std::to_string(num_heads - 1 - j),
               "input_layers=encoder_layer" + std::to_string(i) +
                 "/multi_head_attention/q_fc" +
                 std::to_string(num_heads - 1 - j) + ",encoder_layer" +
                 std::to_string(i) + "/multi_head_attention/k_fc" +
                 std::to_string(num_heads - 1 - j),
               "transpose_key=true", "scaled_dot_product=true",
               "activation=softmax"});
          model->addLayer(multi_head_attention_bwdp1);

          std::shared_ptr<ml::train::Layer> multi_head_attention_bwdp2 =
            ml::train::layer::BatchwiseDotproduct(
              {"name=encoder_layer" + std::to_string(i) +
                 "/multi_head_attention/bwdp2" +
                 std::to_string(num_heads - 1 - j),
               "input_layers=encoder_layer" + std::to_string(i) +
                 "/multi_head_attention/bwdp1" +
                 std::to_string(num_heads - 1 - j) + ",encoder_layer" +
                 std::to_string(i) + "/multi_head_attention/v_fc" +
                 std::to_string(num_heads - 1 - j)});
          model->addLayer(multi_head_attention_bwdp2);

          std::shared_ptr<ml::train::Layer> multi_head_attention_attention =
            ml::train::layer::Identity(
              {"name=encoder_layer" + std::to_string(i) +
                 "/multi_head_attention/attention" +
                 std::to_string(num_heads - 1 - j),
               "input_layers=encoder_layer" + std::to_string(i) +
                 "/multi_head_attention/bwdp2" +
                 std::to_string(num_heads - 1 - j)});
          model->addLayer(multi_head_attention_attention);
        } else {
          std::shared_ptr<ml::train::Layer> multi_head_attention_attention =
            ml::train::layer::Attention(
              {"name=encoder_layer" + std::to_string(i) +
                 "/multi_head_attention/attention" +
                 std::to_string(num_heads - 1 - j),
               "input_layers=encoder_layer" + std::to_string(i) +
                 "/multi_head_attention/q_fc" +
                 std::to_string(num_heads - 1 - j) + ",encoder_layer" +
                 std::to_string(i) + "/multi_head_attention/v_fc" +
                 std::to_string(num_heads - 1 - j) + ",encoder_layer" +
                 std::to_string(i) + "/multi_head_attention/k_fc" +
                 std::to_string(num_heads - 1 - j),
               "scaled_dot_product=true"});
          model->addLayer(multi_head_attention_attention);
        }

        concat_input += "encoder_layer" + std::to_string(i) +
                        "/multi_head_attention/attention" + std::to_string(j);
        if (j != num_heads - 1) {
          concat_input += ",";
        }
      }

      std::shared_ptr<ml::train::Layer> multi_head_attention_concat =
        ml::train::layer::Concat({"name=encoder_layer" + std::to_string(i) +
                                    "/multi_head_attention/concat",
                                  "input_layers=" + concat_input, "axis=3"});
      model->addLayer(multi_head_attention_concat);

      std::shared_ptr<ml::train::Layer> multi_head_attention_fc =
        ml::train::layer::FullyConnected(
          {"name=encoder_layer" + std::to_string(i) +
             "/multi_head_attention/fc",
           "input_layers=encoder_layer" + std::to_string(i) +
             "/multi_head_attention/concat",
           "unit=" + std::to_string(model_dim)});
      model->addLayer(multi_head_attention_fc);

      std::shared_ptr<ml::train::Layer> multi_head_attention =
        ml::train::layer::Identity(
          {"name=encoder_layer" + std::to_string(i) + "/multi_head_attention",
           "input_layers=encoder_layer" + std::to_string(i) +
             "/multi_head_attention/fc"});
      model->addLayer(multi_head_attention);
    } else {
      std::shared_ptr<ml::train::Layer> multi_head_attention =
        ml::train::layer::MultiHeadAttention(
          {"name=encoder_layer" + std::to_string(i) + "/multi_head_attention",
           "input_layers=encoder_layer" + std::to_string(i) +
             "/multi_out1(0), encoder_layer" + std::to_string(i) +
             "/multi_out1(1), encoder_layer" + std::to_string(i) +
             "/multi_out1(2)",
           "num_heads=" + std::to_string(num_heads)});
      model->addLayer(multi_head_attention);
    }

    std::shared_ptr<ml::train::Layer> add1 = ml::train::layer::Addition(
      {"name=encoder_layer" + std::to_string(i) + "/add1",
       "input_layers=encoder_layer" + std::to_string(i) + "/multi_out1(" +
         std::to_string(3 * num_heads) + "), encoder_layer" +
         std::to_string(i) + "/multi_head_attention"});
    model->addLayer(add1);

    std::shared_ptr<ml::train::Layer> ln1 =
      ml::train::layer::LayerNormalization(
        {"name=encoder_layer" + std::to_string(i) + "/ln1", "axis=3",
         "epsilon=1e-5"});
    model->addLayer(ln1);

    std::shared_ptr<ml::train::Layer> multi_out2 = ml::train::layer::MultiOut(
      {"name=encoder_layer" + std::to_string(i) + "/multi_out2"});
    model->addLayer(multi_out2);

    std::shared_ptr<ml::train::Layer> fc1 = ml::train::layer::FullyConnected(
      {"name=encoder_layer" + std::to_string(i) + "/fc1",
       "input_layers=encoder_layer" + std::to_string(i) + "/multi_out2(0)",
       "unit=" + std::to_string(fc_unit), "activation=relu"});
    model->addLayer(fc1);

    std::shared_ptr<ml::train::Layer> fc2 = ml::train::layer::FullyConnected(
      {"name=encoder_layer" + std::to_string(i) + "/fc2",
       "unit=" + std::to_string(model_dim)});
    model->addLayer(fc2);

    std::shared_ptr<ml::train::Layer> add2 = ml::train::layer::Addition(
      {"name=encoder_layer" + std::to_string(i) + "/add2",
       "input_layers=encoder_layer" + std::to_string(i) +
         "/multi_out2(1), encoder_layer" + std::to_string(i) + "/fc2"});
    model->addLayer(add2);

    std::shared_ptr<ml::train::Layer> ln2 =
      ml::train::layer::LayerNormalization(
        {"name=encoder_layer" + std::to_string(i) + "/ln2", "axis=3",
         "epsilon=1e-5"});
    model->addLayer(ln2);
  }

  std::shared_ptr<ml::train::Layer> encoder_layer_normalization =
    ml::train::layer::LayerNormalization(
      {"name=encoder_layer_normalization", "axis=3", "epsilon=1e-5"});
  model->addLayer(encoder_layer_normalization);

  std::shared_ptr<ml::train::Layer> encoder_output =
    ml::train::layer::MultiOut({"name=encoder_output"});
  model->addLayer(encoder_output);

  model->setOptimizer(
    ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  model->setProperty(
    {"input_layers=encoder_input, decoder_input", "label_layers=loss"});

  return model;
}

int main() {
  auto model = genModel();

  try {
    model->compile();
  } catch (const std::exception &e) {
    std::cerr << "Error during compile: " << e.what() << "\n";
    return 1;
  }

  try {
    model->initialize();
  } catch (const std::exception &e) {
    std::cerr << "Error during initialize: " << e.what() << "\n";
    return 1;
  }

  std::string NNTRAINER_BASE_PATH = ".";
  std::string base = NNTRAINER_BASE_PATH + "/build/Applications/Transformer/";

  std::string weight_file_name =
    base + (optimize ? "transformer_optimize.bin" : "transformer_origin.bin");
  std::string train_dataset_file_name = base + "transformer_input.dat";

  model->load(weight_file_name, ml::train::ModelFormat::MODEL_FORMAT_BIN);

  if (!training) {
    const unsigned int ENCODER_INPUT_SIZE =
      batch_size * encoder_timestep * model_dim;
    const unsigned int DECODER_INPUT_SIZE =
      batch_size * decoder_timestep * model_dim;

    std::ifstream model_file(train_dataset_file_name,
                             std::ios::in | std::ios::binary);

    float *encoder_input = new float[ENCODER_INPUT_SIZE];
    float *decoder_input = new float[DECODER_INPUT_SIZE];
    float *label = new float[DECODER_INPUT_SIZE];

    model_file.read((char *)encoder_input, ENCODER_INPUT_SIZE * sizeof(float));
    model_file.read((char *)decoder_input, DECODER_INPUT_SIZE * sizeof(float));
    model_file.read((char *)label, DECODER_INPUT_SIZE * sizeof(float));

    std::vector<float *> ret;
    ret = model->inference(batch_size, {encoder_input, decoder_input}, {label});

    nntrainer::Tensor output(
      nntrainer::TensorDim(batch_size, 1, decoder_timestep, model_dim), ret[0]);
    std::cerr << output << "\n";
  } else {
    std::shared_ptr<ml::train::Dataset> train_dataset;
    train_dataset = ml::train::createDataset(ml::train::DatasetType::FILE,
                                             train_dataset_file_name.c_str());

    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                      std::move(train_dataset));

    model->train({"epochs=3"});
  }

  return 0;
}