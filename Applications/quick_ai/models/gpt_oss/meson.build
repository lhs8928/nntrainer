gpt_oss_src = [
  meson.current_source_dir() / 'gptoss_causallm.cpp',
]

gpt_oss_inc = include_directories('.')

# Define Layers
quick_ai_gptoss_moe_layer_src_abs = [meson.current_source_dir() / 'gpt_oss_moe_layer.cpp']

quick_ai_gptoss_moe_layer = shared_library(
    'gptoss_moe_layer',
    quick_ai_gptoss_moe_layer_src_abs,
    include_directories: [quick_ai_layer_inc, gpt_oss_inc],
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

casuallm_gptoss_moe_layer_dep = declare_dependency(
    link_with: quick_ai_gptoss_moe_layer,
    include_directories: gpt_oss_inc
)

quick_ai_src += gpt_oss_src
quick_ai_inc += gpt_oss_inc
quick_ai_layer_dependencies += [casuallm_gptoss_moe_layer_dep]
