gpt_oss_cached_slim_src = [
  meson.current_source_dir() / 'gptoss_cached_slim_causallm.cpp',
]

gpt_oss_cached_slim_inc = include_directories('.')

quick_ai_gptoss_moe_layer_cached_src_abs = [meson.current_source_dir() / 'gpt_oss_moe_layer_cached.cpp']

quick_ai_cached_slim_gptoss_moe_layer = shared_library(
    'cached_slim_gptoss_moe_layer',
    quick_ai_gptoss_moe_layer_cached_src_abs,
    include_directories: [quick_ai_layer_inc, gpt_oss_cached_slim_inc],
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

quick_ai_cached_slim_gpt_oss_moe_layer_dep = declare_dependency(
    link_with: quick_ai_cached_slim_gptoss_moe_layer,
    include_directories: gpt_oss_cached_slim_inc
)

quick_ai_src += gpt_oss_cached_slim_src
quick_ai_inc += gpt_oss_cached_slim_inc
quick_ai_layer_dependencies += [quick_ai_cached_slim_gpt_oss_moe_layer_dep]
