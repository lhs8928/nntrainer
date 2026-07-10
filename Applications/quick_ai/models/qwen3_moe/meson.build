qwen3_moe_src = [
    meson.current_source_dir() / 'qwen3_moe_causallm.cpp',
]

qwen3_moe_inc = include_directories('.')

# Define Layers
quick_ai_moe_layer_src_abs = [meson.current_source_dir() / 'qwen_moe_layer.cpp']

quick_ai_moe_layer = shared_library(
    'qwen_moe_layer',
    quick_ai_moe_layer_src_abs,
    include_directories: [quick_ai_layer_inc, qwen3_moe_inc],
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

quick_ai_moe_layer_dep = declare_dependency(
    link_with: quick_ai_moe_layer,
    include_directories: qwen3_moe_inc
)

quick_ai_src += qwen3_moe_src
quick_ai_inc += qwen3_moe_inc
quick_ai_layer_dependencies += [quick_ai_moe_layer_dep]
