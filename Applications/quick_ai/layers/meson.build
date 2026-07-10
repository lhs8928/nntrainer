quick_ai_layer_inc_abs = [meson.current_source_dir()]
quick_ai_layer_inc = [include_directories('.')]

quick_ai_rms_norm_src_abs = [meson.current_source_dir() / 'rms_norm.cpp']
quick_ai_swiglu_src_abs = [meson.current_source_dir() / 'swiglu.cpp']
quick_ai_tie_word_embedding_src_abs = [meson.current_source_dir() / 'tie_word_embedding.cpp']
quick_ai_lmhead_src_abs = [meson.current_source_dir() / 'lm_head.cpp']
quick_ai_mha_core_abs = [meson.current_source_dir()/ 'mha_core.cpp']
quick_ai_embedding_src_abs = [meson.current_source_dir() / 'embedding_layer.cpp']
quick_ai_reshaped_rms_norm_src_abs = [meson.current_source_dir() / 'reshaped_rms_norm.cpp']
quick_ai_qkv_layer_src_abs = [meson.current_source_dir() / 'qkv_layer.cpp']
quick_ai_rms_reverse_norm_src_abs = [meson.current_source_dir() / 'rms_reverse_norm.cpp']
quick_ai_embedding_pooling_src_abs = [meson.current_source_dir() / 'embedding_pooling_layer.cpp']
quick_ai_embedding_normalize_src_abs = [meson.current_source_dir() / 'embedding_normalize_layer.cpp']
quick_ai_deberta_attention_src_abs = [meson.current_source_dir() / 'deberta_attention_layer.cpp']
quick_ai_shared_fully_connected_src_abs = [meson.current_source_dir() / 'shared_fully_connected_layer.cpp']
quick_ai_per_layer_slice_src_abs = [meson.current_source_dir() / 'per_layer_slice.cpp']
quick_ai_scalar_multiply_src_abs = [meson.current_source_dir() / 'scalar_multiply.cpp']
quick_ai_logit_softcapping_src_abs = [meson.current_source_dir() / 'logit_softcapping.cpp']

quick_ai_rms_norm = shared_library(
    'rms_norm_layer', 
    quick_ai_rms_norm_src_abs, 
    include_directories: quick_ai_layer_inc, 
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
quick_ai_rms_norm_dep = declare_dependency(
    link_with: quick_ai_rms_norm,
    include_directories: quick_ai_layer_inc
)

quick_ai_swiglu = shared_library(
    'swiglu_layer',
    quick_ai_swiglu_src_abs,
    include_directories: quick_ai_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
quick_ai_swiglu_dep = declare_dependency(
    link_with: quick_ai_swiglu,
    include_directories: quick_ai_layer_inc
)

quick_ai_tie_word_embedding = shared_library(
    'tie_word_embedding_layer',
    quick_ai_tie_word_embedding_src_abs,
    include_directories: quick_ai_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
quick_ai_tie_word_embedding_dep = declare_dependency(
    link_with: quick_ai_tie_word_embedding,
    include_directories: quick_ai_layer_inc
)

quick_ai_lm_head = shared_library(
    'lm_head',
    quick_ai_lmhead_src_abs,
    include_directories: quick_ai_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
quick_ai_lm_head_dep = declare_dependency(
    link_with: quick_ai_lm_head,
    include_directories: quick_ai_layer_inc
)

quick_ai_mha_core = shared_library(
    'mha_core_layer',
    quick_ai_mha_core_abs,
    include_directories: quick_ai_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
quick_ai_mha_core_dep = declare_dependency(
    link_with: quick_ai_mha_core,
    include_directories: quick_ai_layer_inc
)

quick_ai_embedding_layer = shared_library(
    'embedding_layer',
    quick_ai_embedding_src_abs,
    include_directories: quick_ai_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
quick_ai_embedding_layer_dep = declare_dependency(
    link_with: quick_ai_embedding_layer,
    include_directories: quick_ai_layer_inc
)

quick_ai_embedding_pooling_layer = shared_library(
    'embedding_pooling_layer',
    quick_ai_embedding_pooling_src_abs,
    include_directories: quick_ai_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
quick_ai_embedding_pooling_layer_dep = declare_dependency(
    link_with: quick_ai_embedding_pooling_layer,
    include_directories: quick_ai_layer_inc
)

quick_ai_embedding_normalize_layer = shared_library(
    'embedding_normalize_layer',
    quick_ai_embedding_normalize_src_abs,
    include_directories: quick_ai_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

quick_ai_embedding_normalize_layer_dep = declare_dependency(
    link_with: quick_ai_embedding_normalize_layer,
    include_directories: quick_ai_layer_inc
)

quick_ai_reshaped_rms_norm = shared_library(
    'reshaped_rms_norm_layer', 
    quick_ai_reshaped_rms_norm_src_abs, 
    include_directories: quick_ai_layer_inc, 
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
quick_ai_reshaped_rms_norm_dep = declare_dependency(
    link_with: quick_ai_reshaped_rms_norm,
    include_directories: quick_ai_layer_inc
)

quick_ai_qkv_layer = shared_library(
    'qkv_layer',
    quick_ai_qkv_layer_src_abs,
    include_directories: quick_ai_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

quick_ai_qkv_layer_dep = declare_dependency(
    link_with: quick_ai_qkv_layer,
    include_directories: quick_ai_layer_inc
)

quick_ai_deberta_attention_layer = shared_library(
    'deberta_attention_layer',
    quick_ai_deberta_attention_src_abs,
    include_directories: quick_ai_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

quick_ai_deberta_attention_layer_dep = declare_dependency(
    link_with: quick_ai_deberta_attention_layer,
    include_directories: quick_ai_layer_inc
)

quick_ai_shared_fully_connected_layer = shared_library(
    'shared_fully_connected_layer',
    quick_ai_shared_fully_connected_src_abs,
    include_directories: quick_ai_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

quick_ai_shared_fully_connected_layer_dep = declare_dependency(
    link_with: quick_ai_shared_fully_connected_layer,
    include_directories: quick_ai_layer_inc
)

quick_ai_rms_reverse_norm = shared_library(
    'rms_reverse_norm_layer',
    quick_ai_rms_reverse_norm_src_abs,
    include_directories: quick_ai_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

quick_ai_rms_reverse_norm_dep = declare_dependency(
    link_with: quick_ai_rms_reverse_norm,
    include_directories: quick_ai_layer_inc
)

quick_ai_per_layer_slice = shared_library(
    'per_layer_slice_layer',
    quick_ai_per_layer_slice_src_abs,
    include_directories: quick_ai_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)

quick_ai_per_layer_slice_dep = declare_dependency(
    link_with: quick_ai_per_layer_slice,
    include_directories: quick_ai_layer_inc
)

quick_ai_scalar_multiply = shared_library(
    'scalar_multiply_layer',
    quick_ai_scalar_multiply_src_abs,
    include_directories: quick_ai_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
quick_ai_scalar_multiply_dep = declare_dependency(
    link_with: quick_ai_scalar_multiply,
    include_directories: quick_ai_layer_inc
)

quick_ai_logit_softcapping = shared_library(
    'logit_softcapping_layer',
    quick_ai_logit_softcapping_src_abs,
    include_directories: quick_ai_layer_inc,
    dependencies: [nntrainer_dep, nntrainer_ccapi_dep],
    install: true,
    install_dir: application_install_dir
)
quick_ai_logit_softcapping_dep = declare_dependency(
    link_with: quick_ai_logit_softcapping,
    include_directories: quick_ai_layer_inc
)
