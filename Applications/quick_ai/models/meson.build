quick_ai_inc += include_directories('.')

# Transformer base class (required by CED, TimmViT, and LLM models)
quick_ai_src += [
  meson.current_source_dir() / 'transformer.cpp',
]

# CED model (audio classification) - always included
subdir('ced')

# LLM/CausalLM models - only when tokenizer is enabled
quick_ai_use_tokenizer_models = get_option('quick_ai-tokenizer')
if quick_ai_use_tokenizer_models
  quick_ai_src += [
    meson.current_source_dir() / 'causal_lm.cpp',
    meson.current_source_dir() / 'sentence_transformer.cpp',
  ]

  subdir('deberta_v2')
  subdir('gpt_oss')
  subdir('qwen2')
  subdir('qwen3')
  subdir('qwen3_moe')
  subdir('qwen3_slim_moe')
  subdir('gemma3')
  subdir('gemma4')
  subdir('lfm2')
  if get_option('platform') != 'windows'
    subdir('bert')
    subdir('xlm_roberta')
    subdir('gpt_oss_cached_slim')
    subdir('qwen3_cached_slim_moe')
  endif
endif

# Vision models (independent)
subdir('timm_vit')  # Required by CedTransformer
subdir('YOLOv7/jni')
subdir('YOLOv7Pose/jni')
subdir('FastViTKeyword/jni')
