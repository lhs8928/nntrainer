cl_layer_sources = [
  'fc_layer_cl.cpp',
  'addition_layer_cl.cpp',
  'swiglu_cl.cpp',
  'reshape_cl.cpp',
  'rmsnorm_layer_cl.cpp',
  'concat_cl.cpp',
]

foreach s : cl_layer_sources
  nntrainer_sources += meson.current_source_dir() / s
endforeach
