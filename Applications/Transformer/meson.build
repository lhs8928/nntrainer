transformer_sources = [
  'main.cpp',
]

transformer_dependencies = [app_utils_dep,
  iniparser_dep,
  nntrainer_dep,
  nntrainer_ccapi_dep
]

if get_option('enable-test')
  transformer_dependencies += [gtest_dep]
endif

e = executable('nntrainer_transformer',
  transformer_sources,
  include_directories: [include_directories('.')],
  dependencies: transformer_dependencies,
  install: get_option('install-app'),
  install_dir: application_install_dir
)
