# Nemotron 3.5 Transcribe: audio front-end, audio tower and the pipeline that
# feeds the audio embeddings to the Nemotron-H language model (models/nemotron_h).
nemotron3_5_transcribe_src = [
    meson.current_source_dir() / 'audio_frontend.cpp',
    meson.current_source_dir() / 'audio_encoder.cpp',
    meson.current_source_dir() / 'nemotron3_5_transcribe.cpp',
]

nemotron3_5_transcribe_app_src = [
    meson.current_source_dir() / 'main_transcribe.cpp',
]

causallm_src += nemotron3_5_transcribe_src
causallm_inc += include_directories('.')
