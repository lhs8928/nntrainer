pushd ../../../test/input_gen
python3 transformer.py
mv transformer_* ../../build/Applications/Transformer
popd