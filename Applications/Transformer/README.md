# NNTrainer Transformer
There are 4 options you can control
1. swap: whether enable memory swap
2. optimize: There are 2 implementations of multi head attention. One for normal multi head attention layer and for other is subgraph composed fc layers
3. optimize_attention: There are 2 implementations of scaled dot product attention. One for normal attention layer and for other is subgraph composed fc layers
4. training: indicate inference or training

## how to generate dataset and weight files
As described at [NNTrainer Transformer](#nntrainer-transformer) there are 2 implementations of multi head attention and these 2 have different weight format.
So you need to set optimize variable in test/input_gen/transLayer_v2.py file. If you want to run with normal multi head attention layer set optimize variable as True else False.
Run gen_input.sh script located in Applications/Transformer/res directory. 
