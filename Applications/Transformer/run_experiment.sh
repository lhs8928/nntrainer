#!/bin/bash

OPTIMIZE=false
SWAP=true

if [ "$OPTIMIZE" == "true" ]
then
    if [ "$SWAP" == "true" ]
    then
        RESULT_PATH=disassemble_mha_enable_swap_result.txt
    else
        RESULT_PATH=disassemble_mha_disable_swap_result.txt
    fi
else
    if [ "$SWAP" == "true" ]
    then
        RESULT_PATH=normal_mha_enable_swap_result.txt
    else
        RESULT_PATH=normal_mha_disable_swap_result.txt
    fi
fi

NNTRAINER_DIR=~/workspace/git/nntrainer
pushd $NNTRAINER_DIR
    source venv3_7/bin/activate

    rm $RESULT_PATH
    rm -rf build
    meson build -Dbuildtype=release
    ninja -C build

    commit_id=$(git log --oneline -1)
    commit_id=${commit_id:0:8}

    echo $commit_id >> $RESULT_PATH

    if [ "$OPTIMIZE" == "false" ]
    then
        sed -i "s/optimize = True/optimize = False/" test/input_gen/transLayer_v2.py
        sed -i "s/optimize = true/optimize = false/" Applications/Transformer/main.cpp
    fi

    if [ "$SWAP" == "true" ]
    then
        sed -i "s/swap = false/swap = true/" Applications/Transformer/main.cpp
    fi

    for ((i = 1; i < 256; i *= 2))
    do
        batch=$(($i))
        sed -i "s/batch_size = 128/batch_size = $batch/" test/input_gen/transformer.py
        git diff >> $RESULT_PATH
        pushd Applications/Transformer/res
            ./gen_input.sh
        popd
        sed -i "s/batch_size = $batch/batch_size = 128/" test/input_gen/transformer.py

        sed -i "s/batch_size = 128/batch_size = $batch/" Applications/Transformer/main.cpp
        git diff >> $RESULT_PATH
        ninja -C build && Applications/utils/mem_usage.sh build/Applications/Transformer/nntrainer_transformer &>> $RESULT_PATH
        sed -i "s/batch_size = $batch/batch_size = 128/" Applications/Transformer/main.cpp
    done

    if [ "$SWAP" == "true" ]
    then
        sed -i "s/swap = true/swap = false/" Applications/Transformer/main.cpp
    fi

    if [ "$OPTIMIZE" == "false" ]
    then
        sed -i "s/optimize = false/optimize = true/" Applications/Transformer/main.cpp
        sed -i "s/optimize = False/optimize = True/" test/input_gen/transLayer_v2.py
    fi

popd
rm log_nntrainer_2023*