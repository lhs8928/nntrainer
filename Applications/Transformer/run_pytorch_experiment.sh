#!/bin/bash

function expriment() {
    exp_num=$1

    RESULT_PATH=pytorch_result_$exp_num.txt

    rm $RESULT_PATH

    commit_id=$(git log --oneline -1)
    commit_id=${commit_id:0:8}

    echo $commit_id >> $RESULT_PATH

    for ((i = 1; i < 128; i *= 2))
    do
        batch=$(($i))
        sed -i "s/write_fn(inputs)/# write_fn(inputs)/" test/input_gen/transformer.py
        sed -i "s/write_fn(labels)/# write_fn(labels)/" test/input_gen/transformer.py
        sed -i "s/write_fn(list/# write_fn(list/" test/input_gen/transformer.py
        sed -i "s/print(outputs)/# print(outputs)/" test/input_gen/transformer.py

        sed -i "s/batch_size = 128/batch_size = $batch/" test/input_gen/transformer.py
        git diff >> $RESULT_PATH
        ./Applications/utils/mem_usage.sh python3 test/input_gen/transformer.py &>> $RESULT_PATH
        sed -i "s/batch_size = $batch/batch_size = 128/" test/input_gen/transformer.py

        sed -i "s/# print(outputs)/print(outputs)/" test/input_gen/transformer.py
        sed -i "s/# write_fn(list/write_fn(list/" test/input_gen/transformer.py
        sed -i "s/# write_fn(labels)/write_fn(labels)/" test/input_gen/transformer.py
        sed -i "s/# write_fn(inputs)/write_fn(inputs)/" test/input_gen/transformer.py
    done

}

NNTRAINER_DIR=~/workspace/git/nntrainer
pushd $NNTRAINER_DIR

    for ((en = 1; en <= 3; en++))
    do
        exp_num=$(($en))
        expriment $exp_num
    done

popd
rm log_nntrainer_2023*