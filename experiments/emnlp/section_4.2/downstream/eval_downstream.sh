#!/bin/bash

if [ -z $1 ]; then
    echo "❗Requires <model_arch> as \$1"
    exit
fi
model_arch=$1

if [ -z $2 ]; then
    echo "❗Requires <model_name> as \$2"
    exit
fi
model_name=$2

if [ -z $3 ]; then
    echo "❗Requires <quant_config> as \$3"
    exit
fi
quant_config=$3

if [ -z $4 ]; then
    echo " ❗Requires <tag> as \$4"
    exit
fi
tag=$4

if [ -z "$5" ]; then
    echo " ❗Requires <tasks> as \$5"
    exit
fi
tasks=$5

if [ -z $6 ]; then
    echo " batch_size = 4 by default"
    batch_size=4
else
    echo " batch_size = $6"
    batch_size=$6
fi

work_dir=$HOME/Projects/llm-mixed-q
env_name=llm-mixed-q
run_dir=$work_dir/experiments/emnlp/section_4.2/downstream
save_dir=$work_dir/checkpoints/emnlp/section_4.2/downstream/${model_name//\//_}/$tag && mkdir -p $save_dir

cd $run_dir

echo ========== Downstream: $model_name ==========
cp $quant_config $save_dir/quant_config.toml

conda run -n $env_name --no-capture-output python eval_downstream.py \
    --model_wrapper llm-mixed-q \
    --model_name $model_name \
    --model_arch $model_arch \
    --quant_config $quant_config \
    --tasks $tasks \
    --batch_size $batch_size \
    --save_dir $save_dir
