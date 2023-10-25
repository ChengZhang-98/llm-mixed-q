#!/bin/bash

if [ -z $1 ]; then
    echo "❗Requires <model_name> as \$1"
    exit
fi
model_name=$1

if [ -z $2 ]; then
    echo "❗Requires <load_in_n_bit> as \$2"
    exit
fi
load_in_n_bit=$2

if [ -z $3 ]; then
    echo " ❗Requires <tag> as \$3"
    exit
fi
tag=$3

if [ -z $4 ]; then
    echo " 🟡 model_parallelism is disabled by default"
else
    echo " 🟢 model_parallelism is enabled"
    model_parallelism=1
fi

if [ -z "$5" ]; then
    device_map="auto"
else
    device_map=$5
fi

work_dir=$HOME/Projects/llm-mixed-q
env_name=llm-mixed-q
run_dir=$work_dir/experiments/emnlp/section_4.2/perplexity
save_dir=$work_dir/checkpoints/emnlp/section_4.2/perplexity/${model_name//\//_}/$tag && mkdir -p $save_dir

cd $run_dir

echo ========== Perplexity: $model_name ==========

task=wikitext2
batch_size=1
max_length=2048

if [ -z $model_parallelism ]; then

    conda run -n $env_name --no-capture-output python eval_wikitext2_llm_int8.py \
        --model_name $model_name \
        --task $task \
        --save_dir $save_dir \
        --load_in_n_bit $load_in_n_bit \
        --batch_size $batch_size \
        --max_length $max_length \
        --dataset_split test # --model_parallelism
else
    conda run -n $env_name --no-capture-output python eval_wikitext2_llm_int8.py \
        --model_name $model_name \
        --task $task \
        --save_dir $save_dir \
        --load_in_n_bit $load_in_n_bit \
        --batch_size $batch_size \
        --max_length $max_length \
        --dataset_split test \
        --model_parallelism --device_map "$device_map"
fi

echo ========== Done. ==========
