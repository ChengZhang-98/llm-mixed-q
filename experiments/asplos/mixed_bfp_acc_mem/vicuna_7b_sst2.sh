#!/bin/bash

if [ -z $1 ]; then
    echo "❗Requires <search_tag> as \$0"
    exit
fi

if [ -z $2 ]; then
    echo "❗Requires <search_config> as \$1"
    exit
fi

work_dir=$HOME/Projects/llm-mixed-q
env_name=llm-mixed-q
run_dir=$work_dir/experiments/asplos/mixed_int
cd $run_dir
echo ========== Running Vicuna-7B SST2 ==========
search_tag=$1
search_config=$2

save_dir=$work_dir/checkpoints/asplos/mixed_bfp_acc_mem/vicuna_7b/$search_tag && mkdir -p $save_dir
model_arch=llama
tasks=sst
model_name="lmsys/vicuna-7b-v1.3"
limit=128

conda run -n $env_name --no-capture-output python search_prompting_cls.py \
    --model_arch $model_arch \
    --model_name $model_name \
    --tasks $tasks \
    --limit $limit \
    --save_dir $save_dir \
    --search_config $search_config

echo ========== Done. ==========
