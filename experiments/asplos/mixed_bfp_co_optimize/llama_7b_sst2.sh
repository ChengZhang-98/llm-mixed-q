#!/bin/bash

work_dir=$HOME/Projects/llm-mixed-q
env_name=llm-mixed-q
run_dir=$work_dir/experiments/asplos/mixed_bfp_co_optimize

cd $run_dir
if [ -z $1 ]; then
    echo "❗Requires <search_tag> as \$1"
    exit
fi

if [ -z $2 ]; then
    search_config=$work_dir/experiments/asplos/configs/search/bfp_co_optimize/llama_7b_sst2.toml
    echo "❗<serach_config> (\$2) is not provided. Use the one at $search_config"
else
    search_config=$2
fi

echo ========== Running Llama-7B SST2 ==========
search_tag=$1

save_dir=$work_dir/checkpoints/asplos/mixed_bfp_co_optimize/llama_7b/$search_tag && mkdir -p $save_dir
model_arch=llama
tasks=sst
model_name="huggyllama/llama-7b"
limit=128

conda run -n $env_name --no-capture-output python search_prompting_cls.py \
    --model_arch $model_arch \
    --model_name $model_name \
    --tasks $tasks \
    --limit $limit \
    --save_dir $save_dir \
    --search_config $search_config

echo ========== Done. ==========
