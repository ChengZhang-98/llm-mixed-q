#!/bin/bash

if [ -z $1 ]; then
    echo "❗Requires <profile_tag> as \$1"
    exit
fi

work_dir=$HOME/Projects/llm-mixed-q
env_name=llm-mixed-q
run_dir=$work_dir/experiments/asplos/profile_statistics
cd $run_dir
echo ========== Running Llama-7B SST2 ==========
profile_tag=$1
if [ -z $2 ]; then
    echo "ℹ️ Use bypass.toml as <quant_config> (\$2)"
    quant_config=$work_dir/experiments/asplos/configs/quantize/bypass.toml
else
    quant_config=$2
fi

save_dir=$work_dir/checkpoints/asplos/profile_statistics/llama_7b_wikitext2/$profile_tag && mkdir -p $save_dir

model_arch=llama
model_name=huggyllama/llama-7b
task=wikitext2
batch_size=1
max_length=2048
num_samples=32
act_stats="range_min_max"
weight_stats="range_min_max"

conda run -n $env_name --no-capture-output python profile_statistics_lm.py \
    --model_arch $model_arch \
    --model_name $model_name \
    --quant_config $quant_config \
    --task $task \
    --num_samples $num_samples \
    --batch_size $batch_size \
    --max_length $max_length \
    --save_dir $save_dir \
    --act_stats $act_stats \
    --weight_stats $weight_stats

echo ========== Done. ==========
