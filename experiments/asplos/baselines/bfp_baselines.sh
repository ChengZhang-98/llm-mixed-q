#!/bin/bash

if [ -z $1 ]; then
    echo "❗Requires <profile_tag> as \$1"
    exit
fi

work_dir=$HOME/Projects/llm-mixed-q
env_name=llm-mixed-q
run_dir=$work_dir/experiments/asplos/baselines
cd $run_dir
echo ========== Running Vicuna-7B SST2 ==========
profile_tag=$1
if [ -z $2 ]; then
    echo "ℹ️ Use block_fp.toml as <quant_config> (\$2)"
    quant_config=$work_dir/experiments/asplos/configs/quantize/block_size/block_fp.toml
else
    quant_config=$2
fi

model_arch="bert"
declare -a ModelName=("bert_base" "bert_large")
for model_name in ${ModelName[@]}; do
    ckpt_dir=$work_dir/checkpoints/asplos/fine_tune/${model_name}_sst2
    save_dir=$work_dir/checkpoints/asplos/baselines/bfp_8bit/${model_name//\n/_} && mkdir -p $save_dir
    conda run -n llm-mixed-q --no-capture-output python eval_cls.py \
        --model_arch $model_arch \
        --model_name $ckpt_dir \
        --quant_config $quant_config \
        --task sst2 \
        --batch_size 256 \
        --save_dir $save_dir
done
