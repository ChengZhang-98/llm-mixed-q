#!/bin/bash

if [ -z $1 ]; then
    echo "❗Requires <model_arch> as \$1"
    exit
fi

if [ -z $2 ]; then
    echo "❗Requires <model_name> as \$2"
    exit
fi

if [ -z $3 ]; then
    echo "❗Requires <tag> as \$3"
    exit
fi

model_arch=$1
model_name=$2
tag=$3

work_dir=$HOME/Projects/llm-mixed-q
env_name=llm-mixed-q
run_dir=$work_dir/experiments/asplos/baselines
cd $run_dir
echo ========== Running $model_arch : $tag ==========
quant_config=$work_dir/experiments/asplos/configs/quantize/block_size/block_fp.toml

save_dir=$work_dir/checkpoints/asplos/baselines/bfp_8bit/$tag && mkdir -p $save_dir
conda run -n llm-mixed-q --no-capture-output python eval_prompting_cls.py \
    --model_wrapper llm-mixed-q \
    --model_arch $model_arch \
    --model_name $model_name \
    --quant_config $quant_config \
    --tasks sst \
    --save_dir $save_dir \
    --batch_size 32
