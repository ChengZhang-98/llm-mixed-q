#!/bin/bash

work_dir=$HOME/Projects/llm-mixed-q
run_dir=$work_dir/experiments/asplos/baselines

if [ -z $1 ]; then
    echo "❗Requires <model_arch> as \$1"
    exit
fi

if [ -z $2 ]; then
    echo "❗Requires <model_name> as \$2"
    exit
fi

model_arch=$1
model_name=$2
# ${s//\//_}
tasks=sst

bypass_quant_config=$work_dir/experiments/asplos/configs/quantize/bypass.toml
save_dir=$work_dir/checkpoints/asplos/baselines/fp32/${model_name//\//_} && mkdir -p $save_dir

conda run -n llm-mixed-q --no-capture-output python eval_prompting_cls.py \
    --model_arch $model_arch \
    --model_name $model_name \
    --quant_config $bypass_quant_config \
    --tasks $tasks \
    --save_dir $save_dir
