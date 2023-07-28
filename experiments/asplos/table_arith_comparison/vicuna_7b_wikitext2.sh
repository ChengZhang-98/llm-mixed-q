#!/bin/bash

if [ -z $1 ]; then
    echo "❗Requires <search_tag> as \$1"
    exit
fi

if [ -z $2 ]; then
    echo "❗Requires <quant_config> as \$2"
    exit
fi

work_dir=$HOME/Projects/llm-mixed-q
env_name=llm-mixed-q
run_dir=$work_dir/experiments/asplos/table_arith_comparison
cd $run_dir

echo ========== Running Vicuna-7B WikiText2 ==========
eval_tag=$1
quant_config=$2
save_dir=$work_dir/checkpoints/asplos/table_arith_comparison/vicuna_7b_wikitext2/$eval_tag && mkdir -p $save_dir

model_arch=llama
model_name="lmsys/vicuna-7b-v1.3"
task=wikitext2
# quant_config=$work_dir/experiments/asplos/configs/quantize/arith/block_fp.toml
batch_size=1
max_length=2048

conda run -n llm-mixed-q --no-capture-output python eval_on_wikitext2.py \
    --model_arch $model_arch \
    --model_name $model_name \
    --task $task \
    --quant_config $quant_config \
    --batch_size $batch_size \
    --max_length $max_length \
    --save_dir $save_dir \
    --model_parallelism

echo ========== Done. ==========
