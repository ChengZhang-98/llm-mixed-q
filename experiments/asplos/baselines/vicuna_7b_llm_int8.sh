#!/bin/bash

if [ -z $1 ]; then
    echo "❗Requires <load_in_nbit> as \$1"
    exit
fi

work_dir=$HOME/Projects/llm-mixed-q
env_name=llm-mixed-q
run_dir=$work_dir/experiments/asplos/baselines
cd $run_dir

echo ========== Running Vicuna-7B WikiText2 ==========
load_in_n_bit=$1
if [ "$load_in_n_bit" == "8" ]; then
    echo "Using LLM.int8() quantisation"
elif [ "$load_in_n_bit" == "4" ]; then
    echo "Using LLM.int4() quantisation"
else
    echo "❗Unsupported load_in_n_bit: $load_in_n_bit"
    exit
fi

quant_config=$2
save_dir=$work_dir/checkpoints/asplos/table_arith_comparison/vicuna_7b_wikitext2/llm_int$load_in_n_bit && mkdir -p $save_dir

model_name="lmsys/vicuna-7b-v1.3"
task=wikitext2
batch_size=1
max_length=2048

conda run -n llm-mixed-q --no-capture-output python eval_perplexity_llm_int8.py \
    --model_name $model_name \
    --task $task \
    --load_in_n_bit $load_in_n_bit \
    --batch_size $batch_size \
    --max_length $max_length \
    --save_dir $save_dir \
    --model_parallelism

echo ========== Done. ==========
