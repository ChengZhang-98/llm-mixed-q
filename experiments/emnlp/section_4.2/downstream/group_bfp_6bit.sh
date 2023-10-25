#! /bin/bash

work_dir=$HOME/Projects/llm-mixed-q
run_dir=$work_dir/experiments/emnlp/section_4.2/downstream

declare -a MODEL_NAMES=("huggyllama/llama-7b" "lmsys/vicuna-7b-v1.3" "Cheng98/Acapla-7b")

quant_config=$work_dir/experiments/emnlp/configs/quantization/bfp_6bit.toml
# tasks="arc_easy copa lambada_openai sst piqa"
tasks="sst"

for model_name in "${MODEL_NAMES[@]}"; do
    bash $run_dir/eval_downstream.sh llama $model_name $quant_config bfp_6bit "$tasks" 4

    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo " ❗❗❗ Downstream: $model_name failed ❗❗❗"
        exit $retVal
    fi
done
