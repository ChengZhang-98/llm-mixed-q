#!/bin/bash

work_dir=$HOME/Projects/llm-mixed-q
run_dir=$work_dir/experiments/emnlp/section_4.2/perplexity

# declare -a MODEL_NAMES=("huggyllama/llama-7b" "lmsys/vicuna-7b-v1.3" "Cheng98/Acapla-7b")

# for model_name in "${MODEL_NAMES[@]}"; do
#     bash $run_dir/eval_wikitext2_llm_int8.sh $model_name 8 llm_int8
#     retVal=$?
#     if [ $retVal -ne 0 ]; then
#         echo " ❗❗❗ Perplexity: $model_name failed ❗❗❗"
#         exit $retVal
#     fi
# done

# ========= LLama 13B =========
declare -a MODEL_NAMES=("lmsys/vicuna-13b-v1.5")

for model_name in "${MODEL_NAMES[@]}"; do
    bash $run_dir/eval_wikitext2_llm_int8.sh $model_name 8 llm_int8 1 \
        "{'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 2, 'model.layers.28': 2, 'model.layers.29': 2, 'model.layers.30': 2, 'model.layers.31': 2, 'model.layers.32': 2, 'model.layers.33': 2, 'model.layers.34': 2, 'model.layers.35': 2, 'model.layers.36': 2, 'model.layers.37': 2, 'model.layers.38': 2, 'model.layers.39': 2, 'model.norm': 2, 'lm_head': 2}"

    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo " ❗❗❗ Perplexity: $model_name failed ❗❗❗"
        exit $retVal
    fi
done
