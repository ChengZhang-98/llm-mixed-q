#/bin/bash

work_dir=$HOME/Projects/llm-mixed-q
run_dir=$work_dir/checkpoints/asplos/monitor_gpu && cd $run_dir
log_dir=$work_dir/checkpoints/asplos/monitor_gpu

# bert-base, bert-large, opt-125m, opt-350m, llama-160m

declare -a ModelName=("bert-base-uncased" "bert-large-uncased" "facebook/opt-125m", "facebook/opt-350m", "llama-160m")
batch_size=128
seq_len=196
warm_up=1024

for model_name in ${ModelName[@]}; do
    echo "🚀 Running $model_name "
    conda run -n llm-mixed-q --no-capture-output python monitor_gpu.py \
        --model_name $model_name \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --warm_up $warm_up \
        --log_dir $log_dir
    echo "✅ $model_name is done "
done
