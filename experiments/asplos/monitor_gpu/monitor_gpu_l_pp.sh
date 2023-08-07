#/bin/bash

work_dir=$HOME/Projects/llm-mixed-q
run_dir=$work_dir/experiments/asplos/monitor_gpu && cd $run_dir
log_dir=$work_dir/checkpoints/asplos/monitor_gpu

declare -a ModelName=("facebook/opt-6.7b" "huggyllama/llama-7b" "lmsys/vicuna-7b-v1.3" "Cheng98/Acapla-7b")
batch_size=16
seq_len=196
warm_up=66

for model_name in ${ModelName[@]}; do
    echo "🚀 Running $model_name "
    conda run -n llm-mixed-q --no-capture-output python monitor_gpu.py \
        --model_name "${model_name}" \
        --batch_size $batch_size \
        --seq_len $seq_len \
        --warm_up $warm_up \
        --log_dir $log_dir \
        --model_parallelism
    RESULT=$?
    if [ $RESULT -eq 0 ]; then
        echo "✅ $model_name is done "
    else
        echo "❌ $model_name is failed " && exit 1
    fi

done
