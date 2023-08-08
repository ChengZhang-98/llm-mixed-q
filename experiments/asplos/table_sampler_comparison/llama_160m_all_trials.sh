#!/bin/bash
if [ -z $1 ]; then
    echo "❗Requires <tag> as \$1"
    exit
fi

if [ -z $2 ]; then
    echo "❗Requires <study_pkl> as \$2"
    exit
fi

work_dir=$HOME/Projects/llm-mixed-q
env_name=llm-mixed-q
run_dir=$work_dir/experiments/asplos/table_sampler_comparison
cd $run_dir
echo ========== Running Llama-160M SST2 ==========
tag=$1
study_pkl=$2
save_dir=$work_dir/checkpoints/asplos/table_sampler_comparison/llama_160m_sst2/all_trials/$tag && mkdir -p $save_dir

model_arch=llama
task=sst2
ckpt=$work_dir/checkpoints/asplos/fine_tune/llama_160m_sst2
batch_size=256
max_length=196

conda run -n $env_name --no-capture-output python eval_all_trials.py \
    --model_arch $model_arch \
    --model_name $ckpt \
    --study $study_pkl \
    --task $task \
    --batch_size $batch_size \
    --max_length $max_length \
    --save_dir $save_dir
echo ========== Done. ==========
