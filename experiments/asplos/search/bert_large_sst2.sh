#!/bin/bash

if [ -z $1 ]; then
    echo "❗Requires <search_tag> as \$1"
    exit
fi

if [ -z $2 ]; then
    echo "❗Requires <search_config> as \$2"
    exit
fi

work_dir=$HOME/Projects/llm-mixed-q
env_name=llm-mixed-q
run_dir=$work_dir/experiments/asplos/search
cd $run_dir
echo ========== Running BERT Large SST2 ==========
search_tag=$1
search_config=$2

save_dir=$work_dir/checkpoints/asplos/search/bert_large_sst2/$search_tag && mkdir -p $save_dir
model_arch=bert
task=sst2
ckpt=$work_dir/checkpoints/asplos/fine_tune/bert_large_sst2
batch_size=128
max_length=196

if [ $USER = "cz98" ]; then
    conda run -n $env_name python search_cls.py \
        --model_arch $model_arch \
        --model_name $ckpt \
        --task $task \
        --batch_size $batch_size \
        --padding max_length \
        --max_length $max_length \
        --save_dir $save_dir \
        --search_config $search_config
else
    python search_cls.py \
        --model_arch $model_arch \
        --model_name $ckpt \
        --task $task \
        --batch_size $batch_size \
        --padding max_length \
        --max_length $max_length \
        --save_dir $save_dir \
        --search_config $search_config
fi
echo ========== Done. ==========
