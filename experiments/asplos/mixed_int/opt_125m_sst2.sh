#!/bin/bash

if [ -z $1 ]; then
    echo "❗Requires <search_tag> as \$0"
    exit
fi

if [ -z $2 ]; then
    echo "❗Requires <search_config> as \$1"
    exit
fi

work_dir=$HOME/Projects/llm-mixed-q
env_name=llm-mixed-q
run_dir=$work_dir/experiments/asplos/mixed_int
cd $run_dir
echo ========== Running OPT-125M SST2 ==========
search_tag=$1
search_config=$2

save_dir=$work_dir/checkpoints/asplos/mixed_int/opt_125m/$search_tag && mkdir -p $save_dir
stat_profile=$work_dir/checkpoints/asplos/profile_statistics/opt_125m_sst2/fp32/statistic_profile.toml
model_arch=opt
task=sst2
model_name="/home/zz7522/Projects/llm-mixed-q/checkpoints/asplos/fine_tune/opt_125m_sst2"
batch_size=256
max_length=196

conda run -n $env_name --no-capture-output python search_cls.py \
    --model_arch $model_arch \
    --model_name $model_name \
    --task $task \
    --batch_size $batch_size \
    --padding max_length \
    --save_dir $save_dir \
    --stat_profile $stat_profile \
    --search_config $search_config

echo ========== Done. ==========
