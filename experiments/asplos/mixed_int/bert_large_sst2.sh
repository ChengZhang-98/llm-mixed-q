#!/bin/bash
if [ -z $1 ]; then
    echo "❗Requires <search_tag> as \$1"
    exit
fi

work_dir=$HOME/Projects/llm-mixed-q
env_name=llm-mixed-q
run_dir=$work_dir/experiments/asplos/mixed_int
cd $run_dir
echo ========== Running BERT-large SST2 ==========
search_tag=$1
search_config=$2
save_dir=$work_dir/checkpoints/asplos/mixed_int/bert_large_sst2/$search_tag && mkdir -p $save_dir

model_arch=bert
task=sst2
ckpt=$work_dir/checkpoints/asplos/fine_tune/bert_large_sst2
stat_profile=$work_dir/checkpoints/asplos/profile_statistics/bert_large_sst2/fp32/statistic_profile.toml
batch_size=256
max_length=196

conda run -n $env_name --no-capture-output python search_cls.py \
    --model_arch $model_arch \
    --model_name $ckpt \
    --task $task \
    --batch_size $batch_size \
    --padding max_length \
    --max_length $max_length \
    --save_dir $save_dir \
    --search_config $search_config \
    --stat_profile $stat_profile
echo ========== Done. ==========
