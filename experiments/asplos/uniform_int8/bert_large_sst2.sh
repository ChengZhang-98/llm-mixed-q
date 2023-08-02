#!/bin/bash
if [ -z $1 ]; then
    echo "❗Requires <save_tag> as \$1"
    exit
fi

work_dir=$HOME/Projects/llm-mixed-q
env_name=llm-mixed-q
run_dir=$work_dir/experiments/asplos/uniform_int8
cd $run_dir

echo "🚀 Running Uniform Int8 BERT-large on SST2"
model_arch=bert
model_name=$work_dir/checkpoints/asplos/fine_tune/bert_large_sst2
save_tag=$1
stat_toml=$work_dir/checkpoints/asplos/profile_statistics/bert_large_sst2/fp32/statistic_profile.toml
save_dir=$work_dir/checkpoints/asplos/uniform_int8/bert_large_sst2/$save_tag && mkdir -p $save_dir
quant_config_toml=$save_dir/quant_config.toml

range_entry="range_min_max"
integer_width="8"

echo "🔵 transforming statistic profile to q config"

conda run -n llm-mixed-q --no-capture-output python transform_stat_profile_to_config.py \
    --model_arch $model_arch \
    --model_name $model_name \
    --statistic_profile $stat_toml \
    --range_entry $range_entry \
    --save_name $quant_config_toml \
    --integer_width $integer_width

echo "🔵 evaluating"

task=sst2
batch_size=128
max_length=196

conda run -n llm-mixed-q --no-capture-output python eval_cls.py \
    --model_arch $model_arch \
    --model_name $model_name \
    --quant_config $quant_config_toml \
    --task $task \
    --batch_size $batch_size \
    --max_length $max_length \
    --save_dir $save_dir \
    --dataset_split validation

echo "✅ Done"
