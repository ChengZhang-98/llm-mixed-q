#!/bin/bash
if [ -z $1 ]; then
    echo "❗Requires <save_tag> as \$1"
    exit
fi

work_dir=$HOME/Projects/llm-mixed-q
env_name=llm-mixed-q
run_dir=$work_dir/experiments/asplos/uniform_int8
cd $run_dir

echo "🚀 Running Uniform Int8 OPT-6.7B on Wikitext2"
model_arch=opt
model_name=facebook/opt-6.7b
save_tag=$1
stat_toml=$work_dir/checkpoints/asplos/profile_statistics/opt_6.7b_wikitext2/fp32/statistic_profile.toml
save_dir=$work_dir/checkpoints/asplos/uniform_int8/opt_6.7b/$save_tag && mkdir -p $save_dir
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

tasks=sst
max_length=2048

conda run -n llm-mixed-q --no-capture-output python eval_prompting_cls.py \
    --model_arch $model_arch \
    --model_name $model_name \
    --quant_config $quant_config_toml \
    --tasks $tasks \
    --save_dir $save_dir

echo "✅ Done"
