#!/bin/bash

work_dir=$HOME/Projects/llm-mixed-q
run_dir=$work_dir/experiments/emnlp/fine_tune
cd $run_dir

echo ========== Running OPT 350M SST2 ==========
quant_config=$work_dir/experiments/emnlp/configs/quantization/bfp_4bit.toml
ckpt=$work_dir/checkpoints/emnlp/fine_tune/opt_350m_sst2 && mkdir -p $ckpt

project_name=opt_350m_sst2
model_arch=opt
model_name=facebook/opt-350m
task=sst2
batch_size_train=16
batch_size_eval=32
learning_rate=2e-5
num_train_epochs=4
gradient_accumulation_steps=4
lr_scheduler_type=cosine

conda run -n llm-mixed-q --no-capture-output python fine_tune.py \
    --model_arch $model_arch \
    --model_name $model_name \
    --task $task \
    --quant_config $quant_config \
    --batch_size_train $batch_size_train \
    --batch_size_eval $batch_size_eval \
    --learning_rate $learning_rate \
    --num_train_epochs $num_train_epochs \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --lr_scheduler_type $lr_scheduler_type \
    --output_dir $ckpt \
    --project_name $project_name \
    --with_tracking
echo ========== Done. ==========
