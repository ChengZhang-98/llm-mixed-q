#!/bin/bash

work_dir=$HOME/Projects/llm-mixed-q
run_dir=$work_dir/experiments/asplos/fine_tune
cd $run_dir

echo ========== Running OPT 1.3B SST2 ==========
quant_config=$work_dir/experiments/asplos/configs/quantize/bypass.toml
ckpt=$work_dir/checkpoints/asplos/fine_tune/opt_1.3b_sst2 && mkdir -p $ckpt

project_name=opt_1.3b_sst2
model_arch=opt
model_name=facebook/opt-1.3b
task=sst2
batch_size_train=1
batch_size_eval=32
learning_rate=0.6e-5
num_train_epochs=4
gradient_accumulation_steps=24
lr_scheduler_type=cosine

conda run -n llm-mixed-q --no-capture-output accelerate launch --use_fsdp fine_tune_fsdp.py \
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
