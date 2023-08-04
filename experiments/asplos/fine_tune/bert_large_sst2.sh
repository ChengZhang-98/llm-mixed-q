#!/bin/bash

work_dir=$HOME/Projects/llm-mixed-q
env_name=llm-mixed-q

quant_config=$work_dir/experiments/asplos/configs/quantize/bypass.toml
run_dir=$work_dir/experiments/asplos/fine_tune
ckpt=$work_dir/checkpoints/asplos/fine_tune/bert_large_sst2
mkdir -p $ckpt
cd $run_dir

echo ========== Running BERT large SST2 ==========
conda run -n $env_name --no-capture-output accelerate launch --multi_gpu fine_tune.py \
    --model_arch bert \
    --model_name bert-large-uncased \
    --task sst2 \
    --quant_config $quant_config \
    --batch_size_train 32 \
    --batch_size_eval 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --output_dir $ckpt \
    --project_name bert_large_sst2 \
    --with_tracking
echo ========== Done. ==========
