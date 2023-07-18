work_dir=$HOME/Projects/llm-mixed-q
env_name=mase-sw

quant_config=$work_dir/experiments/asplos/configs/quantize/bypass.toml
run_dir=$work_dir/experiments/asplos/fine_tune
cd $run_dir

# conda run -n $env_name accelerate launch --multi_gpu fine_tune.py \
conda run -n $env_name python fine_tune.py \
    --model_arch bert \
    --model_name bert-base-uncased \
    --task sst2 \
    --quant_config $quant_config \
    --batch_size_train 16 \
    --batch_size_eval 64 \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --output_dir $run_dir/bert_base_sst2 \
    --project_name bert_base_sst2
